"""Compare cut ID logs from two training runs to verify dataloader resumption.

After a full run (run1) and a run resumed from its checkpoint (run2), the
batches yielded after that checkpoint should be identical. Comparison is per
**(rank, worker)** stream, and exact: same IDs, same order, same position within
the batch.

    run1_batches[checkpoint_step * grad_accum // workers_per_rank : ]
    starts with
    run2_batches[:]

Two details that the per-rank version of this script got wrong, both invisible
until num_workers > 1:

  - Each worker numbers its batches from 0. Merging a rank's workers and
    de-duplicating on batch_idx therefore discards all but one worker's stream.
  - PyTorch hands batches to workers round-robin, so a worker consumes only its
    1/num_workers share of the rank's micro-batches. The split index has to be
    divided by the worker count; using the rank-wide count overshoots by exactly
    num_workers — the same off-by-num_workers that issue #46 describes in the
    sampler itself.

Run2 is checked as a *prefix* of run1's post-checkpoint stream rather than for
equal length: workers prefetch, so each run logs however many batches its workers
ran ahead to produce, and a shorter resumed run logs fewer.

Files have the naming pattern:
    cut_ids.rank{rank:05d}-ws{world_size:05d}.worker{worker_id:02d}.pid{pid}.jsonl

Additionally, the script verifies that no cut ID is shared between any two
(rank, worker) pairs within a run — i.e. data-parallel sharding and intra-rank
worker partitioning both produce mutually disjoint subsets of the corpus.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"cut_ids\.rank(?P<rank>\d+)-ws(?P<world_size>\d+)\.worker(?P<worker_id>\d+)\.pid(?P<pid>\d+)\.jsonl"
)


def _parse_path(path: Path) -> tuple[int, int] | None:
    """Return (rank, worker_id) parsed from *path*, or None if unrecognised."""
    m = FILENAME_RE.search(path.name)
    return (int(m.group("rank")), int(m.group("worker_id"))) if m else None


def _read_jsonl(
    path: Path,
) -> list[tuple[int, list[str]]]:
    """Read a single JSONL file and return a list of (batch_idx, cut_ids) tuples."""
    records: list[tuple[int, list[str]]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: {path.name}:{lineno}: JSON decode error: {exc}")
                continue
            records.append((record["batch_idx"], record["cut_ids"]))
    return records


def load_batches_by_worker(
    directory: Path,
) -> dict[tuple[int, int], list[list[str]]]:
    """Return {(rank, worker_id): [cut_id_list, ...]} sorted by batch_idx.

    Each (rank, worker_id) key corresponds to a single JSONL file (or the
    merged contents of multiple files sharing that key, e.g. after a restart).
    """
    by_key: dict[tuple[int, int], list[tuple[int, list[str]]]] = defaultdict(list)

    jsonl_files = list(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {directory}")

    for path in jsonl_files:
        key = _parse_path(path)
        if key is None:
            print(f"  WARNING: could not parse rank/worker from {path.name}, skipping")
            continue
        by_key[key].extend(_read_jsonl(path))

    result: dict[tuple[int, int], list[list[str]]] = {}
    for key, entries in by_key.items():
        entries.sort(key=lambda x: x[0])
        seen: set[int] = set()
        deduped: list[list[str]] = []
        for idx, cut_ids in entries:
            if idx not in seen:
                seen.add(idx)
                deduped.append(cut_ids)
        result[key] = deduped

    return result


# ---------------------------------------------------------------------------
# No-overlap check
# ---------------------------------------------------------------------------

def check_no_shared_cut_ids(directory: Path, label: str) -> bool:
    """Verify that every (rank, worker) pair in *directory* sees disjoint cut IDs.

    This catches two classes of bugs:
      - Cross-rank overlap: the same audio cut consumed by two DDP processes.
      - Cross-worker overlap: the same cut consumed by two dataloader workers
        within the same rank.

    Returns True if all pairs are disjoint, False otherwise.
    """
    print(f"\n[no-overlap] Checking {label} ({directory})")
    by_worker = load_batches_by_worker(directory)

    if not by_worker:
        print("  WARNING: no (rank, worker) data found — skipping overlap check")
        return True

    # Build a flat set of all cut IDs per (rank, worker_id)
    all_ids: dict[tuple[int, int], set[str]] = {}
    for key, batches in sorted(by_worker.items()):
        flat: set[str] = set()
        for batch in batches:
            flat.update(batch)
        all_ids[key] = flat
        print(f"  (rank={key[0]}, worker={key[1]}): {len(flat)} unique cut IDs across {len(batches)} batches")

    keys = sorted(all_ids)
    overlap_found = False
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            shared = all_ids[ka] & all_ids[kb]
            if shared:
                overlap_found = True
                examples = sorted(shared)[:5]
                print(
                    f"  OVERLAP between (rank={ka[0]}, worker={ka[1]}) and "
                    f"(rank={kb[0]}, worker={kb[1]}): "
                    f"{len(shared)} shared ID(s), e.g. {examples}"
                )

    if overlap_found:
        print(f"  FAILED: shared cut IDs detected in {label}.")
        return False

    print(f"  OK: all {len(keys)} (rank, worker) pair(s) have disjoint cut ID sets.")
    return True


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare(
    run1_dir: Path,
    run2_dir: Path,
    checkpoint_step: int,
    grad_accum_steps: int,
    check_overlap: bool = True,
) -> bool:
    """Compare the post-checkpoint segment of run1 against run2.

    When *check_overlap* is True (default) also verifies that no cut ID is
    shared between any two (rank, worker) pairs within each run directory.

    Returns True if all checks pass, False otherwise.
    """
    # Compare per (rank, worker), not per rank. Two reasons, both of which made
    # the per-rank comparison structurally wrong once num_workers > 1:
    #
    #   - each worker numbers its batches from 0, so merging a rank's workers and
    #     de-duplicating on batch_idx silently discards all but one worker's
    #     stream;
    #   - PyTorch hands batches to the workers round-robin, so a worker consumes
    #     only its 1/num_workers share of the rank's micro-batches. Skipping the
    #     rank-wide count in a per-worker stream overshoots by num_workers -- the
    #     same off-by-num_workers that issue #46 describes in the sampler itself.
    print(f"Loading run1 batches from: {run1_dir}")
    run1 = load_batches_by_worker(run1_dir)
    for key, batches in sorted(run1.items()):
        print(f"  rank {key[0]} worker {key[1]}: {len(batches)} batches logged")

    print(f"\nLoading run2 batches from: {run2_dir}")
    run2 = load_batches_by_worker(run2_dir)
    for key, batches in sorted(run2.items()):
        print(f"  rank {key[0]} worker {key[1]}: {len(batches)} batches logged")

    if set(run1.keys()) != set(run2.keys()):
        print(
            f"\nERROR: (rank, worker) mismatch between runs.\n"
            f"  run1: {sorted(run1)}\n"
            f"  run2: {sorted(run2)}"
        )
        return False

    workers_per_rank = len({w for _, w in run1}) or 1
    total_microbatches = checkpoint_step * grad_accum_steps
    split_idx, remainder = divmod(total_microbatches, workers_per_rank)
    print(
        f"\nSplit index into each worker's stream: {split_idx}  "
        f"(checkpoint_step={checkpoint_step} × grad_accum={grad_accum_steps} "
        f"÷ {workers_per_rank} worker(s) per rank)"
    )
    if remainder:
        print(
            f"  NOTE: {total_microbatches} micro-batches do not divide evenly across "
            f"{workers_per_rank} workers; workers 0..{remainder - 1} consumed one more. "
            "Comparing on the floor, so the first batch of some streams may be skipped."
        )

    all_ok = True

    for key in sorted(run1.keys()):
        rank, worker = key
        print(f"\n--- Rank {rank}, worker {worker} ---")
        r1_all = run1[key]
        r2_all = run2[key]

        r1_post = r1_all[split_idx:]

        print(f"  run1 total batches : {len(r1_all)}")
        print(f"  run1 post-ckpt     : {len(r1_post)}  (indices {split_idx}..{len(r1_all)-1})")
        print(f"  run2 total batches : {len(r2_all)}")

        if len(r1_post) == 0:
            print(
                f"  WARNING: run1 has no batches after split_idx={split_idx}. "
                "Was the run long enough?"
            )
            all_ok = False
            continue

        # Lengths need not match: workers prefetch, so each run logs however many
        # batches its workers produced before the run ended, which differs between
        # a full run and a shorter resumed one. What must hold is that run2 is a
        # prefix of run1's post-checkpoint stream.
        n_compare = min(len(r1_post), len(r2_all))
        mismatches = 0
        for i in range(n_compare):
            if r1_post[i] != r2_all[i]:
                mismatches += 1
                print(
                    f"  MISMATCH at position {i} (run1 batch {split_idx + i}):\n"
                    f"    run1: {r1_post[i]}\n"
                    f"    run2: {r2_all[i]}"
                )
                if mismatches >= 5:
                    print("  (stopping after 5 mismatches)")
                    break

        if mismatches == 0:
            print(f"  OK: all {n_compare} compared batches match exactly.")
        else:
            print(
                f"  FAILED: {mismatches} mismatch(es) out of {n_compare} compared batches."
            )
            all_ok = False

    if check_overlap:
        if not check_no_shared_cut_ids(run1_dir, "run1"):
            all_ok = False
        if not check_no_shared_cut_ids(run2_dir, "run2"):
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cut ID logs from two training runs to verify sampler resumption."
    )
    parser.add_argument(
        "--run1-dir",
        required=True,
        type=Path,
        help="Directory containing JSONL cut-ID logs from the first (full) run.",
    )
    parser.add_argument(
        "--run2-dir",
        required=True,
        type=Path,
        help="Directory containing JSONL cut-ID logs from the resumed run.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=50,
        help=(
            "Optimizer step at which the checkpoint was saved (default: 50). "
            "Batches 0..(checkpoint_step*grad_accum-1) from run1 are skipped; "
            "the rest must match run2."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps used during training (default: 1).",
    )
    parser.add_argument(
        "--no-check-overlap",
        action="store_true",
        default=False,
        help=(
            "Skip the check that verifies no cut ID is shared between "
            "different (rank, worker) pairs within each run."
        ),
    )
    args = parser.parse_args()

    ok = compare(
        run1_dir=args.run1_dir,
        run2_dir=args.run2_dir,
        checkpoint_step=args.checkpoint_step,
        grad_accum_steps=args.grad_accum_steps,
        check_overlap=not args.no_check_overlap,
    )

    if ok:
        print("\nRESULT: PASS — sampler state is correctly restored on resume.")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL — batches after resume do not match the original run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
