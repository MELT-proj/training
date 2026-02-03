#!/usr/bin/env python3
"""Check for cross-rank data overlap from MELT debug cut-id logs.

This script consumes the JSONL files produced by setting:
- MELT_DEBUG_CUT_IDS_DIR
- MELT_DEBUG_CUT_IDS_MAX_BATCHES
- (optional) MELT_DEBUG_CUT_IDS_EVERY

Each line is expected to be a JSON object containing at least:
- rank: int
- cut_ids: list[str]

It reports:
- which ranks were seen
- total unique cut_ids
- any cut_ids that appear on >1 rank (overlap)
- optional JSON report output

Example:
  python utils/check_cut_id_overlap.py /path/to/debug_ids --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _iter_jsonl_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted([p for p in path.glob(pattern) if p.is_file()])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check overlap of Lhotse cut IDs across DDP ranks from MELT_DEBUG_CUT_IDS_DIR logs.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Directory containing cut_ids.*.jsonl files (or a single JSONL file).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for *.jsonl under path.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, only read the first N files (after sorting).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="If >0, stop after reading N JSONL lines total.",
    )
    parser.add_argument(
        "--max-cut-ids",
        type=int,
        default=0,
        help="If >0, stop after ingesting N cut_ids total.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=25,
        help="Print up to N overlapping cut_id examples.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON report (overlaps and summary).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any cross-rank overlap is found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    files = _iter_jsonl_files(args.path, recursive=bool(args.recursive))
    if not files:
        print(f"No .jsonl files found under: {args.path}", file=sys.stderr)
        return 2

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    # First-seen map to detect cross-rank overlap without storing *all* per-rank sets.
    first_seen_rank: dict[str, int] = {}
    overlap_ranks: dict[str, set[int]] = {}

    # Within-rank repetition (useful to catch unintended repeats).
    rank_seen: dict[int, set[str]] = {}
    within_rank_dups: Counter[int] = Counter()

    # Summary counters.
    ranks_seen: set[int] = set()
    world_sizes_seen: set[int] = set()
    total_lines = 0
    total_cut_ids = 0
    total_records = 0
    bad_lines = 0

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if args.max_lines and total_lines >= args.max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    total_lines += 1

                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue

                    rank = rec.get("rank", None)
                    if rank is None:
                        bad_lines += 1
                        continue

                    try:
                        rank = int(rank)
                    except Exception:
                        bad_lines += 1
                        continue

                    ranks_seen.add(rank)
                    if "world_size" in rec:
                        try:
                            world_sizes_seen.add(int(rec["world_size"]))
                        except Exception:
                            pass

                    cut_ids = rec.get("cut_ids", None)
                    if not isinstance(cut_ids, list):
                        bad_lines += 1
                        continue

                    total_records += 1

                    rs = rank_seen.get(rank)
                    if rs is None:
                        rs = set()
                        rank_seen[rank] = rs

                    for cid in cut_ids:
                        if not isinstance(cid, str):
                            # Be lenient: skip non-string ids.
                            continue

                        total_cut_ids += 1
                        if args.max_cut_ids and total_cut_ids >= args.max_cut_ids:
                            break

                        # within-rank duplicates
                        if cid in rs:
                            within_rank_dups[rank] += 1
                        else:
                            rs.add(cid)

                        # cross-rank overlap
                        prev_rank = first_seen_rank.get(cid)
                        if prev_rank is None:
                            first_seen_rank[cid] = rank
                        else:
                            if prev_rank != rank:
                                s = overlap_ranks.get(cid)
                                if s is None:
                                    overlap_ranks[cid] = {prev_rank, rank}
                                else:
                                    s.add(rank)

                    if args.max_cut_ids and total_cut_ids >= args.max_cut_ids:
                        break

            if (args.max_lines and total_lines >= args.max_lines) or (
                args.max_cut_ids and total_cut_ids >= args.max_cut_ids
            ):
                break

        except OSError as e:
            print(f"Failed to read {file_path}: {e}", file=sys.stderr)
            return 2

    ranks_sorted = sorted(ranks_seen)

    total_unique_cut_ids = len(first_seen_rank)
    num_overlapping_ids = len(overlap_ranks)
    total_unique_by_rank = {str(r): len(rank_seen.get(r, set())) for r in ranks_sorted}

    print("=== MELT cut-id overlap check ===")
    print(f"Files read: {len(files)}")
    print(f"JSONL lines read: {total_lines}")
    print(f"Valid records read: {total_records}")
    print(f"Bad/ignored lines: {bad_lines}")
    print(f"Ranks seen: {ranks_sorted} (n={len(ranks_sorted)})")
    if world_sizes_seen:
        print(f"world_size values seen in logs: {sorted(world_sizes_seen)}")
    print(f"Total cut_ids ingested: {total_cut_ids}")
    print(f"Total unique cut_ids (global): {total_unique_cut_ids}")
    print(f"Overlapping cut_ids across ranks: {num_overlapping_ids}")

    # Within-rank duplicates summary
    if within_rank_dups:
        worst = within_rank_dups.most_common(10)
        print("Within-rank duplicate cut_ids (counts):")
        for r, c in worst:
            print(f"  rank {r}: {c}")

    if num_overlapping_ids > 0:
        # Show a few examples.
        show_n = max(0, int(args.show_examples))
        if show_n:
            print("\nExamples of overlapping cut_ids:")
            for i, (cid, rs) in enumerate(sorted(overlap_ranks.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
                if i >= show_n:
                    break
                print(f"  {cid} -> ranks={sorted(rs)}")

    if args.out_json is not None:
        report = {
            "files_read": [str(p) for p in files],
            "lines_read": total_lines,
            "records_read": total_records,
            "bad_lines": bad_lines,
            "ranks": ranks_sorted,
            "world_sizes_seen": sorted(world_sizes_seen),
            "total_cut_ids": total_cut_ids,
            "total_unique_cut_ids": total_unique_cut_ids,
            "unique_cut_ids_by_rank": total_unique_by_rank,
            "overlapping_cut_ids_count": num_overlapping_ids,
            "overlaps": {cid: sorted(list(rs)) for cid, rs in overlap_ranks.items()},
            "within_rank_duplicate_counts": {str(r): int(c) for r, c in within_rank_dups.items()},
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote report: {args.out_json}")

    if args.strict and num_overlapping_ids > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
