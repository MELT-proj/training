#!/usr/bin/env python3
"""Convert Shar sources to Lhotse's indexed layout (see issue #52).

Lhotse 2.0's indexed reader partitions the corpus by *sample index* across the
whole (DP rank x dataloader worker) pool, instead of by shard. That is what
makes a nominal epoch mean 100% of the data, and it works regardless of how few
shards a source has. It needs two things on disk per shard:

  * ``cuts.NNNNNN.jsonl``     -- plain, NOT gzipped. The index is a table of byte
                                offsets into this file, so it is the permanent
                                on-disk form; the ``.gz`` is what goes away.
  * ``*.idx``                 -- 8 bytes per record, next to each data file
                                (or under ``--indexes-root``).

The audio tars are never rewritten -- only read, to record member offsets.

Migration is per source directory and idempotent, so an interrupted run can be
restarted. It is NOT safe to leave a source half-migrated *and* read it with
``indexed=True``: a directory holding both ``cuts.000000.jsonl`` and
``cuts.000001.jsonl.gz`` is rejected outright. Each source is therefore staged
and only swapped in once its whole conversion succeeded. Readers that do not ask
for indexing keep working on a migrated source, and ``from_shar``'s default
(``indexed=None``) auto-detects, so sources can be migrated a few at a time.

Cost, measured on artemis against scratch-nyx:

  * gunzip expansion is 4.1x-11.0x depending on corpus, 6.85x weighted overall;
    the cuts manifests are a rounding error next to the audio either way.
  * indexing is I/O bound, not CPU bound -- 3.3 s of CPU per 2 GB tar, against
    ~90 s to read that tar cold over NFS. Wall time is therefore (total audio
    bytes) / (aggregate read bandwidth), and the right --jobs depends entirely
    on where you run it:

      artemis, over NFS : 31 MB/s single-stream, ~100 MB/s plateau at 8-16 jobs
      nyx, local RAID   : 340 MB/s single-stream, ~520 MB/s peak at 2 jobs,
                          falling to 126 MB/s at 8 -- it is HDD-backed, so
                          concurrency destroys sequential locality

    Run it on nyx with --jobs 2. Measured end-to-end on the real workload there:
    226 MB/s at --jobs 2 versus 42 MB/s at --jobs 8, a 5x loss.

    Do not read spare CPU as headroom. At --jobs 2 each worker sits around 43%
    CPU on a machine reporting 95% idle, which looks like room for more workers;
    it is not. That 43% is the gunzip/tar-scan phase of a pipeline whose I/O
    phase is already at the disk's sequential limit, and adding workers only
    interleaves more read streams. Measure throughput on the real workload, not
    sequential `dd`, and not CPU utilisation.

Usage:
    python infra/index_shar.py --config config/train/SFT-v1.3.0.yaml --jobs 8
    python infra/index_shar.py --root /path/to/shar --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CUTS_GZ = re.compile(r"^cuts\.\d+\.jsonl\.gz$")
CUTS_PLAIN = re.compile(r"^cuts\.\d+\.jsonl$")


def source_dirs_from_config(config_path: Path) -> list[Path]:
    """Pull every `shar_path` out of a training config, in file order.

    Deliberately textual rather than an OmegaConf load: the configs interpolate
    `${oc.env:LOCAL_DATASETS_DIR}`, and requiring that to resolve would mean this
    tool only runs on a machine where the data is already mounted at the training
    path. Here the root is supplied separately.
    """
    text = config_path.read_text()
    paths = re.findall(r"shar_path:\s*(\S+)", text)
    out, seen = [], set()
    for p in paths:
        p = p.replace("${oc.env:LOCAL_DATASETS_DIR}/", "").strip()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return [Path(p) for p in out]


def classify(d: Path) -> str:
    """Where does this source dir stand: untouched, done, or half-converted?"""
    names = {p.name for p in d.iterdir()} if d.is_dir() else set()
    gz = {n for n in names if CUTS_GZ.match(n)}
    plain = {n for n in names if CUTS_PLAIN.match(n)}
    if not gz and not plain:
        return "empty"
    if gz and plain:
        return "mixed"
    if plain:
        idx_missing = [
            n for n in plain if not (d / (n + ".idx")).exists()
        ] + [
            p.name for p in d.glob("*.tar") if not (d / (p.name + ".idx")).exists()
        ]
        return "done" if not idx_missing else "plain-unindexed"
    return "gzipped"


def migrate_one(
    d: Path, keep_gz: bool, indexes_root: Path | None, dry_run: bool
) -> tuple[str, str, float, int]:
    """Gunzip the cuts manifests of one source, index it, drop the .gz.

    Returns (path, status, seconds, bytes_added).
    """
    t0 = time.time()
    state = classify(d)
    if state == "done":
        return (str(d), "already-done", 0.0, 0)
    if state == "empty":
        return (str(d), "skipped-empty", 0.0, 0)
    if state == "mixed":
        # A previous run died between writing a .jsonl and removing its .gz.
        # Recoverable, but not silently: which of the two is authoritative is
        # not knowable from here.
        return (str(d), "ERROR-mixed", 0.0, 0)

    gz_files = sorted(p for p in d.iterdir() if CUTS_GZ.match(p.name))
    if dry_run:
        # 6.85x is the measured whole-collection weighted mean; per corpus it
        # ranges 4.1x-11.0x, so treat a single source's figure as indicative.
        added = int(sum(p.stat().st_size for p in gz_files) * 6.85)
        return (str(d), f"would-convert {len(gz_files)} shards", 0.0, added)

    added = 0
    written: list[Path] = []
    try:
        for gz_path in gz_files:
            out = d / gz_path.name[: -len(".gz")]
            tmp = d / (out.name + ".partial")
            # Stage then rename: a reader must never see a truncated manifest,
            # and a crash mid-write must not leave one behind either.
            with gzip.open(gz_path, "rb") as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=8 << 20)
            os.replace(tmp, out)
            written.append(out)
            added += out.stat().st_size
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        for p in d.glob("*.partial"):
            p.unlink(missing_ok=True)
        raise

    from lhotse.indexing import create_shar_index

    out_dir = None
    if indexes_root is not None:
        out_dir = indexes_root / d.relative_to(d.anchor)
        out_dir.mkdir(parents=True, exist_ok=True)
    create_shar_index(d, output_dir=out_dir)

    idx_home = out_dir if out_dir is not None else d
    added += sum(p.stat().st_size for p in idx_home.glob("*.idx"))

    if not keep_gz:
        for gz_path in gz_files:
            removed = gz_path.stat().st_size
            gz_path.unlink()
            added -= removed

    return (str(d), "converted", time.time() - t0, added)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="training config; migrate every shar_path it names")
    src.add_argument("--root", type=Path, help="migrate every Shar source found under this directory")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("LOCAL_DATASETS_DIR"),
                    help="prefix for the relative paths in --config (default: $LOCAL_DATASETS_DIR)")
    ap.add_argument("--jobs", type=int, default=2,
                    help="sources converted in parallel; more is often much worse on "
                         "spinning disks (2 on nyx, 8-16 over NFS) -- see module docstring")
    ap.add_argument("--keep-gz", action="store_true",
                    help="leave the .gz manifests in place (peak usage is then plain + gz)")
    ap.add_argument("--indexes-root", type=Path, default=None,
                    help="write .idx under this root instead of next to the data")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, touch nothing")
    args = ap.parse_args()

    if args.config:
        if not args.data_root:
            ap.error("--config needs --data-root (or $LOCAL_DATASETS_DIR)")
        dirs = [Path(args.data_root) / p for p in source_dirs_from_config(args.config)]
    else:
        # Match already-converted sources too, so a second run over the same root
        # reports "already-done" rather than claiming there is nothing there.
        dirs = sorted(
            {p.parent for p in args.root.rglob("cuts.*.jsonl.gz")}
            | {p.parent for p in args.root.rglob("cuts.*.jsonl")}
        )

    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print("no Shar sources found", file=sys.stderr)
        return 1

    states: dict[str, int] = {}
    for d in dirs:
        states[classify(d)] = states.get(classify(d), 0) + 1
    print(f"{len(dirs)} sources: " + ", ".join(f"{v} {k}" for k, v in sorted(states.items())))

    t0 = time.time()
    total_added = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(migrate_one, d, args.keep_gz, args.indexes_root, args.dry_run): d
            for d in dirs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                path, status, secs, added = fut.result()
            except Exception as e:  # one bad source must not sink the batch
                failures.append((str(d), repr(e)))
                print(f"[{i}/{len(dirs)}] FAILED {d}: {e}", flush=True)
                continue
            if status.startswith("ERROR"):
                failures.append((path, status))
            total_added += added
            print(f"[{i}/{len(dirs)}] {status:>28}  {secs:7.1f}s  {added/2**30:+8.3f} GB  {path}", flush=True)

    print(f"\nnet change: {total_added / 2**30:+.2f} GB in {(time.time() - t0)/60:.1f} min")
    if failures:
        print(f"\n{len(failures)} source(s) need attention:")
        for path, why in failures:
            print(f"  {why}: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
