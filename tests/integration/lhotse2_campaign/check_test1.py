#!/usr/bin/env python3
"""Read Test 1's cut-id dump and answer what starved partitions actually did.

The question is not whether the run crashed — it is where the tiny sources
ended up. Indexed reading partitions by SAMPLE index, so a source with fewer
cuts than `world_size * num_workers` cannot reach every partition, and the
partitions it does reach receive a handful of cuts that `.repeat()` then
cycles.

Cut IDs are matched against the source manifests, so attribution is exact
rather than inferred from an ID prefix.

Usage:
    python check_test1.py --cut-ids-dir <dir> --shar-root <indexed shar root>
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FILENAME_RE = re.compile(
    r"cut_ids\.rank(?P<rank>\d+)-ws(?P<world_size>\d+)"
    r"\.worker(?P<worker_id>\d+)\.pid(?P<pid>\d+)\.jsonl"
)

# Matches test1_starved_partitions.yaml. `expect_starved` is True for sources
# with fewer cuts than the partition count.
SOURCES = {
    "voxpopuli/de/train": False,
    "voxpopuli/es/train": False,
    "voxpopuli/lt/validation": True,
    "voxpopuli/lt/test": False,
}


def manifest_ids(source_dir: Path) -> set[str]:
    """Every cut ID in a Shar source, from the plain or gzipped manifests."""
    ids: set[str] = set()
    for pattern, opener in (("cuts.*.jsonl", open),
                            ("cuts.*.jsonl.gz", gzip.open)):
        for path in sorted(source_dir.glob(pattern)):
            with opener(path, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        ids.add(json.loads(line)["id"])
    return ids


def load_streams(directory: Path) -> dict[tuple[int, int], list[list[str]]]:
    streams: dict[tuple[int, int], list[tuple[int, list[str]]]] = defaultdict(list)
    for path in sorted(directory.glob("*.jsonl")):
        m = FILENAME_RE.fullmatch(path.name)
        if not m:
            continue
        key = (int(m.group("rank")), int(m.group("worker_id")))
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial final line is normal on a killed job
                streams[key].append((rec["batch_idx"], rec["cut_ids"]))
    out = {}
    for key, rows in streams.items():
        rows.sort(key=lambda r: r[0])
        seen, batches = set(), []
        for idx, cut_ids in rows:
            if idx in seen:
                continue
            seen.add(idx)
            batches.append(cut_ids)
        out[key] = batches
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-ids-dir", type=Path, required=True)
    ap.add_argument("--shar-root", type=Path, required=True)
    args = ap.parse_args()

    streams = load_streams(args.cut_ids_dir)
    if not streams:
        print(f"FAIL: no cut-id files in {args.cut_ids_dir}")
        return 1

    ranks = sorted({r for r, _ in streams})
    workers = sorted({w for _, w in streams})
    partitions = len(streams)
    print(f"streams: {partitions}  (ranks {ranks}, workers {workers})")

    print("\nreading source manifests...")
    id_to_source: dict[str, str] = {}
    counts: dict[str, int] = {}
    for rel in SOURCES:
        ids = manifest_ids(args.shar_root / rel)
        counts[rel] = len(ids)
        for cut_id in ids:
            id_to_source.setdefault(cut_id, rel)
        print(f"  {len(ids):>8,} cuts  {rel}")

    per_stream: dict[tuple[int, int], dict[str, int]] = {}
    distinct: dict[tuple[int, int], dict[str, set]] = {}
    unknown = 0
    for key, batches in sorted(streams.items()):
        tally: dict[str, int] = defaultdict(int)
        uniq: dict[str, set] = defaultdict(set)
        for batch in batches:
            for cut_id in batch:
                src = id_to_source.get(cut_id)
                if src is None:
                    unknown += 1
                    continue
                tally[src] += 1
                uniq[src].add(cut_id)
        per_stream[key] = tally
        distinct[key] = uniq

    print(f"\n=== composition per (rank, worker) — emitted cuts by source ===")
    header = f"{'rank':>5} {'wkr':>4} {'total':>7}"
    for rel in SOURCES:
        header += f" {rel.split('/')[-2] + '/' + rel.split('/')[-1]:>16}"
    print(header)
    for key in sorted(per_stream):
        tally = per_stream[key]
        total = sum(tally.values())
        row = f"{key[0]:>5} {key[1]:>4} {total:>7}"
        for rel in SOURCES:
            n = tally.get(rel, 0)
            d = len(distinct[key].get(rel, ()))
            row += f" {f'{n} ({d}u)':>16}"
        print(row)
    print("  (n = cuts emitted, u = distinct cut IDs behind them)")

    ok = True

    print("\n=== starvation ===")
    for rel, expect_starved in SOURCES.items():
        reached = [k for k in per_stream if per_stream[k].get(rel, 0) > 0]
        share = 100.0 * len(reached) / partitions
        note = ""
        if expect_starved and len(reached) == partitions:
            note = "  <-- expected starvation did NOT occur"
        print(f"  {rel:<28} reached {len(reached):>2}/{partitions} partitions "
              f"({share:.0f}%), source has {counts[rel]:,} cuts{note}")

    print("\n=== disjointness across (rank, worker) ===")
    flat = {k: {c for b in v for c in b} for k, v in streams.items()}
    keys = sorted(flat)
    shared_pairs = 0
    shared_ids = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            common = flat[a] & flat[b]
            if common:
                shared_pairs += 1
                shared_ids += len(common)
                if shared_pairs <= 5:
                    print(f"  SHARED {a} & {b}: {len(common)} ids, "
                          f"e.g. {sorted(common)[:3]}")
    pairs = len(keys) * (len(keys) - 1) // 2
    if shared_pairs:
        ok = False
        print(f"  FAIL: {shared_ids} shared IDs across {shared_pairs}/{pairs} pairs")
    else:
        print(f"  PASS: zero shared cut IDs across all {pairs} pairs")

    empty = [k for k, v in streams.items() if not v]
    if empty:
        ok = False
        print(f"\nFAIL: {len(empty)} stream(s) emitted nothing: {empty}")
    else:
        print(f"\nPASS: all {partitions} streams emitted batches")

    if unknown:
        print(f"\nNOTE: {unknown} emitted cuts matched no configured source "
              f"(check SOURCES against the config)")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
