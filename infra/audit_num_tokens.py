#!/usr/bin/env python3
"""Report `custom.num_tokens` coverage across a Shar collection.

The length filters `max_tokens` and `max_tps` read `custom.num_tokens`. Where it
is absent they do not fail — they simply do not filter, so a config that appears
to bound sequence length silently does not, and does so unevenly: a source with
the field is filtered, one without is not, inside the same mixture (issue #59).

Before trusting any mixture, we need to know exactly where the field is missing,
so it can be backfilled. Coverage is reported three ways because they disagree
and each answers a different question:

* **by source** — how many corpora need backfilling (the size of the job).
* **by cut** — how much of the data is already covered (dominated by the few
  huge corpora, so it flatters the situation).
* **partial sources** — a source where only some cuts carry the field is the
  worst case, because nothing about the config or the logs reveals it.

    python3 infra/audit_num_tokens.py \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --sample-cuts 2000 --jobs 16 --json coverage.json

Exits non-zero if any source is missing the field, so it can gate a launch.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _manifests(source: Path) -> list[Path]:
    """Cut manifests in either Shar layout.

    An indexed collection stores plain `cuts.*.jsonl` beside `.idx` byte
    offsets, so it cannot stay compressed; globbing only the gzipped form
    reports an indexed source as empty.
    """
    by_shard: dict[str, Path] = {}
    for pattern in ("cuts.*.jsonl", "cuts.*.jsonl.gz"):
        for path in sorted(source.glob(pattern)):
            by_shard.setdefault(path.name[: path.name.index(".jsonl")], path)
    return [by_shard[k] for k in sorted(by_shard)]


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def scan_source(args: tuple[str, int]) -> dict:
    """Return coverage counts for one source directory."""
    source, sample_cuts = args
    path = Path(source)
    seen = have = 0
    for manifest in _manifests(path):
        if sample_cuts and seen >= sample_cuts:
            break
        try:
            with _open(manifest) as handle:
                for line in handle:
                    if sample_cuts and seen >= sample_cuts:
                        break
                    if not line.strip():
                        continue
                    try:
                        cut = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seen += 1
                    value = (cut.get("custom") or {}).get("num_tokens")
                    if isinstance(value, (int, float)) and value > 0:
                        have += 1
        except OSError:
            continue
    return {"source": source, "seen": seen, "have": have}


def find_sources(root: Path) -> list[str]:
    """Every directory holding cut manifests, at any depth."""
    found = set()
    for pattern in ("cuts.*.jsonl", "cuts.*.jsonl.gz"):
        for hit in glob.iglob(str(root / "**" / pattern), recursive=True):
            found.add(str(Path(hit).parent))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument(
        "--sample-cuts",
        type=int,
        default=2000,
        help="Cuts to read per source (0 = all). A source either carries the "
        "field by construction or does not, so a sample finds the gaps; use 0 "
        "for an exact per-cut census.",
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full per-source table here.")
    args = parser.parse_args()

    sources = find_sources(args.datasets_root)
    if not sources:
        print(f"No Shar sources found under {args.datasets_root}", file=sys.stderr)
        return 2

    print(f"Scanning {len(sources)} sources under {args.datasets_root}")
    print(f"Reading up to {args.sample_cuts or 'all'} cuts each\n")

    tasks = [(s, args.sample_cuts) for s in sources]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(scan_source, tasks, chunksize=4))

    full, partial, none, empty = [], [], [], []
    for row in results:
        seen, have = row["seen"], row["have"]
        if seen == 0:
            empty.append(row)
        elif have == 0:
            none.append(row)
        elif have == seen:
            full.append(row)
        else:
            partial.append(row)

    total_seen = sum(r["seen"] for r in results)
    total_have = sum(r["have"] for r in results)

    def rel(row):
        return row["source"].replace(str(args.datasets_root), "").lstrip("/")

    if partial:
        print("PARTIAL — some cuts carry num_tokens, some do not.")
        print("These are the dangerous ones: nothing in the config or the logs")
        print("reveals that filtering applies to only part of the source.\n")
        for row in sorted(partial, key=lambda r: r["have"] / r["seen"]):
            print(f"  {row['have']:>6}/{row['seen']:<6} {rel(row)}")
        print()

    if none:
        print(f"MISSING — no num_tokens at all ({len(none)} sources):")
        for row in sorted(none, key=lambda r: rel(r)):
            print(f"  {rel(row)}")
        print()

    if empty:
        print(f"NO CUTS READ ({len(empty)} sources) — path typo or unsynced:")
        for row in empty:
            print(f"  {rel(row)}")
        print()

    print("Summary")
    print(f"  sources full     : {len(full)}")
    print(f"  sources partial  : {len(partial)}")
    print(f"  sources missing  : {len(none)}")
    print(f"  sources unreadable: {len(empty)}")
    if total_seen:
        print(
            f"  cuts covered     : {total_have:,}/{total_seen:,} "
            f"({total_have / total_seen:.1%} of cuts sampled)"
        )
    print(
        "\nNote the two numbers disagree by design: a handful of very large\n"
        "corpora carry the field, so per-cut coverage looks high while most\n"
        "*sources* still need backfilling."
    )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nPer-source table written to {args.json}")

    if partial or none:
        print(
            "\nBackfill needed. `preprocessing/data-utils/get_dataset_stats.py`\n"
            "already tokenises with Qwen/Qwen3-1.7B but only aggregates; it has\n"
            "to write the value back per cut.\n"
            "Careful: adding a field rewrites `cuts.*.jsonl`, which invalidates\n"
            "the `.idx` byte offsets beside it. Regenerate them with\n"
            "`infra/index_shar.py` afterwards, or indexed reading will seek to\n"
            "the wrong places."
        )
        return 1

    print("\nEvery source carries num_tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
