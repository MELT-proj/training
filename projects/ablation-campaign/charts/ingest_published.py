#!/usr/bin/env python3
"""Add published numbers of public models to the master from a hand-copied table in ``published/``.

    python ingest_published.py TABLE.csv --who WHO [--dry-run] [--master PATH]

A table is a CSV whose ``#`` lines say where it comes from; its first column names the language and every other
column is a model. ``TABLES`` below says, per table, which set, prompt format and metric its numbers belong to and
describes each model. The copy in ``published/`` stays next to the code so the numbers can be checked against the
paper by eye; to add another published table, drop its CSV there and add an entry to ``TABLES``.

Only numbers that appear in the cited table go in. Nothing is estimated, interpolated or recomputed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import master


HERE = Path(__file__).resolve().parent

LANGUAGE_CODES = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Spanish": "es",
    "Polish": "pl",
    "Ukrainian": "uk",
}

APERTUS_SOURCE = (
    "Apertus 1.5 report, Table 35 (Multilingual Speech Recognition, FLEURS WER); published numbers, copied by hand "
    "from the table as the PI supplied it on 2026-10-16"
)
APERTUS_NOTES = (
    "The Apertus authors ran every model under one greedy decoding setup, so these differ from each model's own "
    "paper. FLEURS split and WER normaliser are not stated in the table."
)

#: table file name -> what its numbers are and who the model columns are.
TABLES = {
    "apertus15-table35-fleurs-wer.csv": {
        "set": "fleurs-reported",
        "fmt": "native",
        "metric": "wer",
        "models": {
            # column -> (label, family, order)
            "Apertus-1.5-8B": ("Apertus 1.5 8B", "apertus-1.5", 1),
            "Apertus-1.5-70B": ("Apertus 1.5 70B", "apertus-1.5", 2),
            "Qwen2.5-Omni-7B": ("Qwen2.5-Omni 7B", "qwen2.5-omni", 3),
            "Qwen2-Audio-7B": ("Qwen2-Audio 7B", "qwen2-audio", 4),
            "Kimi-Audio-7B": ("Kimi-Audio 7B", "kimi-audio", 5),
        },
        "source": APERTUS_SOURCE,
        "notes": APERTUS_NOTES,
    },
}


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """The models (header minus the language column) and the rows of a published table, ``#`` lines skipped."""
    with path.open(newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#") and line.strip()]
    reader = csv.DictReader(lines)
    if not reader.fieldnames or reader.fieldnames[0] != "language":
        raise SystemExit(f"{path}: the first column must be 'language'")
    return list(reader.fieldnames[1:]), list(reader)


def build(path: Path) -> tuple[list[tuple[str, str, str]], dict[str, dict[str, str]]]:
    """``(model, column, value)`` cells and per-model metadata for the table at *path*."""
    spec = TABLES.get(path.name)
    if spec is None:
        raise SystemExit(f"{path.name} is not described in TABLES; add it to ingest_published.py first")
    models, rows = read_table(path)
    if set(models) != set(spec["models"]):
        raise SystemExit(f"{path.name}: columns {models} do not match TABLES {list(spec['models'])}")
    cells, meta = [], {}
    for model in models:
        label, family, order = spec["models"][model]
        meta[model] = {
            "label": label,
            "kind": "baseline",
            "family": family,
            "recipe": "public",
            "order": str(order),
            "source": spec["source"],
            "notes": spec["notes"],
        }
    for row in rows:
        code = LANGUAGE_CODES.get(row["language"])
        if code is None:
            raise SystemExit(f"{path.name}: no language code for {row['language']!r}")
        for model in models:
            if row[model].strip():
                cells.append((model, f"{spec['set']}.{spec['fmt']}.{spec['metric']}.{code}", row[model]))
    return cells, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", type=Path)
    parser.add_argument("--who", required=True)
    parser.add_argument("--master")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cells, meta = build(args.table)
    try:
        report = master.upsert(
            cells,
            who=args.who,
            note=f"published/{args.table.name}",
            meta=meta,
            master=args.master,
            dry_run=args.dry_run,
        )
    except master.MasterError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(("DRY RUN: " if args.dry_run else "") + str(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
