#!/usr/bin/env python3
"""Add the checkpoints of ``projects/self-distill/eval/eu24_report.py`` to the master.

    python ingest_eu24_report.py REPORT.json --fmt {repeatfirst,sft-random} --who WHO
                                 [--models KEY ...] [--dry-run] [--master PATH]

REPORT.json is the ``eu24_report.json`` that eu24_report.py writes for one prompt format. Every per-language number
in it becomes a master cell (WER/CER in percent, ST scores as scored, the selection metric's per-language CER, group
medians and score), one ``master.upsert`` per model, so re-running it on a report that is already in is a no-op and a
changed number is refused unless ``master.py`` is told to overwrite (see PROTOCOL.md). Only reports on a registered
set can be ingested; a test-split report needs its sets registered first (``master.py add-set``).

The ``fmt`` must match the report's own prompt label (``fmt-repeatfirst`` / ``fmt-sft-random``), so a report cannot
be filed under the wrong prompt format by accident.

To add a model that eu24_report.py now scores, add it to ``MODELS`` here with the same key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import master


ST_METRICS = ("chrfpp", "spbleu", "bleu", "chrf")
SOURCE = "melt-eval logs of the saved checkpoint, tabulated by projects/self-distill/eval/eu24_report.py"

#: key -> (label, decoder family, recipe, run = the melt-eval ``sft_run`` metadata, bar order). The keys are
#: eu24_report.MODELS' keys. Recipes: start = standard training on five languages, full-ft = SFT-EU24, adapter-ce =
#: gold-CE control, adapter-sd = instruction-mixed self-distillation, adapter-sd-sift = AZeroS's recipe.
MODELS = {
    "A": ("A (Qwen start)", "qwen3.5-2b", "start", "GOLDW-final-A", 10),
    "SFT-EU24": (
        "SFT-EU24 (Qwen)",
        "qwen3.5-2b",
        "full-ft",
        "SFT-EU24-from-goldw-bd60-ga4-8g-s1337",
        11,
    ),
    "CE24": ("CE24 (Qwen)", "qwen3.5-2b", "adapter-ce", "SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337", 12),
    "SD24": (
        "SD24 (Qwen)",
        "qwen3.5-2b",
        "adapter-sd",
        "SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337",
        13,
    ),
    "SD-SIFT24": (
        "SD-SIFT24 (Qwen)",
        "qwen3.5-2b",
        "adapter-sd-sift",
        "SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337",
        14,
    ),
    "A-EuroLLM": ("A' (EuroLLM start)", "eurollm-1.7b", "start", "EuroLLM-A", 20),
    "SFT-EU24-EuroLLM": (
        "SFT-EU24 (EuroLLM)",
        "eurollm-1.7b",
        "full-ft",
        "SFT-EU24-from-EuroLLM-A-bd60-ga4-8g-s1337",
        21,
    ),
    "CE24-EuroLLM": (
        "CE24 (EuroLLM)",
        "eurollm-1.7b",
        "adapter-ce",
        "SD24-ce-from-EuroLLM-A-2700h-bd64-8g-s1337",
        22,
    ),
    "SD24-EuroLLM": (
        "SD24 (EuroLLM)",
        "eurollm-1.7b",
        "adapter-sd",
        "SD24-azeros-from-EuroLLM-A-mix025-2700h-bd64-8g-s1337",
        23,
    ),
    "SD-SIFT24-EuroLLM": (
        "SD-SIFT24 (EuroLLM)",
        "eurollm-1.7b",
        "adapter-sd-sift",
        "SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337",
        24,
    ),
}


def cells_for(report: dict, key: str, fmt: str) -> list[tuple[str, str, float]]:
    """Every master cell of one model's entry in *report*."""
    entry = report["models"][key]
    asr_set, xen_set, enx_set = (name.strip() for name in report["meta"]["sets_note"].split(","))
    cells: list[tuple[str, str, float]] = []
    if "asr" in entry:
        for metric in ("wer", "cer"):
            cells += [(key, f"{asr_set}.{fmt}.{metric}.{lang}", v * 100) for lang, v in entry["asr"][metric].items()]
    for direction, set_name in (("xen", xen_set), ("enx", enx_set)):
        st = entry.get(direction)
        if not st:
            continue
        for case, suffix in (("cased", ""), ("lower", "-lc")):
            for lang, scores in st[case].items():
                cells += [(key, f"{set_name}.{fmt}.{m}{suffix}.{lang}", scores[m]) for m in ST_METRICS]
        cells.append((key, f"{set_name}.{fmt}.nocaps.all", st["no_capitals_share"] * 100))
    selection = entry.get("selection")
    if selection:
        for group, per_language in selection["languages"].items():
            set_name = "selection-id" if group == "ID" else "selection-fleurs26-dev200"
            cells += [(key, f"{set_name}.{fmt}.cer.{lang}", v * 100) for lang, v in per_language.items()]
        cells.append((key, f"selection-metric.{fmt}.score.all", selection["score"]))
        cells += [(key, f"selection-metric.{fmt}.median.{g.lower()}", v) for g, v in selection["medians"].items()]
        runaway = selection.get("runaway")
        if runaway:
            for group, set_name in (("ID", "selection-id"), ("FLEURS", "selection-fleurs26-dev200")):
                share = 100 * runaway[group]["runaways"] / runaway[group]["samples"]
                cells.append((key, f"{set_name}.{fmt}.runaway.all", share))
    return cells


def note_for(report: dict, key: str, path: Path) -> str:
    """The changelog note: the report, how it was made, and the eval logs the model's numbers came from."""
    meta, entry = report["meta"], report["models"][key]
    logs = ", ".join(f"{d}={entry[d]['log']}" for d in ("asr", "xen", "enx") if d in entry)
    label_map = f", label overrides {meta['label_map']}" if meta.get("label_map") else ""
    return (
        f"{path.name} generated {meta['generated']}, prompt label {meta['label']}{label_map}, melt-eval PR #28 "
        f"@ {meta['eval_code']}; logs: {logs}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--fmt", required=True, choices=("repeatfirst", "sft-random"))
    parser.add_argument("--who", required=True)
    parser.add_argument("--models", nargs="*", help="keys to ingest (default: every model in the report)")
    parser.add_argument("--master")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["meta"].get("label") != f"fmt-{args.fmt}":
        print(f"the report's prompt label is {report['meta'].get('label')!r}, not 'fmt-{args.fmt}'", file=sys.stderr)
        return 2
    keys = args.models or list(report["models"])
    unknown = [k for k in keys if k not in report["models"]]
    unlisted = [k for k in keys if k in report["models"] and k not in MODELS]
    if unknown or unlisted:
        print(f"not in the report: {unknown}; not in MODELS of this script: {unlisted}", file=sys.stderr)
        return 2
    for key in keys:
        label, family, recipe, run, order = MODELS[key]
        meta = {
            key: {
                "label": label,
                "kind": "ours",
                "family": family,
                "recipe": recipe,
                "run": run,
                "order": str(order),
                "source": SOURCE,
            }
        }
        try:
            result = master.upsert(
                cells_for(report, key, args.fmt),
                who=args.who,
                note=note_for(report, key, args.report),
                meta=meta,
                master=args.master,
                dry_run=args.dry_run,
            )
        except master.MasterError as exc:
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(f"{key:<20} {'DRY RUN: ' if args.dry_run else ''}{result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
