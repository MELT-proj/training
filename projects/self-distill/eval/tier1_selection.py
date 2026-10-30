#!/usr/bin/env python3
"""The campaign's selection score for every scored checkpoint of the MA-utility arms (Tier 1), as ``selection_tier1.json``.

Reads the melt-eval ``.eval`` logs that ``tier1_queue.py`` submitted by their metadata (``sft_run``, ``sft_label``, frozen
set), applies the campaign's own scorer (``projects/ablation-campaign/selection-metric/score.py``, imported through
``eu24_report.py``, never re-implemented) and writes one row per (arm, step): ``{arm, step, run, label, result}`` with
``result`` None while a checkpoint has no complete set of logs. ``tier1_tables.py --selection`` joins them to the tables.
Run in the melt-eval venv (it needs ``inspect_ai``).

    tier1_selection.py --out FILE [--logs-dir DIR] [--since 2026-10-20]

A's final model (step 11,499) was scored for the master under the label ``eu24-repeatfirst``; its row uses that label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tier1_queue import BY_KEY, LABEL, label_of, plan, run_of


FINAL_LABELS = {("A", 11499): LABEL}  # the label the master's scoring of A's final model carries


def scan(logs_dir: str, since: str) -> dict[tuple[str, str, str], Path]:
    """Newest successful log per (sft_run, sft_label, selection set), one pass over the headers."""
    from inspect_ai.log import read_eval_log

    found: dict[tuple[str, str, str], Path] = {}
    for path in sorted(Path(logs_dir).glob("*.eval")):
        if path.name < since:
            continue
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception:  # noqa: BLE001 - a log a running job is still writing is not a complete zip yet
            continue
        meta = log.eval.metadata or {}
        if log.status != "success" or "sft_run" not in meta:
            continue
        frozen = Path(str((log.eval.task_args or {}).get("frozen_set") or "")).name
        if frozen.startswith("selection-"):
            found[(str(meta["sft_run"]), str(meta.get("sft_label")), frozen)] = path
    return found


def evaluate(found: dict, run: str, label: str, report, selection) -> dict | None:
    """Score one run's checkpoint from its logs, or None when its sets are not all there yet."""
    view = {(r, f): p for (r, lab, f), p in found.items() if r == run and lab == label}
    inputs = report.selection_inputs(view, run, selection, None)
    if inputs is None:
        return None
    sets, summaries = inputs
    result = selection.score_checkpoint(sets)
    return {
        "score": result["score"],
        "medians": result["medians"],
        "runaway": selection.runaway_report(summaries),
        "logs": {f: p.name for (r, f), p in view.items()},
    }


def main(argv: list[str] | None = None) -> int:
    import eu24_report as report

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logs-dir", default=report.DEFAULT_LOGS)
    parser.add_argument(
        "--since", default="2026-10-13", help="ignore logs started before this date (file name prefix)"
    )
    args = parser.parse_args(argv)
    selection = report.load_selection_module()
    found = scan(args.logs_dir, args.since)
    rows = []
    for arm, step in [*plan(), (BY_KEY["A"], 11499)]:  # and A's final model, for the T0-against-A comparison
        label = FINAL_LABELS.get((arm.key, step), label_of(step))
        rows.append(
            {
                "arm": arm.key,
                "step": step,
                "run": run_of(arm),
                "label": label,
                "result": evaluate(found, run_of(arm), label, report, selection),
            }
        )
    Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
    scored = [r for r in rows if r["result"]]
    print(f"{len(scored)} of {len(rows)} rows scored; logs scanned: {len(found)}")
    for r in scored:
        print(f"  {r['arm']:3s} @{r['step']:<6d} {r['result']['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
