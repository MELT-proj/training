#!/usr/bin/env python3
"""Report how many audio hours a run actually fed the model, per language and task.

The ablation campaign compares languages against each other, which is only
meaningful if each language was trained on the same number of hours. Nothing
enforces that by construction: hours come from sampling weights plus a step
budget, so the realized figure drifts with bucketing, filtering, and the length
distribution of each corpus. Two known defects make the drift worse — issue #59
(``max_tokens``/``max_tps`` are inert wherever ``custom.num_tokens`` is absent,
which is most non-YODAS sources) and the epoch-length under-reporting on #46.

So the design must be *checked*, not trusted. Run training with
``MELT_EXPOSURE_DIR`` set, then point this at that directory.

    MELT_EXPOSURE_DIR=/workspace/outputs/$EXP/exposure  # during training
    python3 infra/exposure_audit.py --dir outputs/$EXP/exposure --tolerance 0.05

Exits non-zero if any language deviates from the mean by more than the
tolerance, so it can gate "this arm is done".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(directory: Path) -> tuple[dict[str, float], dict[str, int], int]:
    """Sum every worker's running totals.

    Each file is one worker's snapshot, rewritten in place, so summing across
    files gives the run total. A worker that never flushed contributes nothing,
    which slightly understates the total on short runs.
    """
    seconds: dict[str, float] = {}
    cuts: dict[str, int] = {}
    files = sorted(directory.glob("exposure.*.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key, value in (payload.get("seconds") or {}).items():
            seconds[key] = seconds.get(key, 0.0) + float(value)
        for key, value in (payload.get("cuts") or {}).items():
            cuts[key] = cuts.get(key, 0) + int(value)
    return seconds, cuts, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed relative deviation from the mean, within a task.",
    )
    parser.add_argument(
        "--expected-hours",
        type=float,
        default=None,
        help="Design hours per language. Defaults to the observed mean.",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"error: {args.dir} is not a directory")
        return 2

    seconds, cuts, n_files = load(args.dir)
    if not seconds:
        print(f"No exposure records under {args.dir}. Was MELT_EXPOSURE_DIR set?")
        return 2

    print(f"Exposure from {n_files} worker files under {args.dir}\n")

    by_task: dict[str, list[tuple[str, float]]] = {}
    for key, secs in seconds.items():
        task = key.split(":", 1)[0]
        by_task.setdefault(task, []).append((key, secs / 3600.0))

    failed = False
    total_hours = 0.0

    for task in sorted(by_task):
        entries = sorted(by_task[task], key=lambda kv: kv[0])
        hours = [h for _, h in entries]
        mean = sum(hours) / len(hours)
        target = args.expected_hours if args.expected_hours else mean
        total_hours += sum(hours)

        print(f"[{task}]  {len(entries)} groups, mean {mean:,.1f} h")
        for key, h in entries:
            label = key.split(":", 1)[1]
            deviation = (h - target) / target if target else 0.0
            flag = " " if abs(deviation) <= args.tolerance else "*"
            if flag == "*":
                failed = True
            print(
                f"  {flag} {label:<10} {h:9,.1f} h  "
                f"{cuts.get(key, 0):>9,} cuts  {deviation:+6.1%}"
            )
        print()

    print(f"Total audio seen: {total_hours:,.1f} h")

    if failed:
        print(
            f"\nFAIL: at least one group deviates by more than "
            f"{args.tolerance:.0%}. Cross-language comparisons from this run "
            "are confounded by unequal exposure — report the realized hours "
            "alongside any result, or rebalance and rerun."
        )
        return 1

    print(f"\nOK: every group is within {args.tolerance:.0%} of target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
