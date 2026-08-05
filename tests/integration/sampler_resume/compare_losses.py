"""Compare metric histories from two training runs to verify resume correctness.

compare_cut_ids.py answers "did the resumed run see the same batches?".
This script answers the stronger question: "did it end up in the same place?".
Identical batches are necessary but not sufficient — the optimizer, scheduler
and RNG state must also be restored, and only the metrics reveal that.

Both runs are read from HF `trainer_state.json` files (written inside every
checkpoint directory). Only steps STRICTLY AFTER the checkpoint step are
compared: `Trainer` seeds the resumed run's `log_history` with the history it
restored from the checkpoint, so the entries at or below that step are copies,
not independent measurements, and comparing them would always pass.

Exact equality is the wrong bar. Two from-scratch runs of this pipeline, same
seed and verifiably the same batches, are observed to disagree on the loss from
the first logged step onward, and the gap compounds through the optimizer.
Demanding bit-identical losses from a resume therefore measures the training
loop's nondeterminism, not the resume.

Pass --baseline-state with a third run — from scratch, same config as run 1 —
to measure that floor instead of assuming it. The resume passes when it
reproduces run 1 at least as closely as the replica does. Without a baseline
the comparison falls back to exact equality, which is only meaningful for a
fully deterministic stack.

Usage:
    python compare_losses.py \
        --run1-state RUN1/checkpoint-100/trainer_state.json \
        --run2-state RUN2/checkpoint-100/trainer_state.json \
        --baseline-state RUN3/checkpoint-100/trainer_state.json \
        --checkpoint-step 50
"""

import argparse
import json
import sys
from pathlib import Path

# Metrics worth comparing, in report order. `loss` is the headline number;
# `grad_norm` is the sensitive one (it reacts to a single differing sample
# before the averaged loss visibly moves); `learning_rate` catches a scheduler
# that was restarted rather than resumed.
METRIC_KEYS = ("loss", "grad_norm", "learning_rate", "eval_loss")


def load_history(path: Path) -> dict[int, dict[str, float]]:
    """Return {step: {metric: value}} from a trainer_state.json.

    A single step can appear in several `log_history` entries (train metrics
    and eval metrics are logged separately), so entries are merged by step.
    """
    with path.open(encoding="utf-8") as fh:
        state = json.load(fh)

    history: dict[int, dict[str, float]] = {}
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        metrics = {k: entry[k] for k in METRIC_KEYS if k in entry}
        if metrics:
            history.setdefault(step, {}).update(metrics)
    return history


def max_divergence(
    ha: dict[int, dict[str, float]],
    hb: dict[int, dict[str, float]],
    steps: list[int],
) -> dict[str, float]:
    """Largest absolute difference per metric across *steps*."""
    worst: dict[str, float] = {}
    for step in steps:
        ma, mb = ha.get(step, {}), hb.get(step, {})
        for key in METRIC_KEYS:
            if key in ma and key in mb:
                diff = abs(ma[key] - mb[key])
                worst[key] = max(worst.get(key, 0.0), diff)
    return worst


def compare(
    run1_state: Path,
    run2_state: Path,
    checkpoint_step: int,
    tol: float,
    baseline_state: Path | None = None,
) -> bool:
    print(f"Loading run1 metrics from: {run1_state}")
    h1 = load_history(run1_state)
    print(f"Loading run2 metrics from: {run2_state}")
    h2 = load_history(run2_state)
    h3: dict[int, dict[str, float]] | None = None
    if baseline_state is not None:
        print(f"Loading noise-floor baseline from: {baseline_state}")
        h3 = load_history(baseline_state)

    steps1 = {s for s in h1 if s > checkpoint_step}
    steps2 = {s for s in h2 if s > checkpoint_step}
    print(f"\nComparing steps > {checkpoint_step} "
          f"(entries at or below it are restored copies, not measurements).")
    print(f"  run1 logged steps: {sorted(steps1)}")
    print(f"  run2 logged steps: {sorted(steps2)}")

    all_ok = True

    only1 = steps1 - steps2
    only2 = steps2 - steps1
    if only1 or only2:
        print(f"\n  STEP MISMATCH: only in run1 {sorted(only1)}, only in run2 {sorted(only2)}")
        all_ok = False

    shared = sorted(steps1 & steps2)
    if not shared:
        print(f"\n  FAILED: no logged steps after {checkpoint_step} in common. "
              "Did both runs reach max_steps, and is logging_steps small enough?")
        return False

    # Per-step detail. `resume` is |run1 - run2|; `replica` is |run1 - run3|,
    # i.e. what an identical from-scratch re-run costs.
    header = f"\n  {'step':>6}  {'metric':<14} {'run1':>14} {'run2':>14} {'|resume|':>10}"
    if h3 is not None:
        header += f" {'|replica|':>10}"
    print(header)

    for step in shared:
        for key in METRIC_KEYS:
            if key not in h1[step] or key not in h2[step]:
                if key in h1[step] or key in h2[step]:
                    print(f"  {step:>6}  {key:<14} MISSING in "
                          f"{'run2' if key in h1[step] else 'run1'}")
                    all_ok = False
                continue
            v1, v2 = h1[step][key], h2[step][key]
            line = f"  {step:>6}  {key:<14} {v1:>14.6g} {v2:>14.6g} {abs(v1 - v2):>10.3e}"
            if h3 is not None and key in h3.get(step, {}):
                line += f" {abs(v1 - h3[step][key]):>10.3e}"
            print(line)

    if h3 is None:
        # No measured floor: fall back to exact equality.
        worst_resume = max_divergence(h1, h2, shared)
        print(f"\n  No baseline given — requiring exact agreement (tol={tol}).")
        for key, diff in sorted(worst_resume.items()):
            if diff > tol:
                print(f"  FAILED: {key} differs by up to {diff:.3e}.")
                all_ok = False
        if all_ok:
            print("  OK: every compared metric matches exactly.")
        return all_ok

    baseline_steps = [s for s in shared if s in h3]
    if not baseline_steps:
        print("\n  FAILED: the baseline run logged none of the compared steps.")
        return False

    worst_resume = max_divergence(h1, h2, baseline_steps)
    worst_replica = max_divergence(h1, h3, baseline_steps)

    print(f"\n  Worst-case divergence from run1 over steps {baseline_steps}:")
    print(f"  {'metric':<14} {'resume':>12} {'replica':>12}   verdict")
    for key in METRIC_KEYS:
        if key not in worst_resume or key not in worst_replica:
            continue
        resume_d, replica_d = worst_resume[key], worst_replica[key]
        if resume_d <= max(replica_d, tol):
            verdict = "within noise floor"
        else:
            verdict = "EXCEEDS noise floor"
            all_ok = False
        print(f"  {key:<14} {resume_d:>12.3e} {replica_d:>12.3e}   {verdict}")

    print("\n  'replica' is two identical from-scratch runs disagreeing with each"
          "\n  other, so it is the smallest difference this stack can distinguish"
          "\n  from zero. A resume at or below it is as faithful as re-running.")

    final = shared[-1]
    if "eval_loss" not in h1.get(final, {}):
        print(f"\n  NOTE: no eval_loss logged at the final step ({final}). "
              "Only training loss was compared — attach a validation set and "
              "set eval_steps == max_steps to compare model state directly.")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trainer_state.json metric histories from a full run and a resumed run."
    )
    parser.add_argument("--run1-state", required=True, type=Path,
                        help="trainer_state.json from the uninterrupted run.")
    parser.add_argument("--run2-state", required=True, type=Path,
                        help="trainer_state.json from the resumed run.")
    parser.add_argument("--baseline-state", type=Path, default=None,
                        help=(
                            "trainer_state.json from a from-scratch replica of run 1. "
                            "Supplies the measured run-to-run noise floor; without it the "
                            "comparison demands exact equality."
                        ))
    parser.add_argument("--checkpoint-step", type=int, default=50,
                        help="Step the resume started from; only later steps are compared (default: 50).")
    parser.add_argument("--tol", type=float, default=0.0,
                        help=(
                            "Absolute difference always tolerated (default: 0.0). With a "
                            "baseline, a metric passes if it is within max(tol, replica "
                            "divergence), so this only matters when the replica agrees exactly."
                        ))
    args = parser.parse_args()

    paths = [args.run1_state, args.run2_state]
    if args.baseline_state is not None:
        paths.append(args.baseline_state)
    for path in paths:
        if not path.is_file():
            print(f"ERROR: trainer_state.json not found: {path}")
            sys.exit(2)

    ok = compare(args.run1_state, args.run2_state, args.checkpoint_step,
                 args.tol, args.baseline_state)

    if ok:
        print("\nRESULT: PASS — the resumed run reproduces the reference metrics.")
        sys.exit(0)
    print("\nRESULT: FAIL — the resumed run diverged from the reference run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
