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

Usage:
    python compare_losses.py \
        --run1-state RUN1/checkpoint-100/trainer_state.json \
        --run2-state RUN2/checkpoint-100/trainer_state.json \
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


def compare(
    run1_state: Path,
    run2_state: Path,
    checkpoint_step: int,
    tol: float,
) -> bool:
    print(f"Loading run1 metrics from: {run1_state}")
    h1 = load_history(run1_state)
    print(f"Loading run2 metrics from: {run2_state}")
    h2 = load_history(run2_state)

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

    for step in shared:
        m1, m2 = h1[step], h2[step]
        keys = [k for k in METRIC_KEYS if k in m1 or k in m2]
        parts = []
        step_ok = True
        for key in keys:
            if key not in m1 or key not in m2:
                parts.append(f"{key}: MISSING in {'run2' if key in m1 else 'run1'}")
                step_ok = False
                continue
            v1, v2 = m1[key], m2[key]
            diff = abs(v1 - v2)
            if diff > tol:
                parts.append(f"{key}: {v1} vs {v2} (diff {diff:.3e})  <-- MISMATCH")
                step_ok = False
            else:
                parts.append(f"{key}: {v1}")
        marker = "  " if step_ok else "X "
        print(f"{marker}step {step:>5}: " + ", ".join(parts))
        if not step_ok:
            all_ok = False

    # The final step is the one that matters most: it is the only point where
    # every optimizer update since the checkpoint has been folded in.
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
    parser.add_argument("--checkpoint-step", type=int, default=50,
                        help="Step the resume started from; only later steps are compared (default: 50).")
    parser.add_argument("--tol", type=float, default=0.0,
                        help=(
                            "Maximum tolerated absolute difference (default: 0.0, exact). "
                            "Trainer rounds logged losses to 4 decimals, so exact equality "
                            "is the expected outcome for a correct resume."
                        ))
    args = parser.parse_args()

    for path in (args.run1_state, args.run2_state):
        if not path.is_file():
            print(f"ERROR: trainer_state.json not found: {path}")
            sys.exit(2)

    ok = compare(args.run1_state, args.run2_state, args.checkpoint_step, args.tol)

    if ok:
        print("\nRESULT: PASS — the resumed run reproduces the reference metrics.")
        sys.exit(0)
    print("\nRESULT: FAIL — the resumed run diverged from the reference run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
