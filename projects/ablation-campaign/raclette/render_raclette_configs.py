#!/usr/bin/env python3
"""Render the Raclette LR-pilot configs from the live Fondue.yaml.

Brief: ``plan/08-raclette.md``. Each config is Fondue's constant-phase config with only
the edits below, every one asserted to match exactly once so a drifted base fails loudly
instead of rendering something else:

* ``optimization.*_lr``: Fondue's values times the arm's multiplier;
* ``run.stop_after_step`` removed (one phase, no anneal);
* ``trainer.max_steps`` 3,360, ``num_decay_steps`` 672 (the last fifth), ``warmup_steps`` 336;
* ``data.train_ds.total_hours`` / ``total_cuts`` scaled to the hours drawn (logged only);
* ``trainer.gradient_accumulation_steps``: 1,024 micro-batches per step over ``--nodes`` x 4 GPUs;
* ``save_steps`` / ``eval_steps`` 1,344, no retention limit, ``save_only_model``, no eval on start.

``data.train_ds.snapshot_every_n_steps`` must stay 512 (never <= 34: the data-loader stall).

    render_raclette_configs.py --base Fondue.yaml --out-dir Raclette-configs [--nodes 4]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FILL = 26.3 / 30  # audio a micro-batch carries over its batch_duration (build_fondue_mix.FILL)
BATCH_DURATION = 30
MICRO_BATCHES_PER_STEP = 1024  # Fondue: 16 nodes x 4 GPUs x ga 16
GPUS_PER_NODE = 4
STEPS = 3360
DECAY_STEPS = STEPS // 5
WARMUP_STEPS = 336
READ_EVERY = 1344  # save and eval grid: 1,344, 2,688 (end of the plateau), 3,360 (final)
SNAPSHOT_EVERY = 512

# arm -> (encoder, adapter, decoder) LR strings, Fondue's x multiplier; trainer.seed
FONDUE_LRS = (6e-6, 2e-4, 2e-5)
ARMS: dict[str, tuple[tuple[str, str, str], float, int]] = {
    "x0p5": (("3e-6", "1e-4", "1e-5"), 0.5, 1337),
    "x1": (("6e-6", "2e-4", "2e-5"), 1.0, 1337),
    "x1-s43": (("6e-6", "2e-4", "2e-5"), 1.0, 1338),
    "x2": (("1.2e-5", "4e-4", "4e-5"), 2.0, 1337),
    "x4": (("2.4e-5", "8e-4", "8e-5"), 4.0, 1337),
}
MELT_SEEDS = {"x1-s43": 43}  # the data shard seed (launcher); every other arm reads 42


def hours_drawn() -> float:
    """Audio hours the 3,360 steps draw at the measured fill."""
    return STEPS * BATCH_DURATION * FILL * MICRO_BATCHES_PER_STEP / 3600


def grad_accum(nodes: int) -> int:
    """``gradient_accumulation_steps`` that keeps 1,024 micro-batches per optimizer step."""
    per_accum = nodes * GPUS_PER_NODE
    if MICRO_BATCHES_PER_STEP % per_accum:
        raise SystemExit(f"{nodes} nodes x {GPUS_PER_NODE} GPUs does not divide {MICRO_BATCHES_PER_STEP}")
    return MICRO_BATCHES_PER_STEP // per_accum


def sub_once(text: str, pattern: str, repl: str) -> str:
    """Replace the single match of ``pattern`` (MULTILINE), failing if there is not exactly one."""
    new, n = re.subn(pattern, lambda _: repl, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected exactly one match of {pattern!r}, found {n}")
    return new


def render(base: str, arm: str, nodes: int = 4) -> str:
    """The config text of arm ``arm`` from the Fondue base text."""
    (enc, adp, dec), _, seed = ARMS[arm]
    hours = hours_drawn()
    base_hours = float(re.search(r"^    total_hours: ([0-9.]+)", base, flags=re.M).group(1))
    base_cuts = int(re.search(r"^    total_cuts: ([0-9]+)", base, flags=re.M).group(1))

    t = base
    t = sub_once(t, r"^  exp_name: Fondue$", f"  exp_name: Raclette-{arm}")
    t = sub_once(t, r"^  stop_after_step: \d+.*\n", "")
    t = sub_once(t, r"^    total_hours: (?!0\.0$)[0-9.]+$", f"    total_hours: {hours:.1f}")
    t = sub_once(t, r"^    total_cuts: [0-9]+$", f"    total_cuts: {round(base_cuts * hours / base_hours)}")
    t = sub_once(t, r"^  max_steps: \d+$", f"  max_steps: {STEPS}")
    t = sub_once(t, r"^    num_decay_steps: \d+$", f"    num_decay_steps: {DECAY_STEPS}")
    t = sub_once(t, r"^  warmup_steps: 0\.05$", f"  warmup_steps: {WARMUP_STEPS}")
    t = sub_once(t, r"^  gradient_accumulation_steps: 16$", f"  gradient_accumulation_steps: {grad_accum(nodes)}")
    t = sub_once(t, r"^    snapshot_every_n_steps: 512$", f"    snapshot_every_n_steps: {SNAPSHOT_EVERY}")
    t = sub_once(t, r"^  eval_on_start: true$", "  eval_on_start: false")
    t = sub_once(t, r"^  eval_steps: 500$", f"  eval_steps: {READ_EVERY}")
    t = sub_once(t, r"^  save_steps: 1000$", f"  save_steps: {READ_EVERY}")
    t = sub_once(
        t,
        r"^  save_total_limit: 2$",
        "  save_total_limit: null # keep every read step\n  save_only_model: true # no optimizer state: an arm that dies is rerun, not resumed",
    )
    t = sub_once(t, r"^  seed: 1337$", f"  seed: {seed}")
    t = sub_once(t, r"^  encoder_lr: 6e-6$", f"  encoder_lr: {enc}")
    t = sub_once(t, r"^  adapter_lr: 2e-4$", f"  adapter_lr: {adp}")
    t = sub_once(t, r"^  decoder_lr: 2e-5.*$", f"  decoder_lr: {dec}")

    head = (
        f"# Raclette-{arm}.yaml: the LR pilot for Fondue (plan/08-raclette.md).\n"
        f"# Rendered from the live Fondue.yaml by render_raclette_configs.py; only these differ: LRs\n"
        f"# {enc} / {adp} / {dec} (encoder / adapter / decoder), {hours:,.0f} h drawn, max_steps {STEPS}\n"
        f"# with the decay over the last {DECAY_STEPS} and warmup {WARMUP_STEPS}, ga {grad_accum(nodes)} on {nodes} nodes,\n"
        "# no stop_after_step, weights-only saves every 1,344 steps. Do not hand-edit.\n#\n"
    )
    return head + t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path, help="the live Fondue.yaml")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--nodes", type=int, default=4)
    args = ap.parse_args()

    base = args.base.read_text()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{hours_drawn():,.0f} h; max_steps {STEPS}; ga {grad_accum(args.nodes)} on {args.nodes} nodes")
    for arm in ARMS:
        out = args.out_dir / f"Raclette-{arm}.yaml"
        out.write_text(render(base, arm, args.nodes))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
