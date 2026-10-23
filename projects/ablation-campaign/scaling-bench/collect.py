#!/usr/bin/env python3
"""Turn `[perf]` records into the scaling tables.

    collect.py --raw raw/ [--static tables/flops-static.json] [--out tables/]

Reads one `perf.jsonl` per run (written by ``MELTTrainer`` under
``run.perf_logging``), takes the steady-state window, and emits:

    tables/points.tsv     one row per measured point
    tables/walltime.tsv   wall hours and GPU-hours per audio-hour budget

It needs nothing from the cluster -- no sacct, no wandb, no live job -- so a
result can be reproduced from files alone. Job ids are deliberately absent:
they belong in ``arms.tsv``, which is the campaign's ledger.

Three decisions worth knowing before reading the numbers:

1. **The window starts after `drop_first` steps.** The first step of these runs
   has been measured at 13-21 minutes (shuffle-buffer fill, allocator growth,
   autotune) and would dominate a 110-step mean.
2. **Fill is measured, not assumed.** A duration-bucketed batch rarely holds
   the nominal `batch_duration`; the board measured 78-86% on the 2,100 h mix.
   The nominal figure is reported beside it so old extrapolations remain
   comparable.
3. **MFU is reported twice.** `mfu_total` counts everything the model runs,
   `mfu_trainable` only what produces a gradient. With the encoder unfrozen
   they agree; with it frozen the gap is the encoder's forward pass buying
   nothing `--` which is exactly the column that says whether freezing saved
   anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flops  # noqa: E402

EXP_PATTERN = re.compile(r"-n(?P<nodes>\d+)(?:-r(?P<repeat>\d+))?$")

# Audio-hour budgets the report tabulates: the pilot, the request that prompted
# this campaign, and the two pools Fondue actually draws from.
BUDGETS = (25_000, 100_000, 247_000, 330_000)


def load_shape(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["shape"]


def discover(raw: Path) -> list[Path]:
    """Every perf.jsonl under *raw*, whether fetched into a subdir or dropped in."""
    files = sorted(raw.rglob("perf.jsonl"))
    if not files:
        raise SystemExit(f"no perf.jsonl under {raw}")
    return files


def nodes_from(path: Path, override: int | None) -> tuple[int, int]:
    """(nodes, repeat) from the run's directory name, which bench.py sets."""
    if override:
        return override, 1
    for candidate in (path.parent.name, path.parent.parent.name):
        match = EXP_PATTERN.search(candidate)
        if match:
            return int(match["nodes"]), int(match["repeat"] or 1)
    raise SystemExit(
        f"cannot read a node count from {path.parent.name!r}; pass --nodes, "
        "or name the directory the way bench.py does (…-n8, …-n8-r2)"
    )


def steady_state(records: list[dict], drop_first: int) -> list[dict]:
    kept = [r for r in records if int(r.get("step", 0)) > drop_first and r.get("window_steps", 0) > 0]
    if not kept:
        raise SystemExit(
            f"no records above step {drop_first} ({len(records)} present); "
            "the run died before the steady-state window opened"
        )
    return kept


def summarise(path: Path, shape_kwargs: dict, drop_first: int, override_nodes: int | None) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = steady_state(records, drop_first)
    nodes, repeat = nodes_from(path, override_nodes)
    world_size = 4 * nodes

    window_s = sum(r["window_s"] for r in kept)
    steps = sum(r["window_steps"] for r in kept)
    s_per_step = window_s / steps

    window = flops.Window(
        audio_s=sum(r.get("audio_s", 0.0) for r in kept),
        enc_padded_frames=sum(r.get("enc_padded_frames", 0.0) for r in kept),
        enc_valid_frames=sum(r.get("enc_valid_frames", 0.0) for r in kept),
        enc_chunks=sum(r.get("enc_chunks", 0.0) for r in kept),
        dec_positions=sum(r.get("dec_positions", 0.0) for r in kept),
        dec_sequences=sum(r.get("dec_sequences", 0.0) for r in kept),
        sup_tokens=sum(r.get("sup_tokens", 0.0) for r in kept),
        steps=float(steps),
    )

    metrics = flops.point_metrics(
        flops.ModelShape(**shape_kwargs), window, s_per_step=s_per_step, world_size=world_size
    )

    rate = metrics["audio_s_per_wall_s"]
    row = {
        "run": path.parent.name,
        "nodes": nodes,
        "gpus": world_size,
        "repeat": repeat,
        "steps_used": steps,
        "s_per_step": s_per_step,
        "s_per_step_p10": _percentile([r.get("s_per_step") for r in kept], 10),
        "s_per_step_p90": _percentile([r.get("s_per_step") for r in kept], 90),
        **{k: metrics[k] for k in (
            "fill", "audio_s_per_wall_s", "audio_h_per_wall_h",
            "tflops_per_gpu_total", "mfu_total", "mfu_trainable",
            "encoder_share", "flops_per_audio_h",
        )},
        "gpu_h_per_1k_audio_h": 1000.0 / rate * world_size if rate > 0 else float("nan"),
    }
    for budget in BUDGETS:
        out = flops.walltime_for(rate, world_size, budget)
        row[f"wall_h_{budget // 1000}k"] = out["wall_h"]
        row[f"gpu_h_{budget // 1000}k"] = out["gpu_h"]
    return row


def _percentile(values: list[float], pct: int) -> float:
    clean = sorted(v for v in values if isinstance(v, (int, float)))
    if not clean:
        return float("nan")
    return clean[min(len(clean) - 1, (pct * len(clean)) // 100)]


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(_fmt(row[c]) for c in columns) + "\n")
    print(f"wrote {path}")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown(rows: list[dict]) -> str:
    head = (
        "| nodes | GPUs | s/step | p10-p90 | fill | audio-h/wall-h | TFLOP/s/GPU | "
        "MFU | MFU trainable | GPU-h/1K h | 100K h wall |"
    )
    lines = [head, "|" + "---|" * 11]
    for r in sorted(rows, key=lambda r: r["nodes"]):
        lines.append(
            f"| {r['nodes']} | {r['gpus']} | {r['s_per_step']:.1f} | "
            f"{r['s_per_step_p10']:.1f}-{r['s_per_step_p90']:.1f} | {r['fill']:.0%} | "
            f"{r['audio_h_per_wall_h']:.0f} | {r['tflops_per_gpu_total']:.1f} | "
            f"{r['mfu_total']:.2%} | {r['mfu_trainable']:.2%} | "
            f"{r['gpu_h_per_1k_audio_h']:.1f} | {r['wall_h_100k'] / 24:.1f} d |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(HERE / "raw"), help="directory holding <run>/perf.jsonl")
    parser.add_argument("--static", default=str(HERE / "tables" / "flops-static.json"))
    parser.add_argument("--out", default=str(HERE / "tables"))
    parser.add_argument("--drop-first", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=None,
                        help="override the node count read from the directory name")
    args = parser.parse_args()

    shape_kwargs = load_shape(Path(args.static))
    rows = [
        summarise(path, shape_kwargs, args.drop_first, args.nodes)
        for path in discover(Path(args.raw))
    ]

    out = Path(args.out)
    write_tsv(rows, out / "points.tsv")
    write_tsv(rows, out / "walltime.tsv")
    print()
    print(markdown(rows))


if __name__ == "__main__":
    main()
