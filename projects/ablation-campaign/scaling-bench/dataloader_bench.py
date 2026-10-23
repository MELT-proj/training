#!/usr/bin/env python3
"""How fast can the data path alone feed one rank?

The scaling runs measure steps, and a step that is slow because the GPUs are
waiting on batches looks identical to one that is slow because the model is
expensive. This builder measures the first half on its own: it constructs the
same Lhotse dataloader the trainer uses and times nothing but iteration, with no
model and no GPU.

Cheap by construction -- run it on `gp_debug` (CPU-only, 32 nodes, priority
10000) and it costs zero GPU-hours, at every node count up to 32.

    dataloader_bench.py --config ../ABL-IFT-700.yaml \
        --world-size 32 --rank "$SLURM_PROCID" --steps 200 --json-out rank.json

What to do with the answer: compare the median batch time against the per-rank
step time measured by the GPU arm (`per-rank step = s_per_step / grad_accum`).
If a batch takes a large fraction of that, the data path is the ceiling and more
nodes cannot help -- which is the hypothesis the cost-axis study left open when
it found more nodes buying no speed for two encoders.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def parse_argv(argv: list[str]):
    """Let melt's own parser consume the training config from a synthetic argv."""
    saved = sys.argv
    sys.argv = ["dataloader_bench"] + argv
    try:
        from melt.training.config import expand_env_vars_in_config, parse_args_and_load_config

        cfg = parse_args_and_load_config()
        return expand_env_vars_in_config(cfg)
    finally:
        sys.argv = saved


def build_loader(cfg, rank: int, world_size: int):
    from melt.training.data.audio.lhotse import FallbackDataset, SpeechToTextDataset
    from melt.training.data.audio.lhotse.dataloader import get_train_dataloader_from_config
    from melt.training.setup import prepare_processor

    processor = prepare_processor(cfg)
    dataset = SpeechToTextDataset(
        processor=processor,
        config=cfg.data,
        is_train=True,
        return_labels=True,
        return_langs=True,
    )
    dataset = FallbackDataset(dataset)
    return get_train_dataloader_from_config(
        data_config=cfg.data,
        dataset=dataset,
        global_rank=rank,
        world_size=world_size,
    )


def run(args) -> dict:
    cfg = parse_argv(args.train_args)
    loader = build_loader(cfg, args.rank, args.world_size)

    iterator = iter(loader)
    cold_s = None
    times: list[float] = []
    audio_s: list[float] = []
    cuts: list[int] = []

    start_all = time.monotonic()
    for index in range(args.warmup + args.steps):
        t0 = time.monotonic()
        batch = next(iterator)
        elapsed = time.monotonic() - t0
        if index == 0:
            cold_s = elapsed
        if index < args.warmup:
            continue
        times.append(elapsed)
        durations = batch.get("durations")
        audio_s.append(sum(float(d) for d in durations) if durations is not None else 0.0)
        cuts.append(len(batch.get("input_ids", [])) if hasattr(batch.get("input_ids"), "__len__") else 0)

    wall = time.monotonic() - start_all
    total_audio = sum(audio_s)
    return {
        "rank": args.rank,
        "world_size": args.world_size,
        "steps": len(times),
        "cold_first_batch_s": cold_s,
        "batch_s_p50": statistics.median(times),
        "batch_s_p90": sorted(times)[min(len(times) - 1, (9 * len(times)) // 10)],
        "batch_s_max": max(times),
        "audio_s_per_batch": total_audio / len(times),
        "cuts_per_batch": statistics.median(cuts) if cuts else 0,
        # Per rank. Multiply by world_size for the whole job.
        "audio_s_per_wall_s": total_audio / wall if wall > 0 else 0.0,
        "wall_s": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training config to borrow the data section from")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-duration", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    overrides = []
    if args.batch_duration is not None:
        overrides.append(f"--data.train_ds.batch_duration={args.batch_duration}")
    if args.num_workers is not None:
        overrides.append(f"--data.train_ds.num_workers={args.num_workers}")
    args.train_args = [f"--config={args.config}", *overrides]

    result = run(args)
    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
