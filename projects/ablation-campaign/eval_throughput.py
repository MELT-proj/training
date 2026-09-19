#!/usr/bin/env python3
"""Eval throughput of a trained arm, in utterances per second (plan/03 section 4).

The fourth efficiency number. It is only comparable across arms when measured
under fixed conditions, so they are fixed here and written into the output:

    batch 16, duration-sorted, decoder flash_attention_2, bf16, greedy,
    max_new_tokens 256, one GPU, batches collated on the CPU *before* timing

The clock covers `model.generate` on ready batches, so the number is the model's,
not the data loader's or inspect_ai's. melt-eval's end-to-end rate is lower and is
a different quantity.

It also times the audio stack (encoder + adapter) forward on the same batches and
the CPU collate step, separately. The generate time includes the audio stack, so
`audio_stack_share` says how much of an eval batch the encoder is; with two arms
that differ only in encoder it is also the direct test of whether one encoder is
cheaper to run than another (03 section 4, the Whisper cost question).

GPU work: submit through SLURM only, see eval_throughput.sbatch. Never run this
interactively. Untested on hardware at the time it was written.

    python projects/ablation-campaign/eval_throughput.py \
        --run-dir /workspace/outputs/<exp_name> --output /workspace/outputs/throughput/<exp_name>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from melt.logging_utils import configure_logging, get_logger  # noqa: E402
from melt.modeling import MELTForCausalLM  # noqa: E402
from melt.modeling.configuration_melt import MELTConfig  # noqa: E402
from melt.modeling.processing_melt import MELTProcessor  # noqa: E402
from melt.training.data.audio.lhotse.collator import MELTDataCollator  # noqa: E402
from melt.training.data.audio.lhotse.dataloader import (  # noqa: E402
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset  # noqa: E402

logger = get_logger(__name__)


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a finished arm's output dir (weights, config.json, resolved_config.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval-set", help="named validation set from the run's config; default the first")
    ap.add_argument("--n-utts", type=int, default=320, help="utterances measured (20 batches at 16)")
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--decoder-attn", default="flash_attention_2")
    ap.add_argument("--encoder-attn", help="default: what the arm trained with, from resolved_config.json")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU: submit this through sbatch (eval_throughput.sbatch)")
    configure_logging()
    run = Path(args.run_dir)
    resolved = json.loads((run / "resolved_config.json").read_text())
    encoder_attn = args.encoder_attn or resolved["model"]["encoder"].get("attn_implementation", "sdpa")

    # Same load path as train.py's checkpoint branch: a saved config carries no attention
    # implementation, so it is set on the sub-configs, or both silently fall back to sdpa.
    config = MELTConfig.from_pretrained(run)
    config.text_decoder_config._attn_implementation = args.decoder_attn
    config.audio_encoder_config._attn_implementation = encoder_attn
    processor = MELTProcessor.from_pretrained(run)
    if not (run / "model.safetensors").exists():
        sys.exit(f"{run} has no consolidated weights (FSDP2 run?): consolidate first, see infrastructure.md section 5")
    model = MELTForCausalLM.from_pretrained(run, config=config, torch_dtype=torch.bfloat16).cuda().eval()

    data_cfg = resolve_eval_data_config(OmegaConf.create(resolved["data"]))
    named = split_eval_config_by_name(data_cfg) or {"eval": data_cfg}
    set_name = args.eval_set or next(iter(named))
    set_cfg = named[set_name]

    # First n_utts of the set in its own order (deterministic), then sorted by duration.
    cuts = materialize_cuts_for_eval(set_cfg)[: args.n_utts]
    cuts = sorted(cuts, key=lambda c: c.duration, reverse=True)
    if len(cuts) < args.n_utts:
        logger.warning("set %s has only %d cuts", set_name, len(cuts))
    dataset = MELTMapDataset(cuts=cuts, processor=processor, config=set_cfg, is_train=False, return_langs=True)
    collator = MELTDataCollator(processor=processor, config=set_cfg, is_train=False)

    t0 = time.perf_counter()
    batches = [
        collator([dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))])
        for i in range(0, len(dataset), args.batch_size)
    ]
    collate_s = time.perf_counter() - t0

    def to_gpu(b):
        return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    audio_dtype = next(p.dtype for p in model.audio_stack.parameters() if p.is_floating_point())

    def inputs_of(b):
        feats = b["input_features"]
        return dict(
            input_ids=b["prompt_input_ids"],
            attention_mask=b["prompt_attention_mask"],
            input_features=feats.to(audio_dtype) if feats.is_floating_point() else feats,
            features_attention_mask=b.get("features_attention_mask"),
        )

    gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True)
    stack_s = gen_s = 0.0
    new_tokens = utts = 0
    audio_s = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, b in enumerate(batches):
            g = to_gpu(b)
            kw = inputs_of(g)
            timed = i >= args.warmup_batches
            sync(); t = time.perf_counter()
            model.audio_stack(kw["input_features"], features_attention_mask=kw["features_attention_mask"])
            sync(); ts = time.perf_counter() - t
            t = time.perf_counter()
            out = model.generate(**kw, **gen)
            sync(); tg = time.perf_counter() - t
            if timed:
                stack_s += ts
                gen_s += tg
                utts += out.shape[0]
                new_tokens += int((out != processor.tokenizer.pad_token_id).sum())
        # batches after warmup only
    timed_cuts = cuts[args.warmup_batches * args.batch_size :]
    audio_s = sum(c.duration for c in timed_cuts)
    if utts == 0:
        sys.exit("no timed batches: raise --n-utts or lower --warmup-batches")

    result = {
        "exp_name": resolved["run"]["exp_name"],
        "eval_set": set_name,
        "utt_per_s": utts / gen_s,
        "audio_s_per_wall_s": audio_s / gen_s,
        "n_utts_timed": utts,
        "mean_new_tokens": new_tokens / utts,
        "generate_s_per_batch": gen_s / (utts / args.batch_size),
        "audio_stack_s_per_batch": stack_s / (utts / args.batch_size),
        "audio_stack_share": stack_s / gen_s,
        "collate_s_per_batch_cpu": collate_s / len(batches),
        "batch_size": args.batch_size,
        "sorted_by_duration": True,
        "decoder_attn": args.decoder_attn,
        "encoder_attn": encoder_attn,
        "dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "host": platform.node(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
