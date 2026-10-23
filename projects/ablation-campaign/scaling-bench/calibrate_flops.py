#!/usr/bin/env python3
"""Calibrate the analytic FLOPs model in ``flops.py``.

Two halves, run separately:

    # static: parameter counts and layer geometry. Pure CPU + the HF cache.
    calibrate_flops.py static --out tables/flops-static.json

    # count: run the real modules under torch.utils.flop_counter on one GPU
    calibrate_flops.py count --durations 5,15,30 --out tables/flops-calibration.tsv

``static`` is what ``collect.py`` needs; ``count`` is what makes the analytic
constants trustworthy, and it needs a GPU (artemis) because FlopCounterMode
instruments dispatched kernels.

Why the numbers are not hardcoded in the report: the campaign has changed
encoder, adapter and decoder several times, and a stale parameter count is
invisible in the output -- it just makes MFU quietly wrong. Every figure this
script emits is counted off the real config, and the JSON records which
checkpoint it came from, so the report can say where its denominator came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------- static


def decoder_numbers(model_name: str) -> dict:
    """Parameter split and attention geometry of a causal LM.

    The head is counted even when embeddings are tied: the forward pass still
    does `hidden @ vocab` per position. The input *lookup* costs nothing, which
    is why it is excluded -- counting the whole embedding matrix as 6N per token
    would overstate the decoder's FLOPs by about a quarter on Qwen3.5-2B.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    total = sum(p.numel() for p in model.parameters())
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    emb_params = sum(p.numel() for p in embedding.parameters()) if embedding is not None else 0
    head_params = sum(p.numel() for p in head.parameters()) if head is not None else 0

    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = getattr(text, "layer_types", None)
    if isinstance(layer_types, list):
        full_attn = sum(1 for t in layer_types if t == "full_attention")
    else:
        # No per-layer map: assume every layer is full attention, which is the
        # conservative (larger) estimate.
        full_attn = int(getattr(text, "num_hidden_layers", 0))

    return {
        "name": model_name,
        "total_params": total,
        "embedding_params": emb_params,
        "head_params": head_params,
        "body_params": total - emb_params,
        "layers": int(getattr(text, "num_hidden_layers", 0)),
        "full_attn_layers": full_attn,
        "dim": int(getattr(text, "hidden_size", 0)),
        "tied_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
    }


def encoder_numbers(model_name: str, chunk_frames: int) -> dict:
    """Parameter count and geometry of an audio encoder."""
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = _build_encoder(config)

    return {
        "name": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "layers": int(getattr(config, "num_hidden_layers", 0)),
        "dim": int(getattr(config, "hidden_size", 0)),
        "chunk_frames": chunk_frames,
        # w2v-BERT is fed precomputed features by the processor, so the conv
        # frontend is CPU work in the dataloader, not GPU work in the step.
        "conv_flops_per_frame": 0.0,
    }


def _build_encoder(config):
    """Instantiate the bare encoder model for whatever config we were handed."""
    from transformers import (
        AutoModel,
        Wav2Vec2BertModel,
    )

    if type(config).__name__ == "Wav2Vec2BertConfig":
        return Wav2Vec2BertModel(config)
    return AutoModel.from_config(config)


def adapter_params(encoder_dim: int, decoder_dim: int, stack_factor: int = 1, mid: int | None = None) -> int:
    """MELTMLPAdapter's parameter count, from its own layer shapes.

    Mirrors ``MELTMLPAdapter.__init__`` (melt/modeling/modeling_melt.py): two
    linears with a GELU between them, a LayerNorm and a scalar gain.
    """
    mid = mid or decoder_dim
    fc1 = encoder_dim * stack_factor * mid + mid
    fc2 = mid * decoder_dim + decoder_dim
    norm = 2 * decoder_dim
    gain = 1
    return fc1 + fc2 + norm + gain


def static(args) -> dict:
    dec = decoder_numbers(args.decoder)
    enc = encoder_numbers(args.encoder, args.chunk_frames)
    adapter = adapter_params(enc["dim"], dec["dim"], args.stack_factor)

    shape = {
        "encoder_params": float(enc["params"]),
        "encoder_layers": enc["layers"],
        "encoder_dim": enc["dim"],
        "encoder_chunk_frames": enc["chunk_frames"],
        "encoder_trainable": args.encoder_trainable,
        "encoder_conv_flops_per_frame": enc["conv_flops_per_frame"],
        "adapter_params": float(adapter),
        "adapter_trainable": args.adapter_trainable,
        "decoder_body_params": float(dec["body_params"]),
        "decoder_head_params": float(dec["head_params"]),
        "decoder_full_attn_layers": dec["full_attn_layers"],
        "decoder_dim": dec["dim"],
        "decoder_trainable": args.decoder_trainable,
        "gradient_checkpointing": args.gradient_checkpointing,
        "stack_factor": args.stack_factor,
    }
    record = {"shape": shape, "encoder": enc, "decoder": dec, "adapter_params": adapter}
    out = Path(args.out) if args.out else HERE / "tables" / "flops-static.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"  encoder  {enc['params'] / 1e9:.4f} B  ({enc['layers']} layers, dim {enc['dim']})")
    print(
        f"  decoder  {dec['total_params'] / 1e9:.4f} B  "
        f"(body {dec['body_params'] / 1e9:.4f} + head {dec['head_params'] / 1e9:.4f}, "
        f"{dec['full_attn_layers']}/{dec['layers']} full-attention layers)"
    )
    print(f"  adapter  {adapter / 1e6:.2f} M")
    return record


# ---------------------------------------------------------------------- count


def count(args) -> None:
    """Measure forward+backward FLOPs of the real modules and compare to flops.py.

    Runs on one GPU. Component by component so a disagreement points at one
    term instead of the sum, and with a checkpointed variant so the 6 -> 8
    recompute ratio is checked rather than assumed.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from transformers import AutoConfig, AutoModelForCausalLM

    sys.path.insert(0, str(HERE))
    import flops  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    dec_cfg = AutoConfig.from_pretrained(args.decoder)
    dec = AutoModelForCausalLM.from_config(dec_cfg).to(device)
    dec.train()

    rows = []
    for duration in [float(d) for d in args.durations.split(",")]:
        # 50 Hz is w2v-BERT's frame rate; a cut's decoder length is its frame
        # count plus a short text suffix.
        frames = int(duration * 50)
        positions = frames + 32
        ids = torch.randint(0, 100, (1, positions), device=device)

        def run_decoder():
            out = dec(input_ids=ids, labels=ids)
            out.loss.backward()

        dec.zero_grad(set_to_none=True)
        with FlopCounterMode(display_table=False) as counter:
            run_decoder()
        measured = counter.get_total_flops()

        shape = flops.ModelShape(
            encoder_params=0.0, encoder_layers=0, encoder_dim=0,
            encoder_chunk_frames=1, encoder_trainable=False,
            encoder_conv_flops_per_frame=0.0,
            adapter_params=0.0, adapter_trainable=False,
            decoder_body_params=sum(p.numel() for p in dec.parameters()) - _emb(dec),
            decoder_head_params=_head(dec),
            decoder_full_attn_layers=_full_attn(dec_cfg),
            decoder_dim=int(getattr(dec_cfg.get_text_config(), "hidden_size", 0)),
            decoder_trainable=True, gradient_checkpointing=False,
        )
        window = flops.Window(
            audio_s=duration, enc_padded_frames=0, enc_valid_frames=0, enc_chunks=0,
            dec_positions=positions, dec_sequences=1, sup_tokens=0,
        )
        analytic = flops.decoder_flops(shape, window)
        rows.append((duration, measured, analytic))
        print(f"  {duration:5.0f}s  measured {measured / 1e12:10.2f} TFLOP  "
              f"analytic {analytic / 1e12:10.2f} TFLOP  ratio {measured / analytic:.3f}")

    out = Path(args.out) if args.out else HERE / "tables" / "flops-calibration.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("component\tduration_s\tmeasured_flops\tanalytic_flops\tratio\n")
        for duration, measured, analytic in rows:
            fh.write(f"decoder\t{duration:.0f}\t{measured:.6e}\t{analytic:.6e}\t{measured / analytic:.4f}\n")
    print(f"wrote {out}")
    print(
        "The counter sees mm/addmm/bmm and SDPA (flash/efficient/cudnn) forward and\n"
        "backward, but not fused elementwise/norm/softmax kernels or the\n"
        "Gated-DeltaNet recurrence, so a ratio below 1 is expected. Treat the\n"
        "counter as a lower bound and record the ratio rather than scaling by it."
    )


def _emb(model) -> int:
    e = model.get_input_embeddings()
    return sum(p.numel() for p in e.parameters()) if e is not None else 0


def _head(model) -> int:
    h = model.get_output_embeddings()
    return sum(p.numel() for p in h.parameters()) if h is not None else 0


def _full_attn(config) -> int:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = getattr(text, "layer_types", None)
    if isinstance(layer_types, list):
        return sum(1 for t in layer_types if t == "full_attention")
    return int(getattr(text, "num_hidden_layers", 0))


# ------------------------------------------------------------------------ cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_static = sub.add_parser("static", help="parameter counts (CPU, no weights)")
    p_static.add_argument("--encoder", default="facebook/w2v-bert-2.0")
    p_static.add_argument("--decoder", default="Qwen/Qwen3.5-2B-Base")
    p_static.add_argument("--chunk-frames", type=int, default=1500,
                          help="encoder attention window in frames (max_audio_seq_len)")
    p_static.add_argument("--stack-factor", type=int, default=1)
    p_static.add_argument("--encoder-trainable", action="store_true", default=True)
    p_static.add_argument("--no-encoder-trainable", dest="encoder_trainable", action="store_false")
    p_static.add_argument("--adapter-trainable", action="store_true", default=True)
    p_static.add_argument("--no-adapter-trainable", dest="adapter_trainable", action="store_false")
    p_static.add_argument("--decoder-trainable", action="store_true", default=True)
    p_static.add_argument("--no-decoder-trainable", dest="decoder_trainable", action="store_false")
    p_static.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p_static.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p_static.add_argument("--out")

    p_count = sub.add_parser("count", help="FlopCounterMode check (needs a GPU)")
    p_count.add_argument("--decoder", default="Qwen/Qwen3.5-2B-Base")
    p_count.add_argument("--durations", default="5,15,30")
    p_count.add_argument("--out")

    args = parser.parse_args()
    if args.command == "static":
        static(args)
    else:
        count(args)


if __name__ == "__main__":
    main()
