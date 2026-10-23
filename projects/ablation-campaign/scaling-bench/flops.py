#!/usr/bin/env python3
"""FLOPs, throughput and walltime arithmetic for the scaling benchmark.

Pure arithmetic -- no torch, no config parsing, no cluster access -- so it runs
on nyx and is unit-testable. Every input is either a parameter count reported by
``calibrate_flops.py --static`` or a counter from the trainer's perf window
(``run.perf_logging``), which is why the padded shapes are used rather than the
nominal ones: padding is work the GPU actually does.

What it computes, in the order the report needs it:

    step_flops(...)    -> FLOPs of one optimizer step, split encoder/adapter/
                          decoder and total/trainable
    point_metrics(...) -> TFLOP/s per GPU, MFU, audio-h per wall-h
    walltime_for(...)  -> wall hours and GPU-hours for N audio hours

Conventions, stated once so the report can cite them:

* ``k`` (the body multiplier) is 2 for a forward-only component, 6 for
  forward+backward, and 8 when activation checkpointing recomputes the forward
  pass. A frozen module inside a training graph is forward-only.
* Attention costs 4*T^2*d FLOPs per layer forward and 12*T^2*d forward+backward
  (16 with recompute). This is the non-causal count and therefore an upper bound
  for the decoder's causal layers; ``calibrate_flops.py`` measures the residual
  rather than trusting it.
* Only *full* attention layers carry the quadratic term. Qwen3.5-2B's 18
  Gated-DeltaNet layers are linear in sequence length, and their cost is already
  in the body term because every parameter is used once per token.
* The encoder chunks its input at ``encoder_chunk_frames``, so encoder attention
  is quadratic within a chunk, not over the whole cut.
"""

from __future__ import annotations

from dataclasses import dataclass

# H100 SXM5 dense bf16. MN5's acc nodes ship the 64 GB H100; gpu_peak.py measures
# what they actually achieve, and the report carries both denominators when the
# two disagree.
PEAK_BF16_TFLOPS = 989.0


@dataclass(frozen=True)
class ModelShape:
    """Parameter counts and layer geometry of one recipe.

    Non-embedding body parameters only; the embedding and output head are
    separate so a tied/untied head can be counted correctly.
    """

    encoder_params: float
    encoder_layers: int
    encoder_dim: int
    encoder_chunk_frames: int
    encoder_trainable: bool
    # Fused conv frontend, in FLOPs per (padded) input frame. Zero unless
    # calibrate_flops.py supplies a measured value; it is small for w2v-BERT.
    encoder_conv_flops_per_frame: float
    adapter_params: float
    adapter_trainable: bool
    decoder_body_params: float
    decoder_head_params: float
    decoder_full_attn_layers: int
    decoder_dim: int
    decoder_trainable: bool
    gradient_checkpointing: bool
    # Frames concatenated per adapter output position (MELT's stack_factor).
    # 1 in the campaign baseline; anything else divides the adapter's positions.
    stack_factor: int = 1

    def k(self, trainable: bool) -> int:
        """Body FLOPs per parameter use: forward-only, fwd+bwd, or recompute."""
        if not trainable:
            return 2
        return 8 if self.gradient_checkpointing else 6

    def attention_k(self, trainable: bool) -> int:
        """Attention FLOPs per T^2*d, forward-only, fwd+bwd, or recompute."""
        if not trainable:
            return 4
        return 16 if self.gradient_checkpointing else 12


@dataclass(frozen=True)
class Window:
    """Counters accumulated over one logging window, summed across all ranks.

    Every field is what the model *ran on*, padding included. The trainer
    reports them from the ``durations`` / ``input_ids`` / ``input_features``
    tensors it already pops, so nothing is estimated.
    """

    audio_s: float        # sum of cut durations (real audio, no padding)
    enc_padded_frames: float   # sum over cuts of the padded frame count
    enc_valid_frames: float    # sum over cuts of the unmasked frame count
    enc_chunks: float          # sum over cuts of ceil(padded / chunk_frames)
    dec_positions: float       # input_ids.numel(): text + audio + padding
    dec_sequences: float       # number of cut-sequences
    sup_tokens: float          # labels != -100 (supervised text tokens)
    # How many optimizer steps this window covers. Needed because the counters
    # are totals over the window while s_per_step is a *mean*: dividing one by
    # the other without this is how a rate ends up off by the window length.
    steps: float = 1.0

    @property
    def fill(self) -> float:
        """Real audio per padded encoder second, 0..1.

        The padded frames are what the encoder and adapter pay for; the ratio to
        ``50 * audio_s`` is how much of that work is padding.
        """
        if self.audio_s <= 0:
            return 0.0
        return self.enc_valid_frames / self.enc_padded_frames if self.enc_padded_frames else 1.0


def encoder_flops(shape: ModelShape, w: Window) -> float:
    """Encoder body + attention + conv frontend, over the window."""
    k = shape.k(shape.encoder_trainable)
    attn_k = shape.attention_k(shape.encoder_trainable)
    body = k * shape.encoder_params * w.enc_padded_frames
    # Chunked attention: each chunk attends over chunk_frames^2, not T_cut^2.
    attn = (
        attn_k
        * w.enc_chunks
        * float(shape.encoder_chunk_frames) ** 2
        * shape.encoder_dim
        * shape.encoder_layers
    )
    conv = shape.encoder_conv_flops_per_frame * w.enc_padded_frames
    return body + attn + conv


def adapter_flops(shape: ModelShape, w: Window) -> float:
    """Adapter over its output positions, which are the valid (unpadded) frames.

    ``stack_factor`` frames are concatenated into one position, so it divides
    the position count as well as multiplying the first layer's width -- the
    two cancel in the dominant term and only the bias/LayerNorm parts differ.
    """
    positions = w.enc_valid_frames / shape.stack_factor
    return shape.k(shape.adapter_trainable) * shape.adapter_params * positions


def decoder_flops(shape: ModelShape, w: Window) -> float:
    """Decoder body + head + full-attention layers, over the window."""
    k = shape.k(shape.decoder_trainable)
    attn_k = shape.attention_k(shape.decoder_trainable)
    body = k * (shape.decoder_body_params + shape.decoder_head_params) * w.dec_positions
    if w.dec_sequences <= 0 or shape.decoder_full_attn_layers <= 0:
        return body
    mean_len = w.dec_positions / w.dec_sequences
    attn = (
        attn_k
        * w.dec_sequences
        * mean_len**2
        * shape.decoder_dim
        * shape.decoder_full_attn_layers
    )
    return body + attn


def step_flops(shape: ModelShape, w: Window) -> dict[str, float]:
    """FLOPs over the window, split per component and total/trainable.

    ``trainable`` counts only what produces a gradient. The two numbers answer
    different questions: total is what the hardware is doing, trainable is the
    efficiency of the work that actually trains the model. A frozen 580M encoder
    is pure overhead in the second column.
    """
    enc = encoder_flops(shape, w)
    ad = adapter_flops(shape, w)
    dec = decoder_flops(shape, w)
    trainable = (
        (enc if shape.encoder_trainable else 0.0)
        + (ad if shape.adapter_trainable else 0.0)
        + (dec if shape.decoder_trainable else 0.0)
    )
    return {
        "encoder": enc,
        "adapter": ad,
        "decoder": dec,
        "total": enc + ad + dec,
        "trainable": trainable,
    }


def point_metrics(
    shape: ModelShape,
    w: Window,
    s_per_step: float,
    world_size: int,
    peak_tflops: float = PEAK_BF16_TFLOPS,
) -> dict[str, float]:
    """Per-GPU throughput and MFU for one measured point.

    ``s_per_step`` is the *mean* step time and ``w`` the totals over ``w.steps``
    steps; the wall time is therefore their product. Getting that wrong is
    silent -- it just scales every rate by the window length.
    """
    if s_per_step <= 0:
        raise ValueError(f"s_per_step must be positive, got {s_per_step}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if w.steps <= 0:
        raise ValueError(f"window.steps must be positive, got {w.steps}")

    wall_s = s_per_step * w.steps
    flops = step_flops(shape, w)
    flops_per_s = flops["total"] / wall_s
    trainable_per_s = flops["trainable"] / wall_s
    peak = peak_tflops * 1e12

    audio_s_per_wall_s = w.audio_s / wall_s
    return {
        **flops,
        "s_per_step": s_per_step,
        "world_size": float(world_size),
        "tflops_per_gpu_total": flops_per_s / world_size / 1e12,
        "tflops_per_gpu_trainable": trainable_per_s / world_size / 1e12,
        "mfu_total": flops_per_s / world_size / peak,
        "mfu_trainable": trainable_per_s / world_size / peak,
        "encoder_share": flops["encoder"] / flops["total"] if flops["total"] else 0.0,
        "audio_s_per_wall_s": audio_s_per_wall_s,
        "audio_h_per_wall_h": audio_s_per_wall_s,
        "fill": w.fill,
        # Compute per audio hour is node-independent, so it is the one number
        # that compares recipes rather than topologies.
        "flops_per_audio_h": flops["total"] / (w.audio_s / 3600.0) if w.audio_s else 0.0,
    }


def walltime_for(audio_s_per_wall_s: float, gpus: int, hours: float) -> dict[str, float]:
    """Wall hours and GPU-hours to consume ``hours`` of audio at a measured rate.

    ``audio_s_per_wall_s`` is audio-seconds per wall-second, which is the same
    number as audio-hours per wall-hour (both numerator and denominator scale by
    3600), so the audio hours divide by it directly -- no unit factor.
    """
    if audio_s_per_wall_s <= 0:
        raise ValueError(f"audio_s_per_wall_s must be positive, got {audio_s_per_wall_s}")
    wall_h = hours / audio_s_per_wall_s
    return {
        "hours": hours,
        "wall_h": wall_h,
        "wall_days": wall_h / 24.0,
        "gpu_h": wall_h * gpus,
        "gpu_h_per_1k_audio_h": 1000.0 / audio_s_per_wall_s * gpus,
    }


def nominal_audio_s_per_step(batch_duration: float, grad_accum: int, world_size: int) -> float:
    """Audio seconds a step *should* carry, before bucket fill is measured.

    ``quadratic_duration: null`` in every current config, so lhotse charges a
    cut exactly its duration and this product is the exact nominal figure. The
    measured window will be lower, and the gap is the ``fill`` column.
    """
    return float(batch_duration) * float(grad_accum) * float(world_size)
