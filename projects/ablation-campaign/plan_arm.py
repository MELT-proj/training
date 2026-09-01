#!/usr/bin/env python3
"""Compute one ablation arm's identity and CLI overrides from its axes.

Called by launch_MA.sh / launch_IFT.sh (via launch_campaign.sh) as::

    eval "$(python3 plan_arm.py --config ... --stage MA --world-size 8 ...)"

and never invoked directly by a human. It prints bash source-able text on
stdout: scalar ``KEY=VALUE`` assignments plus one ``OVERRIDE_ARGS=(...)``
array, each token quoted with `shlex.quote` so `eval` is safe regardless of
what characters an override value contains (see the `{audio_token}` note
below).

What it does, in order:

1. Read the chosen base config YAML with plain PyYAML (no OmegaConf, no melt
   import -- this has to run in any dev shell, not just inside the training
   container).
2. Compose EXP_NAME from the axes, matching the convention documented in the
   pre-refactor launch_MA_llama32-1b-instruct_mn5.sh header: stage - hours+task
   - encoder(+F/T) - decoder(+F/T) - adapter(+F/T) - lr - seed - world size.
3. Diff each requested architecture value against what the base config
   already declares, and only emit a CLI override where they differ. This is
   what makes the *default* axis values reproduce the two hand-written
   launchers' commands exactly: the 700 h config already has the campaign's
   Llama decoder and chat-template block baked in (so a matching DECODER
   default emits nothing), while the 125 h config still defaults to the
   Qwen-oriented SFT template (so the same default DECODER triggers the full
   decoder+chat-template override bundle) -- see #94/#100 and the 700 h
   config's own header for why declaring the chat template in YAML is
   preferred over overriding it.
4. Derive eval_steps/save_steps from steps-per-epoch at the requested
   world_size, replicating melt/training/data/audio/lhotse/dataloader.py's
   estimate_steps_per_epoch / _effective_duration_inflation exactly (duplicated
   here deliberately -- importing that module drags in torch/omegaconf, which
   is not installed in every shell this needs to run from).

Optimisation axes are ENCODER_LR, DECODER_LR and ADAPTER_LR (one CLI override
each, same inherit-or-override rule as the architecture axes), plus
DECODER_LORA for toggling model.lora.enabled. A config that omits an *_lr key
(several ABL-*-700 configs do, for whichever module that stage freezes) falls
back to melt/training/config.py's DEFAULT_CONFIG value rather than failing --
see DEFAULT_ENCODER_LR/DEFAULT_DECODER_LR/DEFAULT_ADAPTER_LR below.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import sys

import yaml

# Decoders the campaign knows how to point at without a bespoke CLI override:
# eos/pad tokens and the chat-template family that pairs with them. Adding a
# new backbone means adding an entry here (or declaring it in a new base YAML
# the way ABL-MA-700-asr.yaml does, which sidesteps this table entirely).
DECODER_PROFILES = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "eos_token": "<|eot_id|>",
        "pad_token": "<|finetune_right_pad_id|>",
        "chat_template_config": "llama3",
    },
    # Same eos/pad tokens as Qwen/Qwen3-1.7B (ABL-MA-125-asr.yaml,
    # ABL-IFT-125.yaml) -- the whole Qwen3.x line shares them. chatml is
    # correct per chat_templates.py's note on Qwen 3/3.5's empty
    # <think></think> block (masking is inclusive, so the boundary strings
    # still find the right span) and tests/test_chat_template_configs.py's
    # own ("Qwen/Qwen3.5-9B", "chatml") case.
    "Qwen/Qwen3.5-2B": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|text_pad|>",
        "chat_template_config": "chatml",
    },
}

# Short tags for EXP_NAME. Unknown names fall back to a sanitised slug (see
# _slug) rather than failing -- a new model should get *a* short tag, not
# block the launcher.
ENCODER_TAGS = {"facebook/w2v-bert-2.0": "w2vb"}
DECODER_TAGS = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama1bIns",
    "Qwen/Qwen3-1.7B": "qwen1_7b",
    "Qwen/Qwen3.5-2B": "qwen35_2b",
}

# Wall-clock defaults. 08:00:00 for the 125 h arm, preserved from the
# hand-written launcher this replaces. 06:00:00 for the 700 h arm: measured
# at 6.81 s/it on DDP (batch_duration 180, world_size 8), one epoch is ~4.1 h
# of training plus ~9 min startup, comfortably inside a 6 h allocation (and
# MN5's backfill scheduler starts a 6 h request sooner than the original
# 12:00:00, which assumed the slower FSDP2 throughput this arm no longer
# uses -- see README.md's "Why DDP and not FSDP2" section). Still expect to
# resume: infrastructure interrupts long jobs regardless of how well they fit.
# New budgets get a generic fallback -- check it against measured throughput
# before trusting it.
TIME_DEFAULTS = {
    "ABL-MA-125-asr.yaml": "08:00:00",
    "ABL-MA-700-asr.yaml": "06:00:00",
}
DEFAULT_TIME_FALLBACK = "08:00:00"

# Fallback learning rates, mirroring melt/training/config.py's DEFAULT_CONFIG
# optimization block. This script never imports melt (it has to run in any
# dev shell), so it can't read that merge directly -- these constants are the
# values OmegaConf would fill in when a base config omits a key. That
# omission is deliberate in several ABL-*-700 configs: a frozen module's LR
# has no effect, so e.g. ABL-MA-700-asr.yaml comments out encoder_lr/decoder_lr
# and ABL-IFT-700.yaml comments out encoder_lr/adapter_lr. Falling back here
# (instead of dying like the pre-refactor code did) is what lets those two
# configs plan at all -- die()ing on a missing key that the config omitted on
# purpose is not "flagging a mistake", it's just wrong.
DEFAULT_ENCODER_LR = "6e-6"
DEFAULT_DECODER_LR = "2e-5"
DEFAULT_ADAPTER_LR = "2e-4"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def as_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    die(f"expected true/false, got {s!r}")


def _slug(name: str, maxlen: int) -> str:
    base = name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9]", "", base)[:maxlen]


def encoder_tag(name: str) -> str:
    return ENCODER_TAGS.get(name, _slug(name, 12))


def decoder_tag(name: str) -> str:
    return DECODER_TAGS.get(name, _slug(name, 16))


def lr_tag(lr: str, prefix: str = "lr") -> str:
    # LR values like "2e-4" have no decimal point, so PyYAML reads them as
    # plain strings (YAML 1.1 float resolution requires a dot) -- we get the
    # exact text the config author wrote, which is what we want here.
    return prefix + lr.replace(".", "p").replace("-", "").replace("+", "")


def data_tag(config_path: str, stage: str) -> str:
    base = os.path.basename(config_path)
    prefix, suffix = f"ABL-{stage}-", ".yaml"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix): -len(suffix)].replace("-", "")
    # Non-conventional filename: best-effort fallback so this never crashes.
    print(
        f"WARNING: {base!r} does not match ABL-{stage}-<tag>.yaml; "
        "using a sanitised filename as the EXP_NAME data tag instead.",
        file=sys.stderr,
    )
    return re.sub(r"[^A-Za-z0-9]", "", base[: -len(suffix)] if base.endswith(suffix) else base)


def effective_duration_inflation(train_ds: dict) -> float:
    """Port of melt/training/data/audio/lhotse/dataloader.py:_effective_duration_inflation.

    Kept in lockstep with that function on purpose -- see its docstring for
    the derivation (lhotse charges each cut d + d^2/quadratic_duration, so the
    duration-weighted mean cut length over the bucket bins gives the batch
    count inflation). Verified against the two known-good arm step counts:
    2188 for ABL-MA-700-asr.yaml (quadratic_duration unset -> inflation 1.0)
    and 903 for ABL-MA-125-asr.yaml (inflation ~1.5402).
    """
    q = get(train_ds, "quadratic_duration")
    if q is None or float(q) <= 0:
        return 1.0
    q = float(q)

    bins = get(train_ds, "bucket_duration_bins")
    if not bins:
        print(
            f"WARNING: quadratic_duration={q:g} is set but bucket_duration_bins is "
            "not, so the steps-per-epoch estimate cannot be corrected for it "
            "(same caveat as dataloader.py's estimate_steps_per_epoch). Treating "
            "the inflation factor as 1.0, which UNDER-counts steps per epoch.",
            file=sys.stderr,
        )
        return 1.0

    lo = float(get(train_ds, "min_duration", 0.0) or 0.0)
    hi = float(get(train_ds, "max_duration", 0.0) or 0.0)
    edges = [lo] + [float(b) for b in bins]
    if hi > edges[-1]:
        edges.append(hi)
    if len(edges) < 2:
        return 1.0

    midpoints = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
    mean_weighted_duration = sum(midpoints) / len(midpoints)
    return 1.0 + mean_weighted_duration / q


def derive_steps(train_ds: dict, gradient_accumulation_steps: int, world_size: int) -> int:
    """Port of estimate_steps_per_epoch's optimizer-step branch (batch_duration path only:
    every ABL-* config sets batch_size: null, so that branch is not needed here)."""
    total_hours = get(train_ds, "total_hours")
    batch_duration = get(train_ds, "batch_duration")
    if total_hours is None:
        die("data.train_ds.total_hours is not set in the config; cannot derive steps")
    if not batch_duration or float(batch_duration) <= 0:
        die("data.train_ds.batch_duration is not set (or <= 0) in the config; cannot derive steps")

    total_duration = float(total_hours) * 3600.0
    inflation = effective_duration_inflation(train_ds)
    batches_per_epoch = math.ceil(total_duration / float(batch_duration) * inflation)
    batches_per_rank = batches_per_epoch / world_size
    return math.ceil(batches_per_rank / gradient_accumulation_steps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=["MA", "IFT"])
    p.add_argument("--world-size", required=True, type=int)
    p.add_argument("--adapter", required=True)
    p.add_argument("--adapter-freeze", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--encoder-freeze", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--decoder-freeze", required=True)
    p.add_argument("--decoder-lora", required=True, help="empty string means: use the config's own value, no override")
    p.add_argument("--encoder-lr", required=True, help="empty string means: use the config's own value, no override")
    p.add_argument("--decoder-lr", required=True, help="empty string means: use the config's own value, no override")
    p.add_argument("--adapter-lr", required=True, help="empty string means: use the config's own value, no override")
    p.add_argument("--seed", required=True, type=int)
    args = p.parse_args()

    if not os.path.isfile(args.config):
        die(f"config file not found: {args.config}")
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    train_ds = get(cfg, "data.train_ds")
    if train_ds is None:
        die(f"{args.config} has no data.train_ds section")
    grad_accum = get(cfg, "trainer.gradient_accumulation_steps")
    if not grad_accum:
        die(f"{args.config} has no trainer.gradient_accumulation_steps")
    grad_accum = int(grad_accum)

    steps = derive_steps(train_ds, grad_accum, args.world_size)
    # ~11 eval rounds over the run, per the campaign convention documented in
    # the old launchers' headers (both landed close to 11: 903/100~=9,
    # 2188/200~=11 -- this makes the derivation land on ~11 for every arm
    # instead of depending on someone hand-picking a round number). save_steps
    # is set equal to eval_steps so saves always land on eval boundaries.
    eval_steps = max(1, round(steps / 11))
    save_steps = eval_steps

    # Every architecture/optimisation axis below uses the SAME rule, uniformly:
    # an empty string means "inherit the base config's own value" (no CLI
    # override emitted; EXP_NAME still reflects the real, effective value so
    # it can never mislabel an arm). A non-empty value is an explicit request
    # and only turns into an override if it actually differs from the config.
    #
    # This is deliberately NOT "encoder/decoder frozen, adapter trainable" as
    # a hardcoded universal default: MA and IFT invert which module trains
    # (stage 1 trains only the adapter; stage 2 freezes the adapter stage 1
    # produced and trains the decoder on top of it -- see ABL-IFT-125.yaml's
    # own model.adapter/model.decoder comments). A hardcoded MA-shaped default
    # would silently flip IFT's freeze pattern the first time someone ran
    # launch_IFT.sh without overriding it. Inheriting is safe for both stages
    # by construction: it never asks the config to be anything other than
    # what it already declares unless a human says so.
    cfg_adapter_type = get(cfg, "model.adapter._type")
    cfg_adapter_freeze = bool(get(cfg, "model.adapter.freeze"))
    cfg_encoder_name = get(cfg, "model.encoder.name")
    cfg_encoder_freeze = bool(get(cfg, "model.encoder.freeze"))
    cfg_decoder_name = get(cfg, "model.decoder.name")
    cfg_decoder_freeze = bool(get(cfg, "model.decoder.freeze"))
    # model.lora.enabled is a single global toggle in the current model
    # builder (melt/training/train.py), not decoder-scoped -- see README.md's
    # "Decoder LoRA" note. bool(None) correctly reproduces the DEFAULT_CONFIG
    # default (false) for every ABL-*.yaml config, none of which declare a
    # lora: block of their own.
    cfg_decoder_lora = bool(get(cfg, "model.lora.enabled"))
    cfg_encoder_lr = get(cfg, "optimization.encoder_lr", DEFAULT_ENCODER_LR)
    cfg_decoder_lr = get(cfg, "optimization.decoder_lr", DEFAULT_DECODER_LR)
    cfg_adapter_lr = get(cfg, "optimization.adapter_lr", DEFAULT_ADAPTER_LR)

    adapter_effective = args.adapter or cfg_adapter_type
    encoder_effective = args.encoder or cfg_encoder_name
    decoder_effective = args.decoder or cfg_decoder_name
    if adapter_effective is None:
        die(f"{args.config} has no model.adapter._type and ADAPTER was not set")
    if encoder_effective is None:
        die(f"{args.config} has no model.encoder.name and ENCODER was not set")
    if decoder_effective is None:
        die(f"{args.config} has no model.decoder.name and DECODER was not set")
    adapter_freeze = as_bool(args.adapter_freeze) if args.adapter_freeze else cfg_adapter_freeze
    encoder_freeze = as_bool(args.encoder_freeze) if args.encoder_freeze else cfg_encoder_freeze
    decoder_freeze = as_bool(args.decoder_freeze) if args.decoder_freeze else cfg_decoder_freeze
    decoder_lora = as_bool(args.decoder_lora) if args.decoder_lora else cfg_decoder_lora

    overrides: list[str] = []

    if args.encoder and encoder_effective != cfg_encoder_name:
        overrides += ["--model.encoder.name", encoder_effective]
    if args.encoder_freeze and encoder_freeze != cfg_encoder_freeze:
        overrides += ["--model.encoder.freeze", str(encoder_freeze).lower()]

    if args.decoder and decoder_effective != cfg_decoder_name:
        profile = DECODER_PROFILES.get(decoder_effective)
        if profile is None:
            die(
                f"DECODER={decoder_effective!r} differs from the config's own "
                f"{cfg_decoder_name!r} and has no entry in DECODER_PROFILES "
                "(plan_arm.py), so its eos_token/pad_token/chat-template are "
                "unknown. Either add a profile entry, or declare the decoder "
                "block directly in a new base YAML the way ABL-MA-700-asr.yaml "
                "does, or pass the token overrides yourself as trailing args."
            )
        cfg_chat_template_config = get(cfg, "data.chat_template_config")
        if cfg_chat_template_config and cfg_chat_template_config != profile["chat_template_config"]:
            print(
                f"WARNING: overriding data.chat_template_config "
                f"{cfg_chat_template_config!r} -> {profile['chat_template_config']!r} "
                f"because DECODER={decoder_effective!r} was requested explicitly. "
                "If the config's own chat_template_config was a deliberate choice "
                "(e.g. IFT using chatml for a reason MA does not), double-check "
                "this override is actually wanted before submitting.",
                file=sys.stderr,
            )
        overrides += ["--model.decoder.name", decoder_effective]
        overrides += ["--model.decoder.eos_token", profile["eos_token"]]
        overrides += ["--model.decoder.pad_token", profile["pad_token"]]
        overrides += ["--data.apply_chat_template", "true"]
        overrides += ["--data.chat_template_config", profile["chat_template_config"]]
        overrides += ["--data.prompt_template_selection", "custom"]
        # The inner single quotes are load-bearing: a bare {audio_token} misparses
        # through OmegaConf's dotlist as the flow-mapping dict {audio_token: None}.
        # See #94/#100 (and the pre-refactor 125 h launcher's header, which this
        # bundle is a direct port of). No extra shell quoting is needed here --
        # this token goes straight into argv via OVERRIDE_ARGS, not through a
        # second round of shell parsing.
        overrides += ["--data.prompt_template", "'{audio_token}'"]
    if args.decoder_freeze and decoder_freeze != cfg_decoder_freeze:
        overrides += ["--model.decoder.freeze", str(decoder_freeze).lower()]

    if args.decoder_lora and decoder_lora != cfg_decoder_lora:
        overrides += ["--model.lora.enabled", str(decoder_lora).lower()]
    if decoder_lora and decoder_freeze:
        print(
            "WARNING: DECODER_LORA=true but the decoder is frozen "
            f"(DECODER_FREEZE={'true' if args.decoder_freeze else '<inherited>'}). "
            "train.py sets requires_grad from model.decoder.freeze unconditionally, "
            "including on the decoder's own LoRA params, so this combination "
            "trains nothing on the decoder -- a silent no-op ablation. Pass "
            "DECODER_FREEZE=false if the LoRA arm is meant to actually train.",
            file=sys.stderr,
        )

    if args.adapter and adapter_effective != cfg_adapter_type:
        overrides += ["--model.adapter._type", adapter_effective]
    if args.adapter_freeze and adapter_freeze != cfg_adapter_freeze:
        overrides += ["--model.adapter.freeze", str(adapter_freeze).lower()]

    if args.encoder_lr:
        encoder_lr_effective = args.encoder_lr
        overrides += ["--optimization.encoder_lr", args.encoder_lr]
    else:
        encoder_lr_effective = str(cfg_encoder_lr)

    if args.decoder_lr:
        decoder_lr_effective = args.decoder_lr
        overrides += ["--optimization.decoder_lr", args.decoder_lr]
    else:
        decoder_lr_effective = str(cfg_decoder_lr)

    if args.adapter_lr:
        adapter_lr_effective = args.adapter_lr
        overrides += ["--optimization.adapter_lr", args.adapter_lr]
    else:
        adapter_lr_effective = str(cfg_adapter_lr)

    exp_name = "-".join([
        args.stage,
        data_tag(args.config, args.stage),
        f"{encoder_tag(encoder_effective)}{'F' if encoder_freeze else 'T'}",
        f"{decoder_tag(decoder_effective)}{'F' if decoder_freeze else 'T'}" + ("-lora" if decoder_lora else ""),
        f"{adapter_effective}{'F' if adapter_freeze else 'T'}",
        lr_tag(encoder_lr_effective, "elr"),
        lr_tag(decoder_lr_effective, "dlr"),
        lr_tag(adapter_lr_effective, "lr"),
        f"s{args.seed}",
        f"{args.world_size}g",
    ])

    time_default = TIME_DEFAULTS.get(os.path.basename(args.config), DEFAULT_TIME_FALLBACK)

    out = []
    out.append(f"EXP_NAME={shlex.quote(exp_name)}")
    out.append(f"STEPS={steps}")
    out.append(f"EVAL_STEPS={eval_steps}")
    out.append(f"SAVE_STEPS={save_steps}")
    out.append(f"TIME_DEFAULT={shlex.quote(time_default)}")
    quoted = " ".join(shlex.quote(tok) for tok in overrides)
    out.append(f"OVERRIDE_ARGS=({quoted})")
    print("\n".join(out))


if __name__ == "__main__":
    main()
