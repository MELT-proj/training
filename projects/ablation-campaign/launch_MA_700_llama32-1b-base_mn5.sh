#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment) at 700 h per language,
# on MN5, 2 nodes x 4 GPUs.  BASE backbone arm.
#
#   Backbone under test: meta-llama/Llama-3.2-1B            (frozen, BASE)
#   Encoder:             facebook/w2v-bert-2.0              (frozen)
#   Adapter:             mlp                                (TRAINABLE)
#   Data:                ABL-MA-700-asr.yaml — 5 languages x 700 h, ASR only
#
# Run from the repo root ON MN5:
#   bash projects/ablation-campaign/launch_MA_700_llama32-1b-base_mn5.sh
#
# This shares ABL-MA-700-asr.yaml with the instruct arm rather than forking it.
# Both arms apply the chat template with the same empty-instruction prompt, so
# the only thing that differs between them is the decoder -- which is the whole
# point of the pair. Verified on the real tokenizers: with the template borrowed
# (below), base and instruct produce *byte-identical* input_ids and an identical
# label span for the same cut, and neither adds a single new token to the vocab.
#
# Why chat_template_from is not optional here: a base checkpoint ships no chat
# template at all. meta-llama/Llama-3.2-1B has none; its Instruct sibling has
# 3,827 bytes of Jinja. Without this override the run raises
#   ValueError: chat_template_config is 'llama3' and apply_chat_template is on,
#              but this tokenizer has no chat template
# at startup. (It used to get all the way to the first batch before failing --
# past model construction and the queue wait.)
#
# <|eot_id|> and <|finetune_right_pad_id|> both live in the base vocabulary too,
# so the token settings in the YAML carry over untouched.
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable.
# The `-md60` suffix marks this as the re-run at max_duration 60 / max_tokens
# 400. The first MA run filtered at an unintended max_duration 30 with no token
# filter; its output directory (same name without the suffix) is left in place
# and must NOT be reused -- pointing a new run at it would make HF resume from
# its checkpoint-2188 rather than train from scratch.
EXP_NAME="MA-700asr-w2vbF-llama1bBaseF-mlpT-s42-8g-md60"

# --- topology --------------------------------------------------------------
# Identical to the instruct arm, deliberately: same data, same batch_duration,
# same derived one-epoch budget. See launch_MA_700_llama32-1b_mn5.sh for why
# MELT_GPUS_PER_NODE is pinned rather than autodetected (PR #90), and for the
# resume procedure. One epoch is ~4.1 h of training, so it fits in the 6 h
# requested below; resume only if infrastructure interrupts it.
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4          # -> world_size 8
export MELT_QOS="${MELT_QOS:-acc_ehpc}"
export MELT_SEED=42
export MELT_TIME="${MELT_TIME:-6:00:00}"

# --- step budget -----------------------------------------------------------
NUM_TRAIN_EPOCHS=1
EVAL_STEPS=200
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=2

# --- launch ----------------------------------------------------------------
# Note that nothing about the prompt *format* is overridden, and it should stay
# that way: a braced value cannot travel through OmegaConf's dotlist. It reads
# the leading brace as YAML flow-mapping syntax, so '{audio_token}' arrives as
# the dict {audio_token: None} -- a hard error since PR #100 (issue #94). Both
# overrides below are plain strings, which is exactly why this arm needs no
# config file of its own.
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.decoder.name meta-llama/Llama-3.2-1B \
    --model.decoder.chat_template_from meta-llama/Llama-3.2-1B-Instruct \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed 42
