#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment) at 700 h per language,
# on MN5, 2 nodes x 4 GPUs.
#
#   Backbone under test: meta-llama/Llama-3.2-1B-Instruct  (frozen)
#   Encoder:             facebook/w2v-bert-2.0             (frozen)
#   Adapter:             mlp                               (TRAINABLE)
#   Data:                ABL-MA-700-asr.yaml — 5 languages x 700 h, ASR only
#
# Run from the repo root ON MN5:
#   bash projects/ablation-campaign/launch_MA_700_llama32-1b_mn5.sh
#
# Unlike the 125 h launcher, almost nothing is overridden here: the decoder,
# its tokens and the whole chat-template block live in ABL-MA-700-asr.yaml.
# Declaring them in YAML also sidesteps issue #94 entirely -- a braced value
# only misparses when it travels through OmegaConf's dotlist, i.e. the CLI.
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable.
EXP_NAME="MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g"

# --- topology --------------------------------------------------------------
# MELT_GPUS_PER_NODE is pinned, not autodetected: MN5 allocates `acc` nodes
# whole, so SLURM_GPUS_ON_NODE reports the node's full GPU count whatever
# --gpus-per-node asked for. Pinning it is what makes world_size reproducible
# across a resume (bash/run_train.sh, PR #90).
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4          # -> world_size 8
export MELT_QOS="${MELT_QOS:-acc_ehpc}"
export MELT_SEED=42

# This arm does NOT fit in one allocation. At batch_duration 120, world_size 8
# and gradient_accumulation_steps 4, one epoch over 3500 h is
#   ceil(3500 * 3600 / 120 / 8 / 4) = 3282 steps
# and the 125 h arm measured ~160 audio-hours per wall-clock hour on FSDP with
# activation checkpointing, i.e. ~22 h. DDP without checkpointing should beat
# that, but plan on resuming: submit, let it hit the wall clock, resubmit with
# --trainer.resume_from_checkpoint pointed at the RUN DIRECTORY (never at a
# checkpoint-N subdir -- train.py calls get_last_checkpoint on what it is
# given). Raising batch_duration once the smoke run reports peak memory is the
# lever that shortens this.
export MELT_TIME="${MELT_TIME:-12:00:00}"

# --- step budget -----------------------------------------------------------
# One epoch, steps derived, per the campaign convention: pinning max_steps per
# arm is how two arms end up trained for different amounts.
#
# quadratic_duration is unset in this config, so batch_duration budgets real
# audio seconds and estimate_steps_per_epoch's inflation factor is exactly 1.0
# -- the derived 3282 is the true number of batches, not an approximation.
NUM_TRAIN_EPOCHS=1

# ~11 eval rounds over 3282 steps, plus eval_on_start. Each round decodes
# max_samples per named eval set, so it is not free; do not shrink this without
# a reason.
EVAL_STEPS=300
SAVE_STEPS=300
SAVE_TOTAL_LIMIT=2

# --- launch ----------------------------------------------------------------
# config/accelerate/ddp.yaml, not fsdp2.yaml: only the 6.3 M adapter trains, so
# sharding the frozen 1.8 B buys ~5% of a 64 GB card and charges an all-gather
# every forward. It also turns OFF activation checkpointing, which fsdp2.yaml
# had on. See the header of that file.
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed 42
