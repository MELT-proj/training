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
# The `-md60` suffix marks this as the re-run at max_duration 60 / max_tokens
# 400. The first MA run filtered at an unintended max_duration 30 with no token
# filter; its output directory (same name without the suffix) is left in place
# and must NOT be reused -- pointing a new run at it would make HF resume from
# its checkpoint-2188 rather than train from scratch.
EXP_NAME="MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g-md60"

# --- topology --------------------------------------------------------------
# MELT_GPUS_PER_NODE is pinned, not autodetected: MN5 allocates `acc` nodes
# whole, so SLURM_GPUS_ON_NODE reports the node's full GPU count whatever
# --gpus-per-node asked for. Pinning it is what makes world_size reproducible
# across a resume (bash/run_train.sh, PR #90).
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4          # -> world_size 8
export MELT_QOS="${MELT_QOS:-acc_ehpc}"
export MELT_SEED=42

# This arm DOES fit in one allocation. At batch_duration 180, world_size 8 and
# gradient_accumulation_steps 4, one epoch over 3500 h is
#   ceil(3500 * 3600 / 180 / 8 / 4) = 2188 steps
# each carrying 180 * 8 * 4 = 5760 audio-seconds. The completed run measured
# 6.81 s/it in steady state (846 audio-s per wall-second, 4.0x the FSDP
# configuration's 209.7), so one epoch is ~4.1 h of training plus ~9 min of
# one-time startup.
#
# 6 h rather than 12: it fits with room to spare, and MN5's backfill scheduler
# starts short jobs sooner -- every allocation at this length has queued for
# under 7 minutes.
#
# Resume anyway, because infrastructure will interrupt you: the first attempt
# at this arm needed three allocations, losing one to a NODE_FAIL and one to an
# NCCL collective timeout. Resubmit with --trainer.resume_from_checkpoint
# pointed at the RUN DIRECTORY, never at a checkpoint-N subdir (train.py calls
# get_last_checkpoint on whatever it is given).
export MELT_TIME="${MELT_TIME:-6:00:00}"

# --- step budget -----------------------------------------------------------
# One epoch, steps derived, per the campaign convention: pinning max_steps per
# arm is how two arms end up trained for different amounts.
#
# quadratic_duration is unset in this config, so batch_duration budgets real
# audio seconds and estimate_steps_per_epoch's inflation factor is exactly 1.0
# -- the derived 2188 is the true number of batches, not an approximation.
NUM_TRAIN_EPOCHS=1

# ~11 eval rounds over 2188 steps, plus eval_on_start. Each round decodes
# max_samples per named eval set, so it is not free; do not shrink this without
# a reason.
EVAL_STEPS=200
SAVE_STEPS=200
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
