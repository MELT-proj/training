#!/bin/bash
#
# Ablation campaign, stage 2 (IFT / instruction fine-tuning) at 700 h per
# language, on MN5, 2 nodes x 4 GPUs.
#
#   Backbone:  meta-llama/Llama-3.2-1B-Instruct  (TRAINABLE -- ~1.24 B params)
#   Encoder:   facebook/w2v-bert-2.0             (frozen)
#   Adapter:   mlp                               (frozen)
#   Init from: the stage-1 MA run's consolidated weights
#   Data:      ABL-IFT-700.yaml -- 5 ASR languages + 5 ST directions, 6,729.9 h
#
# Run from the repo root ON MN5:
#   bash projects/ablation-campaign/launch_IFT_700_llama32-1b_mn5.sh
#
# This is the mirror image of stage 1: there the adapter trained and everything
# else was frozen; here the decoder trains and everything else is frozen.
# Nothing about the model or the data is overridden below -- it all lives in
# ABL-IFT-700.yaml, including model.ckpt, which points at the stage-1 run
# directory. Stage 1 ran on DDP and therefore saved consolidated weights, so
# there is no merge step to do first.
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable. Note how the T moves from the adapter to
# the decoder relative to the stage-1 name.
EXP_NAME="IFT-700both-w2vbF-llama1bInsT-mlpF-s1337-8g"

# --- topology --------------------------------------------------------------
# MELT_GPUS_PER_NODE is pinned, not autodetected: MN5 allocates `acc` nodes
# whole, so SLURM_GPUS_ON_NODE reports the node's full GPU count whatever
# --gpus-per-node asked for. Pinning it is what makes world_size reproducible
# across a resume (bash/run_train.sh, PR #90).
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4          # -> world_size 8
export MELT_QOS="${MELT_QOS:-acc_ehpc}"

# 1337, not stage 1's 42. run_train.sh feeds MELT_SEED to
# --data.{train,validation}_ds.shard_seed, so this is what actually decides
# which cuts are streamed; leaving it at 42 would replay stage 1's slice of the
# shared ASR corpora against a model that has already seen it.
export MELT_SEED=1337

# --- wall clock ------------------------------------------------------------
# This arm does NOT fit in one allocation. One epoch is 6310 steps (derived,
# exact -- quadratic_duration is null, so the inflation factor is 1.0), each
# carrying 120 * 8 * 4 = 3840 audio-seconds.
#
# 6 h rather than 12: MN5's backfill scheduler starts short jobs sooner, and
# stage 1's three allocations queued for ~5 min each at this length. Two of
# those three were then lost to infrastructure (a NODE_FAIL and an NCCL
# collective timeout), which is the other reason to prefer several short
# allocations over one long one -- less is in flight when a node dies.
#
# To continue in the same directory, resubmit this same script with
#   --trainer.resume_from_checkpoint True
# appended (HF then scans output_dir for the last checkpoint itself). Never
# point it at a checkpoint-N/ subdirectory: train.py calls get_last_checkpoint
# on whatever it is given.
export MELT_TIME="${MELT_TIME:-6:00:00}"

# --- step budget -----------------------------------------------------------
# One epoch, steps derived, per the campaign convention: pinning max_steps per
# arm is how two arms end up trained for different amounts.
NUM_TRAIN_EPOCHS=1

# ~12 rounds over 6310 steps, plus eval_on_start. Each round decodes 200
# utterances for each of 6 named eval sets.
EVAL_STEPS=500

# Every 150 steps, keeping 2. These checkpoints are much larger than stage 1's:
# a trainable decoder means AdamW's moments are saved alongside the weights,
# so budget ~22 GB each, ~45 GB for the two kept.
#
# 150 rather than 500 because every previous attempt at this arm died of the
# host-RAM wall somewhere around step 340-360, which is *before* the first
# checkpoint at 500 -- so not one of them ever wrote anything and every retry
# restarted from zero. The wall itself is fixed now (the unbounded Shar reader
# cache, see _bound_indexed_reader_handles), but this is the first run of the
# arm since that landed, so the cheap insurance stays until it is proven at
# MN5 scale. Raise it back to 500 once this arm completes an allocation.
SAVE_STEPS=150
SAVE_TOTAL_LIMIT=2

# --- launch ----------------------------------------------------------------
# config/accelerate/ddp.yaml, as in stage 1. FSDP2 would shard the 1.24 B
# trainable decoder and its optimizer state across 8 ranks, which is the case
# sharding is actually for -- but it also turns activation checkpointing on and
# leaves the weights sharded at the end, needing a consolidation job before
# stage 3 or eval could read them (#91). DDP fits: the projected peak is
# ~46.8 GB of 64 (see the batch_duration comment in the YAML), and
# run.memory_preallocation checks that on step 1.
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-IFT-700.yaml \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed 1337 \
    "$@"
