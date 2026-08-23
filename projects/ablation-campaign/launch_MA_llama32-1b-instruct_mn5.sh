#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment) on MN5, 2 nodes x 4 GPUs.
#
#   Backbone under test: meta-llama/Llama-3.2-1B-Instruct  (frozen)
#   Encoder:             facebook/w2v-bert-2.0             (frozen)
#   Adapter:             mlp                               (TRAINABLE — the only trained module)
#   Data:                ABL-MA-125-asr.yaml — 5 languages x 125 h, ASR only
#
# Run from the repo root ON MN5 (it submits with sbatch):
#   bash projects/ablation-campaign/launch_MA_llama32-1b-instruct_mn5.sh
#
# Overrides live here rather than in a new YAML so the difference from the
# shipped ABL-MA-125-asr.yaml stays visible in one place and in git, per
# docs/hpc_runbook.md ("prefer command-line overrides to new YAML files").
# Everything below is also recorded in the run's resolved_config.json.
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# Name encodes, in order: stage - hours+task - encoder - decoder - adapter -
# seed - world size.  Trailing F = frozen, T = trainable.
#   MA        modality alignment (stage 1)
#   125asr    125 h per language, ASR-only mix
#   w2vbF     w2v-bert-2.0 encoder, Frozen
#   llama1bInsF  Llama-3.2-1B-Instruct decoder, Frozen
#   mlpT      MLP adapter, Trainable
#   s42       seed 42
#   8g        world_size 8 (2 nodes x 4 GPUs) — recorded because a resume must
#             reuse the same topology or the lhotse sampler refuses to restore.
EXP_NAME="MA-125asr-w2vbF-llama1bInsF-mlpT-s42-8g"

# --- topology --------------------------------------------------------------
# MELT_GPUS_PER_NODE is pinned rather than left to autodetection: MN5 allocates
# `acc` nodes whole, so SLURM_GPUS_ON_NODE reports the node's full GPU count
# whatever --gpus-per-node asked for.  Pinning it is what makes world_size
# reproducible across a resume (see bash/run_train.sh and PR #90).
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4          # -> world_size 8
export MELT_TIME="${MELT_TIME:-08:00:00}"
export MELT_QOS="${MELT_QOS:-acc_ehpc}"
export MELT_SEED=42

# --- step budget -----------------------------------------------------------
# Campaign convention (2026-08-23): every arm, MA and IFT alike, runs ONE epoch
# and lets the trainer derive the step count.  No max_steps here on purpose --
# pinning it per-arm is how two arms end up trained for different amounts.
#
# What the trainer derives, at world_size 8 with quadratic_duration 35:
#   batches/epoch = ceil(625*3600 / 120 * 1.540) = 28879
#   steps/epoch   = ceil(28879 / world_size / 4) =   903
#
# The 1.540 is estimate_steps_per_epoch's correction for quadratic_duration:
# lhotse charges each cut `d + d^2/q`, so batch_duration budgets *effective*
# seconds and a batch holds less than 120 s of real audio.  Before that
# correction existed this config derived 586 steps, which covered only ~57% of
# the 625 h mix while the trainer logged `epoch: 1.0`.  903 steps covers ~88%.
#
# The residual ~12% is the correction's known bias: it reads the duration
# distribution off bucket_duration_bins via midpoints, which under-reads the
# mass in each bucket's upper half.  Measured truth at q=35 is 0.612 h/step, so
# a true single pass is ~1022 steps.  Under-correcting is the safe direction and
# the bias is identical for every arm, so arms stay comparable step-for-step.
NUM_TRAIN_EPOCHS=1

# eval_steps 100 gives ~9 rounds over the ~903 steps, plus eval_on_start.
# A round is cheap and gets cheaper: measured on 44947472 at world_size 8, a
# full 5-set round (5 x 200 = 1000 generations) took 240 s at step 0 and fell to
# 131 s by step 300 as the model learned to emit <|eot_id|> and stopped spending
# the whole 256-token budget on every sample.  So ~10 rounds costs ~25 min
# against a ~3.5 h run.  Halving this to 50 would double that overhead for
# resolution the WER curve does not need.
EVAL_STEPS=100

# save_steps scaled to the run: 200 gives 4 saves plus the final one, and is a
# multiple of EVAL_STEPS so saves land on eval boundaries.
# save_total_limit=2 keeps the two most recent (NOT the best-scoring) ones.
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=2

# --- launch ----------------------------------------------------------------
# Paths in overrides are CONTAINER paths (/workspace/outputs/...), not host ones.
#
# Decoder token overrides: ABL-MA-125-asr.yaml carries Qwen-oriented values.
# <|endoftext|> does not exist in Llama's vocabulary and would be *appended* as
# a brand-new token that the frozen decoder has never emitted; <|eot_id|> is the
# turn terminator Llama 3 actually uses, and it lands inside the loss under the
# llama3 boundary config, so the model learns to stop.  <|finetune_right_pad_id|>
# is the pad slot Meta reserved for exactly this, so nothing has to grow the
# vocabulary.  bos_token is inert in the current code path but is set correctly
# so resolved_config.json does not record a Qwen token for a Llama run.
#
# Chat template: the campaign runs MA on *instruct* backbones through the chat
# template with NO task instruction — the user turn carries only the audio
# (history/16-ablation-campaign.md §7).  chat_template_config must be llama3;
# under chatml the assistant span is simply never found and the run trains on
# nothing (the pairing is checked at startup, PR #89).
#
# The inner single quotes around {audio_token} are load-bearing.  CLI overrides
# become an OmegaConf dotlist, whose parser reads a bare {audio_token} as YAML
# flow-mapping syntax -- i.e. the dict {audio_token: None} -- not as a string.
# resolve_custom_template then rejects it ("Task 'asr' not found in
# prompt_template dict"), so the run dies at the first batch rather than
# training on something wrong.  Quoting forces a string.  NOTE this means the
# command printed in history/16-ablation-campaign.md §7 does not work as written.
infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
    --config projects/ablation-campaign/ABL-MA-125-asr.yaml \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --model.decoder.name meta-llama/Llama-3.2-1B-Instruct \
    --model.decoder.bos_token '<|begin_of_text|>' \
    --model.decoder.eos_token '<|eot_id|>' \
    --model.decoder.pad_token '<|finetune_right_pad_id|>' \
    --data.apply_chat_template true \
    --data.chat_template_config llama3 \
    --data.prompt_template_selection custom \
    --data.prompt_template "'{audio_token}'" \
    --trainer.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed 42
