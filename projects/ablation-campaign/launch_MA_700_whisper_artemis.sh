#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment), Whisper encoder arm,
# on artemis. Verification run: this is the first arm to use an encoder other
# than w2v-BERT, so it exists to prove the audio stack is correct before any
# full-length arm is committed to.
#
#   Encoder:  openai/whisper-large-v3  (encoder only, frozen)
#   Decoder:  meta-llama/Llama-3.2-1B-Instruct  (frozen)
#   Adapter:  mlp                               (TRAINABLE)
#   Data:     ABL-MA-700-asr.yaml, unmodified -- same mixture, same bucket bins,
#             same max_duration 60 and max_tokens 400 as the w2v-BERT arm, so
#             the two differ only in the encoder.
#
# Run from the repo root ON ARTEMIS (sync first from nyx: infra/sync_repo.sh artemis):
#   bash projects/ablation-campaign/launch_MA_700_whisper_artemis.sh
#
# Smoke test before anything longer:
#   MELT_QOS=gpu-debug MELT_PARTITION=a6000 MELT_GPUS_PER_NODE=1 MELT_TIME=00:30:00 \
#   bash projects/ablation-campaign/launch_MA_700_whisper_artemis.sh \
#       --trainer.max_steps 10 --trainer.eval_on_start false \
#       --data.train_ds.buffer_size 2000 \
#       --run.exp_name whisper-smoke --trainer.output_dir /workspace/outputs/whisper-smoke
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable. `whlv3` = whisper-large-v3.
EXP_NAME="${EXP_NAME:-MA-700asr-whlv3F-llama1bInsF-mlpT-s42-2g-probe}"

# --- topology --------------------------------------------------------------
# artemis defaults to the h100 partition / gpu-h100 QOS (see
# infra/runners/sites/artemis.sh). Whisper-large-v3's encoder is 0.64 B on top
# of the 1.24 B decoder, both frozen, so 2 x H100 is comfortable; the a6000
# debug queue is the right place for the 10-step smoke.
export MELT_NODES="${MELT_NODES:-1}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-2}"
export MELT_SEED="${MELT_SEED:-42}"
export MELT_TIME="${MELT_TIME:-04:00:00}"

# --- step budget -----------------------------------------------------------
# NOT one derived epoch, unlike the w2v-BERT arms. batch_size below caps items
# per batch, and whenever that cap binds a batch holds less audio than
# batch_duration budgeted -- so estimate_steps_per_epoch, which divides total
# hours by batch_duration alone, UNDER-counts the steps in an epoch. Drive this
# arm by an explicit step count until the real audio-per-step is measured.
MAX_STEPS="${MAX_STEPS:-1500}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# --- launch ----------------------------------------------------------------
# Two encoder overrides carry the whole arm:
#
#   max_audio_seq_len 3000 -- Whisper's encoder accepts exactly 3000 mel frames
#   (30 s at its 10 ms frame) and raises on anything else, so this is not a
#   tuning knob. MELTAudioEncoder folds longer inputs into whole windows at this
#   size, which is how max_duration can stay at 60 s; MELTAudioEncoder rejects
#   any other value for this encoder rather than letting it fail mid-forward.
#
#   batch_size 8 -> lhotse max_cuts. Load-bearing: every clip costs a full 30 s
#   encoder window however short it is, so batch_duration alone is the wrong
#   budget -- a 2.6 s bucket at batch_duration 150 would pack ~57 clips and ask
#   the encoder for ~1700 s of work. 8 caps that at 240 s. This is a starting
#   value; take the real one from the memory_preallocation peak in the smoke run.
infra/runners/submit-container.sh artemis config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.encoder.name openai/whisper-large-v3 \
    --model.encoder.max_audio_seq_len 3000 \
    --data.train_ds.batch_size 8 \
    --data.validation_ds.batch_size 8 \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.max_steps "${MAX_STEPS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed "${MELT_SEED}" \
    "$@"   # extra args pass through, e.g.
           #   --trainer.resume_from_checkpoint /workspace/outputs/<EXP_NAME>
