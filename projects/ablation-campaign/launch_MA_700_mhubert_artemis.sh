#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment), mHuBERT encoder arm,
# on artemis. Verification run: this is the first arm to use a RAW-WAVEFORM
# encoder, so it exists to prove the audio stack is correct before any
# full-length arm is committed to.
#
#   Encoder:  utter-project/mHuBERT-147  (frozen)
#   Decoder:  meta-llama/Llama-3.2-1B-Instruct  (frozen)
#   Adapter:  mlp                               (TRAINABLE)
#   Data:     ABL-MA-700-asr.yaml, unmodified -- same mixture, same bucket bins,
#             same max_duration 60 and max_tokens 400 as the w2v-BERT and
#             Whisper arms, so the arms differ only in the encoder.
#
# Run from the repo root ON ARTEMIS (sync first from nyx: infra/sync_repo.sh artemis):
#   bash projects/ablation-campaign/launch_MA_700_mhubert_artemis.sh
#
# Smoke test before anything longer:
#   MELT_TIME=01:00:00 bash projects/ablation-campaign/launch_MA_700_mhubert_artemis.sh \
#       --trainer.max_steps 50 --trainer.eval_on_start false \
#       --data.train_ds.buffer_size 2000 \
#       --run.exp_name mhubert-smoke --trainer.output_dir /workspace/outputs/mhubert-smoke
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable. `mhub147` = mHuBERT-147.
EXP_NAME="${EXP_NAME:-MA-700asr-mhub147F-llama1bInsF-mlpT-s42-2g-probe}"

# --- topology --------------------------------------------------------------
# artemis defaults to the h100 partition / gpu-h100 QOS (see
# infra/runners/sites/artemis.sh). mHuBERT-147 is a 95 M base model -- an order
# of magnitude smaller than either of the other two encoder arms -- so 2 x H100
# is generous and the memory ceiling here is the decoder, not the encoder.
export MELT_NODES="${MELT_NODES:-1}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-2}"
export MELT_SEED="${MELT_SEED:-42}"
export MELT_TIME="${MELT_TIME:-04:00:00}"

# --- step budget -----------------------------------------------------------
# An explicit step count, not one derived epoch: the point of this arm is to
# measure the encoder's real audio-per-step before committing to a length.
MAX_STEPS="${MAX_STEPS:-1500}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# --- launch ----------------------------------------------------------------
# One encoder override carries the whole arm, and its UNITS are the thing to
# read twice:
#
#   max_audio_seq_len 960000 -- for a raw-waveform encoder this counts SAMPLES,
#   not frames. mHuBERT's conv frontend lives inside the encoder, so what MELT
#   chunks is the waveform itself: 960000 = 60 s x 16 kHz, exactly the data's
#   max_duration, so nothing is ever chunked. MELTAudioEncoder rejects a
#   frame-sized value (the 1500 the other arms use) rather than silently
#   slicing every clip into 94 ms fragments.
#
# batch_duration is left alone, unlike the Whisper arm: there is no fixed
# encoder window here, so seconds of audio are again a truthful budget and the
# config's `batch_size: null` (pure dynamic batching) still holds.
#
# Known confound, accepted deliberately: mHuBERT-147 ships
# `apply_spec_augment: true` where w2v-BERT and Whisper both ship it false, so
# this arm's FROZEN encoder applies time masking during training and the other
# two do not. Left at the checkpoint default; revisit before drawing
# encoder-vs-encoder conclusions from the loss curves.
infra/runners/submit-container.sh artemis config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.encoder.name utter-project/mHuBERT-147 \
    --model.encoder.max_audio_seq_len 960000 \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.max_steps "${MAX_STEPS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed "${MELT_SEED}" \
    "$@"   # extra args pass through, e.g.
           #   --trainer.resume_from_checkpoint /workspace/outputs/<EXP_NAME>
