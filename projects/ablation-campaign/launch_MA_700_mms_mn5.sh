#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment), MMS encoder arm, on MN5.
#
#   Encoder:  facebook/mms-1b                    (frozen)
#   Decoder:  meta-llama/Llama-3.2-1B-Instruct   (frozen)
#   Adapter:  mlp                                (TRAINABLE)
#   Data:     ABL-MA-700-asr.yaml, unmodified -- same mixture, same bucket bins,
#             same max_duration 60 and max_tokens 400 as the w2v-BERT, Whisper
#             and mHuBERT arms, so the arms differ only in the encoder.
#
# Run from the repo root ON MN5 (sync first from nyx: infra/sync_repo.sh mn5):
#   bash projects/ablation-campaign/launch_MA_700_mms_mn5.sh
#
# Smoke test before anything longer (acc_debug caps at 2 h but schedules
# immediately, and allows one job at a time -- check `squeue` is clear):
#   MELT_QOS=acc_debug MELT_TIME=00:40:00 MELT_NODES=1 \
#   bash projects/ablation-campaign/launch_MA_700_mms_mn5.sh \
#       --trainer.max_steps 50 --trainer.logging_steps 1 \
#       --run.exp_name mms-smoke --trainer.output_dir /workspace/outputs/mms-smoke
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# --- run identity ----------------------------------------------------------
# stage - hours+task - encoder - decoder - adapter - seed - world size.
# Trailing F = frozen, T = trainable.
EXP_NAME="${EXP_NAME:-MA-700asr-mms1bF-llama1bInsF-mlpT-s42-8g}"

# --- topology --------------------------------------------------------------
export MELT_NODES="${MELT_NODES:-2}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"   # pinned: SLURM_GPUS_ON_NODE
                                                       # lies on MN5's acc (PR #90)
export MELT_SEED="${MELT_SEED:-42}"
export MELT_TIME="${MELT_TIME:-06:00:00}"

MAX_STEPS="${MAX_STEPS:-1500}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# --- what makes this arm different -----------------------------------------
# Three encoder overrides, and the reason for each:
#
#   name -- facebook/mms-1b is a 962 M-parameter wav2vec2 checkpoint covering
#   1000+ languages. It is the only encoder in this campaign that can run flash
#   attention: w2v-BERT applies a relative-position bias inside self-attention
#   and transformers declares no flash support for it at all, while wav2vec2
#   injects position once before the stack (Wav2Vec2PositionalConvEmbedding) and
#   leaves the attention itself vanilla.
#
#   max_audio_seq_len 960000 -- SAMPLES, not frames, for a raw-waveform encoder:
#   60 s x 16 kHz, exactly the data's max_duration, so nothing is ever chunked.
#   Keep it equal to data.train_ds.max_duration x 16000. MELTAudioEncoder rejects
#   a frame-sized value (the 1500 the w2v-BERT arms use) rather than silently
#   slicing every clip into 94 ms fragments. Same rule as the mHuBERT arm.
#
#   attn_implementation flash_attention_2 -- the point of the arm. The config's
#   default is sdpa, which is all w2v-BERT can do; swap this to sdpa to measure
#   the two against each other on identical everything else.
#
# What does NOT change, and why the bins and max_tokens are untouched:
# MMS emits 2999 frames for 60 s of 16 kHz audio (verified against the real
# checkpoint), i.e. 49.98 Hz, against w2v-BERT's frame-synchronous 3000 at 50 Hz.
# One frame in 3000 is not a difference worth re-deriving a config for. And
# max_tokens is a filter on `custom.num_tokens`, a TEXT count, so the encoder's
# frame rate never entered it in the first place.
#
# Known confound, the same one the mHuBERT arm carries: mms-1b ships
# `apply_spec_augment: true` where w2v-BERT and Whisper both ship it false, so
# this arm's FROZEN encoder applies time masking during training and those two
# do not. Left at the checkpoint default; revisit before drawing
# encoder-vs-encoder conclusions from the loss curves.
#
# Unlike every other raw-waveform checkpoint MELT has met, mms-1b IS handed the
# attention mask: it is `feat_extract_norm: layer`, and encoder_specs.py reads
# that off the config. Measured on the real checkpoint, dropping the mask moves a
# padded clip's embeddings by 2.76 in absolute terms. Nothing to set here -- it
# is derived -- but it is the reason a mixed-length batch is trustworthy.
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.encoder.name facebook/mms-1b \
    --model.encoder.max_audio_seq_len 960000 \
    --model.encoder.attn_implementation flash_attention_2 \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.max_steps "${MAX_STEPS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed "${MELT_SEED}" \
    "$@"   # extra args pass through, e.g.
           #   --trainer.resume_from_checkpoint True
