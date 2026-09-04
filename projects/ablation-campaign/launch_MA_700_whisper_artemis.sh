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
# hours by batch_duration alone, UNDER-counts the steps in an epoch.
#
# At the old cap of 8 that error was large: the cap bound for 81.6% of cuts and
# an epoch was 1.91x the w2v-BERT/mHuBERT arms' 501,037 steps. At 48 it binds
# for 22.1% and an epoch is 1.03x, so the estimate is now within a few percent
# -- but still low, so keep driving this arm by an explicit step count.
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
#   batch_size 48 -> lhotse max_cuts, per rank (dataloader.py:930; the
#   world-size multiplication there is deliberately commented out). This is a
#   MEMORY ceiling, not a compute budget -- an earlier comment here claimed the
#   cap was needed because "batch_duration alone is the wrong budget", which is
#   misleading. Total encoder work over an epoch is
#       sum(ceil(d/30)*30) = 3.20 x real audio
#   for this mixture, and that factor is a property of the 30 s window and the
#   duration distribution. No choice of cap changes it. All the cap decides is
#   how much REAL audio rides along with each 30 s window already being paid
#   for: at the old value of 8 we bought 240 encoder-seconds per micro-batch
#   and used 72 of them.
#
#   Lowering the cap therefore does not save compute, it only spends the same
#   compute on less data -- and with num_train_epochs 1 that shows up as extra
#   optimizer steps over the same corpus, which is a difference in optimisation
#   sitting inside an *encoder* ablation. Measured against the uncapped
#   w2v-BERT/mHuBERT arms (1103.0 audio-s per optimizer step, 501,037
#   steps/epoch), simulated over the config's own bins and the real filtered
#   distribution (55.8 M cuts / 153,507 h):
#
#     cap          audio/step   peak encoder   measured peak
#                  (% of base)  per micro-batch  (real training)
#     24            87.7%        720 s           33.7 GB
#     32            92.7%        960 s           35.7 GB
#     48 (chosen)   96.8%       1440 s           37.2 GB
#     none         100.0%       2970 s        not probed
#
#   Peaks are the steady-state maximum over ~30 logged training rows, probed on
#   artemis gpu-h100 (2xH100, jobs 329324/329325/329326) with
#   --data.train_ds.max_duration 6.0 to force every batch into the cap-bound
#   short buckets, where the peak lives. No OOM in any training step.
#
#   Memory is NOT the binding constraint, which is the probe's main result:
#   doubling the cap from 24 to 48 moved the steady-state peak by 3.5 GB, on an
#   80 GB H100. Per-cut marginal cost is ~0.15 GB; the ~33 GB floor is weights
#   plus decoder, not encoder padding. Higher caps were also faster per step
#   (60 steps in 818 s at 24 vs 697 s at 48) while doing more work per step.
#
#   Raising the cap is safe because the 30 s padding never reaches the decoder:
#   the processor's windowed branch leaves the tail window short, so
#   features_attention_mask counts only real mel frames (a 2.6 s clip gives a
#   (1, 128, 3000) tensor with mask.sum() == 260), and _encoder_output_mask maps
#   those lengths through. Decoder cost tracks real audio, which batch_duration
#   already bounds. The encoder itself is frozen and nothing else in the graph
#   requires grad, so no encoder activations are stored -- the windows are a
#   transient forward under flash_attention_2.
#
#   The validation cap is set to the same value only for symmetry with the
#   train override -- it does NOT govern eval. get_eval_dataloader builds a
#   stock PyTorch DataLoader from MELTMapDataset, not a Lhotse sampler, so eval
#   batching is per_device_eval_batch_size (4 in this config) and eval memory
#   is unaffected by this cap. On 64 GB H100s 4 is known-good and 16 OOMs --
#   see the guard in trainer.py that refuses the -1 sentinel for eval.
#
#   NOTE: the preallocation warmup will log an OOM warning at this cap. It is
#   the synthetic batch -- every utterance at max_tokens+32 = 432 text tokens,
#   which no real batch can produce because max_tokens is a per-cut filter
#   (dataloader.py _token_filter), not a batch budget. It is non-fatal by
#   design; the numbers above come from gpu_peak_gb on real training rows.
infra/runners/submit-container.sh artemis config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.encoder.name openai/whisper-large-v3 \
    --model.encoder.max_audio_seq_len 3000 \
    --data.train_ds.batch_size 48 \
    --data.validation_ds.batch_size 48 \
    --run.exp_name "${EXP_NAME}" \
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}" \
    --trainer.max_steps "${MAX_STEPS}" \
    --trainer.eval_steps "${EVAL_STEPS}" \
    --trainer.save_steps "${SAVE_STEPS}" \
    --trainer.save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --trainer.seed "${MELT_SEED}" \
    "$@"   # extra args pass through, e.g.
           #   --trainer.resume_from_checkpoint /workspace/outputs/<EXP_NAME>
