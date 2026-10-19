#!/bin/bash
#
# Self-distillation phase 2, step 1a: the GOLD warm start on Whisper (GOLDW).
#
# Standard MA (melt.training.train, CE on gold transcripts) with the
# five-language screen's Whisper recipe -- ABL-MA-700-asr.yaml data, frozen
# Whisper-large-v3 encoder, MLP adapter at stack_factor 5 (10 Hz), WSD schedule
# with 3% warmup -- and the decoder swapped to Qwen/Qwen3.5-2B, the decoder the
# self-distillation teacher is. Every phase-2 method arm continues from one of
# this run's checkpoints, so methods are compared past the alignment transition
# (phase 1 compared them before it; see README.md).
#
# The overrides are the ones plan_arm.py emits for this arm (DRY_RUN of
# launch_MA.sh with ENCODER/DECODER set), with three deliberate differences:
#   * an explicit step budget instead of one derived epoch, and
#     num_decay_steps at 20% of it, as the screen's WSD arms use;
#   * warmup as a fractional warmup_steps (transformers 5 drops warmup_ratio);
#   * eval every 50 h and a checkpoint every 100 h, to locate the transition
#     and to keep warm starts at several points along it.
#
#   # print the command, submit nothing
#   DRY_RUN=1 bash projects/self-distill/launch_goldw.sh
#
#   # submit (artemis h200 by default), extra train.py overrides pass through
#   bash projects/self-distill/launch_goldw.sh
#   bash projects/self-distill/launch_goldw.sh --trainer.resume_from_checkpoint /workspace/outputs/<EXP_NAME>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

SITE="${SITE:-artemis}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-projects/ablation-campaign/ABL-MA-700-asr.yaml}"
ENCODER="${ENCODER:-openai/whisper-large-v3}"
DECODER="${DECODER:-Qwen/Qwen3.5-2B}"
ADAPTER_LR="${ADAPTER_LR:-1e-3}"
# Seconds of audio per rank per optimizer step (grad_accum 1).
BATCH_DURATION="${BATCH_DURATION:-150}"
MAX_STEPS="${MAX_STEPS:-12000}"
EVAL_STEPS="${EVAL_STEPS:-600}"
SAVE_STEPS="${SAVE_STEPS:-1200}"
# Utterances per language (per named eval set) in each in-training eval.
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
export MELT_PARTITION="${MELT_PARTITION:-h200}"
export MELT_QOS="${MELT_QOS:-gpu-h200}"
export MELT_NODES="${MELT_NODES:-1}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-2}"
export MELT_TIME="${MELT_TIME:-24:00:00}"
export MELT_SEED="$SEED"
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))
DECAY_STEPS=$(( MAX_STEPS / 5 ))
HOURS=$(( MAX_STEPS * BATCH_DURATION * WORLD_SIZE / 3600 ))

# Decoder and encoder overrides from the campaign's own tables (no second copy here).
mapfile -t STACK_OVERRIDES < <(python3 - "${REPO_ROOT}/projects/ablation-campaign" "$DECODER" "$ENCODER" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from plan_arm import DECODER_PROFILES, ENCODER_ATTN_IMPLEMENTATION, ENCODER_WINDOW_FRAMES
decoder, encoder = sys.argv[2], sys.argv[3]
if decoder not in DECODER_PROFILES:
    sys.exit(f"DECODER={decoder!r} is not in plan_arm.DECODER_PROFILES")
p = DECODER_PROFILES[decoder]
pairs = [("model.decoder.name", decoder), ("model.decoder.eos_token", p["eos_token"]),
         ("model.decoder.pad_token", p["pad_token"]), ("data.chat_template_config", p["chat_template_config"])]
if "chat_template_from" in p:
    pairs.append(("model.decoder.chat_template_from", p["chat_template_from"]))
pairs.append(("model.encoder.name", encoder))
if encoder in ENCODER_WINDOW_FRAMES:
    pairs.append(("model.encoder.max_audio_seq_len", ENCODER_WINDOW_FRAMES[encoder]))
if encoder in ENCODER_ATTN_IMPLEMENTATION:
    pairs.append(("model.encoder.attn_implementation", ENCODER_ATTN_IMPLEMENTATION[encoder]))
for key, value in pairs:
    print(f"--{key}")
    print(value)
PY
)

case "$ENCODER" in
    openai/whisper-large-v3) ENCODER_TAG=whisperlarge ;;
    *) ENCODER_TAG="$(basename "$ENCODER" | tr -d '.')" ;;
esac
case "$DECODER" in
    Qwen/Qwen3.5-2B) DECODER_TAG=qwen35_2bIns ;;
    *) DECODER_TAG="$(basename "$DECODER" | tr -d '.')" ;;
esac
EXP_NAME="${EXP_NAME:-GOLDW-700asr-${ENCODER_TAG}F-${DECODER_TAG}F-mlpT-sk5-bd${BATCH_DURATION}-ga1-wsd-wus0p03-lr$(echo "$ADAPTER_LR" | tr -d '-')-${HOURS}h-s${SEED}-${WORLD_SIZE}g}"

CMD=(
    infra/runners/submit-container.sh "$SITE" config/accelerate/ddp.yaml
    --config "$CONFIG"
    --run.exp_name "$EXP_NAME"
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}"
    "${STACK_OVERRIDES[@]}"
    --model.encoder.eval_when_frozen true
    --data.apply_chat_template true
    --data.prompt_template_selection custom
    # Quoted inside the value, as plan_arm emits it: a bare {audio_token} is a
    # YAML flow mapping to OmegaConf's dotlist parser, and the run dies at startup.
    --data.prompt_template "'{audio_token}'"
    --model.adapter.stack_factor 5
    --data.train_ds.batch_duration "$BATCH_DURATION"
    --trainer.gradient_accumulation_steps 1
    --optimization.adapter_lr "$ADAPTER_LR"
    --trainer.max_steps "$MAX_STEPS"
    --trainer.lr_scheduler_type warmup_stable_decay
    --trainer.lr_scheduler_kwargs.num_decay_steps "$DECAY_STEPS"
    --trainer.lr_scheduler_kwargs.min_lr_ratio 0.1
    --trainer.lr_scheduler_kwargs.decay_type cosine
    --trainer.warmup_steps 0.03
    --trainer.eval_steps "$EVAL_STEPS"
    --trainer.save_steps "$SAVE_STEPS"
    --trainer.save_total_limit 20
    --trainer.eval_on_start false
    --data.validation_ds.max_samples "$EVAL_SAMPLES"
    --trainer.seed "$SEED"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] ${HOURS} h of audio over ${MAX_STEPS} steps (${BATCH_DURATION} s x ${WORLD_SIZE} ranks/step), decay ${DECAY_STEPS} steps"
    echo "[DRY_RUN] MELT_PARTITION=${MELT_PARTITION} MELT_QOS=${MELT_QOS} MELT_GPUS_PER_NODE=${MELT_GPUS_PER_NODE} MELT_TIME=${MELT_TIME}"
    printf '[DRY_RUN] would run:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
