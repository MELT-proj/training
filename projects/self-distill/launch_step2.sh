#!/bin/bash
#
# Self-distillation phase 2, step 2: does on-policy distillation add anything
# over CE for ASR, from an aligned start? One launcher, four arms, all from the
# same GOLDW checkpoint, same data, steps, LR schedule and instruction:
#
#   ARM=ce          standard MA (melt.training.train): CE on the gold transcript
#   ARM=soft        forward KL on the teacher's greedy repeat (off-policy, soft labels)
#   ARM=opd         reverse KL on student samples (on-policy)
#   ARM=azeros      CE on the teacher's greedy reply (off-policy, hard labels: AZeroS)
#   ARM=<arm>_anchor  any distillation arm + GOLD_CE_WEIGHT x the gold CE (opd_anchor, ...)
#
# Step 3 (instruction mix): TRANSLATE_FRAC > 0 gives that share of utterances
# a "translate into <language>" instruction (ST_TEMPLATE) instead of the repeat
# one, for student and teacher alike (distill.translate_frac); the ce arm has
# no gold for those and is refused. The run name then starts S3- and says mix<frac>.
#
# The teacher is the frozen decoder reading the transcript under the student's
# own instruction (distill.teacher_prompt=mirror). The instruction is the one
# the teacher audit found the teacher reproduces exactly (README.md); with it
# the teacher's greedy repeat is the transcript, so an offline teacher-target
# CE arm would be the ce arm again.
#
#   DRY_RUN=1 ARM=opd WARM_START=/workspace/outputs/<GOLDW run>/checkpoint-N bash projects/self-distill/launch_step2.sh
#   ARM=opd WARM_START=... bash projects/self-distill/launch_step2.sh [extra train overrides]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

ARM="${ARM:?set ARM=ce|soft|opd|azeros, or a distillation arm with _anchor}"
WARM_START="${WARM_START:?set WARM_START to a GOLDW checkpoint path as the container sees it (/workspace/outputs/...)}"
SITE="${SITE:-artemis}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-projects/ablation-campaign/ABL-MA-700-asr.yaml}"
ENCODER="${ENCODER:-openai/whisper-large-v3}"
DECODER="${DECODER:-Qwen/Qwen3.5-2B}"
BATCH_DURATION="${BATCH_DURATION:-30}"
MAX_STEPS="${MAX_STEPS:-3000}"
ADAPTER_LR="${ADAPTER_LR:-2e-4}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-448}"
GOLD_CE_WEIGHT="${GOLD_CE_WEIGHT:-0.5}"
# Literal \n: the value is handed to OmegaConf as a YAML double-quoted scalar,
# which turns them into the newlines the teacher audit tested.
TEMPLATE="${TEMPLATE:-Repeat the following content exactly, word for word, and write nothing else.\\n\\n{audio_token}}"
ST_TEMPLATE="${ST_TEMPLATE:-Translate the following content into {tgt_lang}, and write nothing else.\\n\\n{audio_token}}"
TRANSLATE_FRAC="${TRANSLATE_FRAC:-0}"
export MELT_PARTITION="${MELT_PARTITION:-h200}"
export MELT_QOS="${MELT_QOS:-gpu-h200}"
export MELT_NODES="${MELT_NODES:-1}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-2}"
export MELT_TIME="${MELT_TIME:-24:00:00}"
export MELT_SEED="$SEED"
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))

MIX=0
awk "BEGIN { exit !(${TRANSLATE_FRAC} > 0) }" && MIX=1
BASE_ARM="${ARM%_anchor}"
case "$BASE_ARM" in
    ce)
        [[ "$ARM" == ce ]] || { echo "ERROR: ce has no anchor variant: it is the anchor" >&2; exit 1; }
        (( MIX == 0 )) || { echo "ERROR: TRANSLATE_FRAC > 0 with ARM=ce: translations have no gold" >&2; exit 1; }
        export MELT_TRAIN_MODULE=melt.training.train
        ARM_OVERRIDES=() ;;
    soft)
        export MELT_TRAIN_MODULE=melt.training.train_self_distill
        ARM_OVERRIDES=(--distill.lmbda 0 --distill.loss jsd --distill.beta 0 --distill.temperature 0) ;;
    opd)
        export MELT_TRAIN_MODULE=melt.training.train_self_distill
        ARM_OVERRIDES=(--distill.lmbda 1 --distill.loss jsd --distill.beta 1 --distill.temperature 1) ;;
    azeros)
        export MELT_TRAIN_MODULE=melt.training.train_self_distill
        ARM_OVERRIDES=(--distill.lmbda 0 --distill.loss ce --distill.temperature 0) ;;
    *) echo "ERROR: unknown ARM=$ARM" >&2; exit 1 ;;
esac
[[ "$ARM" != "$BASE_ARM" ]] && ARM_OVERRIDES+=(--distill.gold_ce_weight "$GOLD_CE_WEIGHT")
if [[ "$MELT_TRAIN_MODULE" == melt.training.train_self_distill ]]; then
    ARM_OVERRIDES+=(--distill.teacher_prompt mirror --distill.max_new_tokens "$MAX_NEW_TOKENS"
                    --distill.top_p 1.0 --distill.eval_alignment_gap true --run.memory_preallocation false
                    --distill.translate_frac "$TRANSLATE_FRAC")
    # TRL is not in the image; .trl-overlay (pip --target trl==0.29.1) is, under the repo root.
    [[ -d .trl-overlay/trl ]] || { echo "ERROR: .trl-overlay/trl missing under $REPO_ROOT" >&2; exit 1; }
    export SINGULARITYENV_PYTHONPATH=/workspace/training/.trl-overlay
fi

# The weights and model config come from WARM_START, but train.py still builds a
# processor from model.encoder/decoder.name before loading it, reads the encoder
# attention backend from the YAML (a checkpoint does not record one), and saves
# resolved_config.json from these values -- so describe the warm start's stack
# exactly as launch_goldw.sh did, or the run says w2v-BERT + Llama-1B.
mapfile -t STACK_OVERRIDES < <(python3 - "${REPO_ROOT}/projects/ablation-campaign" "$DECODER" "$ENCODER" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from plan_arm import DECODER_PROFILES, ENCODER_ATTN_IMPLEMENTATION, ENCODER_WINDOW_FRAMES
decoder, encoder = sys.argv[2], sys.argv[3]
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

# GOLDW-checkpoint-1200 for a mid-run checkpoint, GOLDW-final for a finished run's root.
case "$(basename "$WARM_START")" in
    checkpoint-*|warmstart-*) WARM_TAG="$(basename "$(dirname "$WARM_START")" | cut -c1-5)-$(basename "$WARM_START")" ;;
    *) WARM_TAG="$(basename "$WARM_START" | cut -c1-5)-final" ;;
esac
if (( MIX )); then
    STAGE=S3; PROMPT_TAG="mix$(echo "$TRANSLATE_FRAC" | tr -d '.')"
    # The training template becomes a per-task mapping; eval stays on the repeat instruction.
    TEMPLATE_OVERRIDES=(--data.prompt_template.asr "\"${TEMPLATE}\"" --data.prompt_template.st "\"${ST_TEMPLATE}\"")
else
    STAGE=S2; PROMPT_TAG=repeatfirst
    TEMPLATE_OVERRIDES=(--data.prompt_template "\"${TEMPLATE}\"")
fi
EXP_NAME="${EXP_NAME:-${STAGE}-${ARM}-from-${WARM_TAG}-${PROMPT_TAG}-bd${BATCH_DURATION}-lr$(echo "$ADAPTER_LR" | tr -d '-')-${MAX_STEPS}st-s${SEED}-${WORLD_SIZE}g}"

CMD=(
    infra/runners/submit-container.sh "$SITE" config/accelerate/ddp.yaml
    --config "$CONFIG"
    --run.exp_name "$EXP_NAME"
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}"
    --model.ckpt "$WARM_START"
    "${STACK_OVERRIDES[@]}"
    --model.adapter.stack_factor 5
    --model.encoder.eval_when_frozen true
    --data.apply_chat_template true
    --data.prompt_template_selection custom
    "${TEMPLATE_OVERRIDES[@]}"
    --data.validation_ds.prompt_template "\"${TEMPLATE}\""
    --data.train_ds.batch_duration "$BATCH_DURATION"
    --trainer.gradient_accumulation_steps 1
    --optimization.adapter_lr "$ADAPTER_LR"
    --trainer.max_steps "$MAX_STEPS"
    --trainer.lr_scheduler_type warmup_stable_decay
    --trainer.lr_scheduler_kwargs.num_decay_steps "$(( MAX_STEPS / 5 ))"
    --trainer.lr_scheduler_kwargs.min_lr_ratio 0.1
    --trainer.lr_scheduler_kwargs.decay_type cosine
    --trainer.warmup_steps 0.05
    --trainer.eval_steps "$EVAL_STEPS"
    --trainer.save_steps "$SAVE_STEPS"
    --trainer.save_total_limit 10
    --trainer.eval_on_start true
    --data.validation_ds.max_samples "$EVAL_SAMPLES"
    --trainer.seed "$SEED"
    "${ARM_OVERRIDES[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] MELT_TRAIN_MODULE=${MELT_TRAIN_MODULE} SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH:-} WORLD_SIZE=${WORLD_SIZE}"
    printf '[DRY_RUN] would run:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

exec "${CMD[@]}"
