#!/bin/bash
#
# Self-distillation arm (online AZeroS) -- OFF the campaign ledger.
#
# Submits melt.training.train_self_distill with self-distill.yaml, an overlay
# on ABL-MA-700-asr.yaml that matches MA-700-screen-w2vb-lr2e3-b300-evalfrozen
# except for the objective. Not routed through campaign.py: plan_arm.py has no
# entrypoint axis yet, and adding the arm to the grid is a PI decision. Until
# then this is the debugging escape hatch the README describes, nothing more.
#
#   # print the command, submit nothing
#   DRY_RUN=1 bash projects/ablation-campaign/launch_self_distill.sh
#
#   # online AZeroS control: teacher responses, hard-label cross-entropy
#   LMBDA=0 LOSS=ce bash projects/ablation-campaign/launch_self_distill.sh
#
#   # a short trial, extra train.py overrides after the env vars
#   MELT_TIME=02:00:00 MELT_QOS=acc_debug bash projects/ablation-campaign/launch_self_distill.sh \
#       --trainer.max_steps 500
#
# DECODER (default Qwen/Qwen3.5-2B) must be a key of plan_arm.DECODER_PROFILES, which
# supplies its eos/pad tokens and chat-template family. It must be an Instruct
# checkpoint: the teacher is the decoder answering the transcript.
#
# LMBDA, LOSS, BETA, MAX_NEW_TOKENS override the overlay's distill.* values
# ("" = inherit) and are tagged into EXP_NAME by effective value. The image must
# carry TRL (pyproject.toml's `distill` extra); the entrypoint checks before
# loading any model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-projects/ablation-campaign/self-distill.yaml}"
SITE="${SITE:-mn5}"
SEED="${SEED:-42}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-config/accelerate/ddp.yaml}"
# Per-rank seconds of audio x accumulation, as in plan_arm.py. The Qwen3.5-2B MA
# arm uses 30 s x 20; a rollout step holds the decoder twice, so trials start at 30 x 1.
BATCH_DURATION="${BATCH_DURATION:-30}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DECODER="${DECODER:-Qwen/Qwen3.5-2B}"
export MELT_NODES="${MELT_NODES:-1}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"
export MELT_SEED="$SEED"
export MELT_TRAIN_MODULE=melt.training.train_self_distill
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))

# Effective distill.* values: the env var if set, else the overlay's own.
read -r EFF_LMBDA EFF_LOSS EFF_BETA EFF_MAX_NEW < <(
    python3 - "$CONFIG" "${LMBDA:-}" "${LOSS:-}" "${BETA:-}" "${MAX_NEW_TOKENS:-}" <<'PY'
import sys, yaml
distill = (yaml.safe_load(open(sys.argv[1])) or {}).get("distill") or {}
keys = ("lmbda", "loss", "beta", "max_new_tokens")
missing = [k for k in keys if k not in distill]
if missing:
    sys.exit(f"{sys.argv[1]}: distill.{missing} not stated; the launcher names runs from the file")
values = [env or distill[k] for k, env in zip(keys, sys.argv[2:])]
# 1, 1.0 and "1.0" must give one name: format numbers, leave the loss name alone.
print(*[v if k == "loss" else f"{float(v):g}" for k, v in zip(keys, values)])
PY
)

# Decoder overrides from the campaign's own profile table (no second copy of it here).
mapfile -t DECODER_OVERRIDES < <(python3 - "$SCRIPT_DIR" "$DECODER" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from plan_arm import DECODER_PROFILES
name = sys.argv[2]
if name not in DECODER_PROFILES:
    sys.exit(f"DECODER={name!r} is not in plan_arm.DECODER_PROFILES")
p = DECODER_PROFILES[name]
for k, v in [("name", name), ("eos_token", p["eos_token"]), ("pad_token", p["pad_token"])]:
    print(f"--model.decoder.{k}")
    print(v)
print("--data.chat_template_config")
print(p["chat_template_config"])
if "chat_template_from" in p:
    print("--model.decoder.chat_template_from")
    print(p["chat_template_from"])
PY
)
OVERRIDES=("${DECODER_OVERRIDES[@]}")
[[ -n "${LMBDA:-}" ]] && OVERRIDES+=(--distill.lmbda "$LMBDA")
[[ -n "${LOSS:-}" ]] && OVERRIDES+=(--distill.loss "$LOSS")
[[ -n "${BETA:-}" ]] && OVERRIDES+=(--distill.beta "$BETA")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && OVERRIDES+=(--distill.max_new_tokens "$MAX_NEW_TOKENS")

case "$DECODER" in
    meta-llama/Llama-3.2-1B-Instruct) DECODER_TAG=llama1bInsF ;;
    Qwen/Qwen3.5-2B) DECODER_TAG=qwen35-2bInsF ;;
    *) DECODER_TAG="$(basename "$DECODER" | tr -d '.')F" ;;
esac
tag() { echo "$1" | sed 's/\./p/g'; }
BETA_TAG=""
[[ "$EFF_LOSS" == "jsd" ]] && BETA_TAG="-b$(tag "$EFF_BETA")"
EXP_NAME="${EXP_NAME:-SD-700asr-w2vbF-evalfrozen-${DECODER_TAG}-mlpT-sk5-bd${BATCH_DURATION}-ga${GRAD_ACCUM_STEPS}-lmb$(tag "$EFF_LMBDA")-${EFF_LOSS}${BETA_TAG}-gen${EFF_MAX_NEW}-s${SEED}-${WORLD_SIZE}g}"

CMD=(
    infra/runners/submit-container.sh "$SITE" "$ACCELERATE_CONFIG"
    --config "$CONFIG"
    --run.exp_name "$EXP_NAME"
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}"
    --data.train_ds.batch_duration "$BATCH_DURATION"
    --trainer.gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --trainer.seed "$SEED"
    "${OVERRIDES[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] MELT_TRAIN_MODULE=${MELT_TRAIN_MODULE} MELT_NODES=${MELT_NODES} MELT_GPUS_PER_NODE=${MELT_GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
    printf '[DRY_RUN] would run:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
