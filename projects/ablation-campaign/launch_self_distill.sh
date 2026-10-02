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
# The reference arm's batch: 75 s x 1 x 4 ranks = 300 s. Paired, as in plan_arm.py.
BATCH_DURATION="${BATCH_DURATION:-75}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
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

OVERRIDES=()
[[ -n "${LMBDA:-}" ]] && OVERRIDES+=(--distill.lmbda "$LMBDA")
[[ -n "${LOSS:-}" ]] && OVERRIDES+=(--distill.loss "$LOSS")
[[ -n "${BETA:-}" ]] && OVERRIDES+=(--distill.beta "$BETA")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && OVERRIDES+=(--distill.max_new_tokens "$MAX_NEW_TOKENS")

tag() { echo "$1" | sed 's/\./p/g'; }
BETA_TAG=""
[[ "$EFF_LOSS" == "jsd" ]] && BETA_TAG="-b$(tag "$EFF_BETA")"
EXP_NAME="${EXP_NAME:-SD-700asr-w2vbF-evalfrozen-llama1bInsF-mlpT-sk5-bd${BATCH_DURATION}-ga${GRAD_ACCUM_STEPS}-lmb$(tag "$EFF_LMBDA")-${EFF_LOSS}${BETA_TAG}-gen${EFF_MAX_NEW}-s${SEED}-${WORLD_SIZE}g}"

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
