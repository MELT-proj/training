#!/bin/bash
#
# Shared core for the ablation campaign's launchers, on MN5 (2 nodes x 4
# GPUs) by default; set SITE=artemis for a 1-node smoke test before
# committing an MN5 allocation. Not meant to be run directly -- launch_MA.sh
# and launch_IFT.sh set STAGE and a per-stage CONFIG default and exec this.
# See projects/ablation-campaign/README.md for the axes this exposes and the
# two correctness rules that matter more than the layout.
#
# Replaces launch_MA_llama32-1b-instruct_mn5.sh and launch_MA_700_llama32-1b_mn5.sh
# (both 125 h and 700 h are now the same launcher with a different CONFIG).
#
#   CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml \
#   ADAPTER=mlp DECODER=meta-llama/Llama-3.2-1B-Instruct \
#   ADAPTER_LR=2e-5 SEED=42 \
#   bash projects/ablation-campaign/launch_MA.sh
#
# --- axes -------------------------------------------------------------------
#   Data           CONFIG               one YAML per budget x task (~550 lines,
#                                        rendered by build_campaign_config.py --
#                                        never write a new one per arm)
#   Architecture   ADAPTER, ADAPTER_FREEZE, ENCODER, ENCODER_FREEZE,
#                  DECODER, DECODER_FREEZE, DECODER_LORA     (2-7 CLI overrides)
#   Optimisation   ENCODER_LR, DECODER_LR, ADAPTER_LR         (0-3 CLI overrides)
#
# Every axis below defaults to EMPTY, which plan_arm.py reads as "inherit
# the chosen CONFIG's own value" -- no CLI override, no assumption about
# which module should be frozen. This is deliberate, not lazy: MA trains only
# the adapter (encoder+decoder frozen) while IFT freezes the adapter stage 1
# produced and trains the decoder on top of it (see ABL-IFT-125.yaml's
# model.adapter/model.decoder comments) -- a single hardcoded "sensible
# default" would be right for one stage and silently wrong for the other.
# Leaving an axis unset always reproduces exactly what the base config already
# says; set it explicitly to run an ablation. Either way the *effective* value
# (whatever ends up training) is what lands in EXP_NAME, so an arm can never
# be mislabelled by hand -- plan_arm.py owns this, and it also skips emitting
# an override whenever the requested value already matches the base config,
# which is what makes the parity check in the PR description work (see its
# docstring).
#
# --run.exp_name and --trainer.output_dir are ALWAYS set from the composed
# EXP_NAME; eval_steps/save_steps are ALWAYS derived from the config's own
# total_hours/batch_duration/quadratic_duration/gradient_accumulation_steps at
# the actual world_size (see plan_arm.py's derive_steps / effective_duration_inflation,
# a deliberate duplicate of melt/training/data/audio/lhotse/dataloader.py's
# estimate_steps_per_epoch so this can run without importing melt).
set -euo pipefail

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }
: "${STAGE:?launch_campaign.sh must be run via launch_MA.sh or launch_IFT.sh (STAGE unset)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:?set CONFIG=<path to an ABL-*.yaml>}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-config/accelerate/ddp.yaml}"

# Empty = inherit from CONFIG (see the header note above); non-empty
# overrides that key and always feeds the *requested* value into EXP_NAME.
ADAPTER="${ADAPTER:-}"
ADAPTER_FREEZE="${ADAPTER_FREEZE:-}"
ENCODER="${ENCODER:-}"
ENCODER_FREEZE="${ENCODER_FREEZE:-}"
DECODER="${DECODER:-}"
DECODER_FREEZE="${DECODER_FREEZE:-}"
DECODER_LORA="${DECODER_LORA:-}"
ENCODER_LR="${ENCODER_LR:-}"
DECODER_LR="${DECODER_LR:-}"
ADAPTER_LR="${ADAPTER_LR:-}"
SEED="${SEED:-42}"

# --- site / topology ----------------------------------------------------------
# SITE selects infra/runners/sites/<site>.sh (see submit-container.sh). Topology
# and QoS defaults below match each site's own campaign convention -- mn5's
# full 2-node x 4-GPU arms vs. a 1-node x 2-GPU smoke test on artemis, which is
# what MELT_QOS/MELT_PARTITION default to under SITE=artemis (its own site file
# defaults the same way; set here too so DRY_RUN's world_size is right without
# depending on sourcing order).
#
# MELT_GPUS_PER_NODE is pinned rather than left to autodetection: MN5 allocates
# `acc` nodes whole, so SLURM_GPUS_ON_NODE reports the node's full GPU count
# whatever --gpus-per-node asked for. Pinning it is what makes world_size
# reproducible across a resume (see bash/run_train.sh and PR #90).
SITE="${SITE:-mn5}"
if [[ "$SITE" == "artemis" ]]; then
    export MELT_NODES="${MELT_NODES:-1}"
    export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-2}"
    export MELT_QOS="${MELT_QOS:-gpu-h100}"
    export MELT_PARTITION="${MELT_PARTITION:-h100}"
else
    export MELT_NODES="${MELT_NODES:-2}"
    export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"
    export MELT_QOS="${MELT_QOS:-acc_ehpc}"
fi
WORLD_SIZE=$((MELT_NODES * MELT_GPUS_PER_NODE))
export MELT_SEED="$SEED"

# --- derive EXP_NAME, eval/save steps, and the architecture override bundle --
PLAN="$(python3 "${SCRIPT_DIR}/plan_arm.py" \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --world-size "$WORLD_SIZE" \
    --adapter "$ADAPTER" --adapter-freeze "$ADAPTER_FREEZE" \
    --encoder "$ENCODER" --encoder-freeze "$ENCODER_FREEZE" \
    --decoder "$DECODER" --decoder-freeze "$DECODER_FREEZE" \
    --decoder-lora "$DECODER_LORA" \
    --encoder-lr "$ENCODER_LR" --decoder-lr "$DECODER_LR" --adapter-lr "$ADAPTER_LR" \
    --seed "$SEED")" || { echo "ERROR: plan_arm.py failed (see above)" >&2; exit 1; }
eval "$PLAN"
# PLAN sets: EXP_NAME, STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT,
#            TIME_DEFAULT, OVERRIDE_ARGS[]

export MELT_TIME="${MELT_TIME:-$TIME_DEFAULT}"

# --- launch --------------------------------------------------------------
# Paths in overrides are CONTAINER paths (/workspace/outputs/...), not host
# ones (see infra/runners/submit-container.sh).
#
# One epoch, steps derived: campaign convention (2026-08-23) is that every
# arm, MA and IFT alike, lets the step count fall out of num_train_epochs=1
# rather than pinning max_steps per-arm, which is how two arms end up trained
# for different amounts by accident.
CMD=(infra/runners/submit-container.sh "$SITE" "$ACCELERATE_CONFIG"
    --config "$CONFIG"
    --run.exp_name "$EXP_NAME"
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}"
    "${OVERRIDE_ARGS[@]}"
    --trainer.num_train_epochs 1
    --trainer.eval_steps "$EVAL_STEPS"
    --trainer.save_steps "$SAVE_STEPS"
    --trainer.save_total_limit "$SAVE_TOTAL_LIMIT"
    --trainer.seed "$SEED"
    "$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] STEPS=${STEPS} (derived) EVAL_STEPS=${EVAL_STEPS} SAVE_STEPS=${SAVE_STEPS} WORLD_SIZE=${WORLD_SIZE}"
    printf '[DRY_RUN] would run:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

# Capture sbatch's stdout (which includes submit-container.sh's own
# "[submit-container] ... sbatch ..." echo of the full command, and sbatch's
# own "Submitted batch job N") while still showing it live, so arms.tsv can
# be filled in from what actually ran rather than from shell history.
SUBMIT_OUTPUT="$("${CMD[@]}" | tee /dev/stderr)"
JOB_ID="$(printf '%s\n' "$SUBMIT_OUTPUT" | sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' | tail -n1)"
JOB_ID="${JOB_ID:-UNKNOWN}"

ARMS_TSV="${SCRIPT_DIR}/arms.tsv"
if [[ ! -f "$ARMS_TSV" ]]; then
    printf 'timestamp_utc\texp_name\tjob_id\tcommand\n' > "$ARMS_TSV"
fi
EFFECTIVE_CMD="$(printf '%q ' "${CMD[@]}")"
printf '%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$EXP_NAME" \
    "$JOB_ID" \
    "$EFFECTIVE_CMD" \
    >> "$ARMS_TSV"
echo "Recorded arm in ${ARMS_TSV} (job ${JOB_ID})"
