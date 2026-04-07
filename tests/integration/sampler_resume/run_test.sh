#!/bin/bash
# Integration test: sampler state restoration correctness
#
# Pipeline:
#   Run 1  — train from scratch for 100 steps; checkpoint saved at step 50.
#            Every batch's cut IDs are logged to $OUTPUT_DIR/cut_ids_run1/.
#   Run 2  — resume from the step-50 checkpoint for the remaining 50 steps.
#            Cut IDs are logged to $OUTPUT_DIR/cut_ids_run2/.
#   Check  — compare_cut_ids.py verifies that run2's batches 0..49 match
#            run1's batches 50..99 exactly (same cut IDs in the same order).
#
# Required env vars (override as needed):
#   VENV_PATH          path to the virtualenv activate script
#   LOCAL_DATASETS_DIR path to the root of the local shar datasets
#   OUTPUT_DIR         directory where all test artefacts are written
#   HF_HOME            (optional) HuggingFace cache directory
#
# Usage (from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate \
#   LOCAL_DATASETS_DIR=/path/to/shar \
#   OUTPUT_DIR=/tmp/melt_sampler_test \
#   bash tests/integration/sampler_resume/run_test.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/melt_sampler_resume_test}"
HF_HOME="${HF_HOME:-}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
ACCEL_CONFIG="$SCRIPT_DIR/accelerate_1gpu.yaml"

RUN1_DIR="$OUTPUT_DIR/run1"
RUN2_DIR="$OUTPUT_DIR/run2"
CUT_IDS_RUN1="$OUTPUT_DIR/cut_ids_run1"
CUT_IDS_RUN2="$OUTPUT_DIR/cut_ids_run2"

CHECKPOINT_STEP=50
GRAD_ACC_STEPS=1

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
source "$VENV_PATH"
cd "$REPO_ROOT"

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$CUT_IDS_RUN1" "$CUT_IDS_RUN2"

# Export env vars needed by the config (resolved via ${oc.env:VAR} in config.yaml)
export LOCAL_DATASETS_DIR
[[ -n "$HF_HOME" ]] && export HF_HOME

echo "================================================================="
echo "MELT Sampler Resume Integration Test"
echo "================================================================="
echo "Repo root:          $REPO_ROOT"
echo "Config:             $CONFIG"
echo "Accelerate config:  $ACCEL_CONFIG"
echo "Run 1 output:       $RUN1_DIR"
echo "Run 2 output:       $RUN2_DIR"
echo "Cut IDs run 1:      $CUT_IDS_RUN1"
echo "Cut IDs run 2:      $CUT_IDS_RUN2"
echo "Checkpoint step:    $CHECKPOINT_STEP"
echo "================================================================="

# Helper: run the trainer with the given extra arguments.
# All runs share the same base config; callers pass per-run overrides.
run_trainer() {
    local extra_args=("$@")
    WANDB_MODE=disabled \
    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        --num_machines 1 \
        --num_processes 1 \
        --module melt.training.train \
        --config "$CONFIG" \
        --trainer.gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        "${extra_args[@]}"
}

# ---------------------------------------------------------------------------
# Run 1: full 100-step training run
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 1: Training from scratch for 100 steps ---"
echo "    Checkpoint will be saved at step $CHECKPOINT_STEP."
echo "    Cut IDs logged to: $CUT_IDS_RUN1"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN1" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=200 \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN1_DIR" \
    --trainer.max_steps 100 \
    --trainer.save_steps "$CHECKPOINT_STEP" \
    --trainer.overwrite_output_dir true

echo ""
echo "Run 1 complete."

# ---------------------------------------------------------------------------
# Validate checkpoint
# ---------------------------------------------------------------------------
CHECKPOINT="$RUN1_DIR/checkpoint-${CHECKPOINT_STEP}"

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Expected checkpoint directory not found: $CHECKPOINT"
    echo "Contents of $RUN1_DIR:"
    ls -la "$RUN1_DIR" || true
    exit 1
fi

SAMPLER_STATE="$CHECKPOINT/sampler/sampler_state_rank0.pt"
if [[ ! -f "$SAMPLER_STATE" ]]; then
    echo "ERROR: Sampler state file not found: $SAMPLER_STATE"
    echo "Contents of $CHECKPOINT:"
    ls -la "$CHECKPOINT/" || true
    exit 1
fi

echo "Checkpoint OK: $CHECKPOINT"
echo "Sampler state: $SAMPLER_STATE"

# ---------------------------------------------------------------------------
# Run 2: resume from step-50 checkpoint, train remaining 50 steps
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 2: Resuming from $CHECKPOINT ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN2"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN2" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=200 \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN2_DIR" \
    --trainer.max_steps 100 \
    --trainer.overwrite_output_dir true \
    --trainer.resume_from_checkpoint "$CHECKPOINT"

echo ""
echo "Run 2 complete."

# ---------------------------------------------------------------------------
# Compare cut IDs
# ---------------------------------------------------------------------------
echo ""
echo "--- Comparing cut IDs ---"
echo "    run1 batches[$CHECKPOINT_STEP:100]  vs  run2 batches[0:50]"
echo ""

python "$SCRIPT_DIR/compare_cut_ids.py" \
    --run1-dir "$CUT_IDS_RUN1" \
    --run2-dir "$CUT_IDS_RUN2" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --grad-accum-steps "$GRAD_ACC_STEPS"

echo ""
echo "================================================================="
echo "SAMPLER RESUME TEST PASSED"
echo "================================================================="
