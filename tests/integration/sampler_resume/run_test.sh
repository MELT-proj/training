#!/bin/bash
# Integration test: sampler state restoration correctness
#
# Pipeline:
#   Run 1  — train from scratch for MAX_STEPS steps; checkpoints saved every
#            save_steps (set in config). All checkpoints except the reference
#            one at CHECKPOINT_STEP are removed after training.
#            Every batch's cut IDs are logged to $OUTPUT_DIR/cut_ids_run1/.
#   Run 2  — resume from the checkpoint at CHECKPOINT_STEP and train for the
#            remaining (MAX_STEPS - CHECKPOINT_STEP) steps.
#            Cut IDs are logged to $OUTPUT_DIR/cut_ids_run2/.
#   Check  — compare_cut_ids.py verifies that run2's batches match
#            run1's batches [CHECKPOINT_STEP:MAX_STEPS] exactly.
#
# Required env vars (override as needed):
#   VENV_PATH          path to the virtualenv activate script
#   LOCAL_DATASETS_DIR path to the root of the local shar datasets
#   OUTPUT_DIR         directory where all test artefacts are written
#   MAX_STEPS          total training steps for run 1 (default: 100)
#   CHECKPOINT_STEP    step at which the reference checkpoint was saved (default: 50)
#   HF_HOME            (optional) HuggingFace cache directory
#
# Usage (from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate \
#   LOCAL_DATASETS_DIR=/path/to/shar \
#   OUTPUT_DIR=/tmp/melt_sampler_test \
#   MAX_STEPS=500 CHECKPOINT_STEP=100 \
#   bash tests/integration/sampler_resume/run_test.sh

set -euo pipefail

source /etc/profile.d/02-lmod.sh
module load cuda

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
ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/accelerate_1gpu.yaml}"

RUN1_DIR="$OUTPUT_DIR/run1"
RUN2_DIR="$OUTPUT_DIR/run2"
RUN3_DIR="$OUTPUT_DIR/run3"
CUT_IDS_RUN1="$OUTPUT_DIR/cut_ids_run1"
CUT_IDS_RUN2="$OUTPUT_DIR/cut_ids_run2"
CUT_IDS_RUN3="$OUTPUT_DIR/cut_ids_run3"

MAX_STEPS="${MAX_STEPS:-100}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-50}"
GRAD_ACC_STEPS=1

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
source "$VENV_PATH"
cd "$REPO_ROOT"

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$RUN3_DIR" "$CUT_IDS_RUN1" "$CUT_IDS_RUN2" "$CUT_IDS_RUN3"

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
echo "Run 3 output:       $RUN3_DIR"
echo "Cut IDs run 1:      $CUT_IDS_RUN1"
echo "Cut IDs run 2:      $CUT_IDS_RUN2"
echo "Cut IDs run 3:      $CUT_IDS_RUN3"
echo "Max steps:          $MAX_STEPS"
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
# Run 1: full MAX_STEPS-step training run
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 1: Training from scratch for $MAX_STEPS steps ---"
echo "    Reference checkpoint at step $CHECKPOINT_STEP."
echo "    Cut IDs logged to: $CUT_IDS_RUN1"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN1" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 )) \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN1_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.save_steps "$CHECKPOINT_STEP" \
    --trainer.save_total_limit $(( MAX_STEPS / CHECKPOINT_STEP )) \
    --trainer.overwrite_output_dir true

# Remove every checkpoint except the reference so that HF's trainer picks up
# exactly checkpoint-${CHECKPOINT_STEP} when resuming in Run 2.
echo "Removing all checkpoints except checkpoint-${CHECKPOINT_STEP} ..."
for ckpt in "$RUN1_DIR"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    if [[ "$(basename "$ckpt")" != "checkpoint-${CHECKPOINT_STEP}" ]]; then
        echo "  Removing $ckpt"
        rm -rf "$ckpt"
    fi
done

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
# Run 2: resume from the reference checkpoint, train remaining steps
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 2: Resuming from $CHECKPOINT (steps $CHECKPOINT_STEP → $MAX_STEPS) ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN2"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN2" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 )) \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN2_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.overwrite_output_dir true \
    --trainer.resume_from_checkpoint "$RUN1_DIR"

echo ""
echo "Run 2 complete."

# ---------------------------------------------------------------------------
# Compare cut IDs
# ---------------------------------------------------------------------------
echo ""
echo "--- Comparing cut IDs ---"
echo "    run1 batches[$CHECKPOINT_STEP:$MAX_STEPS]  vs  run2 batches[0:$(( MAX_STEPS - CHECKPOINT_STEP ))]"
echo ""

python "$SCRIPT_DIR/compare_cut_ids.py" \
    --run1-dir "$CUT_IDS_RUN1" \
    --run2-dir "$CUT_IDS_RUN2" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --grad-accum-steps "$GRAD_ACC_STEPS"

# ---------------------------------------------------------------------------
# Run 3: fresh run from scratch — determinism check against run 1
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 3: Training from scratch for $MAX_STEPS steps (determinism check) ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN3"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN3" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 )) \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN3_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.save_strategy '"no"' \
    --trainer.overwrite_output_dir true

echo ""
echo "Run 3 complete."

# ---------------------------------------------------------------------------
# Compare cut IDs: run 3 vs run 1 (full run, all batches must match)
# ---------------------------------------------------------------------------
echo ""
echo "--- Comparing cut IDs (determinism) ---"
echo "    run1 batches[0:$MAX_STEPS]  vs  run3 batches[0:$MAX_STEPS]"
echo ""

python "$SCRIPT_DIR/compare_cut_ids.py" \
    --run1-dir "$CUT_IDS_RUN1" \
    --run2-dir "$CUT_IDS_RUN3" \
    --checkpoint-step 0 \
    --grad-accum-steps "$GRAD_ACC_STEPS"

echo ""
echo "================================================================="
echo "SAMPLER RESUME TEST PASSED"
echo "================================================================="
