#!/bin/bash
# Integration test: sampler state restoration correctness — 2-GPU DDP variant
#
# Identical pipeline to run_test.sh, but launches two DDP processes each with
# one dataloader worker (num_workers=1, prefetch_factor=2).
#
# Pipeline:
#   Run 1  — train from scratch for MAX_STEPS steps; checkpoints saved every
#            save_steps (set in config). All checkpoints except the reference
#            one at CHECKPOINT_STEP are removed after training.
#            Every batch's cut IDs are logged to $OUTPUT_DIR/cut_ids_run1/.
#   Run 2  — resume from the checkpoint at CHECKPOINT_STEP and train for the
#            remaining (MAX_STEPS - CHECKPOINT_STEP) steps.
#            Cut IDs are logged to $OUTPUT_DIR/cut_ids_run2/.
#   Check  — compare_cut_ids.py verifies that:
#              (a) run2's batches match run1's batches [CHECKPOINT_STEP:MAX_STEPS] per rank;
#              (b) no cut ID is shared between any two (rank, worker) pairs.
#            compare_losses.py additionally verifies that run2 reproduces run1's
#            training loss (and eval loss, when the config evaluates) at every
#            step after the checkpoint. Same batches is necessary but not
#            sufficient — matching metrics also cover optimizer/scheduler state.
#
# Required env vars (override as needed):
#   VENV_PATH          path to the virtualenv activate script
#   LOCAL_DATASETS_DIR path to the root of the local shar datasets
#   OUTPUT_DIR         directory where all test artefacts are written
#   MAX_STEPS          total training steps for run 1 (default: 100)
#   CHECKPOINT_STEP    step at which the reference checkpoint was saved (default: 50)
#   CONFIG             training config (default: config_2gpu.yaml; use
#                      config_2gpu_eval.yaml to also compare eval loss)
#   SHARD_SEED         override data.train_ds.shard_seed (default: whatever the
#                      config sets). Use to test 'randomized' vs 'trng' vs an int.
#   SKIP_RUN3          set to 1 to skip the from-scratch determinism re-run
#   HF_HOME            (optional) HuggingFace cache directory
#
# Usage (from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate \
#   LOCAL_DATASETS_DIR=/path/to/shar \
#   OUTPUT_DIR=/tmp/melt_sampler_test_2gpu \
#   MAX_STEPS=500 CHECKPOINT_STEP=100 \
#   bash tests/integration/sampler_resume/run_test_2gpu.sh

# set -euo pipefail

source /etc/profile.d/02-lmod.sh
module load cuda

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/melt_sampler_resume_test_2gpu}"
HF_HOME="${HF_HOME:-}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config_2gpu.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/accelerate_2gpu.yaml}"
NUM_PROCESSES=2

RUN1_DIR="$OUTPUT_DIR/run1"
RUN2_DIR="$OUTPUT_DIR/run2"
RUN3_DIR="$OUTPUT_DIR/run3"
CUT_IDS_RUN1="$OUTPUT_DIR/cut_ids_run1"
CUT_IDS_RUN2="$OUTPUT_DIR/cut_ids_run2"
CUT_IDS_RUN3="$OUTPUT_DIR/cut_ids_run3"

MAX_STEPS="${MAX_STEPS:-100}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-50}"
GRAD_ACC_STEPS=1
SKIP_RUN3="${SKIP_RUN3:-0}"

# trainer_state.json lives inside each checkpoint dir. run1 prunes its
# checkpoints before run2 resumes, so the histories are copied out first.
STATE_RUN1="$OUTPUT_DIR/trainer_state_run1.json"
STATE_RUN2="$OUTPUT_DIR/trainer_state_run2.json"
STATE_RUN3="$OUTPUT_DIR/trainer_state_run3.json"
rm -f "$STATE_RUN1" "$STATE_RUN2" "$STATE_RUN3"

# Extra args shared by every run. SHARD_SEED, when set, overrides the config so
# one config can be exercised under 'randomized', 'trng' and a fixed integer.
COMMON_ARGS=()
if [[ -n "${SHARD_SEED:-}" ]]; then
    COMMON_ARGS+=(--data.train_ds.shard_seed "$SHARD_SEED")
fi

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
echo "MELT Sampler Resume Integration Test — 2-GPU DDP"
echo "================================================================="
echo "Repo root:          $REPO_ROOT"
echo "Config:             $CONFIG"
echo "Accelerate config:  $ACCEL_CONFIG"
echo "Num processes:      $NUM_PROCESSES"
echo "Run 1 output:       $RUN1_DIR"
echo "Run 2 output:       $RUN2_DIR"
echo "Run 3 output:       $RUN3_DIR"
echo "Cut IDs run 1:      $CUT_IDS_RUN1"
echo "Cut IDs run 2:      $CUT_IDS_RUN2"
echo "Cut IDs run 3:      $CUT_IDS_RUN3"
echo "Max steps:          $MAX_STEPS"
echo "Checkpoint step:    $CHECKPOINT_STEP"
echo "Shard seed:         ${SHARD_SEED:-<from config>}"
echo "================================================================="

# Helper: run the trainer with the given extra arguments.
run_trainer() {
    local extra_args=("$@")
    WANDB_MODE=disabled \
    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        --num_machines 1 \
        --num_processes "$NUM_PROCESSES" \
        --module melt.training.train \
        --config "$CONFIG" \
        --trainer.gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"
}

# Helper: copy a run's final metric history out of its last checkpoint.
save_trainer_state() {
    local run_dir="$1" dest="$2"
    local src="$run_dir/checkpoint-${MAX_STEPS}/trainer_state.json"
    if [[ -f "$src" ]]; then
        cp "$src" "$dest"
        echo "Saved metric history: $dest"
    else
        echo "WARNING: no trainer_state.json at $src — loss comparison will be skipped."
    fi
}

# Helper: one truthful verdict line per check, then the overall result.
# The old unconditional "PASSED" banner printed even when a comparison failed,
# because this script deliberately does not run under `set -e`.
SUMMARY_VERDICT="PASSED"

summary_row() {
    local label="$1" status="$2"
    if [[ -z "$status" ]]; then
        printf '  %-30s %s\n' "$label" "SKIPPED"
    elif [[ "$status" -eq 0 ]]; then
        printf '  %-30s %s\n' "$label" "PASS"
    else
        printf '  %-30s %s\n' "$label" "FAIL"
        SUMMARY_VERDICT="FAILED"
    fi
}

print_summary() {
    echo ""
    echo "================================================================="
    echo "SAMPLER RESUME TEST (2-GPU DDP) — shard_seed=${SHARD_SEED:-<from config>}"
    echo "-----------------------------------------------------------------"
    summary_row "cut IDs after resume"       "${STATUS_CUTIDS-}"
    summary_row "determinism (run3 vs run1)" "${STATUS_DETERMINISM-}"
    summary_row "metrics after resume"       "${STATUS_LOSSES-}"
    echo "-----------------------------------------------------------------"
    echo "  OVERALL: $SUMMARY_VERDICT"
    echo "================================================================="
}

# ---------------------------------------------------------------------------
# Run 1: full MAX_STEPS-step training run
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 1: Training from scratch for $MAX_STEPS steps (2 GPUs) ---"
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

save_trainer_state "$RUN1_DIR" "$STATE_RUN1"

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
# Validate checkpoint — both ranks must have saved sampler state
# ---------------------------------------------------------------------------
CHECKPOINT="$RUN1_DIR/checkpoint-${CHECKPOINT_STEP}"

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Expected checkpoint directory not found: $CHECKPOINT"
    echo "Contents of $RUN1_DIR:"
    ls -la "$RUN1_DIR" || true
    exit 1
fi

for rank in $(seq 0 $((NUM_PROCESSES - 1))); do
    SAMPLER_STATE="$CHECKPOINT/sampler/sampler_state_rank${rank}.pt"
    if [[ ! -f "$SAMPLER_STATE" ]]; then
        echo "ERROR: Sampler state file not found for rank ${rank}: $SAMPLER_STATE"
        echo "Contents of $CHECKPOINT/sampler/:"
        ls -la "$CHECKPOINT/sampler/" || true
        exit 1
    fi
    echo "Sampler state rank ${rank}: $SAMPLER_STATE"
done

echo "Checkpoint OK: $CHECKPOINT"

# ---------------------------------------------------------------------------
# Run 2: resume from the reference checkpoint, train remaining steps
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 2: Resuming from $CHECKPOINT (steps $CHECKPOINT_STEP → $MAX_STEPS, 2 GPUs) ---"
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

save_trainer_state "$RUN2_DIR" "$STATE_RUN2"

# ---------------------------------------------------------------------------
# Compare cut IDs (resume correctness + no-overlap between ranks/workers)
# ---------------------------------------------------------------------------
echo ""
echo "--- Comparing cut IDs ---"
echo "    run1 batches[$CHECKPOINT_STEP:$MAX_STEPS]  vs  run2 batches[0:$(( MAX_STEPS - CHECKPOINT_STEP ))]  (per rank)"
echo "    Also verifying no cut-ID overlap between (rank, worker) pairs."
echo ""

python "$SCRIPT_DIR/compare_cut_ids.py" \
    --run1-dir "$CUT_IDS_RUN1" \
    --run2-dir "$CUT_IDS_RUN2" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --grad-accum-steps "$GRAD_ACC_STEPS"
STATUS_CUTIDS=$?

if [[ "$SKIP_RUN3" == "1" ]]; then
    echo ""
    echo "--- RUN 3 skipped (SKIP_RUN3=1) ---"
    echo "    Without it there is no noise floor to compare the metrics against,"
    echo "    so they are checked for exact equality."
fi

# ---------------------------------------------------------------------------
# Run 3: fresh run from scratch — determinism check against run 1
#
# Run 3 does double duty. Its batches must match run 1's (the sampler is
# deterministic), and its metrics give the run-to-run noise floor: two
# identical from-scratch runs are observed NOT to produce bit-identical losses
# even when their batches match exactly, so the training loop is not
# reproducible to the last digit. Judging the resume against exact equality
# would therefore report that nondeterminism as a resume bug; judging it
# against run 3 asks the question that can be answered — is resuming any worse
# than re-running?
# ---------------------------------------------------------------------------
if [[ "$SKIP_RUN3" != "1" ]]; then
echo ""
echo "--- RUN 3: Training from scratch for $MAX_STEPS steps (determinism check, 2 GPUs) ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN3"
echo ""

MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN3" \
MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 )) \
MELT_DEBUG_CUT_IDS_EVERY=1 \
run_trainer \
    --trainer.output_dir "$RUN3_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.save_steps "$MAX_STEPS" \
    --trainer.overwrite_output_dir true

echo ""
echo "Run 3 complete."

save_trainer_state "$RUN3_DIR" "$STATE_RUN3"

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
STATUS_DETERMINISM=$?

fi  # SKIP_RUN3

# ---------------------------------------------------------------------------
# Compare metrics: did the resume land in the same model state?
# ---------------------------------------------------------------------------
echo ""
echo "--- Comparing metrics ---"
echo "    run1 vs run2 at every logged step after $CHECKPOINT_STEP."
echo ""

BASELINE_ARGS=()
if [[ -f "$STATE_RUN3" ]]; then
    BASELINE_ARGS=(--baseline-state "$STATE_RUN3")
fi

# The noise floor is only a noise floor if run 3 saw run 1's batches. When the
# determinism check fails, run 3 trained on different data, so |run1 - run3|
# measures that difference rather than the stack's nondeterminism and the
# comparison cannot separate the two. Report it as inconclusive instead of
# passing the resume against a meaningless baseline.
if [[ "${STATUS_DETERMINISM:-0}" -ne 0 ]]; then
    echo "SKIPPED: run3 did not reproduce run1's batches, so it is not a valid"
    echo "         noise-floor baseline. Fix the determinism failure first."
elif [[ -f "$STATE_RUN1" && -f "$STATE_RUN2" ]]; then
    python "$SCRIPT_DIR/compare_losses.py" \
        --run1-state "$STATE_RUN1" \
        --run2-state "$STATE_RUN2" \
        --checkpoint-step "$CHECKPOINT_STEP" \
        "${BASELINE_ARGS[@]}"
    STATUS_LOSSES=$?
else
    echo "SKIPPED: metric histories were not captured from both runs."
    STATUS_LOSSES=1
fi

print_summary
exit $(( STATUS_CUTIDS != 0 || STATUS_LOSSES != 0 || ${STATUS_DETERMINISM:-0} != 0 ))
