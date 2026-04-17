#!/bin/bash
# Integration test: sampler state restoration correctness — 2-node × 2-GPU FSDP2 variant
#
# Submits a SLURM job that allocates 2 nodes (2 GPUs each, 4 ranks total) and
# runs the same 3-phase pipeline as run_test_2gpu.sh, but using FSDP2 via
# Accelerate and srun for multi-node coordination.
#
# Pipeline:
#   Run 1  — train from scratch for MAX_STEPS steps; checkpoints saved every
#            save_steps (set in config).  All checkpoints except the reference
#            one at CHECKPOINT_STEP are removed after training.
#            Every batch's cut IDs are logged to $OUTPUT_DIR/cut_ids_run1/.
#   Run 2  — resume from the checkpoint at CHECKPOINT_STEP and train for the
#            remaining (MAX_STEPS - CHECKPOINT_STEP) steps.
#            Cut IDs are logged to $OUTPUT_DIR/cut_ids_run2/.
#   Check  — compare_cut_ids.py verifies that:
#              (a) run2's batches match run1's batches [CHECKPOINT_STEP:MAX_STEPS] per rank;
#              (b) no cut ID is shared between any two (rank, worker) pairs.
#   Run 3  — fresh run from scratch (no checkpointing) — determinism check
#            against run 1.
#
# IMPORTANT: OUTPUT_DIR must be on a shared filesystem (NFS/Lustre) visible
# from all allocated nodes; both training ranks and the compare step write
# there.
#
# Required env vars:
#   REPO_ROOT          absolute path to the repo root (MUST be set — see note below)
#   VENV_PATH          path to the virtualenv activate script
#   LOCAL_DATASETS_DIR path to the root of the local shar datasets (shared FS)
#   OUTPUT_DIR         directory for all test artefacts (shared FS)
# Optional:
#   HF_HOME            HuggingFace cache directory
#   MAX_STEPS          total training steps for run 1 (default: 100)
#   CHECKPOINT_STEP    step at which the reference checkpoint was saved (default: 50)
#   MASTER_PORT        port for c10d rendezvous (default: 29500)
#
# NOTE on REPO_ROOT: this cluster copies the batch script to the SLURM job
# directory before execution, so ${BASH_SOURCE[0]} and SLURM_SUBMIT_DIR cannot
# be used to auto-detect the repo root.  Always pass REPO_ROOT explicitly:
#
# Usage (from the repo root):
#   REPO_ROOT=$PWD \
#   VENV_PATH=/path/to/venv/bin/activate \
#   LOCAL_DATASETS_DIR=/path/to/shar \
#   OUTPUT_DIR=/shared/fs/melt_sampler_test_2node \
#   sbatch --partition=<partition> --time=02:00:00 \
#          tests/integration/sampler_resume/run_test_2node_fsdp2.sbatch

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one accelerate-launch process per node
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=melt-sampler-test-2node
#SBATCH --output=logs/sampler_test_2node.%j.out
#SBATCH --error=logs/sampler_test_2node.%j.err
# Cluster-specific directives (uncomment / adjust as needed):
# #SBATCH --partition=gpu
# #SBATCH --time=02:00:00
# #SBATCH --qos=gpu-normal

# set -euo pipefail

source /etc/profile.d/02-lmod.sh 2>/dev/null || true
module load cuda 2>/dev/null || true

# ---------------------------------------------------------------------------
# Resolve paths
#
# This cluster copies the batch script into the SLURM job directory before
# execution, so neither ${BASH_SOURCE[0]} nor SLURM_SUBMIT_DIR can be used
# to locate the repo root.  REPO_ROOT must be passed as an env var at
# submission time (e.g. "REPO_ROOT=$PWD sbatch ...").
# ---------------------------------------------------------------------------
if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "ERROR: REPO_ROOT is not set."
    echo "  Submit with: REPO_ROOT=\$PWD sbatch ... run_test_2node_fsdp2.sbatch"
    exit 1
fi
SCRIPT_DIR="$REPO_ROOT/tests/integration/sampler_resume"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/melt_sampler_resume_test_2node}"
HF_HOME="${HF_HOME:-}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config_2node_fsdp2.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/accelerate_2node_fsdp2.yaml}"

# SLURM-derived distributed settings
GPUS_PER_NODE=2
NUM_NODES="${SLURM_NNODES:-2}"
NUM_PROCESSES=$(( NUM_NODES * GPUS_PER_NODE ))
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

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
# Source venv on the master node for the compare/cleanup steps.
source "$VENV_PATH"
cd "$REPO_ROOT"

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$RUN3_DIR" "$CUT_IDS_RUN1" "$CUT_IDS_RUN2" "$CUT_IDS_RUN3" "$REPO_ROOT/logs"

export LOCAL_DATASETS_DIR
export REPO_ROOT
export VENV_PATH
export CONFIG
export ACCEL_CONFIG
export GRAD_ACC_STEPS
export NUM_NODES
export NUM_PROCESSES
export MASTER_ADDR
export MASTER_PORT
[[ -n "$HF_HOME" ]] && export HF_HOME

echo "================================================================="
echo "MELT Sampler Resume Integration Test — 2-node × 2-GPU FSDP2"
echo "================================================================="
echo "Repo root:          $REPO_ROOT"
echo "Config:             $CONFIG"
echo "Accelerate config:  $ACCEL_CONFIG"
echo "Num nodes:          $NUM_NODES"
echo "GPUs per node:      $GPUS_PER_NODE"
echo "Total processes:    $NUM_PROCESSES"
echo "Master addr:        $MASTER_ADDR"
echo "Master port:        $MASTER_PORT"
echo "Run 1 output:       $RUN1_DIR"
echo "Run 2 output:       $RUN2_DIR"
echo "Run 3 output:       $RUN3_DIR"
echo "Cut IDs run 1:      $CUT_IDS_RUN1"
echo "Cut IDs run 2:      $CUT_IDS_RUN2"
echo "Cut IDs run 3:      $CUT_IDS_RUN3"
echo "Max steps:          $MAX_STEPS"
echo "Checkpoint step:    $CHECKPOINT_STEP"
echo "SLURM job ID:       ${SLURM_JOB_ID:-n/a}"
echo "Nodelist:           ${SLURM_JOB_NODELIST:-n/a}"
echo "================================================================="

# ---------------------------------------------------------------------------
# Helper: run the trainer across all nodes.
#
# Builds a per-run accelerate command string and exports it as MELT_TRAINER_CMD.
# Each srun task (one per node) sources the venv, then eval-s the command.
#
# NOTE: \$SLURM_PROCID is intentionally left un-expanded here — it is
# evaluated inside each remote bash -c by the per-task environment, giving
# the correct machine_rank (0 for node 0, 1 for node 1).
# ---------------------------------------------------------------------------
run_trainer() {
    local extra_args_str=""
    for arg in "$@"; do
        extra_args_str+=" $(printf '%q' "$arg")"
    done

    export MELT_TRAINER_CMD="accelerate launch \
        --config_file '$ACCEL_CONFIG' \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_machines $NUM_NODES \
        --num_processes $NUM_PROCESSES \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank \$SLURM_PROCID \
        --rdzv_backend c10d \
        --module melt.training.train \
        --config '$CONFIG' \
        --trainer.gradient_accumulation_steps $GRAD_ACC_STEPS \
        $extra_args_str"

    WANDB_MODE=disabled \
    srun \
        --nodes="$NUM_NODES" \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash -c '
            source "$VENV_PATH"
            source /etc/profile.d/02-lmod.sh 2>/dev/null || true
            module load cuda 2>/dev/null || true
            cd "$REPO_ROOT"
            eval "$MELT_TRAINER_CMD"
        '
}

# ---------------------------------------------------------------------------
# Run 1: full MAX_STEPS-step training run
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 1: Training from scratch for $MAX_STEPS steps (2 nodes × 2 GPUs, FSDP2) ---"
echo "    Reference checkpoint at step $CHECKPOINT_STEP."
echo "    Cut IDs logged to: $CUT_IDS_RUN1"
echo ""

export MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN1"
export MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 ))
export MELT_DEBUG_CUT_IDS_EVERY=1
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
# Validate checkpoint — all ranks must have saved sampler state
# ---------------------------------------------------------------------------
CHECKPOINT="$RUN1_DIR/checkpoint-${CHECKPOINT_STEP}"

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Expected checkpoint directory not found: $CHECKPOINT"
    echo "Contents of $RUN1_DIR:"
    ls -la "$RUN1_DIR" || true
    exit 1
fi

for rank in $(seq 0 $(( NUM_PROCESSES - 1 ))); do
    SAMPLER_STATE="$CHECKPOINT/sampler/sampler_state_rank${rank}.pt"
    if [[ ! -f "$SAMPLER_STATE" ]]; then
        echo "ERROR: Sampler state file not found for rank ${rank}: $SAMPLER_STATE"
        echo "Contents of $CHECKPOINT/sampler/:"
        ls -la "$CHECKPOINT/sampler/" 2>/dev/null || echo "(directory missing)"
        exit 1
    fi
    echo "Sampler state rank ${rank}: $SAMPLER_STATE"
done

echo "Checkpoint OK: $CHECKPOINT"

# ---------------------------------------------------------------------------
# Run 2: resume from the reference checkpoint, train remaining steps
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 2: Resuming from $CHECKPOINT (steps $CHECKPOINT_STEP → $MAX_STEPS, 2 nodes) ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN2"
echo ""

export MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN2"
export MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 ))
export MELT_DEBUG_CUT_IDS_EVERY=1
run_trainer \
    --trainer.output_dir "$RUN2_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.overwrite_output_dir true \
    --trainer.resume_from_checkpoint "$RUN1_DIR"

echo ""
echo "Run 2 complete."

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

# ---------------------------------------------------------------------------
# Run 3: fresh run from scratch — determinism check against run 1
# ---------------------------------------------------------------------------
echo ""
echo "--- RUN 3: Training from scratch for $MAX_STEPS steps (determinism check, 2 nodes) ---"
echo "    Cut IDs logged to: $CUT_IDS_RUN3"
echo ""

export MELT_DEBUG_CUT_IDS_DIR="$CUT_IDS_RUN3"
export MELT_DEBUG_CUT_IDS_MAX_BATCHES=$(( MAX_STEPS * GRAD_ACC_STEPS * 2 ))
export MELT_DEBUG_CUT_IDS_EVERY=1
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
echo "SAMPLER RESUME TEST (2-NODE FSDP2) PASSED"
echo "================================================================="
