#!/bin/bash
#SBATCH --job-name=lhotse2-test2-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --account=epor48
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#
# Test 2 — resume across two nodes, 8 ranks x 2 workers = 16 streams.
#
# Adapted from tests/integration/sampler_resume/run_test_2node_fsdp2.sbatch,
# which targets a host venv and 2 GPUs/node and carries no account or QoS. Two
# things changed beyond the cluster settings:
#   * launch goes through the container (bash/run_train_singularity.sbatch is
#     driven indirectly — here we call bash/run_train.sh inside singularity, the
#     same path submit-container.sh takes);
#   * the original prints PASSED unconditionally. This one accounts for status.
#
# Submit with an explicit QoS and wall (SLURM bills the wall you ASK for):
#   sbatch --qos=acc_debug --time=02:00:00 \
#       tests/integration/lhotse2_campaign/run_test2_2node_mn5.sbatch
#
# Cost at those settings: 16 GPUh.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT" || exit 1

CONFIG="${CONFIG:-tests/integration/lhotse2_campaign/test2_2node_resume.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-config/accelerate/fsdp2.yaml}"

MAX_STEPS="${MAX_STEPS:-10}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126_lhotse2_td.sif}"
export SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
export HF_HOME="${HF_HOME:-/gpfs/scratch/epor48/hf_cache}"
export TMPDIR_HOST="${TMPDIR_HOST:-/gpfs/projects/epor48/melt-data/tmp}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export MASTER_PORT="${MASTER_PORT:-60001}"

# Must be on GPFS: every rank appends to the same cut-ids directory.
BASE="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}/lhotse2-test2-${SLURM_JOB_ID}"
RUN1_DIR="$BASE/run1"
RUN2_DIR="$BASE/run2"
CUTIDS1="$BASE/cut_ids_run1"
CUTIDS2="$BASE/cut_ids_run2"
mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$CUTIDS1" "$CUTIDS2" logs

GPUS_PER_NODE="$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'gpu:\K[0-9]+' | head -1)"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_NNODES:-2}"
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "================================================================="
echo "Test 2 — two-node resume"
echo "  nodes=$NUM_NODES gpus/node=$GPUS_PER_NODE world_size=$WORLD_SIZE"
echo "  num_workers=2 -> $((WORLD_SIZE * 2)) streams"
echo "  max_steps=$MAX_STEPS checkpoint_step=$CHECKPOINT_STEP grad_acc=$GRAD_ACC_STEPS"
echo "  data=$LOCAL_DATASETS_DIR"
echo "  base=$BASE"
echo "================================================================="

# Cut-id logging has to survive --cleanenv; run_train_singularity.sbatch has an
# allowlist for exactly these three, and we call the container directly here, so
# forward them as SINGULARITYENV_*.
run_phase() {
    local cut_ids_dir="$1"; shift
    export SINGULARITYENV_MELT_DEBUG_CUT_IDS_DIR="$cut_ids_dir"
    export SINGULARITYENV_MELT_DEBUG_CUT_IDS_MAX_BATCHES=$((MAX_STEPS * GRAD_ACC_STEPS * 4))
    export SINGULARITYENV_MELT_DEBUG_CUT_IDS_EVERY=1
    export SINGULARITYENV_LOCAL_DATASETS_DIR=/workspace/shar
    export SINGULARITYENV_OUTPUT_DIR=/workspace/outputs
    export SINGULARITYENV_HF_HOME=/workspace/hf_cache
    export SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_WANDB_MODE=disabled
    export SINGULARITYENV_MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    export SINGULARITYENV_MASTER_PORT="$MASTER_PORT"

    local binds="$REPO_ROOT:/workspace/training"
    binds+=",$TMPDIR_HOST:/workspace/tmp"
    binds+=",$LOCAL_DATASETS_DIR:/workspace/shar"
    binds+=",$BASE:/workspace/outputs"
    binds+=",$HF_HOME:/workspace/hf_cache"
    binds+=",$cut_ids_dir:$cut_ids_dir"

    srun --nodes="$NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=60 \
        bash -c '
            exec "$SINGULARITY_BIN" exec --nv \
                --bind "'"$binds"'" \
                --pwd /workspace/training \
                --cleanenv \
                --env SLURM_PROCID="${SLURM_PROCID:-0}" \
                --env SLURM_NODEID="${SLURM_NODEID:-0}" \
                --env SLURM_LOCALID="${SLURM_LOCALID:-0}" \
                --env SLURM_STEP_ID="${SLURM_STEP_ID:-}" \
                --env SLURM_NNODES="${SLURM_NNODES:-1}" \
                --env SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-4}" \
                --env SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
                "$SINGULARITY_IMG" \
                bash /workspace/training/bash/run_train.sh "$@"
        ' _ "$ACCEL_CONFIG" "$@"
}

# ---- run 1: train to MAX_STEPS, checkpointing at CHECKPOINT_STEP ------------
echo ""
echo "[run1] training to step $MAX_STEPS, saving every $CHECKPOINT_STEP"
run_phase "$CUTIDS1" \
    --config "$CONFIG" \
    --trainer.output_dir /workspace/outputs/run1 \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.save_steps "$CHECKPOINT_STEP" \
    --trainer.save_total_limit $((MAX_STEPS / CHECKPOINT_STEP)) \
    --trainer.gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --run.exp_name lhotse2-test2-run1
STATUS_RUN1=$?

# HF's get_last_checkpoint picks the highest-numbered dir, so drop the later one.
if [[ $STATUS_RUN1 -eq 0 ]]; then
    for ckpt in "$RUN1_DIR"/checkpoint-*; do
        [[ -d "$ckpt" ]] || continue
        if [[ "$(basename "$ckpt")" != "checkpoint-$CHECKPOINT_STEP" ]]; then
            echo "[prune] removing $(basename "$ckpt")"
            rm -rf "$ckpt"
        fi
    done
fi

# Sampler state must exist for every rank or the resume is not a resume.
STATUS_SAMPLER=0
CKPT="$RUN1_DIR/checkpoint-$CHECKPOINT_STEP"
for ((r = 0; r < WORLD_SIZE; r++)); do
    f="$CKPT/sampler/sampler_state_rank${r}.pt"
    if [[ ! -f "$f" ]]; then
        echo "[validate] MISSING $f"
        STATUS_SAMPLER=1
    fi
done
[[ $STATUS_SAMPLER -eq 0 ]] && echo "[validate] sampler state present for all $WORLD_SIZE ranks"

# ---- run 2: resume from the checkpoint -------------------------------------
STATUS_RUN2=""
if [[ $STATUS_RUN1 -eq 0 && $STATUS_SAMPLER -eq 0 ]]; then
    echo ""
    echo "[run2] resuming from checkpoint-$CHECKPOINT_STEP"
    run_phase "$CUTIDS2" \
        --config "$CONFIG" \
        --trainer.output_dir /workspace/outputs/run2 \
        --trainer.max_steps "$MAX_STEPS" \
        --trainer.save_strategy '"no"' \
        --trainer.gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        --trainer.resume_from_checkpoint /workspace/outputs/run1 \
        --run.exp_name lhotse2-test2-run2
    STATUS_RUN2=$?
fi

# ---- compare ---------------------------------------------------------------
STATUS_CUTIDS=""
if [[ "${STATUS_RUN2:-1}" -eq 0 ]]; then
    echo ""
    echo "[compare] run1 vs run2"
    "$SINGULARITY_BIN" exec \
        --bind "$REPO_ROOT:/workspace/training,$BASE:/workspace/outputs" \
        --pwd /workspace/training --cleanenv "$SINGULARITY_IMG" \
        bash -c "source /workspace/venv/bin/activate && \
            python tests/integration/sampler_resume/compare_cut_ids.py \
                --run1-dir /workspace/outputs/cut_ids_run1 \
                --run2-dir /workspace/outputs/cut_ids_run2 \
                --checkpoint-step $CHECKPOINT_STEP \
                --grad-accum-steps $GRAD_ACC_STEPS"
    STATUS_CUTIDS=$?
fi

SUMMARY_VERDICT="PASSED"
summary_row() {
    local label="$1" status="$2"
    if [[ -z "$status" ]]; then
        printf '  %-34s %s\n' "$label" "SKIPPED"
        SUMMARY_VERDICT="FAILED"
    elif [[ "$status" -eq 0 ]]; then
        printf '  %-34s %s\n' "$label" "PASS"
    else
        printf '  %-34s %s\n' "$label" "FAIL"
        SUMMARY_VERDICT="FAILED"
    fi
}

echo ""
echo "================================================================="
echo "TEST 2 — TWO-NODE RESUME (world_size=$WORLD_SIZE, $((WORLD_SIZE * 2)) streams)"
echo "-----------------------------------------------------------------"
summary_row "run1 (train to $MAX_STEPS)"        "${STATUS_RUN1-}"
summary_row "sampler state for all ranks"       "${STATUS_SAMPLER-}"
summary_row "run2 (resume from $CHECKPOINT_STEP)" "${STATUS_RUN2-}"
summary_row "cut IDs: resume + disjointness"    "${STATUS_CUTIDS-}"
echo "-----------------------------------------------------------------"
echo "  OVERALL: $SUMMARY_VERDICT"
echo "  artifacts: $BASE"
echo "================================================================="

[[ "$SUMMARY_VERDICT" == "PASSED" ]] || exit 1
