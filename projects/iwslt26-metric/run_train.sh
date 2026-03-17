#!/bin/bash

set -euo pipefail

# --------------------------
# User-overridable defaults
# --------------------------
# Override any of these at invocation time, e.g.:
#   SCRATCH=/path WANDB_PROJECT=foo ./bash/run_train.sh ...

echo "------------------------------------------------------------------"
echo "WE ARE INSIDE RUN_TRAIN.SH"
echo "------------------------------------------------------------------"

# Two crucial environment variables to set up:
# 1) VENV_PATH: path to the python virtualenv activate script
# 2) LOCAL_DATASETS_DIR: path to local datasets storage
VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"
echo "Using VENV_PATH: $VENV_PATH"
echo "Using LOCAL_DATASETS_DIR: $LOCAL_DATASETS_DIR"

WANDB_PROJECT="${WANDB_PROJECT:-iwslt26-metric}"
WANDB_MODE="${WANDB_MODE:-online}"
TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

echo "Activating virtualenv at $VENV_PATH"
source "$VENV_PATH"
echo "Python is at:"
command -v python || true

if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found (may be running in container)"

# Usage: $0 <accelerate_config> [extra_args...]
# The config system uses OmegaConf, so we pass arguments with dot notation.
# Example: $0 config/accelerate/zero3.yaml --config config/train/asr.yaml --trainer.max_steps 1000

if [ -z \"${1:-}\" ]; then
    echo \"Usage: $0 <accelerate_config> [train_args...]\"
    echo \"Example: $0 config/accelerate/zero3.yaml --config config/train/asr.yaml\"
    echo \"Example: $0 config/accelerate/zero3.yaml --config config/train/asr.yaml --trainer.max_steps 1000\"
    exit 1
fi

ACCELERATE_CONFIG=$1
shift  # Remove accelerate config from args, rest are passed to train.py

# Parse gradient_accumulation_steps from CLI args if provided, else default to 4
GRAD_ACC_STEPS=4
for arg in "$@"; do
    if [[ "$arg" == "--trainer.gradient-accumulation-steps" ]]; then
        # Next arg should be the value
        get_next=1
    elif [[ "${get_next:-0}" == "1" ]]; then
        GRAD_ACC_STEPS=$arg
        get_next=0
    elif [[ "$arg" == --trainer.gradient-accumulation-steps=* ]]; then
        GRAD_ACC_STEPS="${arg#*=}"
    fi
done
echo "Using accelerate config: $ACCELERATE_CONFIG"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"

# setup run-specific envs
export VENV_PATH
export TMPDIR
export LOCAL_DATASETS_DIR
export HF_HOME
export WANDB_PROJECT 
export WANDB_MODE
export TORCHDYNAMO_VERBOSE
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export HF_HUB_ENABLE_HF_TRANSFER
export ACCELERATE_LOG_LEVEL
export TRANSFORMERS_VERBOSITY

echo "*** ENV VARIABLES ***"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_MODE=$WANDB_MODE"
echo "TORCHDYNAMO_VERBOSE=$TORCHDYNAMO_VERBOSE"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
echo "ACCELERATE_LOG_LEVEL=$ACCELERATE_LOG_LEVEL"
echo "TRANSFORMERS_VERBOSITY=$TRANSFORMERS_VERBOSITY"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "HF_HOME=$HF_HOME"
echo "LOCAL_DATASETS_DIR=$LOCAL_DATASETS_DIR"
echo "TMPDIR=$TMPDIR"
echo "---------------------"

echo "Listing what I see under ${HF_HOME}/hub:"
ls -l "${HF_HOME}/hub" || echo "No models cached yet."
echo "---------------------"

RUNNING_UNDER_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUNNING_UNDER_SLURM=1
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
if [[ "$RUNNING_UNDER_SLURM" -eq 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)
        if [[ -n "${DETECTED_GPUS:-}" && "${DETECTED_GPUS:-0}" -gt 0 ]]; then
            GPUS_PER_NODE=$DETECTED_GPUS
        fi
    fi
else
    # Use SLURM's GPUS per node setting
    GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
fi

NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT=${MASTER_PORT:-6000}
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # When running inside containers, `scontrol` might not be available.
    # Allow passing MASTER_ADDR from the host/wrapper.
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi

echo "Running under SLURM: $RUNNING_UNDER_SLURM"
echo "GPUS per node: $GPUS_PER_NODE"
echo "Num nodes: $NUM_NODES"
echo "World size (total GPUs): $WORLD_SIZE"
echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Machine rank (SLURM_PROCID): ${SLURM_PROCID:-N/A}"

if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # For multi-node SLURM: use c10d rendezvous backend (works with torchrun/FSDP)
    # Note: DeepSpeed ignores these flags and uses its own launcher.
    # For multi-node DeepSpeed, use a hostfile or switch to FSDP.
    export LAUNCHER="accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_machines $NUM_NODES \
        --num_processes $WORLD_SIZE \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank \${SLURM_PROCID} \
        --rdzv_backend c10d \
        --rdzv_conf rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --max_restarts 0 \
        --tee 3 \
        "
else
    # Local run (no SLURM): run on a single machine.
    # If your accelerate config already sets num_processes, you can ignore GPUS_PER_NODE.
    export LAUNCHER="accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_machines 1 \
        --num_processes $GPUS_PER_NODE \
        --max_restarts 0 \
        --tee 3 \
        "
fi
echo "Launcher: $LAUNCHER"

if [[ "${LAUNCHER%% *}" == "python" ]]; then
    export CMD="-m projects.iwslt26-metric.train $@"
else
    # accelerate launch
    export CMD="--module projects.iwslt26-metric.train $@"
fi

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # If we're already inside an interactive srun allocation (a Slurm "step"),
    # re-invoking srun from inside that step can fail or create nested steps.
    # Detect that situation via SLURM_STEP_ID (set for srun steps) or SLURM_PROCID
    # and simply run the launcher directly instead of wrapping it with srun.
    if [[ -n "${SLURM_STEP_ID:-}" || -n "${SLURM_PROCID:-}" ]]; then
        echo "Detected running inside an interactive srun step (SLURM_STEP_ID or SLURM_PROCID present); launching directly"
        bash -c "$LAUNCHER $CMD" 2>&1
    else
        srun $SRUN_ARGS --jobid "$SLURM_JOB_ID" bash -c "$LAUNCHER $CMD" 2>&1
    fi
else
# python -m pdb -c continue src/train.py --config $CONFIG_FILE --dry_run
    bash -c "$LAUNCHER $CMD" 2>&1
fi

