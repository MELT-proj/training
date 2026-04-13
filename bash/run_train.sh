#!/bin/bash

set -euo pipefail

is_master_node() {
    [[ -z "${SLURM_PROCID:-}" || "${SLURM_PROCID}" == "0" ]]
}

log_master() {
    if is_master_node; then
        echo "$@"
    fi
}

# --------------------------
# User-overridable defaults
# --------------------------
# Override any of these at invocation time, e.g.:
#   SCRATCH=/path WANDB_PROJECT=foo ./bash/run_train.sh ...

log_master "------------------------------------------------------------------"
log_master "WE ARE INSIDE RUN_TRAIN.SH"
log_master "------------------------------------------------------------------"

# Two crucial environment variables to set up:
# 1) VENV_PATH: path to the python virtualenv activate script
# 2) LOCAL_DATASETS_DIR: path to local datasets storage
VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"
log_master "Using VENV_PATH: $VENV_PATH"
log_master "Using LOCAL_DATASETS_DIR: $LOCAL_DATASETS_DIR"

WANDB_PROJECT="${WANDB_PROJECT:-melt}"
WANDB_MODE="${WANDB_MODE:-online}"
TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

log_master "Activating virtualenv at $VENV_PATH"
source "$VENV_PATH"
if is_master_node; then
    echo "Python is at:"
    command -v python || true
fi

if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
if is_master_node; then
    echo "CUDA version:"
    nvcc --version 2>/dev/null || echo "nvcc not found (may be running in container)"
fi

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
log_master "Using accelerate config: $ACCELERATE_CONFIG"
log_master "Gradient Accumulation Steps: $GRAD_ACC_STEPS"

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

log_master "*** ENV VARIABLES ***"
log_master "WANDB_PROJECT=$WANDB_PROJECT"
log_master "WANDB_MODE=$WANDB_MODE"
log_master "TORCHDYNAMO_VERBOSE=$TORCHDYNAMO_VERBOSE"
log_master "TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
log_master "ACCELERATE_LOG_LEVEL=$ACCELERATE_LOG_LEVEL"
log_master "TRANSFORMERS_VERBOSITY=$TRANSFORMERS_VERBOSITY"
log_master "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
log_master "HF_HOME=$HF_HOME"
log_master "LOCAL_DATASETS_DIR=$LOCAL_DATASETS_DIR"
log_master "TMPDIR=$TMPDIR"
log_master "---------------------"

if is_master_node; then
    echo "Listing what I see under ${HF_HOME}/hub:"
    ls -l "${HF_HOME}/hub" || echo "No models cached yet."
    echo "---------------------"
fi

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

log_master "Running under SLURM: $RUNNING_UNDER_SLURM"
log_master "GPUS per node: $GPUS_PER_NODE"
log_master "Num nodes: $NUM_NODES"
log_master "World size (total GPUs): $WORLD_SIZE"
log_master "Master addr: $MASTER_ADDR"
log_master "Master port: $MASTER_PORT"
log_master "Machine rank (SLURM_PROCID): ${SLURM_PROCID:-N/A}"

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
log_master "Launcher: $LAUNCHER"

if [[ "${LAUNCHER%% *}" == "python" ]]; then
    export CMD="-m src.training.train $@"
else
    # accelerate launch
    export CMD="--module src.training.train $@"
fi

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# ------------------------------------------------------------------
# GPU memory monitoring (per node, runs for the duration of training)
# ------------------------------------------------------------------
_NVIDIA_SMI_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    mkdir -p logs
    _GPU_LOG="logs/gpu_mem_${SLURM_JOB_ID:-local}_node${SLURM_NODEID:-0}.csv"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
        --format=csv -l 30 > "$_GPU_LOG" 2>/dev/null &
    _NVIDIA_SMI_PID=$!
    trap '[[ -n "${_NVIDIA_SMI_PID}" ]] && kill "${_NVIDIA_SMI_PID}" 2>/dev/null; exit' EXIT
    log_master "GPU memory log: $_GPU_LOG"
fi

if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # If we're already inside an interactive srun allocation (a Slurm "step"),
    # re-invoking srun from inside that step can fail or create nested steps.
    # Detect that situation via SLURM_STEP_ID (set for srun steps) or SLURM_PROCID
    # and simply run the launcher directly instead of wrapping it with srun.
    if [[ -n "${SLURM_STEP_ID:-}" || -n "${SLURM_PROCID:-}" ]]; then
        log_master "Detected running inside an interactive srun step (SLURM_STEP_ID or SLURM_PROCID present); launching directly"
        bash -c "$LAUNCHER $CMD" 2>&1
    else
        srun $SRUN_ARGS --jobid "$SLURM_JOB_ID" bash -c "$LAUNCHER $CMD" 2>&1
    fi
else
# python -m pdb -c continue src/train.py --config $CONFIG_FILE --dry_run
    bash -c "$LAUNCHER $CMD" 2>&1
fi

