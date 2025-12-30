#!/bin/bash
#SBATCH --job-name=MELT_PoC
#SBATCH --output=./logs/%A.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=gpu-short
#SBATCH --partition=a6000
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1

set -euo pipefail

# --------------------------
# User-overridable defaults
# --------------------------
# Override any of these at invocation time, e.g.:
#   SCRATCH=/path WANDB_PROJECT=foo ./bash/train_with_config.sh ...

VENV_PATH_DEFAULT="${VENV_PATH_DEFAULT:-$HOME/mydata/venvs/speech_lm/bin/activate}"

SCRATCH="${SCRATCH:-/mnt/home/giuseppe/myscratch}"
BASEDIR="${BASEDIR:-$SCRATCH/speech_lm}"

TMPDIR="${TMPDIR:-/tmp}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/mnt/scratch-artemis/shared/datasets}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_home_speech}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BASEDIR/hf_datasets_cache}"

WANDB_PROJECT="${WANDB_PROJECT:-speech_lm}"
TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_PATH_DEFAULT" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH_DEFAULT"
fi
echo "Python is at:"
command -v python || true

if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda
fi

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <config_file> <accelerate_config>"
    echo "Tip: run under SLURM with sbatch, or run locally on a node by executing this script directly."
    exit 1
fi

CONFIG_FILE=$1
ACCELERATE_CONFIG=$2
GRAD_ACC_STEPS=$(grep -m 1 'gradient_accumulation_steps' "$CONFIG_FILE" | awk '{print $2}' || true)
if [[ -z "${GRAD_ACC_STEPS:-}" ]]; then
    GRAD_ACC_STEPS=1
    echo "Warning: could not parse gradient_accumulation_steps from $CONFIG_FILE; defaulting to $GRAD_ACC_STEPS"
fi
echo "Using config file: $CONFIG_FILE"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"

# setup run-specific envs
# export LOCAL_DATASETS_DIR="/mnt/home/giuseppe/myscratch/speech_lm/datasets"
export SCRATCH
export BASEDIR
export TMPDIR
export LOCAL_DATASETS_DIR
export HF_HOME
export HF_DATASETS_CACHE
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT
# export WANDB_MODE=offline
export TORCHDYNAMO_VERBOSE
export TORCH_NCCL_ASYNC_ERROR_HANDLING

export HF_HUB_ENABLE_HF_TRANSFER
export ACCELERATE_LOG_LEVEL
export TRANSFORMERS_VERBOSITY

RUNNING_UNDER_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUNNING_UNDER_SLURM=1
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
if [[ "$RUNNING_UNDER_SLURM" -eq 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)
        if [[ -n "${DETECTED_GPUS:-}" && "${DETECTED_GPUS:-0}" -gt 0 ]]; then
            GPUS_PER_NODE=$DETECTED_GPUS
        fi
    fi
fi

NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT=${MASTER_PORT:-6000}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    export LAUNCHER="accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_machines $NUM_NODES \
        --num_processes $WORLD_SIZE \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank \$SLURM_PROCID \
        --rdzv_conf \"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \
        --max_restarts 1 \
        --role \$(hostname -s): \
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
        --max_restarts 1 \
        --tee 3 \
        "
fi

export CMD="src/train.py --config-file $CONFIG_FILE"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    srun $SRUN_ARGS --jobid "$SLURM_JOB_ID" bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1
else
# python -m pdb -c continue src/train.py --config-file $CONFIG_FILE --dry_run
    bash -c "$LAUNCHER $CMD" 2>&1
fi

