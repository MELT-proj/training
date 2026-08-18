#!/bin/bash
#
# THE training launcher for every context: local machine, SLURM (native venv),
# and inside a Singularity/Apptainer container (invoked by the sbatch shim).
# It knows nothing about containers — it just activates a venv (if present) and
# runs `accelerate launch python -m melt.training.train`.
#
# Usage:
#   ./bash/run_train.sh <accelerate_config> [train_args...]
#
# Examples:
#   ./bash/run_train.sh config/accelerate/zero3.yaml --config config/train/asr.yaml
#   sbatch [flags] bash/run_train.sh config/accelerate/zero3.yaml --config config/train/asr.yaml --trainer.max_steps 1000
#
# The config system uses OmegaConf, so extra args use dot notation
# (e.g. --trainer.max_steps 1000).

set -euo pipefail

is_master_node() {
    [[ -z "${SLURM_PROCID:-}" || "${SLURM_PROCID}" == "0" ]]
}

log_master() {
    if is_master_node; then
        echo "$@"
    fi
}

# Arguments
if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <accelerate_config> [train_args...]"
    echo "Example: $0 config/accelerate/zero3.yaml --config config/train/asr.yaml"
    echo "Example: $0 config/accelerate/zero3.yaml --config config/train/asr.yaml --trainer.max_steps 1000"
    exit 1
fi

ACCELERATE_CONFIG="$1"
shift  # remaining args are forwarded to the training entrypoint

# Environment defaults (override any of these at invocation time, e.g.:
#   VENV_PATH=/path WANDB_PROJECT=foo ./bash/run_train.sh ...)
# VENV_PATH:          python virtualenv activate script (container: /workspace/venv/bin/activate)
# LOCAL_DATASETS_DIR: SHAR datasets root; train YAMLs read it via ${oc.env:LOCAL_DATASETS_DIR}
export VENV_PATH="${VENV_PATH:-/workspace/venv/bin/activate}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-./shar}"

# Default the crash-prone vars so `set -u` can't kill a bare local run.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/tmp}"

export WANDB_PROJECT="${WANDB_PROJECT:-melt}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MELT_SEED="${MELT_SEED:-42}"
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

# Activate the virtualenv if one is present; otherwise fall back to the
# current python environment (helps conda / system-python users).
if [[ -f "$VENV_PATH" ]]; then
    log_master "[run_train] activating virtualenv: $VENV_PATH"
    # shellcheck disable=SC1090
    source "$VENV_PATH"
else
    log_master "[run_train] WARNING: VENV_PATH not found ($VENV_PATH); using current python environment"
fi
log_master "[run_train] python: $(command -v python || echo 'not found')"

# Best-effort CUDA toolkit via lmod (native SLURM sites); harmless elsewhere.
if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
if is_master_node; then
    nvcc --version 2>/dev/null | grep -i release || echo "[run_train] nvcc not found (fine unless a config JIT-compiles CUDA ops)"
fi

# Gradient accumulation: parse from the train args so the accelerate flag
# matches the YAML. Matches both spellings and both --flag VAL / --flag=VAL.
# (HF TrainingArguments ultimately governs; the flag is kept for consistency
# with config/accelerate/*.yaml.)
GRAD_ACC_STEPS=4
_expect_grad_acc=0
for arg in "$@"; do
    if [[ "$_expect_grad_acc" -eq 1 ]]; then
        GRAD_ACC_STEPS="$arg"
        _expect_grad_acc=0
    elif [[ "$arg" == "--trainer.gradient_accumulation_steps" || "$arg" == "--trainer.gradient-accumulation-steps" ]]; then
        _expect_grad_acc=1
    elif [[ "$arg" == "--trainer.gradient_accumulation_steps="* || "$arg" == "--trainer.gradient-accumulation-steps="* ]]; then
        GRAD_ACC_STEPS="${arg#*=}"
    fi
done

# Topology: SLURM vs local, GPUs/node, world size, rendezvous endpoint.
RUNNING_UNDER_SLURM=0
CONTEXT=local
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUNNING_UNDER_SLURM=1
    CONTEXT=slurm
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$GPUS_PER_NODE}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)
    [[ "${DETECTED_GPUS:-0}" -gt 0 ]] && GPUS_PER_NODE="$DETECTED_GPUS"
fi

NUM_NODES="${SLURM_NNODES:-1}"
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT="${MASTER_PORT:-6000}"
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # scontrol may be unavailable inside containers; allow MASTER_ADDR from the host.
    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

# Explicit shard_seed override: train YAMLs set an int (not 'randomized',
# which the dataloader rejects for indexed Shar sources -- see
# melt/training/data/audio/lhotse/dataloader.py), so this only needs to change
# the value, not silence a warning. Spliced in ahead of "$@" below so an
# explicit --data.*.shard_seed on the CLI still wins (OmegaConf dotlist merge
# keeps the last occurrence of a key).
SEED_ARGS=(--data.train_ds.shard_seed "$MELT_SEED" --data.validation_ds.shard_seed "$MELT_SEED")

log_master "[run_train] starting (context: $CONTEXT, nodes=$NUM_NODES, gpus/node=$GPUS_PER_NODE, world_size=$WORLD_SIZE)"
log_master "[run_train] accelerate config: $ACCELERATE_CONFIG | grad_accum: $GRAD_ACC_STEPS | master: $MASTER_ADDR:$MASTER_PORT"
if is_master_node; then
    echo "[run_train] environment:"
    for v in VENV_PATH HF_HOME HF_HUB_OFFLINE LOCAL_DATASETS_DIR \
             TMPDIR ACCELERATE_LOG_LEVEL TRANSFORMERS_VERBOSITY TORCHDYNAMO_VERBOSE \
             TORCH_NCCL_ASYNC_ERROR_HANDLING HF_HUB_ENABLE_HF_TRANSFER MELT_SEED; do
        echo "  $v=${!v:-}"
    done
    # Whatever experiment tracker is configured, print its settings: they
    # decide where the run ends up, which is the first thing you want from the
    # log when a run does not appear where you expected.
    echo "[run_train] experiment tracking:"
    env | grep -E '^(WANDB_|MLFLOW_|TRACKIO_|COMET_|NEPTUNE_|TENSORBOARD_)' \
        | grep -viE 'key|token|secret|password' | sort | sed 's/^/  /' \
        || echo "  (none configured)"
    echo "[run_train] cached models under ${HF_HOME}/hub:"
    ls -1 "${HF_HOME}/hub" 2>/dev/null || echo "  (none cached yet)"

    # OpenMP settings. Note this shell sees the allocation, not the affinity the
    # training process ends up with: OMP_PROC_BIND collapses it only once libgomp
    # initialises inside python. The post-import value is logged from train.py.
    echo "[run_train] cpu: $(nproc) allocated" \
         "| OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} OMP_PROC_BIND=${OMP_PROC_BIND:-unset}"
fi

# GPU memory monitoring (per node, runs for the duration of training).
# Opt in with GPU_MEM_MONITORING=1.
_NVIDIA_SMI_PID=""
if [[ "${GPU_MEM_MONITORING:-0}" -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mkdir -p logs
    _GPU_LOG="logs/gpu_mem_${SLURM_JOB_ID:-local}_node${SLURM_NODEID:-0}.csv"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
        --format=csv -l 30 > "$_GPU_LOG" 2>/dev/null &
    _NVIDIA_SMI_PID=$!
    trap '[[ -n "${_NVIDIA_SMI_PID}" ]] && kill "${_NVIDIA_SMI_PID}" 2>/dev/null; exit' EXIT
    log_master "[run_train] GPU memory log: $_GPU_LOG"
fi

# Build the accelerate launch command as an array (preserves arg quoting).
# --machine_rank is intentionally omitted here: under SLURM it must be the
# per-task SLURM_PROCID, which is injected at launch time below.
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    # Multi-node uses the c10d rendezvous backend (torchrun/FSDP). DeepSpeed
    # ignores these flags and uses its own launcher; for multi-node DeepSpeed
    # use a hostfile or switch to FSDP.
    LAUNCH_CMD=(
        accelerate launch
        --config_file "$ACCELERATE_CONFIG"
        --gradient_accumulation_steps "$GRAD_ACC_STEPS"
        --num_machines "$NUM_NODES"
        --num_processes "$WORLD_SIZE"
        --main_process_ip "$MASTER_ADDR"
        --main_process_port "$MASTER_PORT"
        --rdzv_backend c10d
        --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
        --max_restarts 0
        --tee 3
    )
else
    LAUNCH_CMD=(
        accelerate launch
        --config_file "$ACCELERATE_CONFIG"
        --gradient_accumulation_steps "$GRAD_ACC_STEPS"
        --num_machines 1
        --num_processes "$GPUS_PER_NODE"
        --max_restarts 0
        --tee 3
    )
fi

# Launch. Three paths:
#   1. inside an existing srun step  -> run directly (no nested srun)
#   2. sbatch payload                -> fan out with srun, one task per node
#   3. local (no SLURM)              -> run directly
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
    SRUN_ARGS=(--wait=60 --kill-on-bad-exit=1)
    if [[ -n "${SLURM_STEP_ID:-}" || -n "${SLURM_PROCID:-}" ]]; then
        # Re-invoking srun inside a step nests/breaks; SLURM_PROCID is this task's rank.
        log_master "[run_train] inside an srun step; launching directly"
        "${LAUNCH_CMD[@]}" --machine_rank "${SLURM_PROCID:-0}" \
            --module melt.training.train "${SEED_ARGS[@]}" "$@" 2>&1
    else
        # One srun task per node; --machine_rank is evaluated per task inside the
        # step (splice it in right after `accelerate launch`).
        log_master "[run_train] launching via srun"
        srun "${SRUN_ARGS[@]}" --jobid "$SLURM_JOB_ID" bash -c '
            cmd=("$@")
            exec "${cmd[@]:0:2}" --machine_rank "${SLURM_PROCID:-0}" "${cmd[@]:2}"
        ' _ "${LAUNCH_CMD[@]}" --module melt.training.train "${SEED_ARGS[@]}" "$@" 2>&1
    fi
else
    "${LAUNCH_CMD[@]}" --module melt.training.train "${SEED_ARGS[@]}" "$@" 2>&1
fi
