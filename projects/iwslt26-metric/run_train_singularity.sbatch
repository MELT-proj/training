#!/bin/bash
#SBATCH --job-name=melt-qe-train
#SBATCH --time=04:00:00
#SBATCH --qos=gpu-short
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

mkdir -p logs

is_master_node() {
  [[ -z "${SLURM_PROCID:-}" || "${SLURM_PROCID}" == "0" ]]
}

log_master() {
  if is_master_node; then
    echo "$@"
  fi
}

log_master "------------------------------------------------------------------"
log_master "WE ARE INSIDE IWSLT26-METRIC/RUN_TRAIN_SINGULARITY.SBATCH"
log_master "------------------------------------------------------------------"

# ------------------------------------------------------------------
# Host-side setup (modules, etc.)
# ------------------------------------------------------------------
if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load singularity 2>/dev/null || true
  module load cuda || true
fi

# ------------------------------------------------------------------
# Optional: expose the host CUDA toolkit (nvcc, headers, libs) inside
# the container. `--nv` typically forwards driver libraries, not nvcc.
# ------------------------------------------------------------------
HOST_CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "${HOST_CUDA_HOME}" ]] && command -v nvcc >/dev/null 2>&1; then
  NVCC_REALPATH="$(readlink -f "$(command -v nvcc)")"
  HOST_CUDA_HOME="$(dirname "$(dirname "${NVCC_REALPATH}")")"
fi

CUDA_BIND_ARG=""
CONTAINER_CUDA_HOME=""
if [[ -n "${HOST_CUDA_HOME}" ]] && [[ -d "${HOST_CUDA_HOME}" ]]; then
  CONTAINER_CUDA_HOME="/workspace/cuda-host"
  CUDA_BIND_ARG=",${HOST_CUDA_HOME}:${CONTAINER_CUDA_HOME}:ro"

  export APPTAINERENV_CUDA_HOME="${CONTAINER_CUDA_HOME}"
  export SINGULARITYENV_CUDA_HOME="${CONTAINER_CUDA_HOME}"

  export APPTAINERENV_PREPEND_PATH="${CONTAINER_CUDA_HOME}/bin:${APPTAINERENV_PREPEND_PATH:-}"
  export SINGULARITYENV_PREPEND_PATH="${CONTAINER_CUDA_HOME}/bin:${SINGULARITYENV_PREPEND_PATH:-}"

  export APPTAINERENV_PREPEND_LD_LIBRARY_PATH="${CONTAINER_CUDA_HOME}/lib64:${CONTAINER_CUDA_HOME}/lib:${APPTAINERENV_PREPEND_LD_LIBRARY_PATH:-}"
  export SINGULARITYENV_PREPEND_LD_LIBRARY_PATH="${CONTAINER_CUDA_HOME}/lib64:${CONTAINER_CUDA_HOME}/lib:${SINGULARITYENV_PREPEND_LD_LIBRARY_PATH:-}"
else
  log_master "WARNING: CUDA toolkit not detected on host (CUDA_HOME/CUDA_PATH/nvcc). nvcc will likely be unavailable in the container."
fi
log_master "CUDA_BIND_ARG: ${CUDA_BIND_ARG}"

# ------------------------------------------------------------------
# Hugging Face cache: bind host HF_HOME into the container.
# ------------------------------------------------------------------
HOST_HF_HOME="${HF_HOME:-}"
HF_BIND_ARG=""
CONTAINER_HF_HOME=""
if [[ -n "${HOST_HF_HOME}" ]]; then
  mkdir -p "${HOST_HF_HOME}"
  CONTAINER_HF_HOME="/workspace/hf_cache"
  HF_BIND_ARG=",${HOST_HF_HOME}:${CONTAINER_HF_HOME}"

  export APPTAINERENV_HF_HOME="${CONTAINER_HF_HOME}"
  export SINGULARITYENV_HF_HOME="${CONTAINER_HF_HOME}"

  export APPTAINERENV_HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
  export SINGULARITYENV_HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
else
  log_master "WARNING: HF_HOME is not set on host; not binding Hugging Face cache into container."
fi
log_master "HF_BIND_ARG: ${HF_BIND_ARG}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export SINGULARITYENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
log_master "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"

# ------------------------------------------------------------------
# Required: path to the .sif image on the host
# ------------------------------------------------------------------
SINGULARITY_IMG="${SINGULARITY_IMG:-}"
if [[ -z "${SINGULARITY_IMG}" ]]; then
  echo "ERROR: SINGULARITY_IMG is not set"
  echo "Example: SINGULARITY_IMG=/gpfs/projects/.../melt_cuda126.sif sbatch projects/iwslt26-metric/run_train_singularity.sbatch ..."
  exit 2
fi
if [[ ! -f "${SINGULARITY_IMG}" ]]; then
  echo "ERROR: SINGULARITY_IMG not found: ${SINGULARITY_IMG}"
  exit 2
fi

# Pick the container runtime binary.
SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [[ -z "${SINGULARITY_BIN}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY_BIN=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    SINGULARITY_BIN=singularity
  else
    echo "ERROR: neither 'apptainer' nor 'singularity' found in PATH"
    exit 2
  fi
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ------------------------------------------------------------------
# Bind / env mapping
# ------------------------------------------------------------------
HOST_PROJECT_DIR="${SUBMIT_DIR}"
HOST_LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR_HOST:-${LOCAL_DATASETS_DIR:-${SUBMIT_DIR}/shar}}"

CONTAINER_PROJECT_DIR="/workspace/training"
CONTAINER_LOCAL_DATASETS_DIR="/workspace/shar"

HOST_TMPDIR="${TMPDIR_HOST:-/tmp}"
mkdir -p "${HOST_TMPDIR}"

export SINGULARITYENV_LOCAL_DATASETS_DIR="${CONTAINER_LOCAL_DATASETS_DIR}"
export SINGULARITYENV_VENV_PATH="/workspace/venv/bin/activate"

CONTAINER_TMPDIR="/workspace/tmp"
export SINGULARITYENV_TMPDIR="${CONTAINER_TMPDIR}"

export SINGULARITYENV_WANDB_PROJECT="${WANDB_PROJECT:-iwslt26-metric}"
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE:-online}"

HOST_OUTPUT_DIR="${OUTPUT_DIR:-${SUBMIT_DIR}/outputs}"
mkdir -p "${HOST_OUTPUT_DIR}"
CONTAINER_OUTPUT_DIR="/workspace/outputs"

export APPTAINERENV_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}"
export SINGULARITYENV_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}"

HOST_WANDB_DIR="${HOST_OUTPUT_DIR}/wandb"
mkdir -p "${HOST_WANDB_DIR}"
CONTAINER_WANDB_DIR="${CONTAINER_OUTPUT_DIR}/wandb"

export APPTAINERENV_WANDB_DIR="${CONTAINER_WANDB_DIR}"
export SINGULARITYENV_WANDB_DIR="${CONTAINER_WANDB_DIR}"

# Multi-node: compute MASTER_ADDR on the host
MASTER_PORT="${MASTER_PORT:-6000}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  MASTER_ADDR_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  MASTER_ADDR_HOST="${MASTER_ADDR:-127.0.0.1}"
fi
export SINGULARITYENV_MASTER_ADDR="${MASTER_ADDR_HOST}"
export SINGULARITYENV_MASTER_PORT="${MASTER_PORT}"

# SLURM variables for multi-node training
export SINGULARITYENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export SINGULARITYENV_SLURM_NNODES="${SLURM_NNODES:-1}"
export SINGULARITYENV_SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-}"
export SINGULARITYENV_SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}"

export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export APPTAINERENV_SLURM_NNODES="${SLURM_NNODES:-1}"
export APPTAINERENV_SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-}"
export APPTAINERENV_SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}"

log_master "SLURM_NNODES: ${SLURM_NNODES:-1}"
log_master "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-unset}"

BIND_ARGS="${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR},${HOST_TMPDIR}:${CONTAINER_TMPDIR},${HOST_LOCAL_DATASETS_DIR}:${CONTAINER_LOCAL_DATASETS_DIR},${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}:rw${CUDA_BIND_ARG}"
BIND_ARGS="${BIND_ARGS}${HF_BIND_ARG}"

log_master "SINGULARITY_IMG: ${SINGULARITY_IMG}"
log_master "SINGULARITY_BIN: ${SINGULARITY_BIN}"
log_master "Host project: ${HOST_PROJECT_DIR} -> ${CONTAINER_PROJECT_DIR}"
log_master "Host datasets: ${HOST_LOCAL_DATASETS_DIR} -> ${CONTAINER_LOCAL_DATASETS_DIR}"
log_master "Host tmp: ${HOST_TMPDIR} -> ${CONTAINER_TMPDIR}"
log_master "Host outputs: ${HOST_OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}"
log_master "Host wandb: ${HOST_WANDB_DIR} -> ${CONTAINER_WANDB_DIR}"
if [[ -n "${CUDA_BIND_ARG}" ]]; then
  log_master "Host CUDA toolkit: ${HOST_CUDA_HOME} -> ${CONTAINER_CUDA_HOME}"
fi
if [[ -n "${HF_BIND_ARG}" ]]; then
  log_master "Host HF_HOME: ${HOST_HF_HOME} -> ${CONTAINER_HF_HOME}"
fi
log_master "MASTER_ADDR (host): ${MASTER_ADDR_HOST}"
log_master "MASTER_PORT: ${MASTER_PORT}"

TRAIN_ARGS="$*"
log_master "Train args: ${TRAIN_ARGS}"

# ------------------------------------------------------------------
# Run inside srun for multi-node support.
# ------------------------------------------------------------------
SRUN_ARGS=(
  --kill-on-bad-exit=1
  --wait=60
)

log_master "Starting training via srun + ${SINGULARITY_BIN} ..."

srun "${SRUN_ARGS[@]}" \
  bash -c '
    if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
      echo "[srun step] SLURM_PROCID=${SLURM_PROCID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset}"
    fi
    '"${SINGULARITY_BIN}"' exec --nv \
      --bind "'"${BIND_ARGS}"'" \
      --pwd "'"${CONTAINER_PROJECT_DIR}"'" \
      --cleanenv \
      --env SLURM_PROCID="${SLURM_PROCID:-0}" \
      --env SLURM_NODEID="${SLURM_NODEID:-0}" \
      --env SLURM_LOCALID="${SLURM_LOCALID:-0}" \
      --env SLURM_STEP_ID="${SLURM_STEP_ID:-}" \
      "'"${SINGULARITY_IMG}"'" \
      bash --noprofile --norc -c "
        set -euo pipefail
        if [[ \"\${SLURM_PROCID:-0}\" == \"0\" ]]; then
          echo \"[container] SLURM_PROCID=\${SLURM_PROCID:-unset} SLURM_NODEID=\${SLURM_NODEID:-unset}\"
          echo \"[container] pwd: \$(pwd)\"
        fi
        exec /workspace/training/projects/iwslt26-metric/run_train.sh '"${TRAIN_ARGS}"'
      "
  '
