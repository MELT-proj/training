#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output="logs/%x.%j.out"
#SBATCH --error="logs/%x.%j.err"
#SBATCH --job-name=melt-train-container

set -euo pipefail

# ------------------------------------------------------------------
# Host-side setup (modules, etc.)
# ------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module purge || true
  # Many clusters provide either `singularity` or `apptainer` as a module name.
  module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
  module load cuda
fi

# ------------------------------------------------------------------
# Required: path to the .sif image on the host
# ------------------------------------------------------------------
SINGULARITY_IMG="${SINGULARITY_IMG:-}"
if [[ -z "${SINGULARITY_IMG}" ]]; then
  echo "ERROR: SINGULARITY_IMG is not set"
  echo "Example: SINGULARITY_IMG=/gpfs/projects/.../melt_cuda126.sif sbatch bash/run_train_container.sbatch ..."
  exit 2
fi
if [[ ! -f "${SINGULARITY_IMG}" ]]; then
  echo "ERROR: SINGULARITY_IMG not found: ${SINGULARITY_IMG}"
  exit 2
fi

# Pick the container runtime binary.
# (Most sites have `apptainer`; some still provide `singularity`.)
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

# ------------------------------------------------------------------
# Arguments: pass through to your existing runner
#   sbatch bash/run_train_container.sbatch <config_file> <accelerate_config>
# ------------------------------------------------------------------
if [[ -z "${1:-}" || -z "${2:-}" ]]; then
  echo "Usage: $0 <config_file> <accelerate_config>"
  echo "Example: sbatch $0 config/train/LS_asr.yaml config/accelerate/fsdp.yaml"
  exit 2
fi

CONFIG_FILE_HOST="$1"
ACCELERATE_CONFIG_HOST="$2"

# Make these absolute for reliable binds.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${CONFIG_FILE_HOST}" != /* ]]; then
  CONFIG_FILE_HOST="${SUBMIT_DIR}/${CONFIG_FILE_HOST}"
fi
if [[ "${ACCELERATE_CONFIG_HOST}" != /* ]]; then
  ACCELERATE_CONFIG_HOST="${SUBMIT_DIR}/${ACCELERATE_CONFIG_HOST}"
fi

if [[ ! -f "${CONFIG_FILE_HOST}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_FILE_HOST}"
  exit 2
fi
if [[ ! -f "${ACCELERATE_CONFIG_HOST}" ]]; then
  echo "ERROR: accelerate config not found: ${ACCELERATE_CONFIG_HOST}"
  exit 2
fi

# ------------------------------------------------------------------

# Bind / env mapping
# - Intentionally keep bindings minimal to avoid hiding the repo/venv baked into the image.
# - We bind only: TMPDIR, LOCAL_DATASETS_DIR, and the two config files.
# ------------------------------------------------------------------
HOST_LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR_HOST:-${LOCAL_DATASETS_DIR:-${SUBMIT_DIR}/shar}}"

CONTAINER_PROJECT_DIR="/workspace/training"
CONTAINER_LOCAL_DATASETS_DIR="/workspace/shar"

# Ensure host tmp/logs exist (for caches/checkpoints/logs).
HOST_TMPDIR="${TMPDIR_HOST:-${SUBMIT_DIR}/tmp}"
mkdir -p "${HOST_TMPDIR}"
mkdir -p "${SUBMIT_DIR}/logs"

# Export container-visible env vars. Using SINGULARITYENV_ makes this work even with --cleanenv.
export SINGULARITYENV_LOCAL_DATASETS_DIR="${CONTAINER_LOCAL_DATASETS_DIR}"

# Keep the default venv activation path used by bash/run_train.sh.
export SINGULARITYENV_VENV_PATH="${CONTAINER_PROJECT_DIR}/venv/bin/activate"

# Optional: avoid filling /tmp inside the container by setting TMPDIR.
CONTAINER_TMPDIR="/workspace/tmp"
export SINGULARITYENV_TMPDIR="${CONTAINER_TMPDIR}"

# Also pass through commonly-used ML env vars if set on host.
# (Inherited automatically unless --cleanenv is used.)
export SINGULARITYENV_WANDB_PROJECT="${WANDB_PROJECT:-melt}"
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE:-online}"

# Base output directory for runs (consumed in configs via ${oc.env:OUTPUT_DIR,...}).
OUTPUT_DIR="${OUTPUT_DIR:-${SUBMIT_DIR}/outputs}"
mkdir -p "${OUTPUT_DIR}"
export SINGULARITYENV_OUTPUT_DIR="${OUTPUT_DIR}"

# Multi-node: compute MASTER_ADDR on the host (scontrol is usually not available in the container)
MASTER_PORT="${MASTER_PORT:-6000}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  MASTER_ADDR_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  MASTER_ADDR_HOST="${MASTER_ADDR:-127.0.0.1}"
fi
export SINGULARITYENV_MASTER_ADDR="${MASTER_ADDR_HOST}"
export SINGULARITYENV_MASTER_PORT="${MASTER_PORT}"

# Binds:
# - tmp + datasets
# - Config files (explicitly) to keep things simple
BIND_ARGS="${HOST_TMPDIR}:${CONTAINER_TMPDIR},${HOST_LOCAL_DATASETS_DIR}:${CONTAINER_LOCAL_DATASETS_DIR},${CONFIG_FILE_HOST}:${CONTAINER_PROJECT_DIR}/_config.yaml,${ACCELERATE_CONFIG_HOST}:${CONTAINER_PROJECT_DIR}/_accelerate.yaml"

echo "SINGULARITY_IMG: ${SINGULARITY_IMG}"
echo "SINGULARITY_BIN: ${SINGULARITY_BIN}"
echo "Host datasets: ${HOST_LOCAL_DATASETS_DIR} -> ${CONTAINER_LOCAL_DATASETS_DIR}"
echo "Host tmp: ${HOST_TMPDIR} -> ${CONTAINER_TMPDIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MASTER_ADDR (host): ${MASTER_ADDR_HOST}"
echo "MASTER_PORT: ${MASTER_PORT}"

# ------------------------------------------------------------------
# Run inside srun for multi-node support.
# Your bash/run_train.sh already avoids nested srun when SLURM_STEP_ID/SLURM_PROCID exist.
# ------------------------------------------------------------------
SRUN_ARGS=(
  --kill-on-bad-exit=1
  --wait=60
)

echo "Starting training via srun + ${SINGULARITY_BIN} ..."

srun "${SRUN_ARGS[@]}" \
  "${SINGULARITY_BIN}" exec --nv \
  --bind "${BIND_ARGS}" \
  --pwd "${CONTAINER_PROJECT_DIR}" \
  "${SINGULARITY_IMG}" \
  bash --noprofile --norc -c '
    set -euo pipefail
    echo "[container] pwd: $(pwd)"
    echo "[container] listing /workspace/training:"; ls -la /workspace/training || true
    echo "[container] listing /workspace/training/bash:"; ls -la /workspace/training/bash || true
    test -f /workspace/training/bash/run_train.sh
    exec /workspace/training/bash/run_train.sh /workspace/training/_config.yaml /workspace/training/_accelerate.yaml
  '
