#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-train-container

set -euo pipefail

is_master_node() {
  [[ -z "${SLURM_PROCID:-}" || "${SLURM_PROCID}" == "0" ]]
}

log_master() {
  if is_master_node; then
    echo "$@"
  fi
}

log_master "------------------------------------------------------------------"
log_master "WE ARE INSIDE RUN_TRAIN_SINGULARITY.SBATCH"
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
# Optional: expose the host CUDA *toolkit* (nvcc, headers, libs) inside
# the container. `--nv` typically forwards driver libraries, not nvcc.
#
# We detect the toolkit root on the host and bind-mount it to a stable
# path in the container, then set CUDA_HOME/PATH/LD_LIBRARY_PATH.
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

  # Prefer APPTAINERENV_ (new) but also set SINGULARITYENV_ (compat).
  export APPTAINERENV_CUDA_HOME="${CONTAINER_CUDA_HOME}"
  export SINGULARITYENV_CUDA_HOME="${CONTAINER_CUDA_HOME}"

  # Make nvcc and toolkit libs visible.
  export APPTAINERENV_PREPEND_PATH="${CONTAINER_CUDA_HOME}/bin:${APPTAINERENV_PREPEND_PATH:-}"
  export SINGULARITYENV_PREPEND_PATH="${CONTAINER_CUDA_HOME}/bin:${SINGULARITYENV_PREPEND_PATH:-}"

  export APPTAINERENV_PREPEND_LD_LIBRARY_PATH="${CONTAINER_CUDA_HOME}/lib64:${CONTAINER_CUDA_HOME}/lib:${APPTAINERENV_PREPEND_LD_LIBRARY_PATH:-}"
  export SINGULARITYENV_PREPEND_LD_LIBRARY_PATH="${CONTAINER_CUDA_HOME}/lib64:${CONTAINER_CUDA_HOME}/lib:${SINGULARITYENV_PREPEND_LD_LIBRARY_PATH:-}"
else
  log_master "WARNING: CUDA toolkit not detected on host (CUDA_HOME/CUDA_PATH/nvcc). nvcc will likely be unavailable in the container."
fi
log_master "CUDA_BIND_ARG: ${CUDA_BIND_ARG}"

# ------------------------------------------------------------------
# Hugging Face cache: bind host HF_HOME into the container so model/
# tokenizer downloads are shared across runs.
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

  # Be explicit: hub cache lives under HF_HOME/hub
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
#   sbatch bash/run_train_container.sbatch <accelerate_config> [train_args...]
# ------------------------------------------------------------------
if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 <accelerate_config> [train_args...]"
  echo "Example: sbatch $0 config/accelerate/fsdp.yaml --config config/train/LS_asr.yaml"
  echo "Example: sbatch $0 config/accelerate/fsdp.yaml --trainer.max-steps 1000"
  exit 2
fi

ACCELERATE_CONFIG_HOST="$1"
if [[ ! -f "${ACCELERATE_CONFIG_HOST}" ]]; then
  echo "ERROR: accelerate config not found: ${ACCELERATE_CONFIG_HOST}"
  exit 2
fi

shift  # Remove accelerate config, rest are train args
TRAIN_ARGS="$*"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ------------------------------------------------------------------

# Bind / env mapping
# - Bind the host project dir so container runs latest code.
# - Keep the venv outside the project dir to avoid being hidden by the bind mount.
# ------------------------------------------------------------------
HOST_PROJECT_DIR="${SUBMIT_DIR}"
HOST_LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR_HOST:-${LOCAL_DATASETS_DIR:-${SUBMIT_DIR}/shar}}"

CONTAINER_PROJECT_DIR="/workspace/training"
CONTAINER_LOCAL_DATASETS_DIR="/workspace/shar"

# Ensure host tmp/logs exist (for caches/checkpoints/logs).
HOST_TMPDIR="${TMPDIR_HOST:-/tmp}"
mkdir -p "${HOST_TMPDIR}"
mkdir -p "${SUBMIT_DIR}/logs"

# Export container-visible env vars. Using SINGULARITYENV_ makes this work even with --cleanenv.
export SINGULARITYENV_LOCAL_DATASETS_DIR="${CONTAINER_LOCAL_DATASETS_DIR}"

# Use the container venv at /workspace/venv
export SINGULARITYENV_VENV_PATH="/workspace/venv/bin/activate"

# Optional: avoid filling /tmp inside the container by setting TMPDIR.
CONTAINER_TMPDIR="/workspace/tmp"
export SINGULARITYENV_TMPDIR="${CONTAINER_TMPDIR}"

# Also pass through commonly-used ML env vars if set on host.
# (Inherited automatically unless --cleanenv is used.)
export SINGULARITYENV_WANDB_PROJECT="${WANDB_PROJECT:-melt}"
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE:-online}"

# Redirect tool/JIT caches to the writable tmp bind-mount.
# The home dir may be mounted read-only inside the container, which causes
# deepspeed/triton (TRITON_CACHE_DIR), lhotse, and wandb to fail when trying
# to create cache dirs under ~/.triton, ~/.lhotse, ~/.cache, etc.
export SINGULARITYENV_TRITON_CACHE_DIR="${CONTAINER_TMPDIR}/.triton"
export APPTAINERENV_TRITON_CACHE_DIR="${CONTAINER_TMPDIR}/.triton"
export SINGULARITYENV_TRITON_HOME="${CONTAINER_TMPDIR}/.triton"
export APPTAINERENV_TRITON_HOME="${CONTAINER_TMPDIR}/.triton"
# Redirect general XDG/pip/torch caches too
export SINGULARITYENV_XDG_CACHE_HOME="${CONTAINER_TMPDIR}/.cache"
export APPTAINERENV_XDG_CACHE_HOME="${CONTAINER_TMPDIR}/.cache"

# Base output directory for runs.
# IMPORTANT: don't point the container at host paths like /mnt/scratch-... directly.
# Some clusters bind those paths read-only inside the container. Instead, bind-mount
# the host output directory to a known writable container path.
HOST_OUTPUT_DIR="${OUTPUT_DIR:-${SUBMIT_DIR}/outputs}"
mkdir -p "${HOST_OUTPUT_DIR}"
CONTAINER_OUTPUT_DIR="/workspace/outputs"

# Prefer APPTAINERENV_ (new) but also set SINGULARITYENV_ (compat).
export APPTAINERENV_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}"
export SINGULARITYENV_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}"

# Weights & Biases directory: store wandb runs under the output directory
HOST_WANDB_DIR="${HOST_OUTPUT_DIR}/wandb"
mkdir -p "${HOST_WANDB_DIR}"
CONTAINER_WANDB_DIR="${CONTAINER_OUTPUT_DIR}/wandb"

export APPTAINERENV_WANDB_DIR="${CONTAINER_WANDB_DIR}"
export SINGULARITYENV_WANDB_DIR="${CONTAINER_WANDB_DIR}"

# Multi-node: compute MASTER_ADDR on the host (scontrol is usually not available in the container)
MASTER_PORT="${MASTER_PORT:-6000}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  MASTER_ADDR_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  MASTER_ADDR_HOST="${MASTER_ADDR:-127.0.0.1}"
fi
export SINGULARITYENV_MASTER_ADDR="${MASTER_ADDR_HOST}"
export SINGULARITYENV_MASTER_PORT="${MASTER_PORT}"

# ------------------------------------------------------------------
# OpenMP tuning (PyTorch CPU-thread performance guide)
# - OMP_NUM_THREADS: default to SLURM cpus-per-task; override by setting the
#   variable before calling sbatch.
# - OMP_PROC_BIND=CLOSE binds threads to nearby CPUs within the cpuset that
#   SLURM already assigned to this job — no need to specify physical CPU IDs.
# - GOMP_CPU_AFFINITY is intentionally omitted: it requires knowing the exact
#   physical CPU IDs in the job's cpuset and causes libgomp thread-creation
#   failures (EINVAL) when those IDs don't start from 0.
# ------------------------------------------------------------------
_OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

export SINGULARITYENV_OMP_NUM_THREADS="${_OMP_NUM_THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${_OMP_NUM_THREADS}"

export SINGULARITYENV_OMP_SCHEDULE="${OMP_SCHEDULE:-STATIC}"
export APPTAINERENV_OMP_SCHEDULE="${OMP_SCHEDULE:-STATIC}"

export SINGULARITYENV_OMP_PROC_BIND="${OMP_PROC_BIND:-CLOSE}"
export APPTAINERENV_OMP_PROC_BIND="${OMP_PROC_BIND:-CLOSE}"

log_master "OMP_NUM_THREADS: ${_OMP_NUM_THREADS}"
log_master "OMP_SCHEDULE: ${OMP_SCHEDULE:-STATIC}"
log_master "OMP_PROC_BIND: ${OMP_PROC_BIND:-CLOSE}"

export SINGULARITYENV_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export APPTAINERENV_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
log_master "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 1800"

# ------------------------------------------------------------------
# SLURM variables: pass through to container (--cleanenv strips them)
# These are critical for multi-node training to detect world size.
#
# NOTE: Variables set at sbatch time (JOB_ID, NNODES, GPUS_ON_NODE, NODELIST)
# can be exported here. Variables set at srun time (PROCID, NODEID, LOCALID,
# STEP_ID) must be passed via --env flags in the singularity exec command
# to capture the per-node values correctly.
# ------------------------------------------------------------------
export SINGULARITYENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export SINGULARITYENV_SLURM_NNODES="${SLURM_NNODES:-1}"
export SINGULARITYENV_SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-}"
export SINGULARITYENV_SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}"

# Also set APPTAINERENV_ for newer apptainer versions
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export APPTAINERENV_SLURM_NNODES="${SLURM_NNODES:-1}"
export APPTAINERENV_SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-}"
export APPTAINERENV_SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}"

log_master "SLURM_NNODES: ${SLURM_NNODES:-1}"
log_master "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-unset}"

# Binds:
# - project dir (latest code)
# - tmp + datasets
# - Accelerate config file (explicitly) to keep things simple
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
log_master "OUTPUT_DIR (container): ${CONTAINER_OUTPUT_DIR}"
log_master "MASTER_ADDR (host): ${MASTER_ADDR_HOST}"
log_master "MASTER_PORT: ${MASTER_PORT}"
log_master "Train args: ${TRAIN_ARGS}"

# ------------------------------------------------------------------
# Run inside srun for multi-node support.
# Your bash/run_train.sh already avoids nested srun when SLURM_STEP_ID/SLURM_PROCID exist.
# ------------------------------------------------------------------
SRUN_ARGS=(
  --kill-on-bad-exit=1
  --wait=60
)

# ------------------------------------------------------------------
# GPU memory monitoring (host side, master node only)
# Per-node logs are also written from inside the container by run_train.sh.
# Set GPU_MEM_MONITORING=1 to enable.
# ------------------------------------------------------------------
_NVIDIA_SMI_PID=""
if [[ "${GPU_MEM_MONITORING:-0}" -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    _GPU_LOG="${SUBMIT_DIR}/logs/gpu_mem_${SLURM_JOB_ID:-local}_host.csv"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
        --format=csv -l 30 > "$_GPU_LOG" 2>/dev/null &
    _NVIDIA_SMI_PID=$!
    trap '[[ -n "${_NVIDIA_SMI_PID}" ]] && kill "${_NVIDIA_SMI_PID}" 2>/dev/null; exit' EXIT
    log_master "GPU memory log (host): $_GPU_LOG"
fi

log_master "Starting training via srun + ${SINGULARITY_BIN} ..."

# NOTE: SLURM_PROCID, SLURM_NODEID, SLURM_LOCALID, SLURM_STEP_ID are set by srun
# at runtime (different value per node). We wrap the singularity call in a bash -c
# so these variables are evaluated INSIDE the srun step, not at sbatch parse time.
#
# The outer 'bash -c' runs inside srun (where SLURM_PROCID is set correctly).
# We then pass these values to singularity via --env.
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
        exec /workspace/training/bash/run_train.sh '"${ACCELERATE_CONFIG_HOST}"' '"${TRAIN_ARGS}"'
      "
  '
