#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-train-container
#
# Thin container shim: validates the image, computes bind mounts and the
# SINGULARITYENV_* the container needs, then runs bash/run_train.sh INSIDE the
# image under srun. All accelerate/topology logic lives in run_train.sh.
#
# Usage (host side):
#   SINGULARITY_IMG=/path/to/melt_cuda126.sif \
#     sbatch [slurm flags] bash/run_train_singularity.sbatch <accelerate_config> [train_args...]
#
# The accelerate config MUST be a repo-relative path (e.g. config/accelerate/zero3.yaml):
# the repo is bind-mounted at /workspace/training and absolute host paths do not
# resolve inside the container. Likewise, --trainer.output_dir must use the
# CONTAINER path /workspace/outputs/... (the host OUTPUT_DIR is only the bind source).

set -euo pipefail

echo "[container] host-side setup for run_train_singularity.sbatch"

# Container runtime: some sites expose it via environment modules, so load
# that first, then autodetect (apptainer preferred, singularity fallback).
if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi

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

# Validation
SINGULARITY_IMG="${SINGULARITY_IMG:-}"
if [[ -z "${SINGULARITY_IMG}" ]]; then
    echo "ERROR: SINGULARITY_IMG is not set"
    echo "Example: SINGULARITY_IMG=/path/melt_cuda126.sif sbatch bash/run_train_singularity.sbatch config/accelerate/zero3.yaml --config config/train/asr.yaml"
    exit 2
fi
if [[ ! -f "${SINGULARITY_IMG}" ]]; then
    echo "ERROR: SINGULARITY_IMG not found: ${SINGULARITY_IMG}"
    exit 2
fi

if [[ -z "${1:-}" ]]; then
    echo "Usage: sbatch $0 <accelerate_config> [train_args...]"
    echo "Example: sbatch $0 config/accelerate/zero3.yaml --config config/train/LS_asr.yaml"
    exit 2
fi
if [[ "$1" = /* ]]; then
    echo "ERROR: pass a repo-relative accelerate config (e.g. config/accelerate/zero3.yaml), not '$1'."
    echo "       Absolute host paths do not resolve inside the container."
    exit 2
fi
if [[ ! -f "$1" ]]; then
    echo "ERROR: accelerate config not found: $1"
    exit 2
fi

ACCELERATE_CONFIG="$1"
shift
TRAIN_ARGS=("$@")

# Host paths (bind sources). Fixed container paths are /workspace/*.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST_TMPDIR="${TMPDIR_HOST:-/tmp}"
HOST_LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR_HOST:-${LOCAL_DATASETS_DIR:-${SUBMIT_DIR}/shar}}"
HOST_OUTPUT_DIR="${OUTPUT_DIR:-${SUBMIT_DIR}/outputs}"
HOST_HF_HOME="${HF_HOME:-}"
# huggingface-cli login always writes to ~/.cache/huggingface/token regardless
# of HF_HOME (sites point HF_HOME at a shared cache dir, not $HOME), and $HOME
# itself is not bound into the container -- so without reading it here, on the
# host, the container sees neither the real $HOME nor a token. Read here and
# forward explicitly (see container_env HF_TOKEN below).
HOST_HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
[[ -z "${HOST_HF_TOKEN}" && -f "${HOME}/.cache/huggingface/token" ]] \
    && HOST_HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"

# --- open-file limit ------------------------------------------------------
# lhotse's indexed Shar reader caches one IndexedTarReader per shard it touches
# and never evicts it (lhotse/shar/readers/indexed.py, _ensure_indexed_readers:
# a plain dict, no LRU and no close), and each reader holds its tar handle open
# for the life of the process. Any scattered access pattern therefore parks one
# fd per distinct shard: a shuffled training stream, and -- far more abruptly --
# restoring a dynamic-bucketing checkpoint, which re-fetches every cut buffered
# in the sampler (buffer_size, commonly tens of thousands) by random access the
# moment iteration starts. Both blow through the default soft limit of 1024:
#
#   lhotse/indexing.py read_index -> OSError: [Errno 24] Too many open files
#
# See issue #76. Observed cold-start on artemis jobs 327837/327839, and on
# resume in job 328205, where it killed the job seconds after the sampler state
# was restored. Raising the soft limit is a ceiling, not a reservation --
# nothing is allocated when the readers are not open -- so it is safe to apply
# unconditionally. Set MELT_NOFILE to override; the hard limit is the cap.
MELT_NOFILE="${MELT_NOFILE:-65536}"
_nofile_hard="$(ulimit -Hn)"
if [[ "${_nofile_hard}" != "unlimited" && "${MELT_NOFILE}" -gt "${_nofile_hard}" ]]; then
    MELT_NOFILE="${_nofile_hard}"
fi
if ulimit -Sn "${MELT_NOFILE}" 2>/dev/null; then
    echo "[container] nofile:   soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
else
    echo "[container] WARNING: could not raise nofile to ${MELT_NOFILE} (soft=$(ulimit -Sn), hard=${_nofile_hard}); indexed-Shar runs may hit issue #76" >&2
fi

mkdir -p "${HOST_TMPDIR}" "${SUBMIT_DIR}/logs" "${HOST_OUTPUT_DIR}/wandb"
[[ -n "${HOST_HF_HOME}" ]] && mkdir -p "${HOST_HF_HOME}"

# Container env. Apptainer honors the SINGULARITYENV_ prefix for backward
# compatibility, so this one prefix covers both runtimes (no separate
# Apptainer-prefixed duplicates).
container_env() { export "SINGULARITYENV_${1}=${2}"; }

# Fixed container paths.
container_env VENV_PATH          /workspace/venv/bin/activate
container_env TMPDIR             /workspace/tmp
container_env LOCAL_DATASETS_DIR /workspace/shar
container_env OUTPUT_DIR         /workspace/outputs
container_env WANDB_DIR          /workspace/outputs/wandb

# $HOME may be read-only in the container, which breaks triton/lhotse/wandb/pip
# caches under ~/.triton, ~/.cache, etc. Point them at the writable tmp mount.
container_env TRITON_CACHE_DIR /workspace/tmp/.triton
container_env TRITON_HOME      /workspace/tmp/.triton
container_env XDG_CACHE_HOME   /workspace/tmp/.cache
# wandb artifact staging (config_artifact.add_file, etc.) uses WANDB_DATA_DIR
# (defaults to ~/.local/share/wandb/..., i.e. XDG_DATA_HOME), which is a
# separate setting from WANDB_DIR above -- redirect it too or it tries to
# mkdir under the unwritable $HOME.
container_env WANDB_DATA_DIR   /workspace/tmp/.local/share

# HF cache is shared into the container only when HF_HOME is set on the host.
if [[ -n "${HOST_HF_HOME}" ]]; then
    container_env HF_HOME     /workspace/hf_cache
    container_env HF_HUB_CACHE /workspace/hf_cache/hub
else
    echo "WARNING: HF_HOME not set on host; HF cache will not be shared into the container."
fi

# Gated/private HF repos: forward the token, never print it.
if [[ -n "${HOST_HF_TOKEN}" ]]; then
    container_env HF_TOKEN "${HOST_HF_TOKEN}"
    echo "[container] HF_TOKEN: forwarded (from \${HF_TOKEN:-\$HOME/.cache/huggingface/token})"
else
    echo "[container] HF_TOKEN: not set on host and no ~/.cache/huggingface/token found -- gated repos will fail to download"
fi

# NCCL heartbeat headroom for slow collectives.
container_env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 1800

# OpenMP threads. This srun runs ONE task per node and accelerate then forks one
# rank per GPU inside it, so SLURM_CPUS_PER_TASK is the whole node's allocation
# and must be divided by the ranks that will share it -- otherwise every rank
# claims all the cores and they oversubscribe by the GPU count.
_omp_cpus="${SLURM_CPUS_PER_TASK:-1}"
_omp_ranks="${SLURM_GPUS_ON_NODE:-1}"
[[ "${_omp_ranks}" -gt 0 ]] || _omp_ranks=1
_omp_default=$(( _omp_cpus / _omp_ranks ))
[[ "${_omp_default}" -gt 0 ]] || _omp_default=1

# SLURM exports OMP_NUM_THREADS=<cpus-per-task> itself, so a plain
# "${OMP_NUM_THREADS:-default}" never falls through to the default -- it always
# finds SLURM's node-wide value. A value equal to SLURM_CPUS_PER_TASK is
# therefore treated as "SLURM set this", not "a human chose this"; anything else
# is a deliberate choice and wins.
_omp_threads="${_omp_default}"
if [[ -n "${OMP_NUM_THREADS:-}" && "${OMP_NUM_THREADS}" != "${_omp_cpus}" ]]; then
    _omp_threads="${OMP_NUM_THREADS}"
    echo "[container] OMP_NUM_THREADS=${_omp_threads} (explicit)"
else
    echo "[container] OMP_NUM_THREADS=${_omp_threads} (${_omp_cpus} cpus / ${_omp_ranks} ranks)"
fi
container_env OMP_NUM_THREADS "${_omp_threads}"
container_env OMP_SCHEDULE    "${OMP_SCHEDULE:-STATIC}"

# OMP_PROC_BIND is deliberately NOT defaulted. With libgomp and no OMP_PLACES,
# `close` binds the OpenMP master thread -- i.e. the Python main thread -- to a
# SINGLE place, after which sched_getaffinity() returns 1 CPU and DataLoader
# workers forked from that thread inherit the one-core mask. The collapse
# happens at `import torch`, so it is invisible to any shell-level check.
# Set it explicitly on the host if you want binding, and set OMP_PLACES with it.
[[ -n "${OMP_PROC_BIND:-}" ]] && container_env OMP_PROC_BIND "${OMP_PROC_BIND}"

# Job-level SLURM vars (set at sbatch time). Per-task vars (PROCID/NODEID/
# LOCALID/STEP_ID) are set by srun and forwarded via --env in the exec below.
container_env SLURM_JOB_ID       "${SLURM_JOB_ID:-}"
container_env SLURM_NNODES       "${SLURM_NNODES:-1}"
container_env SLURM_GPUS_ON_NODE "${SLURM_GPUS_ON_NODE:-}"
container_env SLURM_JOB_NODELIST "${SLURM_JOB_NODELIST:-}"

# Multi-node: compute MASTER_ADDR on the host (scontrol is usually absent in the container).
# MASTER_PORT is derived from SLURM_JOB_ID (constant per job, unique across
# concurrent jobs) rather than fixed -- a fixed port made a second job landing
# on the same node fail its torch-elastic rendezvous and die before user code
# ran. This has to happen here (inside the job, forwarded into the container
# below) rather than in infra/runners/sites/*.sh, which is sourced on the
# LOGIN node at submit time before SLURM_JOB_ID exists. Same derivation and
# range (20000-29999, clear of privileged and Linux ephemeral ports) as
# bash/run_train.sh -- see the comment there for the full rationale.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 10000 ))}"
else
    MASTER_PORT="${MASTER_PORT:-6000}"
fi
if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    MASTER_ADDR_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
    MASTER_ADDR_HOST="${MASTER_ADDR:-127.0.0.1}"
fi
container_env MASTER_ADDR "${MASTER_ADDR_HOST}"
container_env MASTER_PORT "${MASTER_PORT}"

# Forward optional host-set vars only when present. WANDB_PROJECT/WANDB_MODE
# defaults are applied inside run_train.sh, so they are not duplicated here.
for var in HF_HUB_OFFLINE GPU_MEM_MONITORING MELT_EVAL_PROFILE \
           MELT_DEBUG_CUT_IDS_DIR MELT_DEBUG_CUT_IDS_MAX_BATCHES MELT_DEBUG_CUT_IDS_EVERY; do
    [[ -n "${!var:-}" ]] && container_env "$var" "${!var}"
done

# Experiment-tracker settings are forwarded by PREFIX rather than by name.
# Every tracker configures itself from its own env vars (WANDB_ENTITY,
# MLFLOW_TRACKING_URI, ...), so matching the namespace means switching tracker
# — or adding a setting to the current one — needs no change here and no
# tracker-specific code in train.py. Add a prefix to the list to support one.
#
# WANDB_DIR is set above to a fixed container path and must not be overridden
# from the host, where it would point outside the container.
for prefix in WANDB_ MLFLOW_ TRACKIO_ COMET_ NEPTUNE_ TENSORBOARD_; do
    while IFS='=' read -r var _; do
        [[ "$var" == "WANDB_DIR" ]] && continue
        [[ -n "${!var:-}" ]] && container_env "$var" "${!var}"
    done < <(env | grep "^${prefix}" || true)
done

# Bind mounts (host:container[:mode]).
BINDS=(
    "${SUBMIT_DIR}:/workspace/training"
    "${HOST_TMPDIR}:/workspace/tmp"
    "${HOST_LOCAL_DATASETS_DIR}:/workspace/shar"
    "${HOST_OUTPUT_DIR}:/workspace/outputs:rw"
)
[[ -n "${HOST_HF_HOME}" ]] && BINDS+=("${HOST_HF_HOME}:/workspace/hf_cache")
BIND_ARGS=$(IFS=,; echo "${BINDS[*]}")

echo "[container] runtime:  ${SINGULARITY_BIN}"
echo "[container] image:    ${SINGULARITY_IMG}"
echo "[container] project:  ${SUBMIT_DIR} -> /workspace/training"
echo "[container] datasets: ${HOST_LOCAL_DATASETS_DIR} -> /workspace/shar"
echo "[container] tmp:      ${HOST_TMPDIR} -> /workspace/tmp"
echo "[container] outputs:  ${HOST_OUTPUT_DIR} -> /workspace/outputs (wandb under /workspace/outputs/wandb)"
[[ -n "${HOST_HF_HOME}" ]] && echo "[container] hf_cache: ${HOST_HF_HOME} -> /workspace/hf_cache"
echo "[container] master:   ${MASTER_ADDR_HOST}:${MASTER_PORT}"
echo "[container] nnodes:   ${SLURM_NNODES:-1}, gpus/node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "[container] launching: bash/run_train.sh ${ACCELERATE_CONFIG} ${TRAIN_ARGS[*]:-}"

# Run inside srun (one task per node) -> singularity exec -> run_train.sh.
# SLURM_PROCID/NODEID/LOCALID/STEP_ID are set by srun per task and stripped by
# --cleanenv, so they are forwarded with --env inside the step. Train args are
# passed as positional parameters, which is quoting-safe (args with spaces survive).
export SINGULARITY_BIN SINGULARITY_IMG BIND_ARGS

srun --kill-on-bad-exit=1 --wait=60 bash -c '
    exec "$SINGULARITY_BIN" exec --nv \
        --bind "$BIND_ARGS" \
        --pwd /workspace/training \
        --cleanenv \
        --env SLURM_PROCID="${SLURM_PROCID:-0}" \
        --env SLURM_NODEID="${SLURM_NODEID:-0}" \
        --env SLURM_LOCALID="${SLURM_LOCALID:-0}" \
        --env SLURM_STEP_ID="${SLURM_STEP_ID:-}" \
        "$SINGULARITY_IMG" \
        bash /workspace/training/bash/run_train.sh "$@"
' _ "$ACCELERATE_CONFIG" "${TRAIN_ARGS[@]}"
