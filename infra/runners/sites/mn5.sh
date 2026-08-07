# BSC MareNostrum5 (acc partition). Air-gapped compute nodes: run offline.

# --- storage (host paths) -------------------------------------------------
# `:-` so an exported value wins: `OUTPUT_DIR=/gpfs/... infra/runners/submit-*.sh mn5 …`.
# The project is shared by several accounts, and a directory here belongs to
# whoever created it — the defaults below are readable by the group but writable
# only by their owner. Set your own OUTPUT_DIR and TMPDIR_HOST (both get written
# to); HF_HOME and LOCAL_DATASETS_DIR are read-only in a run and can stay shared.
export HF_HOME="${HF_HOME:-/gpfs/projects/epor48/melt-data/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/projects/epor48/melt-data/outputs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar}"
export TMPDIR_HOST="${TMPDIR_HOST:-/gpfs/projects/epor48/melt-data/tmp}"

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=/gpfs/projects/epor48/melt-data/melt_cuda126.sif
export SINGULARITY_BIN=singularity

# --- code sync (infra/sync_repo.sh) ---------------------------------------
# MN5 has no outbound internet, so it cannot `git pull`. Push to it instead:
# git over SSH needs no internet on the far end. `mn5` is an ssh alias each
# person defines in their own ~/.ssh/config, so no username is baked in.
# REMOTE_REPO is relative to the remote $HOME, so it holds for any account.
export REMOTE_SSH=mn5
export REMOTE_REPO=training

# --- misc -----------------------------------------------------------------
# Compute nodes have no internet: pre-download models (infra/setup/download_hf_models.sh)
# and run fully offline.
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export MASTER_PORT=60001

# --- scheduler ------------------------------------------------------------
SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=4 --account=epor48 --qos=acc_ehpc --cpus-per-task=80)
