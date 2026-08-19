# BSC MareNostrum5 (acc partition). Air-gapped compute nodes: run offline.

# --- storage (host paths) -------------------------------------------------
# `:-` so an exported value wins: `OUTPUT_DIR=/gpfs/... infra/runners/submit-*.sh mn5 …`.
# The project is shared by several accounts, and a directory here belongs to
# whoever created it — the defaults below are readable by the group but writable
# only by their owner. Set your own OUTPUT_DIR and TMPDIR_HOST (both get written
# to); HF_HOME and LOCAL_DATASETS_DIR are read-only in a run and can stay shared.
# 2026-08-07: hf_cache, outputs and the .sif images moved off gpfs_projects to
# gpfs_scratch, which was filling up (projects was at 35.7 TB of a 48.8 TB group
# quota). `shar` and `tmp` stayed behind, so only some of these paths changed --
# check where a thing actually is before assuming this file is right.
export HF_HOME="${HF_HOME:-/gpfs/scratch/epor48/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar}"
export TMPDIR_HOST="${TMPDIR_HOST:-/gpfs/projects/epor48/melt-data/tmp}"

# --- container mode -------------------------------------------------------
# Overridable like the storage paths above, so a trial image can be pointed at
# without editing this file: SINGULARITY_IMG=/path/to/other.sif submit-container.sh …
# Moved to gpfs_scratch on 2026-08-07 along with hf_cache and outputs.
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126.sif}"
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
# MASTER_PORT is deliberately NOT set here. This file is sourced on the LOGIN
# node at submit time, before the job (and its SLURM_JOB_ID) exists, so a
# fixed value here would apply to every job the same way and two jobs landing
# on the same node would collide on the rendezvous port -- the second one's
# torch-elastic agent fails to bind and the job dies before user code runs.
# bash/run_train.sh (and the container shim) derive a per-job default from
# SLURM_JOB_ID instead, at the point where it's actually set. See there for
# the derivation; set MASTER_PORT here (or in the environment) only to force
# a specific value.
# run_train.sh forwards this as an explicit --data.{train,validation}_ds.shard_seed
# override, so no train YAML needs shard_seed: randomized (invalid for indexed
# Shar sources -- see melt/training/data/audio/lhotse/dataloader.py).
export MELT_SEED="${MELT_SEED:-42}"

# --- scheduler ------------------------------------------------------------
# MELT_TIME and MELT_QOS are overridable mainly to pick the right QoS: acc_debug
# caps at 2h but runs at priority 10000 against acc_ehpc's 100, so short jobs
# schedule immediately. Accounting follows ELAPSED time (sacct CPUTimeRAW ==
# Elapsed x AllocCPUS), so a generous wall is not itself a cost -- but the node
# is allocated whole, all 80 CPUs and 4 GPUs, however few you use:
#   MELT_TIME=00:20:00 MELT_QOS=acc_debug infra/runners/submit-container.sh mn5 …
# MELT_NODES scales out; bash/run_train.sh derives world_size from SLURM, so no
# other change is needed to go multi-node. No --partition is passed by default
# (account+QoS already select the right one on MN5); set MELT_PARTITION only
# if you need to override it explicitly.
SBATCH_ARGS=(
    --time="${MELT_TIME:-01:00:00}"
    --nodes="${MELT_NODES:-1}"
    --gpus-per-node="${MELT_GPUS_PER_NODE:-4}"
    --account=epor48
    --qos="${MELT_QOS:-acc_ehpc}"
    --cpus-per-task=80
)
[[ -n "${MELT_PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="${MELT_PARTITION}")
