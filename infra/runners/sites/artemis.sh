# Artemis cluster (a6000 / h100 partitions).

# --- storage (host paths) -------------------------------------------------
# `:-` so an exported value wins: `OUTPUT_DIR=/mnt/... infra/runners/submit-*.sh artemis …`.
export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/giuseppe/.cache/huggingface}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/scratch-artemis/giuseppe/melt-data/outputs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"
export TMPDIR_HOST="${TMPDIR_HOST:-/tmp}"

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=${SINGULARITY_IMG:-/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif}
export SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}

# --- repo sync (infra/sync_repo.sh) ---------------------------------------
# artemis does not share a filesystem with nyx, where development happens, so the
# cluster keeps its own checkout that has to be brought up to date before a submit.
export REMOTE_SSH="${REMOTE_SSH:-artemis}"
export REMOTE_REPO="${REMOTE_REPO:-melt-proj/training}"

# --- native mode ----------------------------------------------------------
# melt-312 pins torchdata 0.10.0 (repo requires >=0.11, see pyproject.toml);
# lhotse2 has the matching stack (torch 2.9.1+cu126, torchdata 0.11.0,
# lhotse 2.0.0a3, transformers 4.57.1) and its `melt` editable install has
# been repointed at this checkout (it previously resolved to a stale,
# disconnected copy at melt-data-adjacent lhotse2-repo).
export VENV_PATH="${VENV_PATH:-/mnt/scratch-artemis/giuseppe/venvs/lhotse2/bin/activate}"

# --- misc -----------------------------------------------------------------
export WANDB_MODE=online
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
# MELT_NODES/MELT_QOS/MELT_TIME/MELT_PARTITION/MELT_GPUS_PER_NODE are overridable
# the same way as on mn5. a6000 nodes cannot run containers (Singularity fails
# with "starter-suid doesn't have setuid bit set" -- confirmed via job 328945),
# so submit-native.sh (not submit-container.sh) is the only option there:
#   MELT_QOS=gpu-debug MELT_PARTITION=a6000 infra/runners/submit-native.sh artemis …
SBATCH_ARGS=(--time="${MELT_TIME:-01:00:00}" --nodes="${MELT_NODES:-1}" --gpus-per-node="${MELT_GPUS_PER_NODE:-2}" --qos="${MELT_QOS:-gpu-h100}" --partition="${MELT_PARTITION:-h100}")
# a6000 debug alternative (native mode only -- see above):
# SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=2 --qos=gpu-debug --partition=a6000)
