# Artemis cluster (a6000 / h100 partitions).

# --- storage (host paths) -------------------------------------------------
# `:-` so an exported value wins: `OUTPUT_DIR=/mnt/... infra/runners/submit-*.sh artemis …`.
export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/giuseppe/melt-data/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/scratch-artemis/giuseppe/melt-data/outputs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"
export TMPDIR_HOST="${TMPDIR_HOST:-/tmp}"

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=${SINGULARITY_IMG:-/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif}
export SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}

# --- native mode ----------------------------------------------------------
export VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/melt-312/bin/activate

# --- misc -----------------------------------------------------------------
export WANDB_MODE=online
export MASTER_PORT=60001
# run_train.sh forwards this as an explicit --data.{train,validation}_ds.shard_seed
# override, so no train YAML needs shard_seed: randomized (invalid for indexed
# Shar sources -- see melt/training/data/audio/lhotse/dataloader.py).
export MELT_SEED="${MELT_SEED:-42}"

# --- scheduler ------------------------------------------------------------
# MELT_NODES/MELT_QOS/MELT_TIME/MELT_PARTITION/MELT_GPUS_PER_NODE are overridable
# the same way as on mn5:
#   MELT_QOS=gpu-debug MELT_PARTITION=a6000 infra/runners/submit-container.sh artemis …
SBATCH_ARGS=(--time="${MELT_TIME:-01:00:00}" --nodes="${MELT_NODES:-1}" --gpus-per-node="${MELT_GPUS_PER_NODE:-2}" --qos="${MELT_QOS:-gpu-h100}" --partition="${MELT_PARTITION:-h100}")
# a6000 debug alternative:
# SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=2 --qos=gpu-debug --partition=a6000)
