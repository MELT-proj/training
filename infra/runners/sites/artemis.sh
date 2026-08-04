# Artemis cluster (a6000 / h100 partitions).

# --- storage (host paths) -------------------------------------------------
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache
export OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs
export LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar
export TMPDIR_HOST=/tmp

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif
export SINGULARITY_BIN=singularity

# --- native mode ----------------------------------------------------------
export VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/melt/bin/activate

# --- misc -----------------------------------------------------------------
export WANDB_MODE=online
export MASTER_PORT=60001

# --- scheduler ------------------------------------------------------------
SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=2 --qos=gpu-h100 --partition=h100)
# a6000 debug alternative:
# SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=2 --qos=gpu-debug --partition=a6000)
