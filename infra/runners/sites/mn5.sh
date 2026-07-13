# BSC MareNostrum5 (acc partition). Air-gapped compute nodes: run offline.

# --- storage (host paths) -------------------------------------------------
export HF_HOME=/gpfs/projects/epor48/melt-data/hf_cache
export OUTPUT_DIR=/gpfs/projects/epor48/melt-data/outputs
export LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar
export TMPDIR_HOST=/gpfs/projects/epor48/melt-data/tmp

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=/gpfs/projects/epor48/melt-data/melt_cuda126.sif
export SINGULARITY_BIN=singularity

# --- misc -----------------------------------------------------------------
# Compute nodes have no internet: pre-download models (infra/setup/download_hf_models.sh)
# and run fully offline.
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export MASTER_PORT=60001

# --- scheduler ------------------------------------------------------------
SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=4 --account=epor48 --qos=acc_ehpc --cpus-per-task=80)
