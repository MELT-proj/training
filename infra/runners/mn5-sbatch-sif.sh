#!/bin/bash

# MN5 cluster - Batch job with Singularity container
# This script sets MN5-specific paths and submits a training job

# Export environment variables for the batch job
export HF_HUB_OFFLINE=1
export HF_HOME=/gpfs/projects/epor48/melt-data/hf_cache
export OUTPUT_DIR=/gpfs/projects/epor48/melt-data/outputs
export LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar
export SINGULARITY_BIN="singularity"
export SINGULARITY_IMG=/gpfs/projects/epor48/melt-data/melt_cuda126.sif
export TMPDIR_HOST=/gpfs/projects/epor48/melt-data/tmp
export WANDB_MODE=offline

# ACCELERATE_CONFIG="config/accelerate/fsdp.yaml"
# ACCELERATE_CONFIG="config/accelerate/zero1.yaml"
ACCELERATE_CONFIG="config/accelerate/zero3.yaml"


# Submit batch job with Artemis-specific SLURM settings
sbatch --time=00:30:00 --nodes=1 --gres=gpu:2 \
    -A epor48 --qos=acc_debug -c 40 \
    ./bash/run_train_singularity.sbatch $ACCELERATE_CONFIG \
    --config-file config/train/asr.yaml
