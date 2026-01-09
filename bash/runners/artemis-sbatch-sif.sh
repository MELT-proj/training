#!/bin/bash

# Artemis cluster - Batch job with Singularity container
# This script sets Artemis-specific paths and submits a training job

# Export environment variables for the batch job
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache
export OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs
export LOCAL_DATASETS_DIR=/mnt/scratch-artemis/giuseppe/melt-data/shar
export SINGULARITY_BIN="singularity"
export SINGULARITY_IMG=/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif
export TMPDIR_HOST=/mnt/scratch-artemis/giuseppe/melt-data/tmp

# Submit batch job with Artemis-specific SLURM settings
sbatch --time=01:00:00 --nodes=1 --gres=gpu:2 --qos=gpu-debug --partition a6000 \
  ./bash/run_train_singularity.sbatch config/accelerate/zero3.yaml \
  --config-file config/train/LS_asr.yaml