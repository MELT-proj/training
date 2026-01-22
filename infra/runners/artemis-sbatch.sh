#!/bin/bash

# Artemis cluster - Batch job with Singularity container
# This script sets Artemis-specific paths and submits a training job

# Export environment variables for the batch job
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache
export HF_DATASETS_CACHE=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache/datasets
export HF_METRICS_CACHE=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache/metrics
export OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs
export LOCAL_DATASETS_DIR=/mnt/scratch-artemis/giuseppe/melt-data/shar
export TMPDIR_HOST=/mnt/scratch-artemis/giuseppe/melt-data/tmp

# Export venv if we are not using a singularity container
export VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/melt/bin/activate

# Submit batch job with Artemis-specific SLURM settings
sbatch --time=01:00:00 --nodes=1 --gres=gpu:2 --qos=gpu-debug --partition a6000 \
  --ntasks-per-node=1 --cpus-per-task=32 \
  --output="logs/%x.%j.out" \
  --error="logs/%x.%j.err" \
  --job-name=melt-train \
  ./bash/run_train.sh config/accelerate/zero3.yaml \
  --config-file config/train/LS_asr.yaml