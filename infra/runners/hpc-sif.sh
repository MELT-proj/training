#!/bin/bash

# MN5 cluster - Batch job with Singularity container
# This script sets MN5-specific paths and submits a training job
# The script assumes we are inside a Singularity container with MELT installed

# Export environment variables for the batch job
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/hf_cache
export OUTPUT_DIR=/workspace/outputs
export LOCAL_DATASETS_DIR=/workspace/shar
export TMPDIR_HOST=/workspace/tmp
export WANDB_MODE=offline

# Export venv if we are not using a singularity container
export VENV_PATH=/workspace/venv

./bash/run_train.sh config/accelerate/zero3.yaml --config config/train/LS_asr.yaml