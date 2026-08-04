#!/bin/bash
#SBATCH --job-name=melt-qe-canary
#SBATCH --time=0:30:00
#SBATCH --qos=gpu-h100
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

source /etc/profile.d/02-lmod.sh
module load cuda

mkdir -p logs

VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/py313/bin/activate
source "$VENV_PATH"

export TMPDIR=/tmp
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache
export OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs
export LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/iwslt-2026/shar

export WANDB_PROJECT="${WANDB_PROJECT:-iwslt26-metric}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=info

echo "Running OOM canary on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvcc --version 2>/dev/null || echo "nvcc not found"

python -m projects.iwslt26-metric.canary \
    --config projects/iwslt26-metric/config.yaml \
    --num_cuts 10
