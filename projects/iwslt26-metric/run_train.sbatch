#!/bin/bash
#SBATCH --job-name=melt-qe-train
#SBATCH --time=04:00:00
#SBATCH --qos=gpu-short
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

mkdir -p logs

VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/melt/bin/activate \
TMPDIR=/tmp \
HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache \
OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs \
LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/iwslt-2026/shar \
bash ./projects/iwslt26-metric/run_train.sh \
    config/accelerate/fsdp2.yaml \
    --config projects/iwslt26-metric/config.yaml \
    --trainer.gradient-accumulation-steps 4 \
    --trainer.output_dir /mnt/scratch-artemis/giuseppe/melt-data/outputs/MELT_QE_v1.1
