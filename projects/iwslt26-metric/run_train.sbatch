#!/bin/bash
#SBATCH --job-name=melt-qe-train
#SBATCH --time=5:00:00
#SBATCH --qos=gpu-h100
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

source /etc/profile.d/02-lmod.sh
module load cuda

mkdir -p logs

VENV_PATH=/mnt/scratch-artemis/giuseppe/venvs/py313/bin/activate \
    TMPDIR=/tmp \
    HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache \
    OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs \
    LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/iwslt-2026/shar \
    bash ./projects/iwslt26-metric/run_train.sh \
    ./projects/iwslt26-metric/fsdp2.yaml \
    --config projects/iwslt26-metric/config.yaml \
    --trainer.gradient-accumulation-steps 4 \
    --trainer.output_dir /mnt/scratch-artemis/giuseppe/melt-data/outputs/MELT_QE_v1.1.6
