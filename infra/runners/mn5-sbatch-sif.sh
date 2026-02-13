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
export MASTER_PORT=60001

ACCELERATE_CONFIG="config/accelerate/fsdp2.yaml"
EXP_NAME=MA-iwslt25-q3-1.7B-1node
TRAIN_CONFIG=./config/train/MA-iwslt25.yaml

# Submit batch job with Artemis-specific SLURM settings
sbatch --time=09:00:00 \
    --nodes=1 --gpus-per-node=4 \
    -A epor48 --qos=acc_ehpc -c 80 \
    ./bash/run_train_singularity.sbatch $ACCELERATE_CONFIG \
    --config ${TRAIN_CONFIG} \
    --trainer.eval_steps 6000 \
    --trainer.save_steps 6000 \
    --trainer.logging_steps 3 \
    --trainer.warmup_steps 50 \
    --optimization.adapter_lr 0.0002 \
    --trainer.num_train_epochs 2 \
    --trainer.output_dir ${OUTPUT_DIR}/${EXP_NAME} \
    --run.exp_name ${EXP_NAME} \
    --model.decoder.name "Qwen/Qwen3-1.7B" \
    --model.decoder.attn_implementation "sdpa"
