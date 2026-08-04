#!/bin/bash

# HPC cluster - Batch job with Singularity container
# This script sets HPC-specific paths and submits a training job

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
EXP_NAME=SFT-v1.2.7-newsif-v2-2nodesv3
TRAIN_CONFIG=./config/train/SFT-v1.2.7.yaml

sbatch --time=01:00:00 \
    --nodes=1 --gpus-per-node=4 \
    -A epor48 --qos=acc_ehpc -c 80 \
    ./bash/run_train_singularity.sbatch $ACCELERATE_CONFIG \
    --config ${TRAIN_CONFIG} \
    --trainer.max_steps 1000 \
    --trainer.eval_steps 200 \
    --trainer.save_steps 200 \
    --trainer.warmup_steps 20 \
    --trainer.report_to "wandb" \
    --trainer.output_dir ${OUTPUT_DIR}/${EXP_NAME} \
    --run.exp_name ${EXP_NAME} \
    --data.train_ds.buffer_size 800000 \
    --trainer.eval_on_start false

    # --trainer.resume_from_checkpoint true \
    # --model.decoder.attn_implementation sdpa
    # --model.ckpt /gpfs/projects/epor48/melt-data/outputs/MA-v1.2.7 \
    # --trainer.num_train_epochs 10 \
