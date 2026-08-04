#!/bin/bash

# Artemis cluster - Batch job with Singularity container
# This script sets Artemis-specific paths and submits a training job

# Export environment variables for the batch job
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache
# export OUTPUT_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/outputs
export OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/outputs
export LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar
export SINGULARITY_BIN="singularity"
export SINGULARITY_IMG=/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif
export TMPDIR_HOST=/tmp
export MASTER_PORT=60001

ACCELERATE_CONFIG="config/accelerate/fsdp2.yaml"
EXP_NAME=SFT-v1.2.7.5
TRAIN_CONFIG=./config/train/SFT-v1.2.7.yaml
# OUTPUT_DIR is the host-side path used for bind-mounting by the sbatch script.
# Inside the container it is always mounted at /workspace/outputs, so train args
# must use the container path rather than the host path.
CONTAINER_OUTPUT_DIR="/workspace/outputs"

sbatch --time=01:00:00 \
    --nodes=1 --gpus-per-node=2 \
    --qos=gpu-h100 --partition h100 \
    ./bash/run_train_singularity.sbatch $ACCELERATE_CONFIG \
    --config ${TRAIN_CONFIG} \
    --trainer.max_steps 1000 \
    --trainer.eval_steps 200 \
    --trainer.save_steps 200 \
    --trainer.warmup_steps 20 \
    --trainer.report_to "wandb" \
    --trainer.output_dir ${CONTAINER_OUTPUT_DIR}/${EXP_NAME} \
    --run.exp_name ${EXP_NAME} \
    --data.train_ds.buffer_size 500000 \
    --trainer.eval_on_start false \
    --run.memory_profiling true \
    --run.memory_preallocation true

    # --trainer.resume_from_checkpoint true \
    # --model.decoder.attn_implementation sdpa
    # --model.ckpt /gpfs/projects/epor48/melt-data/outputs/MA-v1.2.7 \
    # --trainer.num_train_epochs 10 \