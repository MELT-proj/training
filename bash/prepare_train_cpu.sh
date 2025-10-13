#!/bin/bash
#SBATCH --job-name=stats_dataset
#SBATCH --output=./logs/job.%A.out
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=IscrC_ItaLLM_0
#SBATCH --mem=128G

source $FAST/venvs/melt/bin/activate
echo "Python is at:"
echo `whereis python`

# source /etc/profile.d/02-lmod.sh
# module load cuda

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

export WANDB_PROJECT=speech_lm
export WANDB_MODE=offline
# export HF_HOME="${SCRATCH}/hf_home_speech"
export LOCAL_DATASETS_DIR="${SCRATCH}/speech_lm/datasets"
# export LENGHTS_DIR="${WORK}/speech_lm/audio_lengths"
export TORCHDYNAMO_VERBOSE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# export HF_DATASETS_IN_MEMORY_MAX_SIZE=80000000000
# export CUDA_VISIBLE_DEVICES=0

python src/train.py --config-file $1