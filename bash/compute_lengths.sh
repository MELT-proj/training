#!/bin/bash
#SBATCH --job-name=stats_dataset
#SBATCH --output=./logs/job.%A.out
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=IscrC_ItaLLM_0
#SBATCH --mem=30000M

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /leonardo_scratch/fast/IscrC_ItaLLM_0/venvs/melt/bin/activate
python src/compute_lengths.py --config-file $1 --output_dir $WORK/speech_lm/audio_lengths