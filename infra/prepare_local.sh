#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=./logs/job.%A.out
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=IscrC_ItaLLM_0
#SBATCH --mem=30000M

set -e
# export HF_HOME="/mnt/home/giuseppe/myscratch/hf_home_speech"
source /leonardo_scratch/fast/IscrC_ItaLLM_0/venvs/melt/bin/activate
echo "This script has to be run only once with internet access"

# Download datasets
# for dataset_name in voxpopuli fleurs; do
#     echo "Downloading $dataset_name"
#     python src/download_datasets.py --dataset_name $dataset_name --dataset_workers 4
# done

# Download models, tokenizers, and config files
for decoder in "utter-project/EuroLLM-1.7B" "meta-llama/Llama-3.2-1B" "Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B"; do
    python src/prepare_local.py \
        --audio_encoder "facebook/w2v-bert-2.0" \
        --text_decoder $decoder
done