#!/bin/bash

export HF_HOME="/mnt/home/giuseppe/myscratch/hf_home_speech"
export LOCAL_DATASET_DIR="/mnt/home/giuseppe/myscratch/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "This script downloads and saves the datasets locally"
echo "It does so using load_dataset and save_to_disk"
echo "Which means that the datasets are potentially duplicated on your disk"

set -e

# if [ ! -d "$LOCAL_DATASET_DIR/fleurs" ]; then
#     echo "Downloading fleurs dataset..."
#     python scripts/save_local_dataset.py --dataset fleurs \
#         --output_dir "$LOCAL_DATASET_DIR/fleurs"
# else
#     echo "Dataset fleurs already exists in $LOCAL_DATASET_DIR/fleurs. Skipping download."
# fi

for dataset in "cv17"; do
    # if [ ! -d "$LOCAL_DATASET_DIR/$dataset" ]; then
    echo "Downloading $dataset dataset..."
    python scripts/save_local_dataset.py --dataset $dataset \
        --output_dir "$LOCAL_DATASET_DIR/$dataset"
    # else
        # echo "Dataset $dataset already exists in $LOCAL_DATASET_DIR/${dataset}. Skipping download."
    # fi
done