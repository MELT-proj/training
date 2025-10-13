#!/bin/bash

source_dir="$HOME/myscratch/datasets/*"
target_dir="/leonardo_scratch/large/userexternal/gattanas/speech_lm/datasets"
key="/mnt/home/giuseppe/.ssh/cineca.key"
user="gattanas"

rsync -Pravz --exclude 'tmp*' \
    --exclude 'cache*' \
    --exclude '*.lock' \
    --exclude '*__pycache__*' --exclude 'models*' \
    -e ssh \
    ${source_dir} leonardo:${target_dir} \
    --delete
    # --exclude 'models--*' \ 
