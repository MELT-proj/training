#!/bin/bash

echo "Syncing local Hugging Face models to HPC cluster..."
rsync -ravhz -P /mnt/scratch-artemis/giuseppe/local_hf mn5:/gpfs/projects/epor48/