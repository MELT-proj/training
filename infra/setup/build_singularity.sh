#!/bin/bash

APPTAINER_TMPDIR=$SCRATCH/tmp TMPDIR=$SCRATCH/tmp \
  apptainer build \
  --bind /mnt/scratch-artemis/giuseppe/.cache/uv:/usr/local/share/uv/cache \
  /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif infra/Singularity.def