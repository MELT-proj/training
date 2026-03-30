#!/bin/bash

APPTAINER_TMPDIR=/mnt/scratch-artemis/giuseppe/tmp \
  TMPDIR=/mnt/scratch-artemis/giuseppe/tmp \
  apptainer build \
  --bind /mnt/scratch-artemis/giuseppe/.cache/uv:/usr/local/share/uv/cache \
  /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif infra/Singularity.def