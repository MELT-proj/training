#!/bin/bash
#SBATCH --job-name=synth-py
#SBATCH --account=epor48
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --output=logs/%x.%j.out
# One python script of synth_st on GPU 0 of a node: SCRIPT="gen_tts_text.py" ARGS="--out ... --n-prompts 400"
# For work that is one process (text generation). Submit from a frozen clone.
set -euo pipefail
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
: "${SCRIPT:?}"
SIF="${SYNTH_SIF:-/gpfs/scratch/epor48/synth.sif}"
export MELT_GPUS_PER_NODE=4
export SINGULARITYENV_HF_HOME=/gpfs/scratch/epor48/hf_cache SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1
export SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID
CUDA_VISIBLE_DEVICES=0 singularity exec --nv --bind /gpfs "$SIF" python "${SLURM_SUBMIT_DIR}/${SCRIPT}" ${ARGS:-}
