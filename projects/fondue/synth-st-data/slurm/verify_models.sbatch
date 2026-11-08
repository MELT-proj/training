#!/bin/bash
#SBATCH --job-name=synth-verify
#SBATCH --account=epor48
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
# Offline model check. Submit from a frozen clone on MN5: cd to the synth_st dir, then: sbatch slurm/verify_models.sbatch
set -euo pipefail
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
SIF="${SYNTH_SIF:-/gpfs/scratch/epor48/synth.sif}"
export SINGULARITYENV_HF_HOME=/gpfs/scratch/epor48/hf_cache
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1
singularity exec --nv --bind /gpfs "$SIF" python -I "${SLURM_SUBMIT_DIR}/verify_models.py"
