#!/bin/bash
#SBATCH --job-name=synth-units
#SBATCH --account=epor48
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --output=logs/%x.%j.out
# One node, 4 GPUs, one process per GPU; GPU g takes UNITS[g], UNITS[g+4], ...
# UNITS: space-separated unit names; the pool of a unit is $UNIT_DIR/<name>.pool.jsonl.
# Chunks whose output exists are skipped, so a resubmitted job only does what is missing.
# Use submit_units.sh rather than calling this directly. Jobs read the submit dir at
# START: submit from a frozen clone and do not touch it while jobs are queued or running.
set -euo pipefail
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
: "${UNITS:?}" "${UNIT_DIR:?}" "${OUT:?}"
SIF="${SYNTH_SIF:-/gpfs/scratch/epor48/synth.sif}"
LEDGER="${LEDGER:-$OUT/ledger.tsv}"
EXTRA="${EXTRA:-}"          # extra run_mt_qe.py args
export MELT_GPUS_PER_NODE=4  # pinned: SLURM_GPUS_ON_NODE lies on acc
export SINGULARITYENV_HF_HOME=/gpfs/scratch/epor48/hf_cache SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1
SD="${SLURM_SUBMIT_DIR}"
mkdir -p "$OUT"
read -r -a ULIST <<< "$UNITS"
for g in 0 1 2 3; do
  (
    for ((i=g; i<${#ULIST[@]}; i+=4)); do
      name="${ULIST[$i]}"
      for stage in mt qe; do
        # one vLLM/inductor compile cache per GPU: four processes sharing one corrupt it
        CUDA_VISIBLE_DEVICES=$g SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g \
        singularity exec --nv --bind /gpfs "$SIF" \
          python "$SD/run_mt_qe.py" --pool "$UNIT_DIR/$name.pool.jsonl" --out-dir "$OUT" --unit "$name" \
          --stage $stage --ledger "$LEDGER" $EXTRA
      done
    done
  ) &
done
wait
