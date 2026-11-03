#!/bin/bash
# Artemis (A6000 etc.) variant of run_units.sbatch: one process per GPU, units from UNITS (names), nyx scratch for everything.
#   sbatch --partition=a6000 --qos=gpu-medium --gres=gpu:4 --cpus-per-task=32 --mem=200G --time=2-00:00:00 \
#          --export=ALL,UNITS="u1 u2 ...",OUT=...,QE_DTYPE=bf16 code/slurm/run_units_artemis.sbatch
# A6000 has 48 GB: XCOMET-XXL does not fit in fp32 there, so QE runs in bf16 (QE_DTYPE).
set -euo pipefail
: "${UNITS:?}" "${OUT:?}"
CODE="${CODE:-/mnt/scratch-nyx/giuseppe/synth/code}"
SIF="${SYNTH_SIF:-/mnt/scratch-nyx/giuseppe/synth/synth.sif}"
UNIT_DIR="${UNIT_DIR:-/mnt/scratch-nyx/giuseppe/synth/units}"
NGPU="${NGPU:-4}"
QE_DTYPE="${QE_DTYPE:-bf16}"
GPU_MEM="${GPU_MEM:-0.88}"
# SingularityCE reads SINGULARITYENV_*, apptainer APPTAINERENV_*: set both
for P in SINGULARITYENV APPTAINERENV; do
  export ${P}_HF_HOME=/mnt/scratch-nyx/giuseppe/synth/hf_cache ${P}_HF_HUB_OFFLINE=1
  export ${P}_TRANSFORMERS_OFFLINE=1 ${P}_VLLM_NO_USAGE_STATS=1
done
mkdir -p "$OUT"
read -r -a ULIST <<< "$UNITS"
for ((g=0; g<NGPU; g++)); do
  (
    for ((i=g; i<${#ULIST[@]}; i+=NGPU)); do
      name="${ULIST[$i]}"
      for stage in mt qe; do
        CUDA_VISIBLE_DEVICES=$g SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g APPTAINERENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g \
        singularity exec --nv --bind /mnt/scratch-nyx "$SIF" \
          python "$CODE/run_mt_qe.py" --pool "$UNIT_DIR/$name.pool.jsonl" --out-dir "$OUT" --unit "$name" \
          --stage $stage --ledger "$OUT/ledger.tsv" --gpu-mem $GPU_MEM --qe-dtype $QE_DTYPE ${EXTRA:-}
      done
    done
  ) &
done
wait
