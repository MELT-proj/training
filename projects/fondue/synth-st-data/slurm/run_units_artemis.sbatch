#!/bin/bash
# Artemis (A6000 etc.) variant of run_units.sbatch: one process per GPU, units from UNITS (names), nyx scratch for everything.
#   sbatch --partition=a6000 --qos=gpu-medium --gres=gpu:4 --cpus-per-task=32 --mem=200G --time=2-00:00:00 \
#          --export=ALL,UNITS="u1 u2 ...",OUT=...,STAGES=mt code/slurm/run_units_artemis.sbatch
# XCOMET-XXL is run in fp32 only (43 GB: H100/H200, not A6000), so the stages are submitted separately:
#   MT on A6000:  --export=ALL,STAGES=mt ...      QE on H100/H200:  --export=ALL,STAGES=qe,QE_DTYPE=fp32 ...
# Both write into the same OUT/<unit> chunks, so the QE job can be chained with --dependency=afterok.
set -euo pipefail
: "${UNITS:?}" "${OUT:?}"
CODE="${CODE:-/mnt/scratch-nyx/giuseppe/synth/code}"
SIF="${SYNTH_SIF:-/mnt/scratch-nyx/giuseppe/synth/synth.sif}"
UNIT_DIR="${UNIT_DIR:-/mnt/scratch-nyx/giuseppe/synth/units}"
NGPU="${NGPU:-4}"
QE_DTYPE="${QE_DTYPE:-fp32}"
STAGES="${STAGES:-mt qe}"
GPU_MEM="${GPU_MEM:-0.88}"
# SingularityCE reads SINGULARITYENV_*, apptainer APPTAINERENV_*: set both
for P in SINGULARITYENV APPTAINERENV; do
  export ${P}_HF_HOME=/mnt/scratch-nyx/giuseppe/synth/hf_cache ${P}_HF_HUB_OFFLINE=1
  export ${P}_TRANSFORMERS_OFFLINE=1 ${P}_VLLM_NO_USAGE_STATS=1
  export ${P}_VLLM_ATTENTION_BACKEND=FLASH_ATTN ${P}_VLLM_FLASH_ATTN_VERSION=2  # vLLM's FlashAttention-2 for MT
done
# flashinfer and others write under $HOME/.cache, which is not writable inside the container on artemis
for P in SINGULARITYENV APPTAINERENV; do
  export ${P}_XDG_CACHE_HOME=/tmp/synth-cache-$SLURM_JOB_ID ${P}_FLASHINFER_WORKSPACE_BASE=/tmp/synth-cache-$SLURM_JOB_ID
  export ${P}_TRITON_CACHE_DIR=/tmp/synth-cache-$SLURM_JOB_ID/triton ${P}_TORCHINDUCTOR_CACHE_DIR=/tmp/synth-cache-$SLURM_JOB_ID/inductor
done
mkdir -p "$OUT"
pids=()
read -r -a ULIST <<< "$UNITS"
for ((g=0; g<NGPU; g++)); do
  (
    for ((i=g; i<${#ULIST[@]}; i+=NGPU)); do
      name="${ULIST[$i]}"
      for stage in $STAGES; do
        CUDA_VISIBLE_DEVICES=$g SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g APPTAINERENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g \
        singularity exec --nv --bind /mnt/scratch-nyx "$SIF" \
          python "$CODE/run_mt_qe.py" --pool "$UNIT_DIR/$name.pool.jsonl" --out-dir "$OUT" --unit "$name" \
          --stage $stage --ledger "$OUT/ledger.tsv" --gpu-mem $GPU_MEM --qe-dtype $QE_DTYPE ${EXTRA:-}
      done
    done
  ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done  # a failed GPU worker must fail the job, or afterok chains run on nothing
exit $rc
