#!/bin/bash
# Artemis (A6000 etc.) variant of run_units.sbatch: one process per GPU, units from UNITS (names), nyx scratch for everything.
#   (UNITS_G0="..." UNITS_G1="..." pin units to GPUs instead of UNITS)
#   sbatch --partition=a6000 --qos=gpu-medium --gres=gpu:4 --cpus-per-task=32 --mem=200G --time=2-00:00:00 \
#          --export=ALL,UNITS="u1 u2 ...",OUT=...,STAGES=mt code/slurm/run_units_artemis.sbatch
# Stages can be submitted separately (they write into the same OUT/<unit> chunks, so QE can be chained with
# --dependency=afterok):  MT:  --export=ALL,STAGES=mt ...     QE (MetricX-24-Hybrid-XXL):  --export=ALL,STAGES=qe ...
# QE_DTYPE=bf16 (default) uses Google's bfloat16 release, 26 GB: fits an A6000. fp32 (52 GB) needs an H100/H200.
set -euo pipefail
: "${OUT:?}"
[ -n "${UNITS:-}" ] || [ -n "${UNITS_G0:-}" ] || { echo "set UNITS (round-robin) or UNITS_G<gpu>" >&2; exit 2; }
CODE="${CODE:-/mnt/scratch-nyx/giuseppe/synth/code}"
SIF="${SYNTH_SIF:-/mnt/scratch-nyx/giuseppe/synth/synth.sif}"
UNIT_DIR="${UNIT_DIR:-/mnt/scratch-nyx/giuseppe/synth/units}"
NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}}"  # one worker per allocated GPU
QE_DTYPE="${QE_DTYPE:-bf16}"
STAGES="${STAGES:-mt qe}"
GPU_MEM="${GPU_MEM:-0.88}"
# SingularityCE reads SINGULARITYENV_*, apptainer APPTAINERENV_*: set both
for P in SINGULARITYENV APPTAINERENV; do
  export ${P}_HF_HOME=/mnt/scratch-nyx/giuseppe/synth/hf_cache ${P}_HF_HUB_OFFLINE=1
  export ${P}_PYTHONPATH=/mnt/scratch-nyx/giuseppe/synth/metricx  # pinned google-research/metricx checkout
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
read -r -a ULIST <<< "${UNITS:-}"
for ((g=0; g<NGPU; g++)); do
  (
    mine=()
    explicit="UNITS_G$g"  # UNITS_G<gpu>="u1 u2 ..." pins units to a GPU, otherwise UNITS is dealt round-robin
    if [ -n "${!explicit:-}" ]; then read -r -a mine <<< "${!explicit}"; else
      for ((i=g; i<${#ULIST[@]}; i+=NGPU)); do mine+=("${ULIST[$i]}"); done
    fi
    [ ${#mine[@]} -gt 0 ] || exit 0  # more GPUs than units
    for stage in $STAGES; do
      # MT reloads vLLM per unit (~1 min); QE takes all of this GPU's units in one process (the MetricX load is a few minutes)
      if [ "$stage" = qe ]; then groups=("$(IFS=,; echo "${mine[*]}")"); else groups=("${mine[@]}"); fi
      for name in "${groups[@]}"; do
        CUDA_VISIBLE_DEVICES=$g SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g APPTAINERENV_VLLM_CACHE_ROOT=/tmp/synth-vllm-$SLURM_JOB_ID-$g \
        singularity exec --nv --bind /mnt/scratch-nyx "$SIF" \
          python "$CODE/run_mt_qe.py" --units-dir "$UNIT_DIR" --out-dir "$OUT" --unit "$name" \
          --stage $stage --ledger "$OUT/ledger.tsv" --gpu-mem $GPU_MEM --qe-dtype $QE_DTYPE ${EXTRA:-}
      done
    done
  ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done  # a failed GPU worker must fail the job, or afterok chains run on nothing
exit $rc
