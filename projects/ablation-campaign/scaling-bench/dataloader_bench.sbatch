#!/bin/bash
#SBATCH --job-name=melt-dlbench
#SBATCH --account=epor48
#SBATCH --qos=gp_debug
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/dlbench.%j.out
#SBATCH --time=00:40:00
#
# Data-path benchmark on the CPU-only partition: no GPU is requested, so this
# costs zero GPU-hours. Submit with the node count, e.g.
#
#   sbatch --nodes=32 projects/ablation-campaign/scaling-bench/dataloader_bench.sbatch
#
# One task per emulated rank (4 per node), matching the GPU sweep's world size so
# the two can be compared directly: `batch_s_p50` here against
# `s_per_step / grad_accum` there.
#
# Needs LOCAL_DATASETS_DIR to point at the same Shar the training runs read.
set -euo pipefail

NODES="${SLURM_NNODES:-1}"
WORLD=$((NODES * 4))
CONFIG="${CONFIG:-projects/ablation-campaign/ABL-IFT-700.yaml}"
STEPS="${STEPS:-200}"
OUTDIR="${OUTDIR:-projects/ablation-campaign/scaling-bench/raw/dlbench-N${NODES}}"
IMAGE="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126.sif}"

mkdir -p "$OUTDIR" logs

echo "[dlbench] nodes=${NODES} world=${WORLD} config=${CONFIG} steps=${STEPS}"

srun --ntasks="$WORLD" --ntasks-per-node=4 --cpus-per-task=2 \
  singularity exec \
    --bind "${LOCAL_DATASETS_DIR:?set LOCAL_DATASETS_DIR}":/workspace/shar \
    --bind "${OUTDIR}":/workspace/dlbench \
    --bind /gpfs/scratch/epor48:/gpfs/scratch/epor48 \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      source /workspace/venv/bin/activate 2>/dev/null || true
      export LOCAL_DATASETS_DIR=/workspace/shar
      export HF_HOME=${HF_HOME:-/gpfs/scratch/epor48/hf_cache}
      export HF_HUB_OFFLINE=1
      python projects/ablation-campaign/scaling-bench/dataloader_bench.py \
        --config '$CONFIG' --rank \"\$SLURM_PROCID\" --world-size '$WORLD' \
        --steps '$STEPS' --json-out /workspace/dlbench/rank-\$SLURM_PROCID.json
    "

echo "[dlbench] done; results in $OUTDIR"
