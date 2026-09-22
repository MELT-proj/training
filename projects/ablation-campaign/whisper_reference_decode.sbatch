#!/bin/bash
#SBATCH --job-name=whisper-ref-decode
#SBATCH --ntasks=1
#SBATCH --output=logs/%x.%j.out
#
# Whisper-large-v3 decoding the step-0b dev sets by itself (plan/timeline.md,
# week 1 Track B). Single process, forward-only: not a campaign arm, off the
# arms.tsv ledger by design, same thin container invocation as
# step0b_diagnostic_full_eval.sbatch. No cluster #SBATCH flags -- pass them on
# the command line.
#
# Usage (repo root on the target site, HF_HOME/LOCAL_DATASETS_DIR/OUTPUT_DIR/
# SINGULARITY_IMG exported as sites/<site>.sh does):
#   mkdir -p logs
#   sbatch --time=01:00:00 --gpus-per-node=1 --cpus-per-task=8 \
#       --qos=gpu-h100 --partition=h100 \
#       projects/ablation-campaign/whisper_reference_decode.sbatch [output_dir] [extra script args]
set -euo pipefail

OUTPUT_DIR_ARG="${1:-/workspace/outputs/whisper_reference_decode}"
shift || true

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root" >&2; exit 1; }

if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi
SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [[ -z "${SINGULARITY_BIN}" ]]; then
    if command -v apptainer >/dev/null 2>&1; then SINGULARITY_BIN=apptainer
    elif command -v singularity >/dev/null 2>&1; then SINGULARITY_BIN=singularity
    else echo "ERROR: neither apptainer nor singularity in PATH" >&2; exit 2; fi
fi

: "${SINGULARITY_IMG:?set SINGULARITY_IMG to the .sif path}"
: "${HF_HOME:?set HF_HOME}"
: "${LOCAL_DATASETS_DIR:?set LOCAL_DATASETS_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" logs
echo "[whisper-ref] image: ${SINGULARITY_IMG}  datasets: ${LOCAL_DATASETS_DIR}  out: ${OUTPUT_DIR_ARG}"
nvidia-smi -L || true

"${SINGULARITY_BIN}" exec --nv \
    --bind "${PWD}:/workspace/training" \
    --bind "${LOCAL_DATASETS_DIR}:/workspace/shar" \
    --bind "${HF_HOME}:/workspace/hf_cache" \
    --bind "${OUTPUT_DIR}:/workspace/outputs:rw" \
    --pwd /workspace/training \
    --env HF_HOME=/workspace/hf_cache \
    --env HF_HUB_CACHE=/workspace/hf_cache/hub \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    --env LOCAL_DATASETS_DIR=/workspace/shar \
    --env PYTHONPATH=/workspace/training \
    "${SINGULARITY_IMG}" \
    bash -c 'source /workspace/venv/bin/activate 2>/dev/null || true; exec python projects/ablation-campaign/whisper_reference_decode.py --output-dir "$@"' _ "${OUTPUT_DIR_ARG}" "$@"
