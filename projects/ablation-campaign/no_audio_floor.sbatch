#!/bin/bash
#SBATCH --job-name=no-audio-floor
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu-h100
#SBATCH --partition=h100
#SBATCH --output=logs/%x.%j.out
#
# One-off diagnostic job (plan/01-interface-recipe.md section 1, week 1
# Track A "No-audio floor"): scores the frozen backbone named in the given
# MA config on its validation transcripts with no audio at all.
#
# Not a campaign arm -- off the arms.tsv ledger by design, same as
# launch_MA.sh/launch_IFT.sh one-offs -- so this runs its own thin container
# invocation instead of infra/runners/submit-container.sh, which drives
# accelerate + train.py for a full training job; this is a single-process
# forward-only score, nothing here trains or needs DDP/FSDP.
#
# Usage (from the repo root on the target site, HF_HOME/LOCAL_DATASETS_DIR/
# OUTPUT_DIR/SINGULARITY_IMG exported the same way sites/<site>.sh does):
#   mkdir -p logs
#   sbatch projects/ablation-campaign/no_audio_floor.sbatch [config] [output]

set -euo pipefail

CONFIG="${1:-projects/ablation-campaign/ABL-MA-700-asr.yaml}"
OUTPUT="${2:-/workspace/outputs/no_audio_floor/no_audio_floor.json}"

[[ -f bash/run_train.sh ]] || { echo "ERROR: run this from the repo root (bash/run_train.sh not found here)" >&2; exit 1; }

if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi

SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [[ -z "${SINGULARITY_BIN}" ]]; then
    if command -v apptainer >/dev/null 2>&1; then
        SINGULARITY_BIN=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        SINGULARITY_BIN=singularity
    else
        echo "ERROR: neither 'apptainer' nor 'singularity' found in PATH" >&2
        exit 2
    fi
fi

: "${SINGULARITY_IMG:?set SINGULARITY_IMG to the .sif path}"
: "${HF_HOME:?set HF_HOME (host path to the shared HF cache)}"
: "${LOCAL_DATASETS_DIR:?set LOCAL_DATASETS_DIR (host path to the shar tree)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR (host path, group-writable outputs dir)}"

mkdir -p "${OUTPUT_DIR}/no_audio_floor" logs

echo "[no_audio_floor] runtime:  ${SINGULARITY_BIN}"
echo "[no_audio_floor] image:    ${SINGULARITY_IMG}"
echo "[no_audio_floor] project:  ${PWD} -> /workspace/training"
echo "[no_audio_floor] datasets: ${LOCAL_DATASETS_DIR} -> /workspace/shar"
echo "[no_audio_floor] hf_cache: ${HF_HOME} -> /workspace/hf_cache"
echo "[no_audio_floor] outputs:  ${OUTPUT_DIR} -> /workspace/outputs"
echo "[no_audio_floor] config:   ${CONFIG}"
echo "[no_audio_floor] output:   ${OUTPUT}"

"${SINGULARITY_BIN}" exec --nv \
    --bind "${PWD}:/workspace/training" \
    --bind "${LOCAL_DATASETS_DIR}:/workspace/shar" \
    --bind "${HF_HOME}:/workspace/hf_cache" \
    --bind "${OUTPUT_DIR}:/workspace/outputs:rw" \
    --pwd /workspace/training \
    --env HF_HOME=/workspace/hf_cache \
    --env HF_HUB_CACHE=/workspace/hf_cache/hub \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    --env LOCAL_DATASETS_DIR=/workspace/shar \
    --env OUTPUT_DIR=/workspace/outputs \
    --env PYTHONPATH=/workspace/training \
    "${SINGULARITY_IMG}" \
    bash -c 'source /workspace/venv/bin/activate 2>/dev/null || true; exec python projects/ablation-campaign/no_audio_floor.py --config "$1" --output "$2"' _ "${CONFIG}" "${OUTPUT}"
