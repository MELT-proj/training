#!/bin/bash
#SBATCH --job-name=step0b-diagnostic
#SBATCH --ntasks=1
#SBATCH --output=logs/%x.%j.out
#
# One-off diagnostic job (plan/01-interface-recipe.md sec2b, pre-flight step
# 1b): full dev-clean/dev-other decoding pass on MA-librispeech-l4-ep3's
# final checkpoint, greedy and then with no_repeat_ngram_size 4, to size
# runaway generation on the whole eval set rather than the 20 logged
# samples. Diagnostic only -- does not gate step 0b's arms.
#
# Same rationale as no_audio_floor.sbatch: not a campaign arm, off the
# arms.tsv ledger by design, single-process (no DDP/FSDP needed for a
# read-only generation pass over one checkpoint), so this runs its own thin
# container invocation instead of infra/runners/submit-container.sh.
#
# Deliberately carries no cluster-specific #SBATCH scheduling flags -- pass
# them on the sbatch command line, same convention as no_audio_floor.sbatch.
#
# Usage (from the repo root on the target site, HF_HOME/LOCAL_DATASETS_DIR/
# OUTPUT_DIR/SINGULARITY_IMG exported the same way sites/<site>.sh does):
#   mkdir -p logs
#   # MN5 (acc_debug: 2h cap, priority 10000 -- schedules immediately):
#   sbatch --time=00:45:00 --gpus-per-node=1 --cpus-per-task=80 \
#       --account=epor48 --qos=acc_debug \
#       projects/ablation-campaign/step0b_diagnostic_full_eval.sbatch \
#       [config] [checkpoint] [output_dir]

set -euo pipefail

CONFIG="${1:-projects/ablation-campaign/ABL-MA-librispeech.yaml}"
CHECKPOINT="${2:-/workspace/outputs/MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s44-8g}"
OUTPUT_DIR_ARG="${3:-/workspace/outputs/step0b_diagnostic}"

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

mkdir -p "${OUTPUT_DIR}/step0b_diagnostic" logs

echo "[step0b-diagnostic] runtime:    ${SINGULARITY_BIN}"
echo "[step0b-diagnostic] image:      ${SINGULARITY_IMG}"
echo "[step0b-diagnostic] project:    ${PWD} -> /workspace/training"
echo "[step0b-diagnostic] datasets:   ${LOCAL_DATASETS_DIR} -> /workspace/shar"
echo "[step0b-diagnostic] hf_cache:   ${HF_HOME} -> /workspace/hf_cache"
echo "[step0b-diagnostic] outputs:    ${OUTPUT_DIR} -> /workspace/outputs"
echo "[step0b-diagnostic] config:     ${CONFIG}"
echo "[step0b-diagnostic] checkpoint: ${CHECKPOINT}"
echo "[step0b-diagnostic] output_dir: ${OUTPUT_DIR_ARG}"

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
    bash -c 'source /workspace/venv/bin/activate 2>/dev/null || true; exec python projects/ablation-campaign/step0b_diagnostic_full_eval.py --config "$1" --checkpoint "$2" --output-dir "$3"' _ "${CONFIG}" "${CHECKPOINT}" "${OUTPUT_DIR_ARG}"
