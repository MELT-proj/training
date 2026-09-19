#!/bin/bash
#SBATCH --job-name=eval-throughput
#SBATCH --ntasks=1
#SBATCH --output=logs/%x.%j.out
#
# Eval throughput of one finished arm (plan/03-audio-stack.md section 4, fourth number).
# Wraps eval_throughput.py the way no_audio_floor.sbatch wraps its script: off the
# arms.tsv ledger, one process, one GPU, no accelerate.
#
# One GPU is the whole point. Throughput at batch 16 is only comparable across arms
# measured on the same GPU type, one arm at a time, so submit one job per arm and do
# not share the node with another job of yours.
#
# Usage (repo root on the target site; SINGULARITY_IMG/HF_HOME/LOCAL_DATASETS_DIR/
# OUTPUT_DIR exported as sites/<site>.sh does):
#   mkdir -p logs
#   # MN5 (H100 64 GB; nodes are whole, so ask for the full node's CPUs):
#   sbatch --time=00:30:00 --gpus-per-node=1 --cpus-per-task=80 \
#       --account=epor48 --qos=acc_debug \
#       projects/ablation-campaign/eval_throughput.sbatch <exp_name>
#   # artemis:
#   sbatch --time=00:30:00 --gpus-per-node=1 --cpus-per-task=8 \
#       --qos=gpu-h100 --partition=h100 \
#       projects/ablation-campaign/eval_throughput.sbatch <exp_name>
#
# Result: $OUTPUT_DIR/throughput/<exp_name>.json, which efficiency.py picks up with
#   --throughput-dir <that directory>
# The run dir must hold consolidated weights (DDP runs do; FSDP2 runs need consolidating
# first, infrastructure.md section 5).

set -euo pipefail

EXP="${1:?usage: eval_throughput.sbatch <exp_name> [extra eval_throughput.py args]}"; shift

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

mkdir -p "${OUTPUT_DIR}/throughput" logs

echo "[eval_throughput] runtime:  ${SINGULARITY_BIN}"
echo "[eval_throughput] image:    ${SINGULARITY_IMG}"
echo "[eval_throughput] project:  ${PWD} -> /workspace/training"
echo "[eval_throughput] datasets: ${LOCAL_DATASETS_DIR} -> /workspace/shar"
echo "[eval_throughput] hf_cache: ${HF_HOME} -> /workspace/hf_cache"
echo "[eval_throughput] outputs:  ${OUTPUT_DIR} -> /workspace/outputs"
echo "[eval_throughput] exp:      ${EXP}"

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
    bash -c 'source /workspace/venv/bin/activate 2>/dev/null || true; exp="$1"; shift; exec python projects/ablation-campaign/eval_throughput.py --run-dir "/workspace/outputs/${exp}" --output "/workspace/outputs/throughput/${exp}.json" "$@"' _ "${EXP}" "$@"
