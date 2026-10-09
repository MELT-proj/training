#!/bin/bash
#
# Run one of this repo's Python scripts on a single GPU inside the training
# image, with the same container paths a training run sees:
#   repo -> /workspace/training     LOCAL_DATASETS_DIR -> /workspace/shar
#   HF_HOME -> /workspace/hf_cache  OUTPUT_DIR -> /workspace/outputs
# so a run's resolved_config.json paths resolve unchanged. TRL comes from
# .trl-overlay in the repo root when present (the image does not carry it).
#
# Run on artemis from the repo root of a checkout the GPU node can see (the
# h200 node does not mount /mnt/home).
#
#   DRY_RUN=1 bash projects/self-distill/launch_script.sh projects/self-distill/alignment_probe.py --ckpt ... --out ...
#   bash projects/self-distill/launch_script.sh projects/self-distill/alignment_probe.py --ckpt ... --out ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="${1:?usage: $0 <repo-relative script.py> [args...]}"; shift
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT not found under $REPO_ROOT" >&2; exit 1; }
PARTITION="${PARTITION:-h200}"
QOS="${QOS:-gpu-h200}"
TIME="${TIME:-02:00:00}"
IMAGE="${SINGULARITY_IMG:-/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif}"
HF_HOME_HOST="${HF_HOME:-/mnt/scratch-artemis/giuseppe/.cache/huggingface}"
DATASETS="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"
OUTPUTS="${OUTPUT_DIR:-/mnt/scratch-artemis/giuseppe/melt-data/outputs}"
JOB_NAME="${JOB_NAME:-sd-$(basename "$SCRIPT" .py)}"

BINDS="${REPO_ROOT}:/workspace/training,${DATASETS}:/workspace/shar,${HF_HOME_HOST}:/workspace/hf_cache,${OUTPUTS}:/workspace/outputs"
ENVS="HF_HOME=/workspace/hf_cache,LOCAL_DATASETS_DIR=/workspace/shar,PYTHONPATH=/workspace/training:/workspace/training/.trl-overlay,TOKENIZERS_PARALLELISM=false,NUMBA_CACHE_DIR=/tmp/numba"
JOB_SCRIPT="#!/bin/bash
set -euo pipefail
echo \"[launch_script] node \$(hostname): ${SCRIPT} \$*\"
singularity exec --nv --cleanenv --bind \"${BINDS}\" --env \"${ENVS}\" --pwd /workspace/training \"${IMAGE}\" /workspace/venv/bin/python ${SCRIPT} \"\$@\""

SBATCH=(sbatch --partition="$PARTITION" --qos="$QOS" --gpus-per-node=1 --cpus-per-task=8 --time="$TIME"
        --job-name="$JOB_NAME" --output="logs/%x.%j.out")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY_RUN] would run:'; printf ' %q' "${SBATCH[@]}" "<job script>" "$@"; printf '\n'
    echo "[DRY_RUN] job script:"; echo "$JOB_SCRIPT"
    exit 0
fi

mkdir -p logs
JOB_FILE="$(mktemp "logs/${JOB_NAME}.sbatch.XXXXXX")"
printf '%s\n' "$JOB_SCRIPT" > "$JOB_FILE"
"${SBATCH[@]}" "$JOB_FILE" "$@"
