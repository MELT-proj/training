#!/bin/bash
#
# Self-distillation phase 2, step 1b: submit teacher_audit.py as a 1-GPU job.
#
# Run on artemis from the repo root of a checkout the GPU node can see: the
# h200 node does not mount /mnt/home, so use the clone on /mnt/scratch-artemis.
# The job script is written next to the results, so the run documents itself.
#
#   DRY_RUN=1 bash projects/self-distill/launch_teacher_audit.sh
#   bash projects/self-distill/launch_teacher_audit.sh                       # full audit
#   bash projects/self-distill/launch_teacher_audit.sh --per-source 4 --fleurs-n 4   # smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-h200}"
QOS="${QOS:-gpu-h200}"
TIME="${TIME:-03:00:00}"
IMAGE="${SINGULARITY_IMG:-/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126.sif}"
HF_HOME_HOST="${HF_HOME:-/mnt/scratch-artemis/giuseppe/.cache/huggingface}"
DATASETS="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"
OUT_ROOT="${OUT_ROOT:-/mnt/scratch-artemis/giuseppe/melt-data/outputs/sd-teacher-audit}"
RUN_NAME="${RUN_NAME:-teacher-audit-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"

BINDS="${REPO_ROOT}:/workspace/training,${DATASETS}:/workspace/shar,${HF_HOME_HOST}:/workspace/hf_cache,${OUT_DIR}:/workspace/out"
ENVS="HF_HOME=/workspace/hf_cache,LOCAL_DATASETS_DIR=/workspace/shar,PYTHONPATH=/workspace/training,TOKENIZERS_PARALLELISM=false"
JOB_SCRIPT="#!/bin/bash
set -euo pipefail
echo \"[teacher-audit] node \$(hostname), out ${OUT_DIR}\"
singularity exec --nv --cleanenv --bind \"${BINDS}\" --env \"${ENVS}\" --pwd /workspace/training \"${IMAGE}\" /workspace/venv/bin/python projects/self-distill/teacher_audit.py --out-dir /workspace/out \"\$@\""

SBATCH=(sbatch --partition="$PARTITION" --qos="$QOS" --gpus-per-node=1 --cpus-per-task=8 --time="$TIME"
        --job-name=sd-teacher-audit --output="logs/%x.%j.out")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] out dir: ${OUT_DIR}"
    printf '[DRY_RUN] would run:'; printf ' %q' "${SBATCH[@]}" "${OUT_DIR}/job.sbatch" "$@"; printf '\n'
    echo "[DRY_RUN] job script:"; echo "$JOB_SCRIPT"
    exit 0
fi

mkdir -p logs "$OUT_DIR"
printf '%s\n' "$JOB_SCRIPT" > "${OUT_DIR}/job.sbatch"
"${SBATCH[@]}" "${OUT_DIR}/job.sbatch" "$@"
