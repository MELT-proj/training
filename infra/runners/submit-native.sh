#!/bin/bash
#
# Submit a SLURM training job on the native host environment (venv, no container).
#
#   infra/runners/submit-native.sh <site> <accelerate_config> [train args...]
#
# Example:
#   infra/runners/submit-native.sh artemis config/accelerate/zero3.yaml \
#     --config config/train/LS_asr.yaml --trainer.max_steps 10
#
# <site> selects infra/runners/sites/<site>.sh, which exports the host paths and
# defines the SBATCH_ARGS array (partition/QoS/account/time). Run from the repo root.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <accelerate_config> [train args...]}"; shift
[[ -f bash/run_train.sh ]] || die "run this from the repo root (bash/run_train.sh not found here)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE}); copy sites/example.sh to add one"
# shellcheck disable=SC1090
source "$SITE_FILE"

mkdir -p logs   # SLURM won't create the --output dir; a missing dir kills the job silently.

echo "[submit-native] site=${SITE} sbatch ${SBATCH_ARGS[*]} bash/run_train.sh $*"
sbatch "${SBATCH_ARGS[@]}" bash/run_train.sh "$@"
