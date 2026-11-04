#!/bin/bash
#
# Submit Fondue on MN5: N_CONSTANT chained jobs of Fondue.yaml (warmup and constant LR,
# each resuming the newest checkpoint of the run's output dir; the phase ends at
# run.stop_after_step), then one job of Fondue-anneal.yaml from the checkpoint at the
# annealing step. 16 nodes, 3-day walls, acc_ehpc.
#
#   submit_fondue_mn5.sh            print the plan, submit nothing
#   submit_fondue_mn5.sh --submit   copy the configs and submit the chain
#
# Run from the frozen clone. The configs the jobs read are COPIES in
# ${OUTPUT_DIR}/Fondue-configs/, read when each job starts: until a job starts, its
# config may still be edited there (the backbone swap, the annealing sources), never
# the clone. A job runs the clone's code as it is when the job starts, so the clone
# is not touched until the last job ends.
#
# N_CONSTANT is set from the dry run's s/step. If the constant phase ends a job early
# (checkpoint-<annealing step> exists while a constant job is still pending), scancel
# that job: a resume past the stop trains one more step, and the annealing job waits on
# it. Cancelled counts as ended for afterany, so the annealing job then becomes eligible.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
HERE=projects/ablation-campaign/fondue
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export LOCAL_ASR_SCORES_DIR="${LOCAL_ASR_SCORES_DIR:-/gpfs/projects/epor48/melt-data/shar-asr-scores}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES="${MELT_NODES:-16}"
export MELT_GPUS_PER_NODE=4
export MELT_QOS="${MELT_QOS:-acc_ehpc}"
export MELT_TIME="${MELT_TIME:-3-00:00:00}"
N_CONSTANT="${N_CONSTANT:-3}"
EXP=Fondue
CONFIGS="${OUTPUT_DIR}/${EXP}-configs"   # host path; /workspace/outputs/${EXP}-configs inside

ANNEAL_STEP=$(sed -n 's/^  stop_after_step: \([0-9]*\).*/\1/p' "$HERE/Fondue.yaml")
[[ -n "$ANNEAL_STEP" ]] || { echo "no run.stop_after_step in $HERE/Fondue.yaml" >&2; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
[[ -z "$(git status --porcelain)" ]] || { echo "the clone has uncommitted changes; submit from a clean commit" >&2; exit 1; }

echo "Fondue from $(pwd) @ ${COMMIT}: ${N_CONSTANT} x Fondue.yaml to step ${ANNEAL_STEP}, then Fondue-anneal.yaml"
echo "  ${MELT_NODES} nodes, ${MELT_TIME}, ${MELT_QOS}; configs read from ${CONFIGS}"
echo "  constant: ${OUTPUT_DIR}/${EXP} (MELT_SEED 42); anneal: ${OUTPUT_DIR}/${EXP}-anneal (MELT_SEED 43)"
(( SUBMIT )) || { echo "(plan only; pass --submit)"; exit 0; }

[[ ! -e "$CONFIGS" ]] || { echo "$CONFIGS exists: a chain was already submitted" >&2; exit 1; }
mkdir -p "$CONFIGS" "${OUTPUT_DIR}/${EXP}-ck${ANNEAL_STEP}"
cp "$HERE/Fondue.yaml" "$HERE/Fondue-anneal.yaml" "$CONFIGS/"
echo "${COMMIT}" > "$CONFIGS/COMMIT"
# The annealing job resumes this directory's only entry. Dangling until the constant
# phase writes it: if the phase never gets there, the job fails at start, loudly.
ln -sfn "../${EXP}/checkpoint-${ANNEAL_STEP}" "${OUTPUT_DIR}/${EXP}-ck${ANNEAL_STEP}/checkpoint-${ANNEAL_STEP}"

# submit-container.sh takes no --dependency, so this is its body with one.
# shellcheck disable=SC1091
source infra/runners/sites/mn5.sh
mkdir -p logs
submit() {  # <dependency or ""> <train args...>; prints the job id
    local dep="$1"; shift
    local extra=(); [[ -n "$dep" ]] && extra=(--dependency="afterany:${dep}")
    sbatch --parsable "${SBATCH_ARGS[@]}" "${extra[@]}" --job-name=fondue \
        bash/run_train_singularity.sbatch config/accelerate/ddp.yaml "$@"
}

prev=""
for k in $(seq 1 "$N_CONSTANT"); do
    prev=$(MELT_SEED=42 submit "$prev" --config "/workspace/outputs/${EXP}-configs/Fondue.yaml" \
        --trainer.output_dir "/workspace/outputs/${EXP}" --run.exp_name "${EXP}")
    echo "constant job ${k}/${N_CONSTANT}: ${prev}"
done
anneal=$(MELT_SEED=43 submit "$prev" --config "/workspace/outputs/${EXP}-configs/Fondue-anneal.yaml" \
    --trainer.resume_from_checkpoint "/workspace/outputs/${EXP}-ck${ANNEAL_STEP}" \
    --trainer.output_dir "/workspace/outputs/${EXP}-anneal" --run.exp_name "${EXP}-anneal")
echo "anneal job: ${anneal}"
