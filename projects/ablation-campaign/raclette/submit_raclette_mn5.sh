#!/bin/bash
#
# Submit the Raclette LR pilot on MN5 (plan/08-raclette.md): one job per arm, 4 nodes x 4 H100,
# acc_ehpc, 36 h wall. Run from the frozen clone ~/training-raclette; never touch it while jobs read it.
#
#   submit_raclette_mn5.sh plan                       print the arms, submit nothing
#   [SMOKE_EXTRA="--a.b c"] submit_raclette_mn5.sh smoke <arm>   one acc_debug job (2 nodes, 40 steps, 1.5 h wall: start-up took 36 min and the final eval ~10 min)
#   submit_raclette_mn5.sh submit --go [--arms "x1 x2"] [--after JOBID] [--nice N]
#
# `submit` needs --go (the PI's go). --after (default $FONDUE_JOB, 48754959) adds
# --dependency=after:JOBID so the arms only become eligible once Fondue's first job has STARTED.
# render_raclette_configs.py writes ${OUTPUT_DIR}/Raclette-configs/ (read when each job starts).
# Each arm's output is ${OUTPUT_DIR}/Raclette-<arm>. MELT_SEED (the data shard seed) is 42, or 43
# for x1-s43, which also takes --trainer.seed 1338 from its config.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
HERE=projects/ablation-campaign/raclette
ALL_ARMS="x0p5 x1 x1-s43 x2 x4"

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export LOCAL_ASR_SCORES_DIR="${LOCAL_ASR_SCORES_DIR:-/gpfs/projects/epor48/melt-data/shar-asr-scores}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES="${MELT_NODES:-4}"
export MELT_GPUS_PER_NODE=4   # pinned: SLURM_GPUS_ON_NODE lies on acc
CONFIGS="${OUTPUT_DIR}/Raclette-configs"   # host path; /workspace/outputs/Raclette-configs inside
FONDUE_JOB="${FONDUE_JOB:-48754959}"

MODE="${1:-plan}"; shift || true
ARMS="$ALL_ARMS"; NICE=""; AFTER="$FONDUE_JOB"; GO=0; SMOKE_ARM=""
if [[ "$MODE" == smoke ]]; then SMOKE_ARM="${1:?smoke needs an arm}"; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)  ARMS="$2"; shift 2 ;;
        --nice)  NICE="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --go)    GO=1; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
COMMIT=$(git rev-parse --short HEAD)
[[ "$MODE" == smoke ]] && export MELT_NODES=2   # before mn5.sh builds SBATCH_ARGS

# shellcheck disable=SC1091
source infra/runners/sites/mn5.sh
mkdir -p logs

seed_of() { [[ "$1" == x1-s43 ]] && echo 43 || echo 42; }

launch() {  # <arm> <qos> <wall> <extra train args...>; prints the job id
    local arm="$1" qos="$2" wall="$3"; shift 3
    local name="Raclette-${arm}"
    local extra=(--time="${wall}" --qos="${qos}" --job-name="raclette-${arm}")
    [[ -n "$NICE" ]] && extra+=(--nice="${NICE}")
    [[ "$qos" == acc_ehpc && -n "$AFTER" ]] && extra+=(--dependency="after:${AFTER}")
    MELT_SEED="$(seed_of "$arm")" sbatch --parsable "${SBATCH_ARGS[@]}" "${extra[@]}" \
        bash/run_train_singularity.sbatch config/accelerate/ddp.yaml \
        --config "/workspace/outputs/Raclette-configs/Raclette-${arm}.yaml" \
        --trainer.output_dir "/workspace/outputs/${name}" --run.exp_name "${name}" "$@"
}

case "$MODE" in
plan)
    echo "Raclette from $(pwd) @ ${COMMIT}: configs read from ${CONFIGS}"
    for arm in $ARMS; do
        echo "  Raclette-${arm}: ${MELT_NODES} nodes, acc_ehpc, 36:00:00, MELT_SEED $(seed_of "$arm"), after:${AFTER:-none}, output ${OUTPUT_DIR}/Raclette-${arm}"
    done
    ;;
smoke)
    # 40 steps at 2 nodes, ga 16 (a smoke of the per-micro-batch cost, not of the 4-node shape):
    # warmup 4 so the peak LR is reached, a save at step 20 to check the weights-only directory.
    # perf_logging prints s/step and gpu_peak_gb. Later dotlist keys win over launch()'s defaults.
    mkdir -p "$CONFIGS"
    python3 "${HERE}/render_raclette_configs.py" --base projects/ablation-campaign/fondue/Fondue.yaml --out-dir "$CONFIGS" --nodes 4 >/dev/null
    tag="Raclette-smoke-${SMOKE_ARM}${SMOKE_TAG:-}"
    id=$(AFTER="" launch "$SMOKE_ARM" acc_debug "${SMOKE_WALL:-01:30:00}" \
        --run.perf_logging true --trainer.max_steps 40 --trainer.warmup_steps 4 \
        --trainer.lr_scheduler_kwargs.num_decay_steps 8 --trainer.gradient_accumulation_steps 16 \
        --trainer.save_steps 20 --trainer.eval_steps 100000 ${SMOKE_EXTRA:-} \
        --trainer.output_dir "/workspace/outputs/${tag}" --run.exp_name "${tag}")
    echo "smoke ${SMOKE_ARM}: ${id} (output ${OUTPUT_DIR}/${tag})"
    ;;
submit)
    (( GO )) || { echo "submit needs --go (the PI's go)" >&2; exit 2; }
    mkdir -p "$CONFIGS"
    python3 "${HERE}/render_raclette_configs.py" --base projects/ablation-campaign/fondue/Fondue.yaml --out-dir "$CONFIGS" --nodes "$MELT_NODES"
    echo "$COMMIT" > "$CONFIGS/COMMIT"
    for arm in $ARMS; do
        [[ -e "${OUTPUT_DIR}/Raclette-${arm}" ]] && { echo "${OUTPUT_DIR}/Raclette-${arm} exists; skipping" >&2; continue; }
        id=$(launch "$arm" acc_ehpc 36:00:00)
        echo "Raclette-${arm}: ${id}"
    done
    ;;
*) echo "usage: $0 plan|smoke <arm>|submit --go" >&2; exit 2 ;;
esac
