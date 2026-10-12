#!/bin/bash
# Submit the selection-metric eval (the five in-domain sets and FLEURS-26, one sbatch each) for cells of the 2,100 h crossing.
# Run ON MN5 from any checkout that holds this script (main) and name the frozen training checkout to score with, e.g.
#   bash <checkout>/projects/ablation-campaign/selection-metric/submit_crossing_eval.sh --code ~/training-eval [options] <cell> ...
#
# <cell> is a crossing row suffix: whisper-conformer, mms1b-qformer, ... The run scored is the one the CURRENT row of
# campaign.yaml names in the --code checkout (its composed run name, so the LR, steps and `qkn` tag are all in it), and only
# once that run has finished: its final checkpoint-<max_steps> and processor_config.json exist. The model scored is the run
# root, the final weights, as for the screen (board 2026-09-24).
#
# Options:
#   --qos Q       acc_ehpc (default) or acc_debug
#   --time T      per-job wall time, HH:MM:SS (default 02:00:00; a FLEURS job takes 35-70 min, an in-domain one 3-6)
#   --sets "..."  only these sets (default: all six)
#   --limit N     score only N samples per set, into a separate log tree (a smoke, not a score)
#   --code DIR    training checkout put on PYTHONPATH (default: the checkout this script is in)
#   --dry-run     print the sbatch commands and submit nothing
#
# Why PYTHONPATH: the eval image installs `melt` as a built package baked in at build time (the v3 image, 2026-09-22, has no MoE
# adapter, an old Conformer and no model.adapter.qk_norm), and the eval runner binds only the melt-eval checkout. The image's
# own `melt` is therefore shadowed by this checkout's, the code the models were trained with, so the image need not be rebuilt
# for every training-side change (the training launcher gets the same effect by running from the bind-mounted checkout).
# SINGULARITYENV_PYTHONPATH carries it into the container; the checkout is under $HOME on GPFS, which the container sees at
# the same path. Each submission is recorded in the sets' eval-jobs.tsv with the code commit.
#
# Queued jobs read the checkouts they were submitted from when they START, hours later on acc_ehpc: leave --code and ~/eval alone
# (no checkout, pull or edit) until the jobs have run.
#
# Idempotent: a set whose log directory already holds a JSON log, or whose job is already queued or running, is skipped, never
# overwritten or doubled. After the jobs finish:
#   python3 <training>/projects/ablation-campaign/selection-metric/score.py /gpfs/scratch/epor48/eval-logs/selection-metric/ma2100-<cell> ...
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd "$HERE/../../.." && pwd)
QOS=acc_ehpc
TIME=02:00:00
SETS_ARG=""
LIMIT=""
DRY=0
CELLS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --qos) QOS=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --sets) SETS_ARG=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --code) CODE=$(cd "$2" && pwd); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) CELLS+=("$1"); shift ;;
  esac
done
[[ ${#CELLS[@]} -gt 0 ]] || { echo "usage: $0 [options] <cell> [<cell> ...]   (try --help)"; exit 2; }

EVAL=$HOME/eval
SETS=$EVAL/frozen-sets/selection-metric
OUTPUTS=/gpfs/scratch/epor48/outputs
LOGS=/gpfs/scratch/epor48/eval-logs/selection-metric
SIF=${MELT_SIF:-/gpfs/scratch/epor48/melt_eval_cuda126.sif}
ALL_SETS="selection-id-en selection-id-de selection-id-es selection-id-fr selection-id-it selection-fleurs26-dev200"
USE_SETS=${SETS_ARG:-$ALL_SETS}
CODE_SHA=$(git -C "$CODE" rev-parse --short HEAD)
[[ -f $EVAL/infra/run_eval_container_mn5.sbatch ]] || { echo "no melt-eval checkout at $EVAL"; exit 1; }
[[ -f $SIF ]] || { echo "no eval image $SIF"; exit 1; }
if [[ -n $LIMIT ]]; then LOGS=$LOGS-smoke; fi
echo "code: $CODE @ $CODE_SHA | image: $(readlink -f "$SIF") | qos: $QOS | time: $TIME | sets: $USE_SETS${LIMIT:+ | LIMIT $LIMIT (smoke)}"

cd "$EVAL"; mkdir -p logs
QUEUED=$(squeue -u "$USER" -h -o %j)
for CELL in "${CELLS[@]}"; do
  PLAN=$(python3 "$CODE/projects/ablation-campaign/campaign.py" run "MA-2100-crossing-$CELL" --dry-run)
  EXP=$(grep -o -- '--run.exp_name [^ ]*' <<<"$PLAN" | cut -d' ' -f2)
  MAXSTEPS=$(grep -o -- '--trainer.max_steps [0-9]*' <<<"$PLAN" | cut -d' ' -f2)
  CKPT=$OUTPUTS/$EXP
  [[ -n $EXP && -n $MAXSTEPS ]] || { echo "$CELL: could not read the run name or max_steps from the plan"; exit 1; }
  if [[ ! -d $CKPT/checkpoint-$MAXSTEPS || ! -f $CKPT/processor_config.json ]]; then
    echo "SKIP $CELL: not finished (no checkpoint-$MAXSTEPS / processor_config.json in $CKPT)"; continue
  fi
  ARM=ma2100-$CELL
  echo "== $CELL -> $EXP (final step $MAXSTEPS)"
  for S in $USE_SETS; do
    OUT=$LOGS/$ARM/$S
    if compgen -G "$OUT/*.json" > /dev/null; then echo "   skip $S: JSON log already in $OUT"; continue; fi
    JOBNAME=sel-$ARM-${S#selection-}${LIMIT:+-smoke}
    if grep -qxF -- "$JOBNAME" <<<"$QUEUED"; then echo "   skip $S: job $JOBNAME is already queued or running"; continue; fi
    CMD=(sbatch --parsable --account=epor48 --qos="$QOS" --nodes=1 --gpus-per-node=1 --cpus-per-task=20 --time="$TIME"
         --job-name="$JOBNAME" infra/run_eval_container_mn5.sbatch
         "$CKPT" "$SETS/$S" -T task_filter=asr --log-format json -M batch_size=4 -M dtype=bfloat16 ${LIMIT:+--limit "$LIMIT"})
    if [[ $DRY == 1 ]]; then echo "   DRY: SINGULARITYENV_PYTHONPATH=$CODE MELT_SIF=$SIF OUTPUT_DIR=$OUT ${CMD[*]}"; continue; fi
    mkdir -p "$OUT"
    JID=$(SINGULARITYENV_PYTHONPATH=$CODE MELT_SIF=$SIF OUTPUT_DIR=$OUT "${CMD[@]}")
    echo -e "$(date -Is)\t$ARM\t$S\t$JID\tcode=$CODE_SHA qos=$QOS${LIMIT:+ limit=$LIMIT}" | tee -a "$SETS/eval-jobs.tsv"
  done
done
