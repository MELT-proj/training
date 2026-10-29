#!/bin/bash
#
# ASR score band smoke, 1 node x 4 GPUs on acc_debug (~3 GPUh each).
#
#   run_smoke_mn5.sh train     40 steps, bands switch at step 25 (see smoke.yaml)
#   run_smoke_mn5.sh resume    resume from train's checkpoint-20, crossing the switch
#
# Needs the score arrays on MN5 (infra/build_asr_scores.py, staged with
# mn5transfer) and a clone of the branch that is not used by any other job.
# Afterwards, on a machine that has shar-stats and the arrays:
#   python check_smoke.py --cut-ids-dir <OUTPUT_DIR>/<exp>-cut_ids ...
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PHASE="${1:?usage: $0 <train|resume>}"
EXP_BASE="${EXP_BASE:-asr-filter-smoke}"

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export LOCAL_ASR_SCORES_DIR="${LOCAL_ASR_SCORES_DIR:-/gpfs/projects/epor48/melt-data/shar-asr-scores}"
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126.sif}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES=1
export MELT_GPUS_PER_NODE=4
export MELT_QOS="${MELT_QOS:-acc_debug}"
export MELT_TIME="${MELT_TIME:-01:00:00}"

CONFIG=tests/integration/asr_filter_smoke/smoke.yaml
COMMON=(--config "$CONFIG"
        --trainer.save_steps 10 --trainer.save_total_limit 10
        --trainer.eval_steps 20 --trainer.logging_steps 1
        --trainer.report_to none)

case "$PHASE" in
train)  EXP="${EXP_BASE}-train" ;;
resume) EXP="${EXP_BASE}-resume"
        # train.py resumes from the newest checkpoint inside a directory, so give it one
        # that holds only checkpoint-20 (a symlink; nothing is copied or touched).
        mkdir -p "${OUTPUT_DIR}/${EXP_BASE}-train-ck20"
        ln -sfn "../${EXP_BASE}-train/checkpoint-20" "${OUTPUT_DIR}/${EXP_BASE}-train-ck20/checkpoint-20"
        COMMON+=(--trainer.resume_from_checkpoint "/workspace/outputs/${EXP_BASE}-train-ck20") ;;
*) echo "unknown phase '$PHASE' (expected train|resume)" >&2; exit 1 ;;
esac

# Created on the host, but the variable carries the CONTAINER path. One line per
# batch, with the step the worker saw when it formed the batch.
mkdir -p "${OUTPUT_DIR}/${EXP}-cut_ids"
export MELT_DEBUG_CUT_IDS_DIR="/workspace/outputs/${EXP}-cut_ids"
export MELT_DEBUG_CUT_IDS_MAX_BATCHES=1000
export MELT_DEBUG_CUT_IDS_EVERY=1

infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
    "${COMMON[@]}" \
    --trainer.output_dir "/workspace/outputs/${EXP}" \
    --run.exp_name "$EXP"
