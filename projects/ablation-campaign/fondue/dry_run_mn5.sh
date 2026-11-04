#!/bin/bash
#
# Fondue dry run on MN5 acc_debug: 8 nodes (the queue's cap), the rendered configs as
# they will run, cut short with run.stop_after_step. Three jobs, one at a time
# (acc_debug takes one submission per user):
#
#   dry_run_mn5.sh fresh    Fondue.yaml from scratch: start-up on the full pool,
#                           InfiniBand (NCCL_DEBUG=INFO), s/step, evals at 0/15/30,
#                           checkpoints at 15 and 30, stops at 30.
#   dry_run_mn5.sh resume   the production chain: same config, same output dir,
#                           resumes the newest checkpoint (30) with the deferred
#                           optimizer load, stops at 45.
#   dry_run_mn5.sh anneal   the phase switch: Fondue-anneal.yaml from fresh's
#                           checkpoint-30 with run.restart_data_stream, its own
#                           output dir and another MELT_SEED, stops at 45.
#
# Run from a clone that no other job reads. What to read in the logs is in README.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PHASE="${1:?usage: $0 <fresh|resume|anneal>}"
EXP_BASE="${EXP_BASE:-fondue-dry}"
HERE=projects/ablation-campaign/fondue

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export LOCAL_ASR_SCORES_DIR="${LOCAL_ASR_SCORES_DIR:-/gpfs/projects/epor48/melt-data/shar-asr-scores}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES="${MELT_NODES:-8}"
export MELT_GPUS_PER_NODE=4
export MELT_QOS="${MELT_QOS:-acc_debug}"
export MELT_TIME="${MELT_TIME:-01:30:00}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

COMMON=(--trainer.save_steps 15 --trainer.eval_steps 15 --trainer.logging_steps 1
        --trainer.report_to none --trainer.save_total_limit 3)

case "$PHASE" in
fresh)
    CONFIG="$HERE/Fondue.yaml"; EXP="${EXP_BASE}"
    COMMON+=(--run.stop_after_step 30) ;;
resume)
    CONFIG="$HERE/Fondue.yaml"; EXP="${EXP_BASE}"
    [[ -d "${OUTPUT_DIR}/${EXP}/checkpoint-30" ]] || { echo "run 'fresh' first: no ${OUTPUT_DIR}/${EXP}/checkpoint-30" >&2; exit 1; }
    COMMON+=(--run.stop_after_step 45 --trainer.resume_from_checkpoint true) ;;
anneal)
    CONFIG="$HERE/Fondue-anneal.yaml"; EXP="${EXP_BASE}-anneal"
    export MELT_SEED="${MELT_SEED:-43}"
    [[ -d "${OUTPUT_DIR}/${EXP_BASE}/checkpoint-30" ]] || { echo "run 'fresh' first" >&2; exit 1; }
    # train.py resumes the newest checkpoint of a directory: give it one holding only
    # checkpoint-30 (a symlink; nothing is copied, and resume's later ones are ignored).
    mkdir -p "${OUTPUT_DIR}/${EXP_BASE}-ck30"
    ln -sfn "../${EXP_BASE}/checkpoint-30" "${OUTPUT_DIR}/${EXP_BASE}-ck30/checkpoint-30"
    COMMON+=(--run.stop_after_step 45 --trainer.resume_from_checkpoint "/workspace/outputs/${EXP_BASE}-ck30") ;;
*) echo "unknown phase '$PHASE' (expected fresh|resume|anneal)" >&2; exit 1 ;;
esac

infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config "$CONFIG" "${COMMON[@]}" \
    --trainer.output_dir "/workspace/outputs/${EXP}" \
    --run.exp_name "${EXP}-${PHASE}"
