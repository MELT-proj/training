#!/bin/bash
#
# Test 2 — resume across two nodes. 8 ranks x num_workers 2 = 16 streams.
#
# Every resume proof so far has been single-node, so world_size has never
# varied in `shard_id = rank * num_workers + worker_id` /
# `num_shards = world_size * num_workers`.
#
# Four phases, each driven through infra/runners/submit-container.sh — the same
# path Test 1 validated, rather than a second copy of the container launcher:
#
#   run_test2_mn5.sh run1      train to step 10, saving at 5 and 10
#   run_test2_mn5.sh prune     drop checkpoint-10 so the run dir resolves to 5
#   run_test2_mn5.sh run2      resume from the run dir and continue to 10
#   run_test2_mn5.sh compare   cut IDs: resume equality + disjointness
#
# `prune` is needed because resume_from_checkpoint is always scanned as a
# PARENT directory (train.py calls get_last_checkpoint on it), so it cannot be
# pointed straight at checkpoint-5.
#
# Cost: 2 nodes x 4 GPUs, 40 min requested per training phase = ~10.7 GPUh.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PHASE="${1:?usage: $0 <run1|prune|run2|compare>}"

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126_lhotse2_td.sif}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES=2
export MELT_QOS="${MELT_QOS:-acc_debug}"
export MELT_TIME="${MELT_TIME:-00:40:00}"

EXP="${EXP:-lhotse2-test2-2node}"
CONFIG=tests/integration/lhotse2_campaign/test2_2node_resume.yaml
MAX_STEPS="${MAX_STEPS:-10}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5}"
GRAD_ACC="${GRAD_ACC:-4}"

SING="${SINGULARITY_BIN:-/apps/GPP/SINGULARITY/3.11.5/bin/singularity}"

submit() {
    local run="$1"; shift
    # Host dir created here; the variable must carry the CONTAINER path, since
    # it is forwarded verbatim and only OUTPUT_DIR is bound.
    mkdir -p "${OUTPUT_DIR}/${EXP}-${run}-cut_ids"
    export MELT_DEBUG_CUT_IDS_DIR="/workspace/outputs/${EXP}-${run}-cut_ids"
    export MELT_DEBUG_CUT_IDS_MAX_BATCHES=$((MAX_STEPS * GRAD_ACC * 4))
    export MELT_DEBUG_CUT_IDS_EVERY=1

    echo "[test2:$run] nodes=2 gpus/node=4 -> 8 ranks x 2 workers = 16 streams"
    echo "[test2:$run] cut ids -> ${OUTPUT_DIR}/${EXP}-${run}-cut_ids"
    infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
        --config "$CONFIG" \
        --trainer.output_dir "/workspace/outputs/${EXP}-${run}" \
        --trainer.gradient_accumulation_steps "$GRAD_ACC" \
        --run.exp_name "${EXP}-${run}" \
        "$@"
}

case "$PHASE" in
run1)
    submit run1 \
        --trainer.max_steps "$MAX_STEPS" \
        --trainer.save_steps "$CHECKPOINT_STEP" \
        --trainer.save_total_limit 2
    ;;
prune)
    RUN1="${OUTPUT_DIR}/${EXP}-run1"
    echo "[test2:prune] checkpoints in $RUN1:"
    ls -d "$RUN1"/checkpoint-* 2>/dev/null || { echo "  none — did run1 finish?"; exit 1; }
    for ckpt in "$RUN1"/checkpoint-*; do
        [[ -d "$ckpt" ]] || continue
        if [[ "$(basename "$ckpt")" != "checkpoint-${CHECKPOINT_STEP}" ]]; then
            echo "  removing $(basename "$ckpt")"
            rm -rf "$ckpt"
        else
            echo "  keeping  $(basename "$ckpt")"
        fi
    done
    echo "[test2:prune] sampler state present for:"
    ls "$RUN1/checkpoint-${CHECKPOINT_STEP}/sampler/" 2>/dev/null | tr '\n' ' '
    echo
    ;;
run2)
    submit run2 \
        --trainer.max_steps "$MAX_STEPS" \
        --trainer.save_strategy '"no"' \
        --trainer.resume_from_checkpoint "/workspace/outputs/${EXP}-run1"
    ;;
compare)
    # In the container: compare_cut_ids.py uses builtin generics, which the
    # login node's python3.9 cannot parse.
    "$SING" exec --bind /gpfs:/gpfs "$SINGULARITY_IMG" bash -c "
        source /workspace/venv/bin/activate
        cd $(pwd)
        python tests/integration/sampler_resume/compare_cut_ids.py \
            --run1-dir ${OUTPUT_DIR}/${EXP}-run1-cut_ids \
            --run2-dir ${OUTPUT_DIR}/${EXP}-run2-cut_ids \
            --checkpoint-step ${CHECKPOINT_STEP} \
            --grad-accum-steps ${GRAD_ACC}"
    ;;
*)
    echo "unknown phase '$PHASE' (expected run1|prune|run2|compare)" >&2
    exit 1
    ;;
esac
