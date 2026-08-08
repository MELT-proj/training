#!/bin/bash
#
# Test 3 — long mixed-task soak, 2 nodes x 4 GPUs.
#
# Three phases, submitted one at a time so a config error cannot burn the big
# allocation:
#   smoke   40 min    5.3 GPUh   catches config errors, exercises eval_on_start
#   legA    20 h    160   GPUh   train from scratch
#   legB    20 h    160   GPUh   resume from legA's last checkpoint
#
# legB is not just more training: resuming at full scale after 20 h is the
# soak test's resume check, and it keeps each job well inside the 3-day wall.
#
#   run_test3_mn5.sh smoke
#   run_test3_mn5.sh legA
#   run_test3_mn5.sh legB
#
# BUDGET: the campaign cap is 500 GPUh. Tests 0-2 spend ~20; these three spend
# ~325. Check `sacct` before resubmitting anything.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PHASE="${1:?usage: $0 <smoke|legA|legB>}"
EXP_BASE="${EXP_BASE:-lhotse2-test3-soak}"

export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126_lhotse2_td.sif}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"
export MELT_NODES=2

CONFIG=tests/integration/lhotse2_campaign/test3_mixed_task_soak.yaml
COMMON=(--config "$CONFIG")

case "$PHASE" in
smoke)
    EXP="${EXP_BASE}-smoke"
    export MELT_QOS="${MELT_QOS:-acc_debug}"
    export MELT_TIME="${MELT_TIME:-00:40:00}"
    # Enough steps to cross one eval and one save, no more.
    # The sampler fills buffer_size before emitting a first batch; the
    # production 300k does not fit in a 40-minute smoke. Nothing this phase
    # checks (eval-set wiring, ST tags, templates) depends on buffer size.
    COMMON+=(--trainer.max_steps 30
             --trainer.eval_steps 15
             --trainer.save_steps 15
             --trainer.save_total_limit 1
             --trainer.logging_steps 1
             --trainer.report_to none
             --data.train_ds.buffer_size 20000)
    # Cut-id logging on for the smoke only: it confirms 16 disjoint streams
    # before the long legs, and costs nothing at 30 steps.
    # Created on the HOST, but the variable must carry the CONTAINER path:
    # it is forwarded verbatim and only OUTPUT_DIR is bound. Not under
    # output_dir either -- HF refuses a non-empty output_dir.
    mkdir -p "${OUTPUT_DIR}/${EXP}-cut_ids"
    export MELT_DEBUG_CUT_IDS_DIR="/workspace/outputs/${EXP}-cut_ids"
    export MELT_DEBUG_CUT_IDS_MAX_BATCHES=200
    export MELT_DEBUG_CUT_IDS_EVERY=1
    ;;
legA)
    EXP="${EXP_BASE}-legA"
    export MELT_QOS="${MELT_QOS:-acc_ehpc}"
    export MELT_TIME="${MELT_TIME:-20:00:00}"
    COMMON+=(--trainer.report_to wandb)
    ;;
legB)
    EXP="${EXP_BASE}-legB"
    export MELT_QOS="${MELT_QOS:-acc_ehpc}"
    export MELT_TIME="${MELT_TIME:-20:00:00}"
    # Resume from legA's run dir; HF picks its highest checkpoint. The sampler
    # refuses to restore under a changed (world_size, num_workers), so legB
    # must keep 2 nodes x 4 GPUs x 2 workers.
    COMMON+=(--trainer.report_to wandb
             --trainer.resume_from_checkpoint "/workspace/outputs/${EXP_BASE}-legA")
    ;;
*)
    echo "unknown phase '$PHASE' (expected smoke|legA|legB)" >&2
    exit 1
    ;;
esac

echo "[test3:$PHASE] exp=$EXP nodes=$MELT_NODES qos=$MELT_QOS time=$MELT_TIME"
echo "[test3:$PHASE] data=$LOCAL_DATASETS_DIR"
GPUH=$(python3 -c "
h, m, s = '${MELT_TIME}'.split(':')
print(f'{(int(h) + int(m)/60 + int(s)/3600) * 4 * ${MELT_NODES}:.1f}')")
echo "[test3:$PHASE] this submission bills ~${GPUH} GPUh of the 500 cap"

infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
    "${COMMON[@]}" \
    --trainer.output_dir "/workspace/outputs/${EXP}" \
    --run.exp_name "$EXP"
