#!/bin/bash
#
# Test 1 — starved partitions. 1 node x 4 GPUs, ~20 min of work.
#
# Submits a 20-step run over a mixture that includes a 3-cut source, with cut-id
# logging on, then leaves check_test1.py to read the dump.
#
#   tests/integration/lhotse2_campaign/run_test1_mn5.sh
#
# Cost: 4 GPUh (1h requested x 4 GPUs). SLURM bills the wall you ASK for.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXP="${EXP:-lhotse2-test1-starved-$(date -u +%Y%m%d-%H%M)}"

# The indexed collection. The shared shar/ tree has no .idx and must stay that
# way — lhotse 1.32 consumers still read it.
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"
export SINGULARITY_IMG="${SINGULARITY_IMG:-/gpfs/scratch/epor48/melt_cuda126_lhotse2_td.sif}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/outputs}"

# acc_debug: 2h cap but priority 10000 vs 100, so it schedules almost at once.
export MELT_QOS="${MELT_QOS:-acc_debug}"
export MELT_TIME="${MELT_TIME:-00:30:00}"

# NOT under the trainer's output_dir: HF refuses to start when that directory
# already exists and is non-empty, and this has to be created before submit.
#
# The directory is created on the HOST but the variable must carry the
# CONTAINER path -- it is forwarded verbatim through SINGULARITYENV_*, and only
# OUTPUT_DIR is bound (to /workspace/outputs).
CUT_IDS_HOST="${OUTPUT_DIR}/${EXP}-cut_ids"
mkdir -p "$CUT_IDS_HOST"
export MELT_DEBUG_CUT_IDS_DIR="/workspace/outputs/${EXP}-cut_ids"
export MELT_DEBUG_CUT_IDS_MAX_BATCHES=400
export MELT_DEBUG_CUT_IDS_EVERY=1

echo "[test1] exp=$EXP"
echo "[test1] data=$LOCAL_DATASETS_DIR"
echo "[test1] cut ids -> $CUT_IDS_HOST (container: $MELT_DEBUG_CUT_IDS_DIR)"

infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
    --config tests/integration/lhotse2_campaign/test1_starved_partitions.yaml \
    --trainer.output_dir "/workspace/outputs/${EXP}" \
    --run.exp_name "$EXP"

echo
echo "[test1] when it finishes:"
echo "  python tests/integration/lhotse2_campaign/check_test1.py \\"
echo "      --cut-ids-dir $CUT_IDS_HOST \\"
echo "      --shar-root $LOCAL_DATASETS_DIR"
