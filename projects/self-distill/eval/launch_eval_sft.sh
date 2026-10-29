#!/bin/bash
#
# Score one SFT model directory with melt-eval on the FLEURS battery, one A6000 job per set.
# Run on artemis (the watcher streams this file there over ssh):
#
#   bash launch_eval_sft.sh <model dir> <processor dir> <tag> [frozen set ...]
#
#   <model dir>      a run's checkpoint-N (model.safetensors + config.json, fp32) or a finished run root
#   <processor dir>  a directory with the tokenizer and processor files: the run's warm start. A
#                    checkpoint-N carries neither; they go to melt-eval as -M processor= and -T tokenizer=
#   <tag>            run and step, e.g. SFT-HR5-from-goldw-bd60-ga16-2g-s1337:ckpt718 or ...:final.
#                    It becomes the SLURM comment (sft-score:<tag>:<set>, shown by squeue while the job lives and kept
#                    by sacct once it ends), the stdout file name, and the eval log's metadata (sft_run, sft_label)
#   frozen set       default: fleurs24-st-xen-dev fleurs24-st-enx-dev fleurs24-asr-dev.
#                    sd-dev-per-source (in-domain ASR) is only for the final model: pass it by name
#
# Prompts come from format-sft-random.yaml (the instruction is drawn per sample from TASK_TEMPLATES,
# seeded by the sample, so every checkpoint sees the same one). Weights load as bfloat16 with
# flash attention, as the step-0 evaluation did: the SFT checkpoints are fp32 and melt-eval casts them
# at load. Both are passed explicitly so each eval log records them under model_args.
# fleurs24-st-enx-dev runs with --no-score (melt-eval's pooled corpus_bleu refuses several target
# languages after the whole generation); summarize_evals.py scores it per target.
#
# Environment (defaults in brackets):
#   TRAINING     the melt checkout melt-eval imports [training-sdw, the clone the step-0 battery used]
#   MELT_EVAL    melt-eval checkout [~/melt-proj/melt-eval]
#   FORMAT_FILE  prompt format [scoring/format-sft-random.yaml next to the SFT outputs]
#   SLURM_LOG_DIR  where the jobs' stdout goes [scoring/slurm]
#   TIME_LIMIT   per-job wall time [02:00:00]; BATCH_SIZE [16]
#   QOS_ST, QOS_ASR  [gpu-short, gpu-medium]; CAP_ST, CAP_ASR  most jobs of this user allowed in that QOS,
#                running plus pending, counting everyone's submissions [4, 3: the QOS limits]
#   DRY_RUN=1    print the sbatch commands only
#
# Prints one line per set: "SUBMITTED <comment> <job id>", or "QOS_FULL <qos> <jobs>/<cap>" (exit 3) when
# the QOS is full, in which case nothing is submitted for that set and later sets are not tried.
set -euo pipefail

MODEL="${1:?usage: $0 <model dir> <processor dir> <tag> [frozen set ...]}"
PROCESSOR="${2:?processor dir}"
TAG="${3:?tag}"
shift 3
if [[ $# -gt 0 ]]; then SETS=("$@"); else SETS=(fleurs24-st-xen-dev fleurs24-st-enx-dev fleurs24-asr-dev); fi

SCORING="${SCORING:-/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring}"
TRAINING="${TRAINING:-/mnt/scratch-artemis/giuseppe/melt-proj/training-sdw}"
MELT_EVAL="${MELT_EVAL:-${HOME}/melt-proj/melt-eval}"
EVAL_SETS="${EVAL_SETS:-/mnt/scratch-artemis/giuseppe/melt-data/eval-sets}"
FORMAT_FILE="${FORMAT_FILE:-${SCORING}/format-sft-random.yaml}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCORING}/slurm}"
QOS_ST="${QOS_ST:-gpu-short}"; QOS_ASR="${QOS_ASR:-gpu-medium}"
CAP_ST="${CAP_ST:-4}"; CAP_ASR="${CAP_ASR:-3}"

die() { echo "ERROR: $*" >&2; exit 1; }

RUN_NAME="${TAG%%:*}"; STEP_LABEL="${TAG#*:}"; [[ "$STEP_LABEL" != "$TAG" ]] || STEP_LABEL=""

[[ -f "$FORMAT_FILE" ]] || die "no $FORMAT_FILE"
for f in model.safetensors config.json; do
    [[ -f "${MODEL}/${f}" ]] || die "${MODEL} has no ${f}"
done
for f in tokenizer.json processor_config.json; do
    [[ -f "${PROCESSOR}/${f}" ]] || die "processor dir ${PROCESSOR} has no ${f}"
done
[[ -d "$MELT_EVAL" ]] || die "no melt-eval checkout at ${MELT_EVAL}"
[[ -d "${TRAINING}/melt" ]] || die "no melt package under ${TRAINING}"

# The melt-eval venv's own copy of melt is stale: the training clone goes first.
export PYTHONPATH="${TRAINING}:${MELT_EVAL}"
export MELT_PARTITION=a6000 MELT_TIME="${TIME_LIMIT:-02:00:00}"
cd "$MELT_EVAL"
echo "[launch_eval_sft] ${TAG}: model ${MODEL}, processor ${PROCESSOR}"
echo "[launch_eval_sft] melt ${TRAINING} @ $(git -C "$TRAINING" rev-parse --short HEAD 2>/dev/null || echo '?'), melt-eval @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

jobs_in_qos() {  # <qos>: this user's jobs in that QOS, running plus pending
    squeue -h -u "$(id -un)" -o '%q' | grep -cx "$1" || true
}

for set in "${SETS[@]}"; do
    case "$set" in *-st-*) task=st; qos="$QOS_ST"; cap="$CAP_ST" ;; *) task=asr; qos="$QOS_ASR"; cap="$CAP_ASR" ;; esac
    case "$set" in *-st-enx-*) noscore=(--no-score) ;; *) noscore=() ;; esac
    [[ -f "${EVAL_SETS}/${set}/frozen_set.json" ]] || die "no frozen set ${EVAL_SETS}/${set}"
    comment="sft-score:${TAG}:${set}"
    stdout="${SLURM_LOG_DIR}/${TAG//:/__}__${set}.%j.out"
    args=(artemis "$MODEL" "${EVAL_SETS}/${set}"
          -T task_filter="$task" -T prompt_style=melt -T format_config="$FORMAT_FILE" -T tokenizer="$PROCESSOR"
          -M processor="$PROCESSOR" -M batch_size="${BATCH_SIZE:-16}" -M dtype=bfloat16 -M attn_implementation=flash_attention_2
          --tags sft-score --metadata "sft_run=${RUN_NAME}" --metadata "sft_label=${STEP_LABEL}"
          ${noscore[@]+"${noscore[@]}"})
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "[DRY_RUN] MELT_QOS=${qos} MELT_JOB_TAG='${comment}' MELT_SLURM_OUT='${stdout}' infra/runners/submit_eval.sh ${args[*]}"
        continue
    fi
    n="$(jobs_in_qos "$qos")"
    if (( n >= cap )); then
        echo "QOS_FULL ${qos} ${n}/${cap}"
        exit 3
    fi
    out="$(MELT_QOS="$qos" MELT_JOB_TAG="$comment" MELT_SLURM_OUT="$stdout" infra/runners/submit_eval.sh "${args[@]}" 2>&1)" \
        || { echo "SUBMIT FAILED (${comment}): ${out}" >&2; exit 1; }
    job="$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | grep -oE '[0-9]+' | tail -n 1)"
    [[ -n "$job" ]] || die "sbatch printed no job id for ${comment}: ${out}"
    echo "SUBMITTED ${comment} ${job}"
done
