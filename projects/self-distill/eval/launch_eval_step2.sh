#!/bin/bash
#
# Score one finished run on the step-2 battery with melt-eval, one job per set:
#
#   sd-dev-per-source    in-domain ASR: en/de/fr/es/it x CV22/MLS/VoxPopuli, 300 each
#   fleurs24-asr-dev     zero-shot ASR: 24 languages x 100, 19 never trained on
#   fleurs24-st-xen-dev  zero-shot ST: 23 languages -> English x 100, translate instruction
#
# FORMAT picks the prompt file next to this script: repeatfirst (default, the
# step-2 arms) or bare (GOLDW). The model directory must be a finished run's root
# (it carries the processor and tokenizer). Run on artemis; the melt package
# comes from this checkout, not from the melt-eval venv's stale copy.
#
#   FORMAT=bare bash projects/self-distill/eval/launch_eval_step2.sh /path/to/outputs/<GOLDW run>
#   bash projects/self-distill/eval/launch_eval_step2.sh /path/to/outputs/<S2 run> fleurs24-st-xen-dev
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING="$(cd "${HERE}/../../.." && pwd)"
MODEL="${1:?usage: $0 <run dir> [frozen set ...]}"; shift
if [[ $# -gt 0 ]]; then SETS=("$@"); else SETS=(sd-dev-per-source fleurs24-asr-dev fleurs24-st-xen-dev); fi
FORMAT_FILE="${HERE}/format-${FORMAT:-repeatfirst}.yaml"
EVAL_SETS="${EVAL_SETS:-/mnt/scratch-artemis/giuseppe/melt-data/eval-sets}"
MELT_EVAL="${MELT_EVAL:-${HOME}/melt-proj/melt-eval}"

[[ -f "$FORMAT_FILE" ]] || { echo "ERROR: no $FORMAT_FILE" >&2; exit 1; }
for f in model.safetensors tokenizer.json processor_config.json; do
    [[ -f "${MODEL}/${f}" ]] || { echo "ERROR: ${MODEL} has no ${f}; pass a finished run's root" >&2; exit 1; }
done

export PYTHONPATH="${TRAINING}:${MELT_EVAL}"
export MELT_PARTITION="${MELT_PARTITION:-h100}" MELT_QOS="${MELT_QOS:-gpu-h100}" MELT_TIME="${MELT_TIME:-04:00:00}"
cd "$MELT_EVAL"
for set in "${SETS[@]}"; do
    case "$set" in *-st-*) task=st ;; *) task=asr ;; esac
    [[ -f "${EVAL_SETS}/${set}/frozen_set.json" ]] || { echo "ERROR: no frozen set ${EVAL_SETS}/${set}" >&2; exit 1; }
    infra/runners/submit_eval.sh artemis "$MODEL" "${EVAL_SETS}/${set}" \
        -T task_filter="$task" -T prompt_style=melt -T format_config="$FORMAT_FILE" \
        -M batch_size="${BATCH_SIZE:-16}"
done
