#!/bin/bash
#
# Pull a finished SD24 / CE24 run (or an alignment start) from MN5 to nyx and score it on the FLEURS-24 dev sets
# (X->en, en->X, ASR) with melt-eval on artemis A6000s, under both prompt formats:
#
#   fmt-repeatfirst   the SD24 / CE24 training instructions ("Repeat the following content ...", "Translate the
#                     following content into X ...")
#   fmt-sft-random    the random task templates the SFT-EU24 reference was trained and scored with
#
#   bash score_sd24_run.sh <run name under the MN5 outputs> [--pull-only | --score-only] [--formats "repeatfirst sft-random"]
#
# The run root must be finished: the pull refuses a run without training_config.yaml, the last file of the final
# save. The weights go to DEST_ROOT/<run>/ on nyx scratch, which artemis mounts. Jobs are tagged
# sft-score:<run>:fmt-<format>:<set> (the same melt-eval tag as the SFT scoring), and the launcher stops with
# exit 3 when a QOS is full: rerun with --score-only and the missing formats once the earlier jobs drain.
#
# Environment: MN5_HOST (mn5transfer), MN5_OUTPUTS (/gpfs/scratch/epor48/outputs), DEST_ROOT
# (/mnt/scratch-nyx/giuseppe/melt/outputs-sd24), SCORING (the SFT scoring directory with bin/launch_eval_sft.sh and
# the format files), SSH_HOST (artemis).
set -euo pipefail

RUN="${1:?usage: $0 <run name> [--pull-only | --score-only] [--formats \"repeatfirst sft-random\"]}"
shift
MODE=both
FORMATS="repeatfirst sft-random"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull-only) MODE=pull ;;
        --score-only) MODE=score ;;
        --formats) FORMATS="${2:?}"; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

MN5_HOST="${MN5_HOST:-mn5transfer}"
MN5_OUT="${MN5_OUTPUTS:-/gpfs/scratch/epor48/outputs}"
DEST_ROOT="${DEST_ROOT:-/mnt/scratch-nyx/giuseppe/melt/outputs-sd24}"
SCORING="${SCORING:-/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring}"
SSH_HOST="${SSH_HOST:-artemis}"
DEST="${DEST_ROOT}/${RUN}"
SMALL_FILES=(config.json generation_config.json chat_template.jinja processor_config.json tokenizer.json tokenizer_config.json resolved_config.json)

if [[ "$MODE" != score ]]; then
    ssh -o BatchMode=yes "$MN5_HOST" "test -f ${MN5_OUT}/${RUN}/training_config.yaml" 2>/dev/null \
        || { echo "ERROR: ${RUN} has no training_config.yaml on MN5 yet: the run is not finished" >&2; exit 1; }
    mkdir -p "$DEST"
    rsync -a --partial --inplace "${MN5_HOST}:${MN5_OUT}/${RUN}/model.safetensors" "${DEST}/"
    for f in "${SMALL_FILES[@]}"; do rsync -a "${MN5_HOST}:${MN5_OUT}/${RUN}/${f}" "${DEST}/"; done
    rsync -a "${MN5_HOST}:${MN5_OUT}/${RUN}/training_config.yaml" "${DEST}/"
    echo "[score_sd24_run] pulled ${RUN} to ${DEST}"
fi

if [[ "$MODE" != pull ]]; then
    [[ -f "${DEST}/training_config.yaml" ]] || { echo "ERROR: ${DEST} is not a pulled run" >&2; exit 1; }
    for fmt in $FORMATS; do
        format_file="${SCORING}/format-${fmt}.yaml"
        [[ -f "$format_file" ]] || { echo "ERROR: no ${format_file}" >&2; exit 1; }
        ssh "$SSH_HOST" "env CAP_ST=4 CAP_ASR=3 FORMAT_FILE=${format_file} bash ${SCORING}/bin/launch_eval_sft.sh ${DEST} ${DEST} ${RUN}:fmt-${fmt}" \
            || { code=$?; echo "[score_sd24_run] the launcher stopped (exit ${code}) at format ${fmt}: rerun with --score-only --formats \"${fmt}\"" >&2; exit "$code"; }
    done
fi
