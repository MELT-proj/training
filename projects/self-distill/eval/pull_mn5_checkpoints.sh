#!/bin/bash
#
# Pull the scheduled checkpoints of the MN5 (EU24) SFT runs to nyx, weights only, so the watcher can score them.
# Run on nyx. MN5 has no outbound internet and its login node must stay idle: everything goes through the
# data-transfer host (ssh alias mn5transfer, about 8 MB/s per stream, so a 10 GB fp32 checkpoint takes ~20 min).
#
#   bash pull_mn5_checkpoints.sh [--dry-run] [--parallel N] [--steps "718 1436 ..."] [--runs-file FILE] [run ...]
#
# Runs: the names given on the command line, else the lines marked "mn5" in the runs file (default:
# scoring/runs.txt next to the SFT outputs, the file the watcher reads too; a line is "<run directory name> [mn5]").
# Default steps: 718 1436 2154 2872 4308 5744 8616 (1, 2, 3, 4, 6, 8 and 12 sixteenths).
#
# For each run and step it
#   1. skips the checkpoint when the nyx copy already carries .transfer_complete;
#   2. skips it (prints "not complete on MN5") unless MN5 has trainer_state.json in it, the trainer's last
#      write after the weights, so a half-saved checkpoint is never pulled; run the script again later;
#   3. rsyncs only model.safetensors (or its shards and index), config.json and generation_config.json into
#      checkpoint-<N>.partial/ (resumable: --partial --inplace, retried three times);
#   4. checks the file sizes against MN5's and the safetensors header against the file length, writes
#      .transfer_complete, and renames the directory to checkpoint-<N>. The watcher
#      (watch_sft_checkpoints.py) accepts .transfer_complete in place of trainer_state.json.
# Optimizer states, sampler states and everything else stay on MN5.
#
# Environment: MN5_HOST (mn5transfer; empty = the source is a local directory, for testing),
# MN5_OUTPUTS (/gpfs/scratch/epor48/outputs), DEST (/mnt/scratch-nyx/giuseppe/melt/outputs-sft).
set -euo pipefail

MN5_HOST="${MN5_HOST-mn5transfer}"
MN5_OUTPUTS="${MN5_OUTPUTS:-/gpfs/scratch/epor48/outputs}"
DEST="${DEST:-/mnt/scratch-nyx/giuseppe/melt/outputs-sft}"
STEPS="718 1436 2154 2872 4308 5744 8616"
PARALLEL=2
DRY_RUN=""
RUNS_FILE="${DEST}/scoring/runs.txt"
RUNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --parallel) PARALLEL="${2:?}"; shift ;;
        --steps) STEPS="${2:?}"; shift ;;
        --runs-file) RUNS_FILE="${2:?}"; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) RUNS+=("$1") ;;
    esac
    shift
done
if [[ ${#RUNS[@]} -eq 0 ]]; then
    [[ -f "$RUNS_FILE" ]] || { echo "ERROR: no runs given and no runs file ${RUNS_FILE}" >&2; exit 2; }
    # the lines whose second field is "mn5", comments stripped
    while read -r name site; do
        if [[ "$site" == "mn5" ]]; then RUNS+=("$name"); fi
    done < <(sed -e 's/#.*//' "$RUNS_FILE")
    [[ ${#RUNS[@]} -gt 0 ]] || { echo "ERROR: ${RUNS_FILE} has no run marked mn5" >&2; exit 2; }
fi

# Run a command on the MN5 side: over ssh, or here when MN5_HOST is empty.
on_mn5() {
    if [[ -n "$MN5_HOST" ]]; then ssh -o BatchMode=yes -o ConnectTimeout=30 "$MN5_HOST" "$1"; else bash -c "$1"; fi
}

# Whole-file check of a safetensors file: its length is 8 + header size + the end of its last tensor.
safetensors_whole() {
    python3 - "$1" <<'EOF'
import json, os, struct, sys
path = sys.argv[1]
with open(path, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(n))
end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
sys.exit(0 if os.path.getsize(path) == 8 + n + end else 1)
EOF
}

pull_one() {  # <run> <step>
    local run="$1" step="$2"
    local src="${MN5_OUTPUTS}/${run}/checkpoint-${step}"
    local dest="${DEST}/${run}/checkpoint-${step}" tmp="${DEST}/${run}/checkpoint-${step}.partial"
    local tag="[${run} ${step}]"
    if [[ -f "${dest}/.transfer_complete" ]]; then echo "${tag} already pulled"; return 0; fi
    # Never write into a directory this script did not create (a locally trained run's checkpoint, say).
    if [[ -e "$dest" ]]; then echo "${tag} ERROR: ${dest} exists without .transfer_complete; not touching it" >&2; return 1; fi
    if ! on_mn5 "test -f '${src}/trainer_state.json'" 2>/dev/null; then echo "${tag} not complete on MN5 (no trainer_state.json)"; return 0; fi
    # sizes of what is about to be copied, to compare afterwards
    local listing
    listing="$(on_mn5 "cd '${src}' && for f in model*.safetensors model.safetensors.index.json config.json generation_config.json; do if [ -f \"\$f\" ]; then stat -c '%s %n' \"\$f\"; fi; done")" \
        || { echo "${tag} ERROR: could not list the weights on MN5" >&2; return 1; }
    grep -q ' model.*safetensors' <<<"$listing" || { echo "${tag} ERROR: no model*.safetensors on MN5" >&2; return 1; }
    grep -q ' config.json' <<<"$listing" || { echo "${tag} ERROR: no config.json on MN5" >&2; return 1; }
    if [[ -n "$DRY_RUN" ]]; then
        echo "${tag} [DRY_RUN] would rsync $(wc -l <<<"$listing") files ($(awk '{s+=$1} END {printf "%.1f GB", s/1e9}' <<<"$listing")) ${MN5_HOST:+${MN5_HOST}:}${src}/ -> ${tmp}/"
        return 0
    fi
    mkdir -p "$tmp"
    local from="${MN5_HOST:+${MN5_HOST}:}${src}/" attempt
    for attempt in 1 2 3; do
        if rsync -a --partial --inplace --timeout=600 \
                --include='model*.safetensors' --include='model.safetensors.index.json' \
                --include='config.json' --include='generation_config.json' --exclude='*' \
                "$from" "${tmp}/"; then
            break
        fi
        echo "${tag} rsync failed (attempt ${attempt}/3)" >&2
        [[ $attempt -lt 3 ]] || return 1
        sleep 30
    done
    local size name
    while read -r size name; do
        [[ "$(stat -c %s "${tmp}/${name}")" == "$size" ]] || { echo "${tag} ERROR: ${name} is $(stat -c %s "${tmp}/${name}") bytes, MN5 has ${size}" >&2; return 1; }
    done <<<"$listing"
    for f in "${tmp}"/model*.safetensors; do
        safetensors_whole "$f" || { echo "${tag} ERROR: ${f} is shorter than its header says" >&2; return 1; }
    done
    touch "${tmp}/.transfer_complete"
    mv -T "$tmp" "$dest"
    echo "${tag} pulled to ${dest}"
}
export -f pull_one on_mn5 safetensors_whole
export MN5_HOST MN5_OUTPUTS DEST DRY_RUN

jobs=()
for run in "${RUNS[@]}"; do
    for step in $STEPS; do jobs+=("${run} ${step}"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -L 1 bash -c 'pull_one "$1" "$2"' _
