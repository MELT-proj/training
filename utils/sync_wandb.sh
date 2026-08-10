#!/bin/bash
#
# Pull offline W&B runs off a remote cluster and sync them to wandb.ai.
#
# MN5 compute nodes have no outbound internet, so runs are written offline to
# $OUTPUT_DIR/wandb/wandb/ and have to be copied somewhere with network access
# (artemis) before `wandb sync` can upload them.
#
# Runs still being written to are synced with --append and left unmarked, so a
# later invocation picks up the rest; finished runs are marked .synced and are
# skipped from then on. That makes the script safe to run mid-training.
#
# Usage:
#   utils/sync_wandb.sh --remote-path <path> [options]
#
# Options (each also settable by environment variable; the flag wins):
#   -r, --remote-path PATH   $WANDB_REMOTE_PATH   remote wandb/wandb directory. REQUIRED.
#   -l, --local-path PATH    $WANDB_LOCAL_PATH    where to mirror it locally.
#                                                 default: /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb
#   -H, --host HOST          $WANDB_REMOTE_HOST   ssh alias of the cluster. default: mn5
#   -v, --venv PATH          $VENV_PATH           virtualenv *activate script* to source.
#                                                 default: none (use the current environment)
#   -t, --threshold MINUTES  $ACTIVE_THRESHOLD_MINUTES
#                                                 a run touched within this many minutes counts
#                                                 as still training. default: 10
#   -n, --dry-run                                 show what would transfer and sync; change nothing
#   -h, --help                                    this message
#
# Examples:
#   # your own runs on MN5 (see docs/hpc_runbook.md B3 for the layout)
#   utils/sync_wandb.sh -r /gpfs/scratch/epor48/$USER/outputs/wandb/wandb
#
#   # a colleague's account, into your own local mirror
#   utils/sync_wandb.sh -r /gpfs/scratch/epor48/<their-mn5-user>/outputs/wandb/wandb \
#                       -l /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb
#
#   # if wandb lives in a venv rather than on PATH
#   utils/sync_wandb.sh -r ... -v /mnt/scratch-artemis/$USER/venvs/melt-312/bin/activate
#
# The remote username is usually NOT your local one. To find it:
#   ssh mn5 'echo $USER'

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
# Print the header comment block (everything from line 3 up to the first line
# that is not a comment), with the leading '#' stripped.
usage() { awk 'NR<3 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"; }

# --- configuration: env vars as defaults, flags override -------------------
REMOTE_HOST="${WANDB_REMOTE_HOST:-${REMOTE_HOST:-mn5}}"
REMOTE_PATH="${WANDB_REMOTE_PATH:-}"
LOCAL_PATH="${WANDB_LOCAL_PATH:-/mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb}"
VENV_PATH="${VENV_PATH:-}"
ACTIVE_THRESHOLD_MINUTES="${ACTIVE_THRESHOLD_MINUTES:-10}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--remote-path) REMOTE_PATH="${2:?--remote-path needs a value}"; shift 2 ;;
        -l|--local-path)  LOCAL_PATH="${2:?--local-path needs a value}";   shift 2 ;;
        -H|--host)        REMOTE_HOST="${2:?--host needs a value}";        shift 2 ;;
        -v|--venv)        VENV_PATH="${2:?--venv needs a value}";          shift 2 ;;
        -t|--threshold)   ACTIVE_THRESHOLD_MINUTES="${2:?--threshold needs a value}"; shift 2 ;;
        -n|--dry-run)     DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

if [[ -z "$REMOTE_PATH" ]]; then
    usage
    echo
    die "--remote-path is required (or set \$WANDB_REMOTE_PATH).
       There is no safe default: outputs are per-account, so the previous
       hardcoded path was wrong for everyone but its author.
       Find yours with:  ssh $REMOTE_HOST 'echo \$USER'"
fi

[[ "$ACTIVE_THRESHOLD_MINUTES" =~ ^[0-9]+$ ]] || die "--threshold must be a whole number of minutes, got '$ACTIVE_THRESHOLD_MINUTES'"

# Normalise: hold both without a trailing slash, add one only where rsync
# needs it. The old version depended on the caller remembering to include it.
REMOTE_PATH="${REMOTE_PATH%/}"
LOCAL_PATH="${LOCAL_PATH%/}"

echo "remote:    $REMOTE_HOST:$REMOTE_PATH"
echo "local:     $LOCAL_PATH"
echo "active if touched within: ${ACTIVE_THRESHOLD_MINUTES}m"
[[ $DRY_RUN -eq 1 ]] && echo "mode:      DRY RUN (nothing will be written or uploaded)"
echo

ssh "$REMOTE_HOST" "[ -d '$REMOTE_PATH' ]" \
    || die "remote directory not found: $REMOTE_HOST:$REMOTE_PATH
       Check the path and the account it belongs to."

[[ $DRY_RUN -eq 1 ]] || mkdir -p "$LOCAL_PATH"

echo "--- 1. Checking remote for active runs ---"

# Which runs are still being written to? A run whose .wandb file was touched
# recently is mid-training, and must be appended to rather than finalised.
ACTIVE_RUNS=$(ssh "$REMOTE_HOST" \
    "find '$REMOTE_PATH'/offline-run-* -type f -name '*.wandb' -mmin -$ACTIVE_THRESHOLD_MINUTES -printf '%h\n' 2>/dev/null | sort -u | xargs -n1 basename 2>/dev/null" \
    || echo "")

if [ -n "$ACTIVE_RUNS" ]; then
    echo "Active runs detected on remote:"
    echo "$ACTIVE_RUNS" | sed 's/^/  - /'
else
    echo "No active runs detected on remote"
fi

echo ""
echo "--- 2. Syncing files from remote cluster ---"

# Exclude .synced marker files to preserve local sync state
RSYNC_FLAGS=(-ravh --append-verify --exclude='.synced' -e ssh)
[[ $DRY_RUN -eq 1 ]] && RSYNC_FLAGS+=(--dry-run)
rsync "${RSYNC_FLAGS[@]}" "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"

echo ""
echo "--- 3. Activating virtual environment ---"
if [[ -n "$VENV_PATH" ]]; then
    # Repo convention (bash/run_train.sh, sites/*.sh): VENV_PATH *is* the
    # activate script, not the venv root.
    [[ -f "$VENV_PATH" ]] || die "VENV_PATH is not a file: $VENV_PATH
       It must point at the activate script itself, e.g. /path/to/venv/bin/activate"
    # shellcheck disable=SC1090
    source "$VENV_PATH"
    echo "activated: $VENV_PATH"
else
    echo "no --venv given; using the current environment"
fi

if ! command -v wandb >/dev/null 2>&1; then
    MSG="'wandb' not found on PATH. Pass --venv /path/to/venv/bin/activate, or activate an environment that has it."
    # Under --dry-run this is worth knowing but not worth stopping for: the
    # point of a preview is to see the classification without uploading.
    [[ $DRY_RUN -eq 1 ]] && echo "WARNING: $MSG" || die "$MSG"
fi

echo ""
echo "--- 4. Processing Offline Runs ---"

# Counter for summary
SKIPPED=0
ACTIVE_SYNCED=0
FINISHED_SYNCED=0
FAILED=0

for run_dir in "$LOCAL_PATH"/offline-run-*/; do
    # Check if the directory actually exists (handles empty glob)
    [ -d "$run_dir" ] || continue

    run_name=$(basename "$run_dir")

    # Skip if already marked as synced
    if [ -f "$run_dir/.synced" ]; then
        echo ">> Skipping already synced run: $run_name"
        ((SKIPPED++)) || true
        continue
    fi

    # Check for recent local activity (file modified in last N minutes)
    RECENT_LOCAL_ACTIVITY=$(find "$run_dir" -type f -name '*.wandb' -mmin -"$ACTIVE_THRESHOLD_MINUTES" 2>/dev/null)

    # Check if this run is in the active runs list from remote OR has recent local activity
    if echo "$ACTIVE_RUNS" | grep -q "^$run_name$" || [ -n "$RECENT_LOCAL_ACTIVITY" ]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo ">> [dry run] would sync ACTIVE run (appending): $run_name"
            ((ACTIVE_SYNCED++)) || true
        else
            echo ">> Syncing ACTIVE run (appending): $run_name"
            # Don't let wandb sync failure stop the script
            if wandb sync "$run_dir" --include-offline --append; then
                ((ACTIVE_SYNCED++)) || true
            else
                echo "WARNING: Failed to sync $run_name"
                ((FAILED++)) || true
            fi
        fi
    else
        if [[ $DRY_RUN -eq 1 ]]; then
            echo ">> [dry run] would sync FINISHED run (finalizing): $run_name"
            ((FINISHED_SYNCED++)) || true
        else
            echo ">> Syncing FINISHED run (finalizing): $run_name"
            # Don't let wandb sync failure stop the script
            if wandb sync "$run_dir" --include-offline --mark-synced; then
                # Create .synced marker file to prevent re-syncing
                touch "$run_dir/.synced"
                ((FINISHED_SYNCED++)) || true
            else
                echo "WARNING: Failed to sync $run_name"
                ((FAILED++)) || true
            fi
        fi
    fi

    echo ""
done

echo "--- Sync Process Complete ---"
echo "Summary:"
echo "  - Skipped (already synced): $SKIPPED"
echo "  - Active runs synced: $ACTIVE_SYNCED"
echo "  - Finished runs synced: $FINISHED_SYNCED"
echo "  - Failed: $FAILED"
