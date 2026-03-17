#!/bin/bash

# Exit on any error
set -e

# Configuration
REMOTE_HOST="mn5"
REMOTE_PATH="/gpfs/projects/epor48/melt-data/outputs/wandb/wandb/"
LOCAL_PATH="/mnt/scratch-artemis/giuseppe/melt-data/outputs/wandb/wandb/"
ACTIVE_THRESHOLD_MINUTES=10  # Consider runs active if modified in last N minutes

mkdir -p "$LOCAL_PATH"

echo "--- 1. Checking remote for active runs ---"

# Get list of run directories that have been modified recently on remote
# This identifies which runs are still actively training
ACTIVE_RUNS=$(ssh "$REMOTE_HOST" "find $REMOTE_PATH/offline-run-* -type f -name '*.wandb' -mmin -$ACTIVE_THRESHOLD_MINUTES -printf '%h\n' 2>/dev/null | sort -u | xargs -n1 basename 2>/dev/null" || echo "")

if [ -n "$ACTIVE_RUNS" ]; then
    echo "Active runs detected on remote:"
    echo "$ACTIVE_RUNS" | sed 's/^/  - /'
else
    echo "No active runs detected on remote"
fi

echo ""
echo "--- 2. Syncing files from remote cluster ---"

# Exclude .synced marker files to preserve local sync state
rsync -ravh --append-verify --exclude='.synced' -e ssh "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"

echo ""
echo "--- 3. Activating virtual environment ---"
source "$VENV_PATH/bin/activate"

echo ""
echo "--- 4. Processing Offline Runs ---"

# Counter for summary
SKIPPED=0
ACTIVE_SYNCED=0
FINISHED_SYNCED=0
FAILED=0

for run_dir in "$LOCAL_PATH"offline-run-*/; do
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
    RECENT_LOCAL_ACTIVITY=$(find "$run_dir" -type f -name '*.wandb' -mmin -$ACTIVE_THRESHOLD_MINUTES 2>/dev/null)
    
    # Check if this run is in the active runs list from remote OR has recent local activity
    if echo "$ACTIVE_RUNS" | grep -q "^$run_name$" || [ -n "$RECENT_LOCAL_ACTIVITY" ]; then
        echo ">> Syncing ACTIVE run (appending): $run_name"
        # Don't let wandb sync failure stop the script
        if wandb sync "$run_dir" --include-offline --append; then
            ((ACTIVE_SYNCED++)) || true
        else
            echo "WARNING: Failed to sync $run_name"
            ((FAILED++)) || true
        fi
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

    echo ""
done

echo "--- Sync Process Complete ---"
echo "Summary:"
echo "  - Skipped (already synced): $SKIPPED"
echo "  - Active runs synced: $ACTIVE_SYNCED"
echo "  - Finished runs synced: $FINISHED_SYNCED"
echo "  - Failed: $FAILED"
