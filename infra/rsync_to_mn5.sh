#!/usr/bin/env bash
set -euo pipefail

# Simple rsync helper to sync the full melt-data folder to BSC transfer node
# Usage:
#   export MN5USER=myuser
#   export MN5PROJ=MYPROJECT
#   export MELT_DATA_ROOT=/path/to/melt-data  # optional
#   ./rsync_data.sh
#   ./rsync_data.sh --dry-run   # show what would be synced without transferring

: "${MELT_DATA_ROOT:=/mnt/home/giuseppe/myscratch/melt-data}"

# Parse args (support --dry-run / -n, --help, and arbitrary rsync args)
DRY_RUN=false
RSYNC_EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run|-n] [-- <RSYNC ARGS>]"
      echo
      echo "Sync \\${MELT_DATA_ROOT} to ${MN5USER:-}<MN5USER>@transfer1.bsc.es:/gpfs/projects/${MN5PROJ:-}<MN5PROJ>/"
      echo
      echo "Any additional arguments are passed directly to rsync (e.g., --exclude 'path')."
      echo
      exit 0
      ;;
    --)
      # Treat remaining args as rsync args
      shift
      while [ "$#" -gt 0 ]; do
        RSYNC_EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      # Unknown options are treated as rsync args (allows passing --exclude, --include, etc.)
      RSYNC_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

RSYNC_DRY_RUN_FLAG=""
if [ "$DRY_RUN" = true ]; then
  RSYNC_DRY_RUN_FLAG="--dry-run"
fi

if [ -z "${MN5USER:-}" ] || [ -z "${MN5PROJ:-}" ]; then
  echo "Error: MN5USER and MN5PROJ environment variables must be set."
  echo "Example: export MN5USER=giuseppe; export MN5PROJ=MN5PROJNAME"
  exit 1
fi

SRC="${MELT_DATA_ROOT}"
DEST="${MN5USER}@transfer1.bsc.es:/gpfs/projects/${MN5PROJ}/"

echo "Syncing ${SRC} -> ${DEST} (excluding tmp/ and models/)"
# Exclude the 'tmp/' and 'models/' folders inside MELT_DATA_ROOT to avoid copying transient files and large model files.
# Append any user-provided rsync args after the default options. Use array expansion to preserve quoting.
rsync -avh --progress ${RSYNC_DRY_RUN_FLAG} -e ssh --exclude 'tmp/' --exclude 'models/' "${RSYNC_EXTRA_ARGS[@]}" "${SRC}" "${DEST}"

echo "Done."
