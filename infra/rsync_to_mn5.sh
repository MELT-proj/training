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

# Parse args (simple): support --dry-run / -n and --help
DRY_RUN=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run|-n]"
      echo
      echo "Sync \\${MELT_DATA_ROOT} to ${MN5USER:-}<MN5USER>@transfer1.bsc.es:/gpfs/projects/${MN5PROJ:-}<MN5PROJ>/"
      echo
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dry-run|-n]"
      exit 1
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
rsync -avh --progress ${RSYNC_DRY_RUN_FLAG} -e ssh --exclude 'tmp/' --exclude 'models/' "${SRC}" "${DEST}"

echo "Done."
