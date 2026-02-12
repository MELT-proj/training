#!/usr/bin/env bash
set -euo pipefail

# Generic rsync helper to sync a local folder to BSC transfer node
# Usage:
#   export MN5USER=myuser
#   export MN5PROJ=MYPROJECT
#   ./rsync_to_mn5.sh <SRC_FOLDER> [<DEST_FOLDER>]
#   ./rsync_to_mn5.sh <SRC_FOLDER> <DEST_FOLDER> --dry-run

# Parse args (support --dry-run / -n, --help, and arbitrary rsync args)
DRY_RUN=false
RSYNC_EXTRA_ARGS=()
SRC_FOLDER=""
DEST_FOLDER=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 <SRC_FOLDER> [<DEST_FOLDER>] [--dry-run|-n] [-- <RSYNC ARGS>]"
      echo
      echo "Sync a local folder to ${MN5USER:-<MN5USER>}@transfer1.bsc.es:/gpfs/projects/${MN5PROJ:-<MN5PROJ>}/"
      echo
      echo "Arguments:"
      echo "  SRC_FOLDER    Local source folder to sync (required)"
      echo "  DEST_FOLDER   Destination path on remote (optional, defaults to basename of SRC_FOLDER)"
      echo
      echo "Options:"
      echo "  -n, --dry-run Show what would be synced without transferring"
      echo "  -h, --help    Show this help message"
      echo "  --            Treat remaining arguments as rsync options"
      echo
      echo "Environment variables:"
      echo "  MN5USER       Username for BSC transfer node (required)"
      echo "  MN5PROJ       Project name on BSC (required)"
      echo
      echo "Examples:"
      echo "  $0 /path/to/data"
      echo "  $0 /path/to/data remote-data-folder"
      echo "  $0 /path/to/data --dry-run"
      echo "  $0 /path/to/data -- --exclude '*.tmp'"
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
    -*)
      # Unknown options are treated as rsync args (allows passing --exclude, --include, etc.)
      RSYNC_EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      # Positional arguments
      if [ -z "$SRC_FOLDER" ]; then
        SRC_FOLDER="$1"
      elif [ -z "$DEST_FOLDER" ]; then
        DEST_FOLDER="$1"
      else
        # Extra positional args treated as rsync args
        RSYNC_EXTRA_ARGS+=("$1")
      fi
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

if [ -z "$SRC_FOLDER" ]; then
  echo "Error: SRC_FOLDER argument is required."
  echo "Usage: $0 <SRC_FOLDER> [<DEST_FOLDER>] [OPTIONS]"
  echo "Run '$0 --help' for more information."
  exit 1
fi

# Default DEST_FOLDER to basename of SRC_FOLDER if not provided
if [ -z "$DEST_FOLDER" ]; then
  DEST_FOLDER="$(basename "$SRC_FOLDER")"
fi

SRC="${SRC_FOLDER}"
DEST="${MN5USER}@transfer1.bsc.es:/gpfs/projects/${MN5PROJ}/${DEST_FOLDER}"

echo "Syncing ${SRC} -> ${DEST}"
if [ "$DRY_RUN" = true ]; then
  echo "(Dry-run mode: no files will be transferred)"
fi

# Use array expansion to preserve quoting for user-provided rsync args.
rsync -avh --progress ${RSYNC_DRY_RUN_FLAG} -e ssh "${RSYNC_EXTRA_ARGS[@]}" "${SRC}/" "${DEST}/"

echo "Done."




