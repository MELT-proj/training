#!/bin/bash
#
# Sync this repo to a cluster that cannot reach the internet (e.g. BSC MN5).
#
#   infra/sync_repo.sh <site> [options]
#
# The far end does NOT need internet -- git over SSH only needs an SSH
# connection and `git` on the remote. That makes the cluster a normal git
# remote you push to, so there is no bundle/sneakernet step to keep in sync.
#
# Modes:
#   (default)   git push: sends commits and updates the remote working tree
#               in place. Refuses if the remote tree has uncommitted changes,
#               so it can never silently clobber work done on the cluster.
#   --dirty     rsync the working tree instead, including uncommitted edits.
#               For fast "test this one edit" loops. No provenance: prefer the
#               default when the run's checkpoints need to be traceable.
#
# Options:
#   --dirty       rsync the working tree instead of pushing commits
#   --no-runs     skip the runs/ sync (see below)
#   --delete      (--dirty only) remove remote files absent locally. Off by
#                 default: it is destructive and cannot distinguish your stale
#                 files from someone else's work.
#   --init        create the remote repo if missing, then push
#   -n|--dry-run  show what would transfer, change nothing
#
# runs/ is gitignored but holds the configs you launch with, so it is rsync'd
# alongside the code in BOTH modes. Disable with --no-runs.
#
# The site file (infra/runners/sites/<site>.sh) must define:
#   REMOTE_SSH    ssh alias or user@host for the cluster login node
#   REMOTE_REPO   path to the repo checkout on the cluster. Prefer a path
#                 RELATIVE to the remote $HOME (e.g. "training") -- git, rsync
#                 and ssh all resolve it there, so it holds for any account.
# Use an ssh *alias* so no usernames are baked in -- each person configures
# their own ~/.ssh/config. See infra/runners/sites/example.sh.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE=""
MODE=git
SYNC_RUNS=1
DO_DELETE=0
DO_INIT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dirty)       MODE=rsync; shift ;;
        --no-runs)     SYNC_RUNS=0; shift ;;
        --delete)      DO_DELETE=1; shift ;;
        --init)        DO_INIT=1; shift ;;
        -n|--dry-run)  DRY_RUN=1; shift ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        -*)            die "unknown option '$1' (try --help)" ;;
        *)             [[ -z "$SITE" ]] && SITE="$1" || die "unexpected argument '$1'"; shift ;;
    esac
done

[[ -n "$SITE" ]] || die "usage: $0 <site> [--dirty] [--no-runs] [--delete] [--init] [--dry-run]"
[[ -f bash/run_train.sh ]] || die "run this from the repo root (bash/run_train.sh not found here)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/runners/sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE})"
# shellcheck disable=SC1090
source "$SITE_FILE"

[[ -n "${REMOTE_SSH:-}" ]]  || die "site '${SITE}' does not define REMOTE_SSH (see sites/example.sh)"
[[ -n "${REMOTE_REPO:-}" ]] || die "site '${SITE}' does not define REMOTE_REPO (see sites/example.sh)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" != "HEAD" ]] || die "detached HEAD: check out a branch before syncing"

echo "[sync] site:   ${SITE}"
echo "[sync] remote: ${REMOTE_SSH}:${REMOTE_REPO}"
echo "[sync] branch: ${BRANCH}"
echo "[sync] mode:   ${MODE}$( [[ $DRY_RUN -eq 1 ]] && echo ' (dry run)')"

ssh "$REMOTE_SSH" true 2>/dev/null \
    || die "cannot ssh to '${REMOTE_SSH}'. Configure a Host block in ~/.ssh/config and make sure your key is in the remote ~/.ssh/authorized_keys."

remote_repo_exists() {
    ssh "$REMOTE_SSH" "test -d '${REMOTE_REPO}/.git'" 2>/dev/null
}

if ! remote_repo_exists; then
    if [[ $DO_INIT -eq 1 ]]; then
        echo "[sync] remote repo missing -> initialising ${REMOTE_REPO}"
        [[ $DRY_RUN -eq 1 ]] || ssh "$REMOTE_SSH" "
            set -e
            mkdir -p '${REMOTE_REPO}'
            git init -q '${REMOTE_REPO}'
            git -C '${REMOTE_REPO}' config receive.denyCurrentBranch updateInstead
        "
    else
        die "no git repo at ${REMOTE_SSH}:${REMOTE_REPO} (pass --init to create it)"
    fi
fi

# Pushing to a checked-out branch is refused unless the remote opts in.
# updateInstead also updates the remote working tree, and aborts if it is dirty.
if [[ "$MODE" == git && $DRY_RUN -eq 0 ]]; then
    ssh "$REMOTE_SSH" "git -C '${REMOTE_REPO}' config receive.denyCurrentBranch updateInstead"
fi

sync_runs() {
    [[ $SYNC_RUNS -eq 1 ]] || return 0
    [[ -d runs ]] || { echo "[sync] no runs/ directory here, skipping"; return 0; }
    echo "[sync] runs/ -> ${REMOTE_REPO}/runs/  (gitignored, so rsync'd separately)"
    local args=(-rlptvh --exclude '__pycache__/' --exclude '*.zip' --exclude 'debug_cut_ids/')
    [[ $DRY_RUN -eq 1 ]] && args+=(--dry-run)
    rsync "${args[@]}" runs/ "${REMOTE_SSH}:${REMOTE_REPO}/runs/"
}

case "$MODE" in
git)
    echo "[sync] pushing ${BRANCH} -> ${REMOTE_SSH}:${REMOTE_REPO}"
    if [[ $DRY_RUN -eq 1 ]]; then
        git push --dry-run "${REMOTE_SSH}:${REMOTE_REPO}" "HEAD:refs/heads/${BRANCH}"
    else
        # A remote tree with uncommitted edits makes updateInstead refuse; say so plainly.
        git push "${REMOTE_SSH}:${REMOTE_REPO}" "HEAD:refs/heads/${BRANCH}" || die \
            "push rejected. If the remote working tree is dirty, commit/stash it there, or use --dirty to rsync over it."
        ssh "$REMOTE_SSH" "git -C '${REMOTE_REPO}' checkout -q '${BRANCH}' 2>/dev/null || true"
    fi
    if git status --porcelain | grep -q .; then
        echo "[sync] NOTE: you have uncommitted local changes; they were NOT pushed (use --dirty)."
    fi
    ;;
rsync)
    echo "[sync] rsyncing working tree (uncommitted changes included)"
    args=(-rlptvh --exclude '.git/' --filter=':- .gitignore')
    [[ $DO_DELETE -eq 1 ]] && args+=(--delete)
    [[ $DRY_RUN -eq 1 ]] && args+=(--dry-run)
    rsync "${args[@]}" ./ "${REMOTE_SSH}:${REMOTE_REPO}/"
    ;;
esac

sync_runs

echo "[sync] done."
if [[ $DRY_RUN -eq 0 ]]; then
    ssh "$REMOTE_SSH" "git -C '${REMOTE_REPO}' log --oneline -1 2>/dev/null; git -C '${REMOTE_REPO}' status -sb 2>/dev/null | head -3" || true
fi
