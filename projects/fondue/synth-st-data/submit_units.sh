#!/bin/bash
# Submit synthetic-ST work units on MN5, four units per one-node job (one per GPU).
# Run ON MN5 from the frozen clone, e.g.
#   bash submit_units.sh --unit-dir /gpfs/scratch/epor48/synth/units --out /gpfs/scratch/epor48/synth/out [options] <unit> ...
#
#   --qos Q        acc_ehpc (default) or acc_debug
#   --time T       wall per job, HH:MM:SS (default 05:00:00; short walls backfill sooner)
#   --max-nodes N  stop while N synth jobs are queued/running (default 8: the fairshare guard)
#   --dry-run      print the sbatch commands, submit nothing
#
# Idempotent: a unit with a DONE marker is skipped; units whose group is already
# queued or running are skipped; every submission is a line in $OUT/submit.tsv.
# Jobs read this directory at START, hours later on acc_ehpc: do not edit the clone
# until they have run. Never submit while a Fondue job or resume is pending (ask first).
set -euo pipefail
QOS=acc_ehpc; TIME=05:00:00; MAXN=8; DRY=0; UNIT_DIR=""; OUT=""; UNITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --qos) QOS=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --max-nodes) MAXN=$2; shift 2 ;;
    --unit-dir) UNIT_DIR=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) UNITS+=("$1"); shift ;;
  esac
done
[[ -n $UNIT_DIR && -n $OUT && ${#UNITS[@]} -gt 0 ]] || { echo "usage: $0 --unit-dir D --out D [options] <unit> ...  (try --help)"; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHA=$(cat "$HERE/REVISION" 2>/dev/null || git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)
mkdir -p "$OUT" "$HERE/logs"
TODO=()
for u in "${UNITS[@]}"; do
  [[ -f $UNIT_DIR/$u.pool.jsonl ]] || { echo "no pool for $u"; exit 1; }
  [[ -f $OUT/$u/DONE ]] && { echo "skip $u: DONE"; continue; }
  TODO+=("$u")
done
QUEUED=$(squeue -u "$USER" -h -o '%j')
# Units held by a job that is still queued or running (submit.tsv: time, jobid, units, info). Regrouping after a
# partial group would otherwise hand a running unit to a second job.
BUSY=" $(squeue -u "$USER" -h -o '%i' | while read -r J; do awk -F'\t' -v j="$J" '$2==j {print $3}' "$OUT/submit.tsv" 2>/dev/null; done | tr '\n' ' ') "
NJOBS=$(grep -c '^synth-' <<<"$QUEUED" || true)
cd "$HERE"
for ((i=0; i<${#TODO[@]}; i+=4)); do
  GROUP=("${TODO[@]:i:4}")
  NAME=synth-${GROUP[0]}
  if grep -qxF -- "$NAME" <<<"$QUEUED"; then echo "skip ${GROUP[*]}: $NAME is queued or running"; continue; fi
  BUSYU=""; for u in "${GROUP[@]}"; do [[ $BUSY == *" $u "* ]] && BUSYU+="$u "; done
  if [[ -n $BUSYU ]]; then echo "skip ${GROUP[*]}: ${BUSYU}already held by an active job"; continue; fi
  if (( NJOBS >= MAXN )); then echo "STOP: $NJOBS synth jobs in flight (--max-nodes $MAXN)"; break; fi
  CMD=(sbatch --parsable --qos="$QOS" --time="$TIME" --job-name="$NAME" slurm/run_units.sbatch)
  if [[ $DRY == 1 ]]; then echo "DRY: UNITS=\"${GROUP[*]}\" UNIT_DIR=$UNIT_DIR OUT=$OUT ${CMD[*]}"; NJOBS=$((NJOBS+1)); continue; fi
  JID=$(UNITS="${GROUP[*]}" UNIT_DIR=$UNIT_DIR OUT=$OUT "${CMD[@]}")
  echo -e "$(date -Is)\t$JID\t${GROUP[*]}\tcode=$SHA qos=$QOS time=$TIME" | tee -a "$OUT/submit.tsv"
  NJOBS=$((NJOBS+1))
done
