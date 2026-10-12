#!/bin/bash
#
# Collect the SFT battery's eval logs and render the A-vs-B tables. Run on nyx (or artemis); CPU only, a few
# minutes the first time, seconds for the logs already cached.
#
#   bash render_sft_tables.sh [out dir]
#
# Writes summaries.json, langshare.json, tables.md and tables.json into the out dir (default: scoring/summaries
# next to the SFT outputs) and prints the tables. See collect_sft_scores.py and sft_tables.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring/summaries}"
PY="${PY:-/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python}"
# The checkout this script sits in supplies melt (BasicTextNormalizer, the runaway ratio); the venv's own copy is stale.
export PYTHONPATH="$(cd "${HERE}/../../.." && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1

mkdir -p "$OUT"
if ! "$PY" "${HERE}/collect_sft_scores.py" --out-dir "$OUT" > "${OUT}/collect.log" 2>&1; then
    tail -30 "${OUT}/collect.log" >&2
    exit 1
fi
grep -E "logs selected|^note:|wrote" "${OUT}/collect.log" || true
"$PY" "${HERE}/sft_tables.py" "${OUT}/summaries.json" "${OUT}/langshare.json" --markdown "${OUT}/tables.md" --json "${OUT}/tables.json"
