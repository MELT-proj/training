#!/usr/bin/env bash
set -euo pipefail

# Prepare CommonVoice22 Sidon WebDataset -> Lhotse SHAR.
#
# Input layout (HF snapshot):
#   ${MLS_SIDON_SNAPSHOT_DIR}/{lang}/train-00000.tar.gz
#   ${MLS_SIDON_SNAPSHOT_DIR}/{lang}/validation-00000.tar.gz
#   ${MLS_SIDON_SNAPSHOT_DIR}/{lang}/test-00000.tar.gz
#
# Output layout:
#   /mnt/home/giuseppe/myscratch/melt-data/shar/cv22_sidon/{lang}/
#
# This script converts ALL splits together per-language (single output dir per lang).
# If you want split-separated outputs, use run_convert_webdataset_all_langs.sh instead.

SIDON_SNAPSHOT_DIR_DEFAULT="/mnt/scratch-nyx/giuseppe/melt/hf_home/hub/datasets--sarulab-speech--mls_sidon/snapshots/a9abb3a85d5ca67f92ed29f65003b420169e2732"
OUTPUT_BASE_DEFAULT="/mnt/scratch-nyx/giuseppe/melt/melt-data/shar/mls_sidon"

MLS_SIDON_SNAPSHOT_DIR="${MLS_SIDON_SNAPSHOT_DIR:-$SIDON_SNAPSHOT_DIR_DEFAULT}"
OUTPUT_BASE="${OUTPUT_BASE:-$OUTPUT_BASE_DEFAULT}"

# Languages to convert (edit as needed)
LANGUAGES=(dutch french german italian polish portuguese spanish english)

# Conversion knobs (can override via env vars)
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SHARD_SIZE="${SHARD_SIZE:-4000}"
AUDIO_FORMAT="${AUDIO_FORMAT:-flac}"
RESAMPLE="${RESAMPLE:-1}"           # 1 enables resample
ORIG_SR="${ORIG_SR:-48000}"
TARGET_SR="${TARGET_SR:-16000}"
FORCE="${FORCE:-0}"                 # 1 forces re-run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/convert_webdataset.py"

if [[ ! -f "$CONVERTER" ]]; then
  echo "Missing converter: $CONVERTER" >&2
  exit 2
fi

if [[ ! -d "$MLS_SIDON_SNAPSHOT_DIR" ]]; then
  echo "Snapshot dir not found: $MLS_SIDON_SNAPSHOT_DIR" >&2
  echo "Set MLS_SIDON_SNAPSHOT_DIR to override." >&2
  exit 2
fi

mkdir -p "$OUTPUT_BASE"

echo "Input snapshot: $MLS_SIDON_SNAPSHOT_DIR"
echo "Output base  : $OUTPUT_BASE"
echo "Languages    : ${LANGUAGES[*]}"

echo "Settings: shard_size=$SHARD_SIZE audio_format=$AUDIO_FORMAT resample=$RESAMPLE orig_sr=$ORIG_SR target_sr=$TARGET_SR force=$FORCE log_level=$LOG_LEVEL"

for lang in "${LANGUAGES[@]}"; do
  lang_dir="$MLS_SIDON_SNAPSHOT_DIR/$lang"
  if [[ ! -d "$lang_dir" ]]; then
    echo "[skip] missing lang dir: $lang_dir" >&2
    continue
  fi

  for split in train dev test; do
    out_dir="$OUTPUT_BASE/$lang/$split"

    cmd=(
      python "$CONVERTER"
      --shards-dir "$lang_dir"
      --output-dir "$out_dir"
      --pattern "${split}*.tar.gz"
      --recursive 
      --shard-size "$SHARD_SIZE"
      --audio-format "$AUDIO_FORMAT"
      --language "$lang"
      --log-level "$LOG_LEVEL"
      --num-workers 8
    )

    if [[ "$FORCE" == "1" ]]; then
      cmd+=(--force)
    fi

    if [[ "$RESAMPLE" == "1" ]]; then
      cmd+=(--resample --orig-sr "$ORIG_SR" --target-sr "$TARGET_SR")
    fi

    echo "[run] lang=$lang split=$split -> $out_dir"
    "${cmd[@]}"

  done
done

echo "Done."
