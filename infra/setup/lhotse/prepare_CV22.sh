#!/usr/bin/env bash
set -euo pipefail

# Prepare CommonVoice22 Sidon WebDataset -> Lhotse SHAR.
#
# Input layout (HF snapshot):
#   ${CV22_SIDON_SNAPSHOT_DIR}/{lang}/train-00000.tar.gz
#   ${CV22_SIDON_SNAPSHOT_DIR}/{lang}/validation-00000.tar.gz
#   ${CV22_SIDON_SNAPSHOT_DIR}/{lang}/test-00000.tar.gz
#
# Output layout:
#   /mnt/home/giuseppe/myscratch/melt-data/shar/cv22_sidon/{lang}/
#
# This script converts ALL splits together per-language (single output dir per lang).
# If you want split-separated outputs, use run_convert_webdataset_all_langs.sh instead.

CV22_SIDON_SNAPSHOT_DIR_DEFAULT="/mnt/scratch-nyx/giuseppe/melt/hf_home/hub/datasets--sarulab-speech--commonvoice22_sidon/snapshots/7c06e40565468fda8c80a57c0ce4a7d9af97c095"
OUTPUT_BASE_DEFAULT="/mnt/scratch-nyx/giuseppe/melt/shar/cv22_sidon"

CV22_SIDON_SNAPSHOT_DIR="${CV22_SIDON_SNAPSHOT_DIR:-$CV22_SIDON_SNAPSHOT_DIR_DEFAULT}"
OUTPUT_BASE="${OUTPUT_BASE:-$OUTPUT_BASE_DEFAULT}"

# Languages to convert (edit as needed)
LANGUAGES=(en es fr it pt de)

# Conversion knobs (can override via env vars)
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SHARD_SIZE="${SHARD_SIZE:-25000}"
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

if [[ ! -d "$CV22_SIDON_SNAPSHOT_DIR" ]]; then
  echo "Snapshot dir not found: $CV22_SIDON_SNAPSHOT_DIR" >&2
  echo "Set CV22_SIDON_SNAPSHOT_DIR to override." >&2
  exit 2
fi

mkdir -p "$OUTPUT_BASE"

echo "Input snapshot: $CV22_SIDON_SNAPSHOT_DIR"
echo "Output base  : $OUTPUT_BASE"
echo "Languages    : ${LANGUAGES[*]}"

echo "Settings: shard_size=$SHARD_SIZE audio_format=$AUDIO_FORMAT resample=$RESAMPLE orig_sr=$ORIG_SR target_sr=$TARGET_SR force=$FORCE log_level=$LOG_LEVEL"

for lang in "${LANGUAGES[@]}"; do
  lang_dir="$CV22_SIDON_SNAPSHOT_DIR/$lang"
  if [[ ! -d "$lang_dir" ]]; then
    echo "[skip] missing lang dir: $lang_dir" >&2
    continue
  fi

  out_dir="$OUTPUT_BASE/$lang"

  cmd=(
    python "$CONVERTER"
    --shards-dir "$lang_dir"
    --output-dir "$out_dir"
    --pattern "*.tar.gz"
    --recursive
    --shard-size "$SHARD_SIZE"
    --audio-format "$AUDIO_FORMAT"
    --language "$lang"
    --log-level "$LOG_LEVEL"
  )

  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--force)
  fi

  if [[ "$RESAMPLE" == "1" ]]; then
    cmd+=(--resample --orig-sr "$ORIG_SR" --target-sr "$TARGET_SR")
  fi

  echo "[run] lang=$lang -> $out_dir"
  "${cmd[@]}"
done

echo "Done."
