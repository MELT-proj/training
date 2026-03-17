#!/usr/bin/env bash
set -euo pipefail

# Check SHAR shard-manifest coverage risk for distributed training with muxed sources.
#
# Usage:
#   LOCAL_DATASETS_DIR=/gpfs/.../shar WORLD_SIZE=64 ./infra/check_shards.sh config/train/MA-v1.2.yaml
#   LOCAL_DATASETS_DIR=/gpfs/.../shar ./infra/check_shards.sh config/train/MA-v1.2.yaml 64
#
# Notes:
# - This script does NOT enforce a hard per-source rule (S_i >= P).
# - It reports per-source risk and aggregate risk where:
#     P = WORLD_SIZE * num_workers
# - For muxed pipelines, aggregate coverage can compensate for small individual sources,
#   but many small sources still increase risk of empty workers.

CFG_PATH="${1:-}"
WORLD_SIZE_INPUT="${2:-${WORLD_SIZE:-}}"

if [[ -z "${CFG_PATH}" || -z "${WORLD_SIZE_INPUT}" ]]; then
  echo "Usage: $0 <train_config.yaml> <world_size>"
  echo "Example: LOCAL_DATASETS_DIR=/gpfs/.../shar $0 config/train/MA-v1.2.yaml 64"
  exit 1
fi

if ! command -v yq >/dev/null 2>&1; then
  echo "ERROR: yq not found. Install mikefarah/yq and retry."
  exit 2
fi

if [[ -z "${LOCAL_DATASETS_DIR:-}" ]]; then
  echo "ERROR: LOCAL_DATASETS_DIR is not set."
  echo "This is required to resolve paths like \${oc.env:LOCAL_DATASETS_DIR}/..."
  exit 2
fi

WORLD_SIZE="${WORLD_SIZE_INPUT}"
NUM_WORKERS="$(yq -r '.data.train_ds.num_workers // 1' "${CFG_PATH}")"
P=$(( WORLD_SIZE * NUM_WORKERS ))

if (( P <= 0 )); then
  echo "ERROR: invalid parallelism factor P=${P} (WORLD_SIZE=${WORLD_SIZE}, num_workers=${NUM_WORKERS})"
  exit 2
fi

echo "Config: ${CFG_PATH}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "train_ds.num_workers: ${NUM_WORKERS}"
echo "LOCAL_DATASETS_DIR: ${LOCAL_DATASETS_DIR}"
echo "P = WORLD_SIZE * num_workers = ${P}"
echo

# Extract train lhotse_shar sources and optional weights.
# If weight is missing, default to 1.0 to match mux default behavior.
mapfile -t ROWS < <(
  yq -r '
    .data.train_ds.input_cfg[]
    | select(.type == "lhotse_shar")
    | [ .shar_path, (.weight // 1.0) ]
    | @tsv
  ' "${CFG_PATH}"
)

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "No lhotse_shar sources found under data.train_ds.input_cfg"
  exit 0
fi

printf "%-3s %-8s %-8s %-8s %-8s %-8s %s\n" "ID" "SHARDS" "P" "S/P" "WEIGHT" "W*S/P" "SOURCE"
echo "------------------------------------------------------------------------------------------------------------"

TOTAL_SHARDS=0
TOTAL_WEIGHT=0
WEIGHTED_SHARDS=0
MISSING=0
VERY_LOW=0
LOW=0

id=0
for row in "${ROWS[@]}"; do
  id=$((id + 1))
  src_raw="$(printf '%s' "${row}" | cut -f1)"
  weight="$(printf '%s' "${row}" | cut -f2)"

  # Resolve OmegaConf env interpolation token explicitly.
  src="${src_raw//\$\{oc.env:LOCAL_DATASETS_DIR\}/${LOCAL_DATASETS_DIR}}"

  # Fallback expansion for any shell-style vars if present.
  eval "src=\"${src}\""

  if [[ ! -d "${src}" ]]; then
    printf "%-3s %-8s %-8s %-8s %-8s %-8s %s\n" "${id}" "MISSING" "${P}" "-" "${weight}" "-" "${src}"
    MISSING=$((MISSING + 1))
    continue
  fi

  shards="$(find "${src}" -maxdepth 1 -type f -name 'cuts.*.jsonl.gz' | wc -l | tr -d ' ')"
  ratio="$(awk -v s="${shards}" -v p="${P}" 'BEGIN { printf "%.3f", s/p }')"
  wratio="$(awk -v s="${shards}" -v p="${P}" -v w="${weight}" 'BEGIN { printf "%.3f", (w*s)/p }')"

  printf "%-3s %-8s %-8s %-8s %-8s %-8s %s\n" "${id}" "${shards}" "${P}" "${ratio}" "${weight}" "${wratio}" "${src}"

  TOTAL_SHARDS=$((TOTAL_SHARDS + shards))
  TOTAL_WEIGHT="$(awk -v a="${TOTAL_WEIGHT}" -v b="${weight}" 'BEGIN { printf "%.6f", a+b }')"
  WEIGHTED_SHARDS="$(awk -v ws="${WEIGHTED_SHARDS}" -v w="${weight}" -v s="${shards}" 'BEGIN { printf "%.6f", ws + (w*s) }')"

  # Heuristic risk bins per source.
  # S/P < 0.25: very high risk source (many workers likely receive zero shards from this source).
  # 0.25 <= S/P < 1.0: partial coverage source.
  if awk -v r="${ratio}" 'BEGIN { exit !(r < 0.25) }'; then
    VERY_LOW=$((VERY_LOW + 1))
  elif awk -v r="${ratio}" 'BEGIN { exit !(r < 1.0) }'; then
    LOW=$((LOW + 1))
  fi
done

echo
AGG_RATIO="$(awk -v ts="${TOTAL_SHARDS}" -v p="${P}" 'BEGIN { printf "%.3f", ts/p }')"
W_AGG_RATIO="$(awk -v ws="${WEIGHTED_SHARDS}" -v p="${P}" 'BEGIN { printf "%.3f", ws/p }')"

echo "Summary"
echo "  Sources: ${#ROWS[@]}"
echo "  Missing dirs: ${MISSING}"
echo "  Total shards (sum S_i): ${TOTAL_SHARDS}"
echo "  Aggregate ratio sum(S_i)/P: ${AGG_RATIO}"
echo "  Weight sum: ${TOTAL_WEIGHT}"
echo "  Weighted aggregate ratio sum(w_i*S_i)/P: ${W_AGG_RATIO}"
echo "  Per-source very low coverage (S_i/P < 0.25): ${VERY_LOW}"
echo "  Per-source partial coverage (0.25 <= S_i/P < 1): ${LOW}"
echo

echo "Interpretation"
if (( MISSING > 0 )); then
  echo "  - CRITICAL: some source directories are missing."
fi
if awk -v r="${W_AGG_RATIO}" 'BEGIN { exit !(r < 1.0) }'; then
  echo "  - HIGH RISK: weighted aggregate shards are below P; empty workers are likely."
elif awk -v r="${W_AGG_RATIO}" 'BEGIN { exit !(r < 2.0) }'; then
  echo "  - MEDIUM RISK: weighted aggregate shards are near P; distribution may be brittle."
else
  echo "  - LOWER RISK: weighted aggregate shards are comfortably above P."
fi

if (( VERY_LOW > 0 )); then
  echo "  - Note: several sources have very low per-source coverage; mux can still work, but fragility increases."
fi

echo
if (( MISSING > 0 )); then
  exit 3
fi

if awk -v r="${W_AGG_RATIO}" 'BEGIN { exit !(r < 1.0) }'; then
  exit 4
fi

exit 0
