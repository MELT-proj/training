#!/bin/bash
# Download Hugging Face models (weights, tokenizers, feature extractors) to HF_HOME cache.
# This enables offline training by pre-populating the cache.

set -euo pipefail

# Set HF_HOME to your cache directory (override if needed)
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "Using HF_HOME: ${HF_HOME}"
mkdir -p "${HF_HOME}"

# Models to download
MODELS=(
  "facebook/w2v-bert-2.0"
  "Qwen/Qwen2.5-0.5B"
)

echo "============================================================"
echo "Downloading Hugging Face models to ${HF_HOME}"
echo "============================================================"

# Check python and huggingface_hub availability
if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: Python 3 is required (python or python3 in PATH)."
  exit 1
fi
PYBIN="$(command -v python || command -v python3)"

if ! ${PYBIN} -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "ERROR: 'huggingface_hub' Python package not found. Install with: pip install 'huggingface-hub'"
  exit 1
fi

# Download each model using huggingface_hub.snapshot_download
for model in "${MODELS[@]}"; do
  echo ""
  echo "------------------------------------------------------------"
  echo "Downloading: ${model}"
  echo "------------------------------------------------------------"
  ${PYBIN} - <<PY
import os
from huggingface_hub import snapshot_download

repo_id = "${model}"
try:
    path = snapshot_download(repo_id=repo_id, repo_type="model")
    print(f"Downloaded {repo_id} -> {path}")
except Exception as e:
    print(f"ERROR downloading {repo_id}: {e}")
    raise
PY
done

echo ""
echo "============================================================"
echo "All models downloaded successfully to ${HF_HOME}"
echo "============================================================"
echo ""
echo "To use offline mode in training, set:"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_HUB_OFFLINE=1"
