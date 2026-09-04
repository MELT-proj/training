#!/bin/bash
# Download Hugging Face models (weights, tokenizers, feature extractors) to HF_HOME cache.
# This enables offline training by pre-populating the cache.

set -euo pipefail

# Set HF_HOME to your cache directory (override if needed)
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "Using HF_HOME: ${HF_HOME}"
mkdir -p "${HF_HOME}"

# Models to download. An entry may carry an optional "|pattern,pattern" suffix that
# restricts the download to matching files -- for repos that ship far more than the
# model itself. Leave it off and the whole repo comes down, as before.
MODELS=(
  "facebook/w2v-bert-2.0"
  "openai/whisper-large-v3"  # speech-encoder ablation arm; only the encoder is used
  # Raw-waveform speech-encoder ablation arm. The repo also carries a 4.5 GB faiss
  # index, a 1.1 GB fairseq checkpoint and 80 dataset manifests, none of which
  # transformers ever reads: 6.3 GB blind vs 360 MB restricted.
  "utter-project/mHuBERT-147|config.json,preprocessor_config.json,model.safetensors"
  "Qwen/Qwen2.5-0.5B"

  # Ablation campaign backbones (base vs instruct). MN5 compute nodes run
  # offline, so every one of these must be in HF_HOME before the first job.
  # Check `tokenizer.chat_template` on each *base* checkpoint after downloading:
  # Qwen 2.5 base ships one, which would make a base-vs-instruct comparison a
  # comparison of formatting rather than of tuning.
  "Qwen/Qwen3.5-2B-Base"
  "Qwen/Qwen3.5-2B"
  "utter-project/EuroLLM-1.7B"
  "utter-project/EuroLLM-1.7B-Instruct"
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-1B-Instruct"
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
for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  patterns=""
  [[ "${entry}" == *"|"* ]] && patterns="${entry#*|}"
  echo ""
  echo "------------------------------------------------------------"
  echo "Downloading: ${model}${patterns:+ (files: ${patterns})}"
  echo "------------------------------------------------------------"
  MELT_ALLOW_PATTERNS="${patterns}" ${PYBIN} - <<PY
import os
from huggingface_hub import snapshot_download

repo_id = "${model}"
allow = [p for p in os.environ.get("MELT_ALLOW_PATTERNS", "").split(",") if p] or None
try:
    path = snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=allow)
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
