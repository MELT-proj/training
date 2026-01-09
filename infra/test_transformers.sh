#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="test-transformers"
REPO_ID="Qwen/Qwen2.5-0.5B"
export HF_HOME="./test_cache"

echo "[1/6] Creating uv venv: ${VENV_DIR}"
uv venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[2/6] Installing transformers==4.57.3 and torch"
uv pip install "transformers==4.57.1" torch

echo "[3/5] Downloading repo via huggingface_hub.snapshot_download: ${REPO_ID}"
REPO_ID="${REPO_ID}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
hf_home = os.environ["HF_HOME"]

print(f"HF_HOME={hf_home}")
path = snapshot_download(repo_id=repo_id)
print(f"snapshot_download OK: {repo_id}")
print(f"Snapshot path: {path}")
PY

echo "[4/5] Exporting HF_HUB_OFFLINE=1"
export HF_HUB_OFFLINE=1

echo "[5/5] Testing transformers from_pretrained calls (AutoConfig/AutoTokenizer/AutoModel) with repo id: ${REPO_ID}"
REPO_ID="${REPO_ID}" python - <<'PY'
import os
import sys
import traceback

repo_id = os.environ["REPO_ID"]
print(f"HF_HOME={os.environ.get('HF_HOME')}")
print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

failures: list[str] = []

def run(name: str, fn) -> None:
    try:
        fn()
        print(f"{name}: OK")
    except Exception:
        tb = traceback.format_exc()
        failures.append(f"{name} FAILED:\n{tb}")
        print(f"{name}: FAILED (full traceback below)")
        print(tb)

from transformers import AutoConfig, AutoTokenizer, AutoModel

run("AutoConfig.from_pretrained", lambda: AutoConfig.from_pretrained(repo_id))
run("AutoTokenizer.from_pretrained", lambda: AutoTokenizer.from_pretrained(repo_id))
run("AutoModel.from_pretrained", lambda: AutoModel.from_pretrained(repo_id))

if failures:
    print("\n=== SUMMARY: one or more calls failed ===")
    for msg in failures:
        print(msg)
        print("-" * 80)
    sys.exit(1)

print("\nAll from_pretrained calls succeeded.")
PY