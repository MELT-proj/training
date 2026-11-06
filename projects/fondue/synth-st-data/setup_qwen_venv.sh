#!/bin/bash
# Build the vLLM >= 0.19 venv used for Qwen3.6-35B-A3B text generation (not the MT/QE image: that one pins vLLM 0.14.1).
# Run on nyx; the venv lives on artemis scratch. Wheels only, nothing is compiled. If a source build is ever needed,
# cap the compile jobs (MAX_JOBS=4) or the machine freezes.
set -euo pipefail
UV=/mnt/scratch-artemis/giuseppe/bin/uv
PY_DIR=/mnt/scratch-artemis/giuseppe/uv-python
VENV=/mnt/scratch-artemis/giuseppe/venvs/vllm-qwen36

# uv's managed interpreter sits in ~/.local on nyx, which artemis compute nodes cannot see: copy it to scratch
# and point the venv at the copy.
mkdir -p "$PY_DIR"
cp -a "$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/." "$PY_DIR/cpython-3.12/"
"$UV" venv --python "$PY_DIR/cpython-3.12/bin/python3.12" "$VENV"
"$UV" pip install --python "$VENV/bin/python" --link-mode=copy vllm==0.19.0 tyro  # torch 2.10, transformers 4.57.6

# smoke test on a compute node (the artemis driver is 550):
#   sbatch --partition=a6000 --qos=gpu-debug --gres=gpu:2 slurm/gen_tts_text_artemis.sbatch
