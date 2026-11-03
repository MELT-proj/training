#!/bin/bash
#SBATCH --job-name=tts-text
#SBATCH --partition=a6000
#SBATCH --qos=gpu-debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/mnt/scratch-nyx/giuseppe/synth/logs/%x.%j.out
# English TTS source text with Qwen3.6-35B-A3B (vLLM >= 0.19 venv from setup_qwen_venv.sh, thinking off).
# The model is 67 GB in bf16: two A6000s with tensor parallel 2, or one H200. Override on the command line:
#   sbatch --export=ALL,TP=2,GPU_MEM=0.85,NP=380,OUT=/path/out.jsonl slurm/gen_tts_text_artemis.sbatch
# (for H200: --partition=h200 --qos=gpu-h200 --gres=gpu:1 with TP=1). The code dir must sit on a shared filesystem.
set -euo pipefail
export HF_HOME=/mnt/scratch-nyx/giuseppe/synth/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/mnt/scratch-nyx/giuseppe/synth/vllm-cache-qwen36-tp${TP:-1}  # keeps the compile cache: load 19 min -> 4 min
CODE="${CODE:-/mnt/scratch-nyx/giuseppe/synth/code-qwen}"
cd "$CODE"
/mnt/scratch-artemis/giuseppe/venvs/vllm-qwen36/bin/python gen_tts_text.py \
  --out "${OUT:-/mnt/scratch-nyx/giuseppe/synth/e/tts_text_qwen36.jsonl}" --n-prompts "${NP:-380}" \
  --seed "${SEED:-0}" --model Qwen/Qwen3.6-35B-A3B --tp "${TP:-1}" --gpu-mem "${GPU_MEM:-0.88}" \
  --ledger /mnt/scratch-nyx/giuseppe/synth/e/ledger-qwen.tsv --unit qwen36-tts-text
