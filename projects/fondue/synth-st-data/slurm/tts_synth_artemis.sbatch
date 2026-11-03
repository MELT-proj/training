#!/bin/bash
#SBATCH --job-name=tts-full
#SBATCH --partition=a6000
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch-nyx/giuseppe/synth/logs/%x.%j.out
set -euo pipefail
S=/mnt/scratch-nyx/giuseppe/synth
SNAP=$(ls -d $S/hf_cache/hub/models--hexgrad--Kokoro-82M/snapshots/*)
singularity exec --nv --bind /mnt/scratch-nyx --env PYTHONPATH=$S/pylibs,HF_HOME=$S/hf_cache,HF_HUB_OFFLINE=1 $S/synth.sif \
  python $S/code-qwen/tts_synth.py --inp $S/e/tts_part${P}.jsonl --out $S/e/tts_en/part${P} \
  --hub-snapshot $SNAP --workers 6 --threads 1 --device cuda
