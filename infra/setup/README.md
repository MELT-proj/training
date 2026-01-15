infra/setup/

Purpose
- Utility scripts and dataset conversion tools (Lhotse/WebDataset converters) used to prepare training data into SHAR archives consumable by the training pipeline.

Contents (examples)
- `lhotse/`: conversion helpers and per-dataset converters (LibriSpeech, People's Speech, MLS, VoxPopuli, FLEURS).
- `lhotse/batch_utils.py`: batched conversion utilities with IO/CPU parallelism.
- `lhotse/convert_webdataset.py`: converter from WebDataset tar shards into Lhotse SHAR archives.
- Runners: `run_convert_webdataset_all_langs.sh`, `prepare_CV22.sh` for dataset-specific orchestration.

Usage
- Prefer batched converters for throughput on machines with sufficient RAM and CPUs (`--batched` / `--io-num-workers`).
- Use the provided runners to convert organized shard trees per-language.

Usage examples
- Convert a single HF-style WebDataset language folder to SHAR (batched, resample to 16k):

```bash
python infra/setup/lhotse/convert_webdataset.py \
	--shards-dir /data/wds/en \
	--output-dir /mnt/home/giuseppe/myscratch/melt-data/shar/mydataset/en \
	--pattern "train-*.tar.gz" --recursive \
	--shard-size 4000 --audio-format flac --language en \
	--resample --orig-sr 48000 --target-sr 16000
```

- Per-dataset commands: run the specific converter or runner for the dataset you need (the earlier `run_convert_webdataset_all_langs.sh` is not part of this repo anymore).

Dataset prepare command examples (run inside the repo root):

- LibriSpeech (batched):

```bash
python infra/setup/lhotse/librispeech.py --batched --batch-size 5000 --num-workers 8 --hf-num-proc 4 --log-level INFO
```

- People's Speech (batched, specific config):

```bash
python infra/setup/lhotse/peoples_speech.py --configs clean --splits train validation test --batched --batch-size 5000 --num-workers 8 --hf-num-proc 4
```

- Multilingual LibriSpeech (MLS):

```bash
# Process these languages by default: en de it fr es pt
python infra/setup/lhotse/multilingual_librispeech.py --configs german french italian spanish portuguese german --splits train dev test --batched --batch-size 5000 --num-workers 8 --io-num-workers 8
```

*Note:* MLS uses language names (e.g., `german`, `italian`) rather than short ISO codes.

- VoxPopuli (per-language configs; optional accented tests):

```bash
# Process these languages by default: en de it fr es pt
python infra/setup/lhotse/voxpopuli.py --configs en de it fr es pt --include-special-configs --batched --batch-size 5000 --num-workers 8 --io-num-workers 8
```

*Note:* VoxPopuli uses short ISO language codes (e.g., `de`, `it`) for configs.

- FLEURS (select a few configs or use `--all-configs` very carefully):

```bash
# Process these languages by default: en de it fr es pt
# FLEURS uses locale-style config names (e.g., `it_it`, `pt_pt`, `en_us`); check available configs here:
# https://huggingface.co/datasets/google/fleurs/blob/main/fleurs.py
python infra/setup/lhotse/fleurs.py --configs en_us de_de it_it fr_fr es_es pt_pt --splits train validation test --batched --batch-size 5000 --num-workers 8
```

- CommonVoice22 (Sidon) — download snapshot from HF hub first, then run the per-language converter or the `prepare_CV22.sh` runner:

```bash
# Download snapshot for selected languages into your HF cache (run once):
HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/mnt/scratch-nyx/giuseppe/melt/hf_home \
  hf download sarulab-speech/commonvoice22_sidon --repo-type dataset --include "it/*" "en/*" "de/*" "es/*" "fr/*" "pt/*" "README.md" --exclude "*invalidated*"

# Convert (per-language) using the main runner in this folder (processes en,de,it,fr,es,pt):
infra/setup/lhotse/prepare_CV22.sh
```

Notes
- These examples assume the converter scripts live under `infra/setup/lhotse/` and that the necessary HF snapshots or local shards are present.
- When converting large datasets prefer `--batched` + `--io-num-workers` to overlap HF IO and local decoding.

Notes
- These examples assume the converter script and runners live under `infra/setup/lhotse/`.
- When converting large datasets prefer `--batched` + `--io-num-workers` to overlap HF IO and local decoding.
