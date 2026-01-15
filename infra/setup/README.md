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

- Run the repo-wide webdataset runner (per-language split runner):

```bash
infra/setup/lhotse/run_convert_webdataset_all_langs.sh \
	--maindir /data/wds --outputdir /mnt/home/giuseppe/myscratch/melt-data/shar/wds_out
```

- Convert CommonVoice22 Sidon snapshot for a fixed list of languages (uses `prepare_CV22.sh`):

```bash
infra/setup/lhotse/prepare_CV22.sh
# or override envvars for one-off runs:
CV22_SIDON_SNAPSHOT_DIR=/mnt/scratch-nyx/... OUTPUT_BASE=/mnt/home/giuseppe/myscratch/melt-data/shar/cv22_sidon \ 
	RESAMPLE=1 TARGET_SR=16000 infra/setup/lhotse/prepare_CV22.sh
```

Notes
- These examples assume the converter script and runners live under `infra/setup/lhotse/`.
- When converting large datasets prefer `--batched` + `--io-num-workers` to overlap HF IO and local decoding.
