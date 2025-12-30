# MELT Training — Getting Started ✅

## Datasets (creating Lhotse SHAR archives) 🔧

This repository includes standalone scripts to download and convert speech datasets to Lhotse SHAR archives. The scripts live in:

```
infra/preparation/lhotse/
```

Each script is a small, self-contained converter (e.g., `librispeech.py`, `peoples_speech.py`) that downloads data from HuggingFace and writes Lhotse-compatible SHAR archives into a local output directory.

Key points:

- Default output directory: the scripts write to the path defined by the environment variable `LHOTSE_DATA_SHAR_ROOT` (if set) or to the default
  `~/myscratch/melt-data/shar` (see each script for the exact default).
- Two processing modes are supported:
  - **Streaming (default):** memory-efficient, processes items one-by-one.
  - **Batched:** faster, uses more RAM and parallel workers. Enable with `--batched` and tune `--batch-size`/`--num-workers`.
- Use `--force` to re-run conversion even when a completion marker or output files already exist.

### Example: Download and convert LibriSpeech 📥

To download and convert LibriSpeech to Lhotse SHAR archives, run the LibriSpeech converter script:

```bash
# Convert all LibriSpeech splits (streaming, memory-efficient)
python infra/preparation/lhotse/librispeech.py

# Convert using batched mode (faster, uses more RAM)
python infra/preparation/lhotse/librispeech.py --batched --batch-size 10000 --num-workers 8

# Re-run conversion even if outputs exist
python infra/preparation/lhotse/librispeech.py --force
```

After running the script, data will be written under:

```
$LHOTSE_DATA_SHAR_ROOT/librispeech/
```

(or the default path if `LHOTSE_DATA_SHAR_ROOT` is not set). Check the corresponding directory for the created SHAR archives and a `.conversion_markers` entry that notes completion.