# AGENTS.md Guide for MELT Project

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/src`: Main source code for the training library
  - `/config.py`: OmegaConf-based configuration (YAML + CLI overrides)
  - `/logging_utils.py`: Central logging utilities (only global-rank-0 logs)
  - `/data/audio/`: Speech data loading utilities
    - `/lhotse/`: Lhotse-based data loading for speech datasets
  - `/melt`: Core model components (configuration, modeling, processing)
  - `/evaluation`: Evaluation utilities and text normalizers
- `/tests`: Unit tests for the project
- `/bash`: Shell scripts for training and data preparation
- `/infra`: Infrastructure scripts for environment setup and synchronization
- `./venv`: Virtual environment to be activate for running tests.

## Coding Conventions

- Code style is enforced using `ruff`.
- PRs should be focused and minimal. Bugfix PRs should be as brief as possible.
- When writing tests, add them to existing test files when appropriate.
- Use type hints for function signatures.
- Document public functions and classes with docstrings.
- Do not use the `typing` module as it is deprecated starting from Python 3.10.
- I use `uv` for package management, so `pip` calls should take that into account.
- The virtual environment is assumed to live in `./venv`.
  - In container runs, the expected venv path is typically `/workspace/venv`.

## Dependencies

- The project depends on `transformers` for model implementations.
- Speech data loading uses `lhotse` for efficient audio processing.
- Training is managed with `accelerate` and optionally `deepspeed` for distributed training.
- Experiment tracking uses `wandb`.

## Testing

Run tests with:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_librispeech_lhotse.py
pytest tests/test_peoples_speech_lhotse.py
```

## Development Setup

1. Create a virtual environment and install the package in editable mode:
   ```bash
  uv venv
  uv pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Key Components

### Data Loading (`src/data/`)
- `audio/supported_datasets.py`: Registry of supported datasets
- `audio/lhotse/`: Lhotse-based data loaders for LibriSpeech, People's Speech, etc.

### Model Components (`src/melt/`)
- `configuration_melt.py`: Model configuration classes
- `modeling_melt.py`: Model architecture implementations
- `processing_melt.py`: Processor for input/output handling

### Training (`src/`)
- `train.py`: Main training entrypoint (OmegaConf config with CLI overrides)
- `trainer.py`: HF Trainer wrapper using Lhotse dataloaders
- `ddp.py`: Lightweight distributed rank helpers

### Training launch scripts (`bash/`)
- `run_train.sh`: Activates venv and launches `accelerate` + `src/train.py`
- `run_train_singularity.sbatch`: SLURM + Singularity/Apptainer wrapper (bind-mounts datasets/cache/outputs)

## Environment Variables

- `WANDB_PROJECT`: W&B project name for experiment tracking
- `HF_HOME`: Hugging Face cache directory
- `HF_HUB_CACHE`: Hugging Face hub cache directory (optional; if set, should be consistent with `HF_HOME`)
- `HF_HUB_OFFLINE`: Set to `1` to force HF Hub offline mode
- `TRANSFORMERS_OFFLINE`: Set to `1` to force Transformers offline mode
- `CUDA_VISIBLE_DEVICES`: GPU device selection

Common launch variables:
- `VENV_PATH`: Path to the venv activation script used by `bash/run_train.sh`
- `LOCAL_DATASETS_DIR`: Where SHAR datasets live (host path; bind-mounted in containers)
- `OUTPUT_DIR`: Output directory for checkpoints/logs (host path; bind-mounted to a writable container path)
- `SINGULARITY_IMG`: Path to the `.sif` image for container runs
- `TMPDIR_HOST`: Host tmp directory to bind into container
