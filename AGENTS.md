# AGENTS.md Guide for MELT Project

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/src`: Main source code for the training library
  - `/data_utils`: Data loading and processing utilities
    - `/lhotse`: Lhotse-based data loading for speech datasets
  - `/melt`: Core model components (configuration, modeling, processing)
  - `/evaluation`: Evaluation utilities and text normalizers
- `/tests`: Unit tests for the project
- `/bash`: Shell scripts for training and data preparation
- `/infra`: Infrastructure scripts for environment setup and synchronization

## Coding Conventions

- Code style is enforced using `ruff`.
- PRs should be focused and minimal. Bugfix PRs should be as brief as possible.
- When writing tests, add them to existing test files when appropriate.
- Use type hints for function signatures.
- Document public functions and classes with docstrings.
- Do not use the `typing` module as it is deprecated starting from Python 3.10.
- I use `uv` for package management, so `pip` calls should take that into account.
- My current virtual environment is in `./venv` so be sure to activate it before running tests that require it.

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
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Key Components

### Data Loading (`src/data_utils/`)
- `supported_datasets.py`: Registry of supported datasets
- `lhotse/`: Lhotse-based data loaders for LibriSpeech, People's Speech, etc.

### Model Components (`src/melt/`)
- `configuration_speechlm.py`: Model configuration classes
- `modeling_speechlm.py`: Model architecture implementations
- `processing_speechlm.py`: Processors for input/output handling

### Training (`src/`)
- `train.py`: Main training script
- `training_utils.py`: Training utilities and helpers
- `evaluate.py`: Evaluation script

## Environment Variables

- `WANDB_PROJECT`: W&B project name for experiment tracking
- `HF_HOME`: Hugging Face cache directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection
