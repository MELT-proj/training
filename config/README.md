config/

Purpose
- Central location for YAML configuration files used by training and launch scripts.

Contents
- `accelerate/` and `train/` subfolders: presets for distributed training (FSDP, Zero, etc.) and dataset/training configs.

Usage
- Loaded and merged with the project’s dataclass-based CLI (tyro) in `src/train.py`.
- Use shell-style env expansion in YAML (e.g. `${VAR:-default}`) for portability.
- Prefer editing small, focused YAMLs and pass overrides via the CLI or environment variables.
