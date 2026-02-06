config/

Purpose
- Central location for YAML configuration files used by training and launch scripts.

Contents
- `accelerate/` and `train/` subfolders: presets for distributed training (FSDP, Zero, etc.) and dataset/training configs.

Usage
- Loaded via OmegaConf in `src/train.py` with support for hierarchical CLI overrides.
- Use shell-style env expansion in YAML (e.g. `${VAR:-default}`) for portability.
- Prefer editing small, focused YAMLs and pass overrides via the CLI or environment variables.
- CLI overrides use dot notation: `--trainer.max_steps 100 --run.exp_name my_exp`
