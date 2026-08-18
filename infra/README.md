infra/

Purpose
- Infrastructure and operational scripts for environment setup, data synchronization, container images, and convenience utilities used during development and HPC runs.

Contents (high-level)
- `check_training_config.py`: verifies a training config's data section against the data it points at — mixture weights, `total_hours` / `total_cuts`, `bucket_duration_bins`, validation `name:` keys, tags and paths — and prints the command or YAML edit for anything that is off. Run it before scheduling a job. See `docs/config_verification.md`.
- `compute_mix_weights.py`: computes the two-tier mixture weights and emits a training config. See `docs/mixture_weights.md`. `estimate_bucket_bins.py`: measures duration distributions and bucket bins; `bucket_bins.py` holds the bin arithmetic the two share.
- `rsync_*` and `sync_*` scripts: rsync helpers for remote storage (MN5, Leonardo, local mirrors).
- `Singularity.def`: container image definition. `runners/`: per-site submit wrappers (`submit-native.sh` / `submit-container.sh`) around the training launchers in `bash/`. See `docs/run_training.md`.

Notes
- Many scripts assume environment variables (e.g., `VENV_PATH`, `OUTPUT_DIR`, `LOCAL_DATASETS_DIR`).
- Read individual scripts for options; most are safe to run locally after reviewing.
