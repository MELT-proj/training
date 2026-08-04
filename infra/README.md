infra/

Purpose
- Infrastructure and operational scripts for environment setup, data synchronization, container images, and convenience utilities used during development and HPC runs.

Contents (high-level)
- `rsync_*` and `sync_*` scripts: rsync helpers for remote storage (MN5, Leonardo, local mirrors).
- `Singularity.def`: container image definition. `runners/`: per-site submit wrappers (`submit-native.sh` / `submit-container.sh`) around the training launchers in `bash/`. See `docs/run_training.md`.

Notes
- Many scripts assume environment variables (e.g., `VENV_PATH`, `OUTPUT_DIR`, `LOCAL_DATASETS_DIR`).
- Read individual scripts for options; most are safe to run locally after reviewing.
