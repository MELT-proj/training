## Running training (config-driven)

We provide several ways to launch training using a YAML config file (`--config`). The canonical helper is `bash/train_with_config.sh`, which wraps `accelerate` and sets useful environment defaults.

### 1) Batch (SLURM + sbatch)
Use this when submitting a job to the cluster scheduler. Example:

```bash
sbatch bash/train_with_config.sh config/train/LS_asr.yaml config/accelerate/zero3.yaml
```

This submits a SLURM job and wraps `accelerate launch` with the correct node/process configuration.

### 2) Interactive (inside an srun allocation)
If you already reserved an interactive allocation with `srun` (or `salloc`) do **not** re-run `srun` from inside the allocation — instead just run the same helper. The script detects interactive srun steps and will launch `accelerate` directly:

```bash
# reserve a node (interactive session)
srun -N1 --gpus-per-node=4 --pty bash
# inside that shell, launch training
bash bash/train_with_config.sh config/train/LS_asr.yaml config/accelerate/zero3.yaml
```

You should see a message like "Detected running inside an interactive srun step..." if the script chooses the direct-launch path.

### 3) Local single-machine run (no SLURM) 
Launch training directly on your machine (it will start `accelerate` with local settings):

```bash
bash bash/train_with_config.sh config/train/LS_asr.yaml config/accelerate/fsdp.yaml
```

This is useful for small-scale debugging and development.

### 4) Direct `accelerate` invocation (advanced)
If you prefer to call `accelerate` yourself, you can run:

```bash
accelerate launch --config_file config/accelerate/zero3.yaml src/train.py --config config/train/LS_asr.yaml
```

This is equivalent to the helper but useful when experimenting with different `accelerate` flags.

---

### Run inside a Singularity / Apptainer container (SLURM)
If you built a container image (e.g., `melt_cuda126.sif`) you can run training inside the image using the provided wrapper `bash/run_train_singularity.sbatch`. The wrapper:

- validates the image path (`SINGULARITY_IMG=/path/to/image.sif`)
- binds your project, tmp and dataset directories into the container
- exports the in-container venv path (`SINGULARITYENV_VENV_PATH`) so the runner auto-activates the virtualenv
- runs the container under `srun` so multi-node jobs behave correctly

Example submission (host-side):

```bash
SINGULARITY_IMG=/gpfs/projects/epor32/hugop/melt_cuda126.sif \
  TMPDIR_HOST=/gpfs/scratch/epor32/hugop/tmp \
  sbatch bash/run_train_singularity.sbatch config/train/LS_asr.yaml config/accelerate/fsdp.yaml
```

Notes:
- The wrapper does **not** automatically set HF-related environment variables (`HF_HOME`, `HF_DATASETS_CACHE`); if you rely on these inside the container, set them explicitly with `SINGULARITYENV_HF_HOME` and `SINGULARITYENV_HF_DATASETS_CACHE` or provide them in a site init script.
- To control where temporary files are stored inside the container, set `TMPDIR_HOST` when submitting; it will be bound into the container at `/workspace/tmp`.

---

### Notes & tips 
- The helper script (`bash/run_train.sh`) exports commonly useful env vars when run directly (e.g., `SCRATCH_DIR`, `LOCAL_DATASETS_DIR`); you can override them at invocation time, e.g. `SCRATCH_DIR=/my/tmp bash bash/train_with_config.sh ...`.
- The script parses `gradient_accumulation_steps` from the YAML to pass that through to `accelerate` when possible.
- For interactive use, the script detects `SLURM_STEP_ID` / `SLURM_PROCID` and avoids creating nested `srun` steps.
- If you need a dry run, use the `--dry_run` flag supported by `src/train.py` (see the training CLI for more options).

If you'd like I can add a short example `sbatch` job script and a one-line `Makefile` target to make launching even easier.

---