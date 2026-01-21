# Runner Scripts

This directory contains environment-specific runner scripts that wrap the main training scripts with hardcoded paths and settings for different HPC environments.

## Purpose

These scripts simplify launching training jobs by pre-configuring environment variables and paths specific to each HPC cluster, eliminating the need to manually set them each time.

## Naming Convention

```
<server>-<launch_method>-<mode>.sh
```

- **`server`**: Name of the HPC cluster (e.g., `artemis`, `leonardo`, `local`)
- **`launch_method`**: Job submission method
  - `sbatch` - Submit batch job via SLURM sbatch
  - `srun` - Interactive execution via SLURM srun
- **`mode`**: Execution environment
  - `sif` - Run inside Singularity/Apptainer container
  - `native` - Run directly on the host environment (no container)

## Examples

- `artemis-sbatch-sif.sh` - Submit batch job on Artemis cluster using Singularity container
- `leonardo-srun-native.sh` - Interactive run on Leonardo cluster without container
- `local-srun-sif.sh` - Local interactive run with container

## Usage

Each script wraps either `bash/run_train.sh` (native) or `bash/run_train_singularity.sbatch` (container) with appropriate environment variables pre-set.

### Running a Script

```bash
# From project root
bash bash/runners/artemis-sbatch-sif.sh

# Or make it executable
chmod +x bash/runners/artemis-sbatch-sif.sh
./bash/runners/artemis-sbatch-sif.sh
```

### Required Environment Variables

These are set by each runner script:

#### For Native Execution (`run_train.sh`)
- `VENV_PATH` - Path to Python virtual environment activation script
- `LOCAL_DATASETS_DIR` - Path to dataset storage
- `OUTPUT_DIR` - Path for training outputs (checkpoints, logs)
- `WANDB_PROJECT` - W&B project name (optional)
- `WANDB_MODE` - W&B mode: online/offline (optional)

#### For Container Execution (`run_train_singularity.sbatch`)
- `SINGULARITY_IMG` - Path to Singularity/Apptainer .sif image
- `LOCAL_DATASETS_DIR` - Path to dataset storage (bind-mounted)
- `OUTPUT_DIR` - Path for training outputs (bind-mounted)
- `TMPDIR_HOST` - Path for temporary files (bind-mounted)

## Creating New Runner Scripts

When working on a new cluster:

1. Copy an existing runner script as template
2. Rename following the convention: `<new_server>-<method>-<mode>.sh`
3. Update the environment variables to match your cluster's paths
4. Update SLURM parameters (partition, QoS, time limits, etc.)
5. Adjust training arguments as needed

### Template

```bash
#!/bin/bash

# Export environment variables
export OUTPUT_DIR=/path/to/outputs
export LOCAL_DATASETS_DIR=/path/to/datasets
export SINGULARITY_IMG=/path/to/container.sif
export TMPDIR_HOST=/path/to/tmp

# Submit job with cluster-specific SLURM settings
sbatch --time=01:00:00 --nodes=1 --gres=gpu:2 --partition=gpu \
  ./bash/run_train_singularity.sbatch config/accelerate/zero3.yaml \
  --config-file config/train/LS_asr.yaml
```

## Notes

- Always use absolute paths in environment variables
- Export variables before calling sbatch/srun to ensure they're available in the job environment
- Keep cluster-specific configuration (partitions, QoS, paths) in these runner scripts
- Keep training configuration (hyperparameters, model settings) in YAML files under `config/train/`
