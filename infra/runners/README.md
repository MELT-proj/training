# Runners

Thin submit wrappers around the training launchers, so you don't retype cluster
paths and SLURM flags every run.

```
infra/runners/
  submit-native.sh      # SLURM job, native host venv (-> bash/run_train.sh)
  submit-container.sh   # SLURM job, Singularity/Apptainer (-> bash/run_train_singularity.sbatch)
  sites/
    example.sh          # commented template — copy this to add a cluster
    artemis.sh          # Artemis (a6000 / h100)
    mn5.sh              # BSC MareNostrum5 (air-gapped, offline)
```

A **site file** (`sites/<name>.sh`) holds only `export`s (host storage paths, the
`.sif` path, the venv path) and one `SBATCH_ARGS` array (partition/QoS/account/time).
It contains no logic and no experiment arguments.

**Experiment args live on the CLI** (and model/data config in `config/train/*.yaml`) —
never in runners. That keeps a runner reusable across every experiment.

Run everything **from the repo root**.

## Usage

```
infra/runners/submit-native.sh    <site> <accelerate_config> [train args...]
infra/runners/submit-container.sh <site> <accelerate_config> [train args...]
```

### Native (host venv)

```bash
EXP=SFT-v1.2.7-smoke
infra/runners/submit-native.sh artemis config/accelerate/fsdp2.yaml \
  --config config/train/SFT-v1.2.7.yaml \
  --trainer.max_steps 1000 \
  --trainer.eval_steps 200 \
  --trainer.save_steps 200 \
  --trainer.warmup_steps 20 \
  --trainer.report_to wandb \
  --trainer.output_dir "$OUTPUT_DIR/$EXP" \
  --run.exp_name "$EXP" \
  --data.train_ds.buffer_size 800000 \
  --trainer.eval_on_start false
```

### Container

Identical, except `--trainer.output_dir` uses the **container** path:

```bash
EXP=SFT-v1.2.7-smoke
infra/runners/submit-container.sh artemis config/accelerate/fsdp2.yaml \
  --config config/train/SFT-v1.2.7.yaml \
  --trainer.max_steps 1000 \
  --trainer.eval_steps 200 \
  --trainer.save_steps 200 \
  --trainer.warmup_steps 20 \
  --trainer.report_to wandb \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name "$EXP" \
  --data.train_ds.buffer_size 800000 \
  --trainer.eval_on_start false
```

> ⚠️ **Container path gotcha.** In container mode the host `OUTPUT_DIR` is only the
> bind *source*; inside the container it is always mounted at `/workspace/outputs`.
> So `--trainer.output_dir` must be `/workspace/outputs/...`, not the host path.
> (Datasets `/workspace/shar`, HF cache `/workspace/hf_cache`, tmp `/workspace/tmp`
> are handled for you — only `--trainer.output_dir` is passed on the CLI.)

Multi-node: add `--nodes=N` to the site's `SBATCH_ARGS`; the launcher derives
world size and per-node rank from SLURM automatically.

## Porting to a new server

1. `cp sites/example.sh sites/<name>.sh` and fill in the storage paths
   (`HF_HOME`, `OUTPUT_DIR`, `LOCAL_DATASETS_DIR`, `TMPDIR_HOST`) and the mode you
   need (`SINGULARITY_IMG` for container, `VENV_PATH` for native).
2. Set `SBATCH_ARGS` for the cluster (partition, QoS, account, time, nodes, gpus).
3. Get the image: build it on the cluster with `infra/setup/build_singularity.sh`,
   or copy an existing `melt_cuda126.sif` over. (Container mode only.)
4. Air-gapped site? Pre-download models with `infra/setup/download_hf_models.sh`,
   then set `export HF_HUB_OFFLINE=1` and `export WANDB_MODE=offline` in the site file
   (see `sites/mn5.sh`).
5. Smoke-test with `--trainer.max_steps 10` before a real run.
