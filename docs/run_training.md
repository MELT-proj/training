# Running training

Training runs in three contexts — a local machine, a SLURM cluster on the native
host venv, and a SLURM cluster inside a Singularity/Apptainer container — all
through **one launcher**, `bash/run_train.sh`. It activates a virtualenv (if
present) and runs `accelerate launch python -m melt.training.train`. Under SLURM
it fans out with `srun` and derives world size / per-node rank automatically.

```
                       (pick your site file)
infra/runners/submit-native.sh    <site> <acc.yaml> [args...]   # SLURM, no container
infra/runners/submit-container.sh <site> <acc.yaml> [args...]   # SLURM + container
        │  sources infra/runners/sites/<site>.sh (host paths + SBATCH_ARGS)
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ bash/run_train_singularity.sbatch   (container mode only, host side)   │
│   validates SINGULARITY_IMG, computes bind mounts + SINGULARITYENV_*   │
│   srun → singularity exec → bash/run_train.sh                          │
└──────────────────────────────────────────────────────────────────────┘
        ▼
bash/run_train.sh <acc.yaml> [train args...]     ← SAME script in all contexts:
   • local:            ./bash/run_train.sh ...
   • SLURM native:     sbatch [flags] bash/run_train.sh ...
   • interactive srun: bash bash/run_train.sh ...   (detects step, no nested srun)
   • inside container: invoked by the sbatch shim
        ▼
accelerate launch → python -m melt.training.train --config config/train/X.yaml ...
```

The first positional argument is always the **accelerate config**; everything
after it is forwarded to the training entrypoint as OmegaConf dot-notation
overrides (e.g. `--trainer.max_steps 1000`).

## How do I run it?

| Context | Command |
|---|---|
| **Local dev** (your machine) | `./bash/run_train.sh config/accelerate/zero1.yaml --config config/train/LS_asr.yaml` |
| **Interactive `srun` step** | Same command, `bash bash/run_train.sh ...`, from inside your `srun`/`salloc` shell. The launcher detects the existing step (`SLURM_STEP_ID`/`SLURM_PROCID`) and runs `accelerate` **directly** — do **not** wrap it in another `srun`. |
| **SLURM native** (host venv) | `infra/runners/submit-native.sh <site> config/accelerate/zero3.yaml --config config/train/X.yaml ...` — or raw: `sbatch [flags] bash/run_train.sh config/accelerate/zero3.yaml --config config/train/X.yaml` |
| **SLURM + container** | `infra/runners/submit-container.sh <site> config/accelerate/zero3.yaml --config config/train/X.yaml ...` — or raw: `SINGULARITY_IMG=/path/melt.sif sbatch [flags] bash/run_train_singularity.sbatch config/accelerate/zero3.yaml --config config/train/X.yaml` |

The `submit-*.sh` runners just source a per-site file for paths and SLURM flags;
see [infra/runners/README.md](../infra/runners/README.md). Accelerate configs live
in [config/accelerate/](../config/accelerate/) (`zero1`, `zero3`, `fsdp2`).

## Environment variables

Set these by exporting before launch (locally) or in a site file (runners). The
launcher fills sane defaults, so a bare local run needs none of them.

| Variable | Consumed by | Default | Notes |
|---|---|---|---|
| `LOCAL_DATASETS_DIR` | train YAML `${oc.env:LOCAL_DATASETS_DIR}` | `./shar` | SHAR datasets root. Container: fixed to `/workspace/shar` (host dir bind-mounted). |
| `OUTPUT_DIR` | train YAML `${oc.env:OUTPUT_DIR}` | — | Checkpoints/logs. Container: fixed to `/workspace/outputs` (host `OUTPUT_DIR` is the bind source). |
| `HF_HOME` | launcher / HF libs | `$HOME/.cache/huggingface` | Model/tokenizer cache. Container: fixed to `/workspace/hf_cache`; host `HF_HOME` is bind-mounted (unset ⇒ not shared, warns). |
| `VENV_PATH` | launcher | `/workspace/venv/bin/activate` | Native: point at your host venv. Missing file ⇒ warn and use current python. Container: set to the in-image venv. |
| `SINGULARITY_IMG` | container shim | — | **Required** for container mode; ignored natively. |
| `SINGULARITY_BIN` | container shim | autodetect (`apptainer`→`singularity`) | Force a specific runtime binary. |
| `TMPDIR_HOST` | container shim | `/tmp` | Host tmp bind-mounted to `/workspace/tmp` (caches/JIT). Native uses `TMPDIR`. |
| `WANDB_PROJECT` | launcher / wandb | `melt` | |
| `WANDB_MODE` | launcher / wandb | `online` | Use `offline` on air-gapped sites. |
| `HF_HUB_OFFLINE` | launcher / HF libs | unset (online) | Set `1` on air-gapped sites (pre-download with `infra/setup/download_hf_models.sh`). |
| `MASTER_ADDR` | launcher | SLURM: first node (`scontrol`); local: `127.0.0.1` | Container mode computes it on the host and forwards it. |
| `MASTER_PORT` | launcher | `6000` | Bump if it clashes on the node. |
| `GPUS_PER_NODE` | launcher | SLURM: `SLURM_GPUS_ON_NODE`; local: `nvidia-smi` count or `1` | Rarely set by hand. |
| `GPU_MEM_MONITORING` | launcher | `0` | `1` ⇒ per-node GPU-memory CSV under `logs/`. |

## Container notes

- **Fixed `/workspace/*` layout** (host paths are bind sources): project
  `/workspace/training`, venv `/workspace/venv`, datasets `/workspace/shar`,
  outputs `/workspace/outputs`, HF cache `/workspace/hf_cache`, tmp `/workspace/tmp`.
- **Output-path gotcha:** `--trainer.output_dir` must use the *container* path,
  e.g. `/workspace/outputs/my_run` — the host `OUTPUT_DIR` only says where that
  mount comes from. (Native runs use the host path, e.g. `$OUTPUT_DIR/my_run`.)
- **`logs/` must exist before `sbatch`:** SLURM won't create the `--output`
  directory and the job dies silently without it. The `submit-*.sh` runners
  `mkdir -p logs` for you; if you call `sbatch` directly, do it yourself.
- The image ships the CUDA toolkit (built from a `-devel` base), so JIT-compiled
  ops (e.g. DeepSpeed) build inside the container — no host toolkit is bind-mounted.
  Rebuild the image with `infra/setup/build_singularity.sh` after pulling changes
  to `infra/Singularity.def`.
- **Smoke test:** add `--run.dry_run true` (or `--trainer.max_steps 10`) to check
  the pipeline end-to-end before a full run.
