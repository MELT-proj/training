# HPC runbook: running MELT training on MN5

Practical guide to taking over training runs on BSC MareNostrum5. For how the
launchers themselves work, see [run_training.md](run_training.md); this document is
the operational side — getting code and images onto an air-gapped cluster and
submitting jobs there.

Two machines are involved:

| | role | why |
|---|---|---|
| **artemis** | builds the container image | has internet + apptainer; MN5 has neither |
| **mn5** (`glogin1.bsc.es`) | runs training | the GPUs |

**MN5 compute and login nodes have no outbound internet.** Everything — code,
images, HF models — is pushed to it from outside. That constraint explains most
of what follows.

---

## 0. Mental model (read this first)

The single most useful thing to understand:

> **The code that runs is the host checkout, not the copy inside the image.**

The sbatch shim bind-mounts your repo over `/workspace/training` and sets the
working directory there, so Python imports resolve to the bind-mounted source.
The repo baked into the `.sif` at build time is only used to install the venv.

Consequences:

- **Changing training code ⇒ sync the repo. No image rebuild.**
- **Changing dependencies** (`pyproject.toml` / `uv.lock`) **⇒ rebuild the image.**

Fixed paths inside the container (host dirs are bind sources, set per site in
`infra/runners/sites/mn5.sh`):

| container | host |
|---|---|
| `/workspace/training` | your repo checkout (submit dir) |
| `/workspace/shar` | `$LOCAL_DATASETS_DIR` |
| `/workspace/outputs` | `$OUTPUT_DIR` |
| `/workspace/hf_cache` | `$HF_HOME` |
| `/workspace/tmp` | `$TMPDIR_HOST` |
| `/workspace/venv` | *(in image)* |

---

## 1. One-time setup

**SSH aliases.** Everything is driven by ssh aliases so no usernames are baked
into the repo. In `~/.ssh/config`:

```
Host mn5
    HostName glogin1.bsc.es
    User <your-bsc-user>

Host mn5transfer
    HostName transfer1.bsc.es
    User <your-bsc-user>
```

Get your key into `mn5:~/.ssh/authorized_keys`. Verify with `ssh mn5 hostname`
→ `glogin1`.

**Clone the repo on MN5.** It has no internet, so you cannot `git clone` there.
Push to it instead (see §2), or bootstrap once with a bundle:

```bash
git bundle create /tmp/melt.bundle <branch>          # on your laptop
scp /tmp/melt.bundle mn5:~/                          # copy over
ssh mn5 'git clone -b <branch> ~/melt.bundle ~/training'
```

**Check the site file** `infra/runners/sites/mn5.sh` matches your project
paths, then run everything below **from the repo root**.

---

## 2. Sync code to MN5

```bash
infra/sync_repo.sh mn5                 # push commits (default)
infra/sync_repo.sh mn5 --dirty         # rsync working tree, uncommitted included
infra/sync_repo.sh mn5 --dry-run       # show what would move
```

Git over SSH needs no internet on the far end, so MN5 is just a git remote.
The default mode pushes commits *and* updates the remote working tree
(`receive.denyCurrentBranch=updateInstead`), and **refuses if that tree is
dirty** rather than clobbering it — if it complains, commit or stash on MN5.

`runs/` is gitignored but holds the configs you launch with, so it is rsynced
in both modes. Without that, you would edit a config locally and silently run
the old one.

Prefer the default over `--dirty` for anything whose results you'll want to
reproduce: a run launched from an rsynced dirty tree has no commit to trace a
checkpoint back to.

---

## 3. Build and ship an image

**Only needed when dependencies change.** Code changes need §2, not this.

```bash
# On artemis, from the repo root
source /etc/profile.d/02-lmod.sh && module load apptainer

BUILD_TMPDIR=/mnt/data-artemis/$USER/tmp \
UV_CACHE_DIR=/mnt/scratch-artemis/$USER/.cache/uv \
  infra/setup/build_singularity.sh /path/to/melt_cuda126.sif
```

Then ship it via the transfer node (not the login node):

```bash
rsync -avh --partial --info=progress2 \
  /path/to/melt_cuda126.sif mn5transfer:/gpfs/projects/epor48/melt-data/
```

Notes that will save you time:

- **Build with `apptainer`, not `singularity`.** The def file's `%setup` needs
  `APPTAINER_ROOTFS`, which SingularityCE does not export. The build script
  loads the module and refuses otherwise.
- On artemis, `apptainer exec` fails (`starter-suid` lacks the setuid bit) —
  use `singularity exec` to *inspect* an image. Build with apptainer, run with
  singularity.
- **Expect ~50 minutes.** Artemis login sessions are capped by systemd at
  `CPUQuotaPerSecUSec=900ms` — 0.9 of a core, shared across *all* your SSH
  sessions on that box. Nothing is wrong; it's just slow.
- Must be run **from the repo root** — the def file rsyncs `.` as its build
  context.
- Result is ~7.6 GB and contains the CUDA toolkit (`nvcc`), since the base is
  `-devel`. No host CUDA is bind-mounted.

Sanity-check a new image before trusting it:

```bash
singularity exec melt_cuda126.sif bash -c \
  'source /workspace/venv/bin/activate; nvcc --version | grep release; \
   python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"'
```

---

## 4. Sync datasets

Shar data lives outside the repo and is synced separately, from artemis:

```bash
SHAR_SRC=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar
SHAR_DST=mn5transfer:/gpfs/projects/epor48/melt-data/shar

rsync -avhn --no-owner --no-group --info=stats2 "$SHAR_SRC"/ "$SHAR_DST"/   # dry run first
rsync -avh  --no-owner --no-group --partial --info=progress2 "$SHAR_SRC"/ "$SHAR_DST"/
```

`--no-owner --no-group` because the source and BSC target have different
groups; plain `-a` spams chgrp failures. No `-z`: shar audio is already
compressed. Run long transfers inside `tmux`. Check `bsc_quota` first — this
data is measured in TB.

---

## 5. Submit a run

```bash
EXP=my-experiment

infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
  --config runs/debug_VP-only.yaml \
  --trainer.max_steps 60 \
  --trainer.eval_steps 30 --trainer.save_steps 30 \
  --trainer.per_device_eval_batch_size 4 \
  --trainer.report_to wandb \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name $EXP
```

Everything after the accelerate config is an OmegaConf dot-notation override.

**Three things that will bite you:**

1. **`--trainer.output_dir` must be the CONTAINER path** (`/workspace/outputs/…`).
   The host `OUTPUT_DIR` is only the bind source; a host path here fails.
2. **Pass `--trainer.per_device_eval_batch_size` explicitly.** The YAMLs set
   `-1` ("Lhotse handles batching"), which the train path understands but the
   eval path does not — it crashes at the first eval ([#32](https://github.com/MELT-proj/training/issues/32)).
   `4` is known-good; `16` OOMs.
3. **Adjust the QoS for the job.** `SBATCH_ARGS` lives in the site file:
   `acc_debug` = 2 h cap, **1 job per user**, high priority (good for smoke
   tests); `acc_ehpc` = 3 days, normal priority (real runs).

Model weights must already be in `$HF_HOME` — compute nodes run with
`HF_HUB_OFFLINE=1`. Pre-download with `infra/setup/download_hf_models.sh`.

---

## 6. Monitor and verify

```bash
squeue -u $USER
tail -f logs/melt-train-container.<jobid>.out
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -X
```

A healthy start looks like this — worth knowing so you can spot where it broke:

```
[container] image:    .../melt_cuda126.sif
[container] project:  <your repo> -> /workspace/training
Cuda compilation tools, release 12.6            <- nvcc FROM THE IMAGE
[run_train] starting (context: slurm, nodes=1, gpus/node=4, world_size=4)
[run_train] inside an srun step; launching directly    <- correct, no nested srun
world_size: 4, ... is_distributed: True
```

A completed run leaves, under `$OUTPUT_DIR/$EXP/`: `checkpoint-N/` dirs (with
`pytorch_model_fsdp_0`, `optimizer_0`, `sampler`, `trainer_state.json`), the
processor/tokenizer files, and `resolved_config.json`. Offline wandb lands in
`$OUTPUT_DIR/wandb/`; sync later from a machine with internet (`wandb sync`).

**GPU memory profiling:** `export GPU_MEM_MONITORING=1` before submitting to get
`logs/gpu_mem_<jobid>_node<N>.csv`. Note it samples every 30 s, so it *aliases*
and under-reports true peaks — treat its numbers as a lower bound.

---

## 7. Gotchas worth internalising

- **Eval dominates wall-clock.** In a 60-step run, three evals took ~71% of the
  job. Budget for it, and keep `eval_steps` sane on short runs.
- **Eval cuts are sorted longest-first**, so the first batches are the peak in
  both memory and time (≈4× memory and ≈3.4× per-batch time vs the tail). A
  spot `nvidia-smi` reading taken mid-eval samples the cheap tail and will
  badly overstate your headroom — don't size batches from it. See
  [#33](https://github.com/MELT-proj/training/issues/33).
- **`logs/` must exist before `sbatch`** or SLURM kills the job silently. The
  `submit-*.sh` runners `mkdir -p logs` for you; raw `sbatch` does not.
- **Site files use bare `export`**, so `VAR=x infra/runners/submit-container.sh …`
  loses to the site value. Edit the site file to override.
- **Smoke-test first:** `--trainer.max_steps 10 --trainer.eval_on_start false`.

## Known issues

| | |
|---|---|
| [#32](https://github.com/MELT-proj/training/issues/32) | eval ignores the `per_device_eval_batch_size=-1` sentinel |
| [#33](https://github.com/MELT-proj/training/issues/33) | eval batches by fixed count over length-sorted cuts |
| [#34](https://github.com/MELT-proj/training/issues/34) | eval dataloader capped at one worker, starving the GPUs |
| [#35](https://github.com/MELT-proj/training/issues/35) | shim oversubscribes `OMP_NUM_THREADS` ~4× |
