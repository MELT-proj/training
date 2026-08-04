# HPC runbook: running MELT training on MN5

Guide for taking over training runs on BSC MareNostrum5. For how the launchers
work internally, see [run_training.md](run_training.md); this is the operational
side — running experiments, monitoring them, and getting results back.

It assumes you have accounts on **artemis** and **MN5**.

**[Part A](#part-a--one-time-preparation) is one-time setup that is normally already done for you.**
Skim it so you know what exists and why, then work from
**[Part B](#part-b--running-an-experiment)**, which is the day-to-day loop.

---

## The two machines

| machine | role | has internet? |
|---|---|---|
| **your laptop** | edits code, pushes to MN5 and GitHub | yes |
| **artemis** (`ssh artemis`) | builds container images; stores results pulled back from MN5 | yes |
| **mn5** (`ssh mn5` → `glogin1.bsc.es`) | runs training on 4×H100 nodes | **no** |
| **mn5transfer** (`transfer1.bsc.es`) | bulk data transfer endpoint for MN5 | — |

**MN5 has no outbound internet.** It cannot `git pull`, download HF models, or
reach W&B. Everything is pushed to it from outside, and results are pulled back.
That single constraint explains most of this document.

Every command below is tagged with where to run it: **[laptop]**, **[artemis]**,
or **[mn5]**.

### Check your access

```bash
# [laptop]
ssh artemis hostname     # -> artemis
ssh mn5 hostname         # -> glogin1
```

If `ssh mn5` fails, add to `~/.ssh/config` and get your public key into
`mn5:~/.ssh/authorized_keys`:

```
Host mn5
    HostName glogin1.bsc.es
    User <your-bsc-user>

Host mn5transfer
    HostName transfer1.bsc.es
    User <your-bsc-user>
```

---

## Mental model (read this once)

The single most useful thing to know:

> **The code that runs is the checkout on MN5, not the copy inside the `.sif`.**

The sbatch shim bind-mounts your MN5 repo over `/workspace/training` and sets
the working directory there, so Python imports resolve to the bind-mounted
source. The repo baked into the image is only used to build the venv.

| you changed… | what to do | cost |
|---|---|---|
| training code, configs | sync the repo (§B1) | seconds |
| `pyproject.toml` / `uv.lock` | rebuild the image (§A1) | ~50 min |

Fixed container paths (host dirs are bind sources, set in
`infra/runners/sites/mn5.sh`):

| inside container | host (MN5) |
|---|---|
| `/workspace/training` | your repo checkout |
| `/workspace/shar` | `$LOCAL_DATASETS_DIR` |
| `/workspace/outputs` | `$OUTPUT_DIR` |
| `/workspace/hf_cache` | `$HF_HOME` |
| `/workspace/tmp` | `$TMPDIR_HOST` |

---

# Part A — one-time preparation

Setup done once, not per experiment. **Ownership splits in two:**

| steps | owner | why |
|---|---|---|
| **A1** container image, **A2** SHAR datasets, **A3** model checkpoints | **the project maintainer** | shared, account-independent artefacts living in the group project space. Done once for everyone; you should not need to repeat them. |
| **A4** repo checkout on MN5 | **each collaborator, for themselves** | lives in *your* MN5 home directory under *your* account, so every person joining the project does this once |

So if you are picking up training runs: **do A4, skim A1–A3** to understand what
exists and why, and only revisit them if something is genuinely missing or a
dependency changed.

## A1. Build and ship the container image

**When:** dependencies changed (`pyproject.toml` / `uv.lock`). **Not** for code changes.

```bash
# [artemis] from the repo root
source /etc/profile.d/02-lmod.sh && module load apptainer

BUILD_TMPDIR=/mnt/data-artemis/$USER/tmp \
UV_CACHE_DIR=/mnt/scratch-artemis/$USER/.cache/uv \
  infra/setup/build_singularity.sh /mnt/scratch-artemis/$USER/melt-data/melt_cuda126.sif
```

Verify before shipping:

```bash
# [artemis]  note: singularity to RUN, apptainer to BUILD
singularity exec /mnt/scratch-artemis/$USER/melt-data/melt_cuda126.sif bash -c \
  'source /workspace/venv/bin/activate; nvcc --version | grep release;
   python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"'
# expect: release 12.6 / 2.9.1+cu126 2.8.3
```

Ship it via the **transfer node**, not the login node (~7.6 GB, ~5 min):

```bash
# [artemis]
rsync -avh --partial --info=progress2 \
  /mnt/scratch-artemis/$USER/melt-data/melt_cuda126.sif \
  mn5transfer:/gpfs/projects/epor48/melt-data/
```

Keep the previous image under a dated name (e.g.
`melt_cuda126_pre-devel-20260411.sif`) until the new one has a successful run.

Things that will otherwise cost you an hour:

- **Build with `apptainer`, not `singularity`** — the def file's `%setup` needs
  `APPTAINER_ROOTFS`, which SingularityCE doesn't export. The build script loads
  the module and refuses otherwise. Conversely `apptainer exec` is broken on
  artemis (`starter-suid` lacks the setuid bit), so *inspect* images with
  `singularity exec`.
- **Run from the repo root** — the def file rsyncs `.` as its build context.
- **~50 minutes is normal.** Artemis login sessions are capped by systemd at
  `CPUQuotaPerSecUSec=900ms` — 0.9 of one core, shared across *all* your SSH
  sessions on that host. Nothing is broken; it's throttled.

## A2. Sync SHAR datasets

```bash
# [artemis]
SHAR_SRC=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar
SHAR_DST=mn5transfer:/gpfs/projects/epor48/melt-data/shar

rsync -avhn --no-owner --no-group --info=stats2 "$SHAR_SRC"/ "$SHAR_DST"/    # dry run
rsync -avh  --no-owner --no-group --partial --info=progress2 "$SHAR_SRC"/ "$SHAR_DST"/
```

`--no-owner --no-group` because source and BSC target have different groups and
plain `-a` spams chgrp failures. No `-z`: shar audio is already compressed. Use
`tmux` — this is measured in TB. Check headroom with `bsc_quota` first.

## A3. Stage the model checkpoints (encoders + text decoders)

**This is the step most likely to block an ablation.** MN5 runs with
`HF_HUB_OFFLINE=1`, so `from_pretrained` never contacts the Hub: any checkpoint
not already in `$HF_HOME` fails at model load, minutes into a queued job. Both
halves of the model come from the Hub — the **audio encoder**
(`model.encoder.name`) and the **text decoder** (`model.decoder.name`) — so
every backbone you intend to ablate must be staged **before** you start.

Download on a machine with internet, then ship the cache:

```bash
# [artemis] or [laptop] -- populates the local HF cache
python -c 'from huggingface_hub import snapshot_download; snapshot_download("<org>/<model-id>")'
# (or use infra/setup/download_hf_models.sh for the standard set)

# [artemis] ship the cache to MN5 via the transfer node
rsync -avh --partial --info=progress2 \
  $HF_HOME/hub/ mn5transfer:/gpfs/projects/epor48/melt-data/hf_cache/hub/
```

Gated repos (e.g. `meta-llama/*`) need `huggingface-cli login` on the machine
doing the download; the token is never needed on MN5.

**Always check the cache before planning a sweep** — this listing is the source
of truth for what will load offline:

```bash
# [mn5]
ls $HF_HOME/hub/ | sed 's/^models--//; s/--/\//'
```

Two things worth checking deliberately, because both fail the same way and only
after the job is scheduled:

- **Both halves of a base-vs-instruct pair.** A sweep with one half missing runs
  the arm that exists and dies on the other.
- **Every encoder in an audio-stack sweep,** not just the default one.

Checkpoints are large and the queue wait is long, so stage everything a planned
set of experiments needs in one pass rather than discovering a gap mid-sweep.

## A4. Get the repo onto MN5

MN5 can't clone from GitHub. Bootstrap once from your laptop:

```bash
# [laptop]
git bundle create /tmp/melt.bundle <branch>
scp /tmp/melt.bundle mn5:~/
ssh mn5 'git clone -b <branch> ~/melt.bundle ~/training'
```

After that, keep it current with `infra/sync_repo.sh` (§B1) — no more bundles.

---

# Part B — running an experiment

The loop you'll actually repeat: **sync → configure → submit → monitor → retrieve.**

## B1. Sync your code and configs to MN5

```bash
# [laptop] from the repo root
infra/sync_repo.sh mn5                 # push commits (default)
infra/sync_repo.sh mn5 --dirty         # rsync working tree, uncommitted included
infra/sync_repo.sh mn5 --dry-run       # show what would move
```

Git over SSH needs no internet on the far end, so MN5 is just a git remote. The
default mode pushes commits **and** updates the remote working tree, and
**refuses if that tree is dirty** rather than clobbering it — if it complains,
commit or stash on MN5 first.

`runs/` is gitignored but holds the configs you launch with, so it is rsynced in
both modes. Without that you'd edit a config locally and silently run the old one.

Use the default for anything you'll want to reproduce: a run launched from a
`--dirty` tree has no commit to trace its checkpoints back to.

## B2. Configure the ablation

See [Configuring ablations](#configuring-ablations) below for what to change and
where.

## B3. Submit

```bash
# [mn5] from ~/training
EXP=ablation-adapter-4layer

infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
  --config runs/my_config.yaml \
  --trainer.max_steps 2000 \
  --trainer.eval_steps 500 --trainer.save_steps 500 \
  --trainer.save_total_limit 3 \
  --trainer.per_device_eval_batch_size 4 \
  --trainer.report_to wandb \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name $EXP
```

**Three things that will bite you:**

1. **`--trainer.output_dir` must be the CONTAINER path** (`/workspace/outputs/…`).
   The host `OUTPUT_DIR` is only the bind source; a host path here fails.
2. **Always pass `--trainer.per_device_eval_batch_size` explicitly.** The YAMLs
   set `-1` ("Lhotse handles batching"), which the train path understands but the
   eval path does not — it crashes at the first eval. `4` is known-good; `16`
   OOMs on 64 GB H100s.
3. **Pick the QoS in the site file** (`SBATCH_ARGS` in `infra/runners/sites/mn5.sh`):

   | QoS | wall limit | jobs/user | use for |
   |---|---|---|---|
   | `acc_debug` | 2 h | **1** | smoke tests (high priority, short queue) |
   | `acc_ehpc` | 3 days | many | real runs |

   Note site files use bare `export`, so `VAR=x infra/runners/submit-container.sh …`
   loses to the site value — edit the file to override.

**Always smoke-test a new config first:** `--trainer.max_steps 10
--trainer.eval_on_start false` on `acc_debug`. It catches config errors in
minutes instead of after a queue wait.

## B4. Monitor

```bash
# [mn5]
squeue -u $USER
tail -f ~/training/logs/melt-train-container.<jobid>.out
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -X
```

A healthy startup looks like this — worth recognising so you can tell *where* a
failure happened:

```
[container] image:    .../melt_cuda126.sif
[container] project:  /gpfs/home/.../training -> /workspace/training
Cuda compilation tools, release 12.6                  <- nvcc FROM THE IMAGE
[run_train] starting (context: slurm, nodes=1, gpus/node=4, world_size=4)
[run_train] inside an srun step; launching directly   <- correct: no nested srun
world_size: 4, ... is_distributed: True
```

**Loss and metrics.** Each eval prints a metrics dict to the log — overall
`eval_loss`, `eval_wer`, `eval_cer`, plus per-language `eval_wer_<lang>` /
`eval_cer_<lang>`. To watch them:

```bash
# [mn5]
grep -oE "'eval_(loss|wer|cer)': [0-9.]+" ~/training/logs/melt-train-container.<jobid>.out
```

W&B runs **offline** (no internet), writing to `$OUTPUT_DIR/wandb/`. To see them
in the browser, sync from artemis (§B5).

**GPU memory:** `export GPU_MEM_MONITORING=1` before submitting to get
`logs/gpu_mem_<jobid>_node<N>.csv`. It samples every 30 s, so it *aliases* and
under-reports true peaks — treat its numbers as a lower bound.

## B5. Retrieve results

**W&B metrics** — there's a helper that pulls offline runs to artemis and syncs
them (edit the paths inside for your account):

```bash
# [artemis]
bash utils/sync_wandb.sh
```

It skips runs still being written to, so it's safe to run mid-training. Manually,
the same thing is:

```bash
# [artemis]
rsync -avh mn5transfer:/gpfs/projects/epor48/melt-data/outputs/wandb/wandb/ \
           /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb/
wandb sync /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb/offline-run-*
```

**Checkpoints** — pull via the transfer node:

```bash
# [artemis]
EXP=ablation-adapter-4layer
rsync -avh --partial --info=progress2 \
  mn5transfer:/gpfs/projects/epor48/melt-data/outputs/$EXP/ \
  /mnt/scratch-artemis/$USER/melt-data/outputs/$EXP/
```

A finished run's output directory contains:

| | |
|---|---|
| `checkpoint-<N>/` | `pytorch_model_fsdp_0/` (sharded), `optimizer_0/`, `sampler`, `scheduler.pt`, `trainer_state.json` |
| processor files | `config.json`, `tokenizer.json`, `preprocessor_config.json`, `chat_template.jinja`, … |
| `resolved_config.json` | **the config that actually ran**, after CLI overrides — the record of what produced these weights |

If you only need the final model, the top-level processor/config files plus the
last checkpoint are enough; skip the intermediate `checkpoint-*/` dirs with
`--exclude 'checkpoint-*'` to save a lot of transfer.

**Checkpoints are FSDP-sharded**, so merge before loading for inference:

```bash
# [artemis]
python utils/merge_fsdp_weight.py \
  --checkpoint_dir .../outputs/$EXP/checkpoint-2000/pytorch_model_fsdp_0 \
  --output_path   .../outputs/$EXP/merged
```

---

## Configuring ablations

Two ways to vary an experiment. Use CLI overrides for anything you'd sweep;
edit a YAML only for structural changes.

**CLI override** (dot notation, appended to the submit command). Preferred — the
config stays fixed and the diff between runs is visible in your shell history
and in `resolved_config.json`:

```bash
--optimization.adapter_lr 5e-4 --model.adapter.num_adapter_layers 4
```

**A new YAML in `runs/`** — for changing dataset mixes or many fields at once.
Copy an existing one, edit, and point `--config` at it. `runs/` is gitignored but
synced by `sync_repo.sh`, so it reaches MN5.

### What the knobs do

| knob | where | notes |
|---|---|---|
| `model.encoder.freeze` / `decoder.freeze` / `adapter.freeze` | YAML | defines the phase: **MA** freezes encoder+decoder and trains the adapter; **IFT** unfreezes the decoder |
| `model.decoder.name` | either | the LM backbone; **must be in `$HF_HOME`** (offline) |
| `model.adapter._type`, `num_adapter_layers`, `adapter_kernel_size`, `adapter_stride` | either | adapter architecture ablations |
| `model.ckpt` | either | start from a previous run — **this is how IFT consumes the MA checkpoint** |
| `data.train_ds.<...>.shar_path` | YAML | dataset mix; paths resolve under `/workspace/shar` |
| `data.train_ds.batch_duration` | either | **seconds of audio per batch** — the real batch-size lever for training (Lhotse dynamic bucketing), not `per_device_train_batch_size` |
| `data.train_ds.buffer_size` | either | shuffle buffer; smaller = faster startup for smoke tests |
| `data.prompt_template` | YAML | e.g. `"{audio_token}{lang}"` |
| `optimization.{encoder,decoder,adapter}_lr` | either | per-component LRs |
| `trainer.max_steps`, `eval_steps`, `save_steps`, `warmup_steps` | CLI | schedule |

### Running the two phases

MA and IFT are two sequential runs; the second points at the first's output:

```bash
# 1) MA — adapter only
--config runs/MA.yaml --run.exp_name MA-v1 --trainer.output_dir /workspace/outputs/MA-v1

# 2) IFT — starts from the MA checkpoint
--config runs/IFT.yaml --run.exp_name IFT-v1 --trainer.output_dir /workspace/outputs/IFT-v1 \
  --model.ckpt /workspace/outputs/MA-v1
```

`--model.ckpt` takes the **container** path, same rule as `output_dir`.

### The planned ablation axes

Four families of experiment, and what each one actually varies. In all cases the
**backbone must already be staged** (§A3) — that is the usual reason an ablation
job dies.

**1. Base vs. instruct text decoders.** Fix the audio stack (w2v-BERT +
conformer), ASR-only data in MA, and vary only the decoder:

```bash
--model.decoder.name Qwen/Qwen2.5-1.5B            --run.exp_name MA-qwen2.5-1.5B-base
--model.decoder.name Qwen/Qwen2.5-1.5B-Instruct   --run.exp_name MA-qwen2.5-1.5B-inst
```

Instruct backbones bring their own chat template; `data.apply_chat_template` and
`data.prompt_template` interact with that, so keep them fixed across a pair or
you are ablating two things at once.

**2. Audio stack.** Fix the decoder and vary the front end:

| component | knob | values |
|---|---|---|
| encoder | `model.encoder.name` | any staged encoder checkpoint |
| projector | `model.adapter._type` | **`conformer`**, **`mlp`**, **`qformer`** (all three are implemented) |
| conformer depth | `model.adapter.num_adapter_layers` | e.g. 2 vs 4 |
| conformer compression | `model.adapter.adapter_stride`, `adapter_kernel_size` | changes how hard the signal is compressed |

```bash
--model.adapter._type mlp --run.exp_name MA-proj-mlp
--model.adapter._type conformer --model.adapter.num_adapter_layers 4 --run.exp_name MA-proj-conf4
```

**3. MA/IFT task mix** (ASR-only, ST-only, ASR+ST in MA). This is a *dataset*
change, so it belongs in a YAML rather than on the CLI — copy a config in `runs/`
and edit the `data.train_ds` source list. Keep the total hours fixed across
variants, otherwise task mix and data volume are confounded.

**4. Backbone prior-knowledge analysis** (task performance vs. perplexity on
target transcripts). The perplexity side is a **separate offline evaluation of
the text backbone**, not a training run — it needs no GPU allocation and no
change here. Only the task-performance side comes from this pipeline; take it
from the eval metrics in §B4.

### Keeping ablations straight

- Use one `EXP` name for `--run.exp_name` **and** the last element of
  `--trainer.output_dir`, so the W&B run and the output directory match.
- Encode the variable in the name (`MA-adapter4layer-lr5e4`), not just a version
  number — you will not remember what `v7` changed.
- `resolved_config.json` in each output dir is the ground truth for what ran.
- Add `--trainer.overwrite_output_dir true` only when you intend to replace a
  previous run of the same name.

---

## Gotchas worth internalising

- **Eval dominates wall-clock.** In a 60-step run, three evals took ~71% of the
  job. Budget for it; keep `eval_steps` large on short runs.
- **Eval cuts are sorted longest-first**, so the first batches are the peak in
  both memory and time (≈4× memory, ≈3.4× per-batch time vs the tail). A spot
  `nvidia-smi` mid-eval samples the cheap tail and badly overstates headroom —
  don't size batches from it.
- **`logs/` must exist before `sbatch`** or SLURM kills the job silently. The
  `submit-*.sh` runners create it; raw `sbatch` does not.
- **Model weights must be pre-downloaded** — compute nodes are offline.

## Troubleshooting

| symptom | cause |
|---|---|
| `batch_size should be a positive integer, but got -1` | missing `--trainer.per_device_eval_batch_size` |
| Job exits instantly, no log | `logs/` didn't exist, or a bad `--output` path |
| `SINGULARITY_IMG not found` | image not shipped, or site-file path is stale |
| Model load fails / tries to reach the Hub | weights not in `$HF_HOME` (§A3) |
| `CUDA out of memory` during eval | eval batch too large — first batches are worst-case |
| Output dir "not empty" | add `--trainer.overwrite_output_dir true`, or pick a new `EXP` |
| Push rejected by `sync_repo.sh` | remote working tree is dirty — commit/stash on MN5 |
| Build writes to host `/workspace` | built with `singularity` instead of `apptainer` |

If something here does not match what you observe, check the repository's open
issues before debugging from scratch — known rough edges are tracked there.
