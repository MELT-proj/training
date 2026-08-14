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
| **mn5** (`ssh mn5` → `alogin1.bsc.es`) | runs training on 4×H100 nodes | **no** |
| **mn5transfer** (`transfer1.bsc.es`) | bulk data transfer endpoint for MN5 | — |

Use the **`alogin`** nodes, not `glogin`. MN5 has two login pools — `glogin*`
for the general-purpose partition and `alogin*` for the accelerated (GPU)
partition. Everything in this document targets the `acc_*` QoS, so submit from
`alogin1`.

**MN5 has no outbound internet.** It cannot `git pull`, download HF models, or
reach W&B. Everything is pushed to it from outside, and results are pulled back.
That single constraint explains most of this document.

Every command below is tagged with where to run it: **[laptop]**, **[artemis]**,
or **[mn5]**.

### Check your access

```bash
# [laptop]
ssh artemis hostname     # -> artemis
ssh mn5 hostname         # -> alogin1
```

If `ssh mn5` fails, add to `~/.ssh/config` and get your public key into
`mn5:~/.ssh/authorized_keys`:

```
Host mn5
    HostName alogin1.bsc.es
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

### Where things actually live on MN5

The project spans **two filesystems**, and things moved between them on
2026-08-07 when `gpfs_projects` began to fill up. Check where something is
before assuming; the group quotas are very different:

| filesystem | epor48 usage | headroom |
|---|---|---|
| `gpfs_projects` | 18.19 TB / 19.53 TB | **93% full — do not write new results here** |
| `gpfs_scratch` | 1.14 TB / 19.53 TB | plenty |

| what | path | notes |
|---|---|---|
| container images | `/gpfs/scratch/epor48/*.sif` | moved off projects |
| HF cache (`HF_HOME`) | `/gpfs/scratch/epor48/hf_cache` | moved off projects |
| outputs (`OUTPUT_DIR`) | `/gpfs/scratch/epor48/<user>/outputs` | **write your own — see §B3** |
| scratch dir (`TMPDIR_HOST`) | `/gpfs/scratch/epor48/<user>/tmp` | **write your own — see §B3** |
| Shar data, streaming | `/gpfs/projects/epor48/melt-data/shar` | read-only, stayed on projects |
| Shar data, **indexed** | `/gpfs/projects/epor48/melt-data/shar-indexed` | read-only, **use this one** |

Never delete anything under either Shar tree. The plain `shar/` tree must stay
free of `.idx` sidecars — lhotse 1.32 consumers and the Smurf collaborators
read it.

### Which container image

**The default is correct — you do not need to set `SINGULARITY_IMG`.** As of
2026-08-10, `melt_cuda126.sif` is a symlink to the lhotse 2 image:

```
melt_cuda126.sif -> melt_cuda126_lhotse2_td.sif
```

so the site-file default resolves to a stack matching `main`:

| image | lhotse | torchdata | works with `main` (≥0.5.0)? |
|---|---|---|---|
| `melt_cuda126.sif` → `…_lhotse2_td.sif` | 2.0.0a3 | 0.11.0 | **yes — the default** |
| `melt_cuda126_lhotse2_td.sif` | 2.0.0a3 | 0.11.0 | yes (same file) |
| `melt_cuda126_pre-lhotse2-20260804.sif` | 1.32.2 | absent | no — kept for reference only |
| `melt_cuda126_pre-devel-20260411.sif` | — | — | no — kept for reference only |

`pyproject.toml` pins `lhotse==2.0.0a3` and `torchdata>=0.11`, which only the
promoted image satisfies. The pre-lhotse2 images are retained deliberately, but
nothing on `main` runs on them.

The `_lhotse2_td.sif` name still resolves — the campaign scripts under
`tests/integration/lhotse2_campaign/` reference it directly — so both names
work and neither costs extra disk.

To pin a specific image (a trial build, or reproducing an old run) override it
as usual, and verify what you pinned before spending an allocation on it:

```bash
# [mn5]
module load singularity
singularity exec --bind /gpfs:/gpfs /gpfs/scratch/epor48/melt_cuda126.sif bash -lc \
  'source /workspace/venv/bin/activate
   python -c "import lhotse, torch, torchdata
print(lhotse.__version__, torch.__version__, torchdata.__version__)"'
# expect: 2.0.0a3 2.9.1+cu126 0.11.0+cpu
```

When you promote a future image, keep the previous one under a dated name
(`melt_cuda126_pre-<reason>-<YYYYMMDD>.sif`) and move the symlink — never
overwrite the file a running job is reading.

### Which Shar tree

Point `LOCAL_DATASETS_DIR` at **`shar-indexed`**. The indexed tree carries
`.idx` sidecars, which let the loader partition by sample index across the
(rank × worker) pool: an epoch is then exactly 100% of the data, and the
sampler position survives a resume. The plain tree streams, and does neither.

```bash
export LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar-indexed
```

Set `data.train_ds.indexed: true` in the config as well. The default (`null`)
auto-detects and will quietly fall back to streaming if `LOCAL_DATASETS_DIR`
points at the wrong tree; `true` fails loudly instead.

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

Ship it via the **transfer node**, not the login node (~7.6 GB, ~5 min), under
a **descriptive name of its own** — never straight onto `melt_cuda126.sif`,
which is a symlink and would be clobbered:

```bash
# [artemis]
NEW=melt_cuda126_<what-changed>.sif
rsync -avh --partial --info=progress2 \
  /mnt/scratch-artemis/$USER/melt-data/melt_cuda126.sif \
  mn5transfer:/gpfs/scratch/epor48/$NEW
```

Images live on **`gpfs_scratch`**, not `gpfs_projects` — they moved on
2026-08-07 and the old directory no longer exists.

**Promote it only after a successful run on it.** `melt_cuda126.sif` is the
name everything defaults to, so switching it is what puts a new image in front
of collaborators:

```bash
# [mn5] 1. prove the new image works, by pinning it for one real run
SINGULARITY_IMG=/gpfs/scratch/epor48/$NEW infra/runners/submit-container.sh mn5 …

# [mn5] 2. only then, retire the current default and move the symlink
cd /gpfs/scratch/epor48
mv -n "$(readlink -f melt_cuda126.sif)" melt_cuda126_pre-<reason>-$(date +%Y%m%d).sif
ln -sfn $NEW melt_cuda126.sif
ls -la melt_cuda126.sif
```

Retire, don't delete: the dated images are the only way to reproduce a run made
against them. Moving a symlink is atomic and never disturbs a job already
running on the old image, whereas overwriting a `.sif` in place corrupts one.

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
  $HF_HOME/hub/ mn5transfer:/gpfs/scratch/epor48/hf_cache/hub/
```

Gated repos (e.g. `meta-llama/*`) need `huggingface-cli login` on the machine
doing the download; the token is never needed on MN5.

**Always check the cache before planning a sweep** — this listing is the source
of truth for what will load offline:

```bash
# [mn5]  (HF_HOME is exported by the site file, not by your login shell)
ls /gpfs/scratch/epor48/hf_cache/hub/ | sed 's/^models--//; s/--/\//'
```

Staged as of 2026-08-10 — encoders and decoders together:

```
Qwen/Qwen2.5-0.5B          Qwen/Qwen3-1.7B     meta-llama/Llama-3.2-1B
Qwen/Qwen2.5-1.5B          Qwen/Qwen3-2B       meta-llama/Llama-3.2-1B-Instruct
Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-4B       utter-project/EuroLLM-1.7B
Qwen/Qwen3.5-2B            facebook/w2v-bert-2.0
CohereLabs/tiny-aya-global Skywork/Skywork-Reward-V2-Qwen3-1.7B
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

**`config/train/ABL-*.yaml` is tracked**, so a committed render travels in both
modes. It was gitignored until 2026-08-14, and that is worth knowing when
reading older MN5 checkouts: `--dirty` feeds `.gitignore` to rsync as a filter,
so a regenerated ablation config used to stay on the laptop while MN5 silently
kept whatever it had. Commit renders now rather than relying on `--dirty`.

Regenerating **on MN5** is still the safer habit for a *new* budget or task
composition — the data is there, so the measurement is real rather than
inherited from a cache someone else built:

```bash
# [mn5] from ~/training
python3 infra/build_campaign_config.py \
  --template config/train/ABL-MA-700.yaml \
  --datasets-root /gpfs/projects/epor48/melt-data/shar \
  --budget-hours 125 --tasks both --exclude-corpus fleurs \
  --cache campaign_hours.json --out config/train/ABL-MA-125.yaml
```

`--tasks` picks the task composition. `both` gives the ASR+ST mix;
`asr` gives the ASR-only modality-alignment arm, which is the same config
minus its ST groups — same languages, same budget, same corpus mix, with the
five ASR groups renormalised from 1/9 to 1/5 each. Render the pair from the
same template and the same `--budget-hours` so the two runs differ only in the
ST data:

```bash
# [mn5] the ASR-only arm at the same budget
python3 infra/build_campaign_config.py \
  --template config/train/ABL-MA-700.yaml \
  --datasets-root /gpfs/projects/epor48/melt-data/shar \
  --budget-hours 125 --tasks asr --exclude-corpus fleurs \
  --cache campaign_hours.json --out config/train/ABL-MA-125-asr.yaml
```

Check `grep -c fleurs config/train/ABL-*.yaml` on MN5 if you are unsure whether
you are looking at a stale render (0 = current, since fleurs is excluded).

Use the default for anything you'll want to reproduce: a run launched from a
`--dirty` tree has no commit to trace its checkpoints back to.

If the push is rejected because the MN5 tree is dirty and you are certain the
uncommitted changes there are disposable:

```bash
# [mn5] in ~/training — DISCARDS local changes, check `git status` first
git -C ~/training status
git -C ~/training checkout -- . && git -C ~/training clean -fd
```

Then re-run `infra/sync_repo.sh mn5` from your laptop and confirm the branch:

```bash
# [mn5]
git -C ~/training log --oneline -1
```

## B2. Configure the ablation

See [Configuring ablations](#configuring-ablations) below for what to change and
where.

## B3. Submit

### Worked example: MA on three VoxPopuli languages

Copy-pasteable and complete. This is the shape every submit takes.

**Once per account** — give yourself somewhere to write:

```bash
# [mn5]
mkdir -p /gpfs/scratch/epor48/$USER/{outputs,tmp}
```

**Smoke test first** — 10 steps on `acc_debug`, catches config errors in
minutes rather than after a queue wait:

```bash
# [mn5] from ~/training
EXP=MA-VP3-smoke
MY=/gpfs/scratch/epor48/$USER

LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar-indexed \
OUTPUT_DIR=$MY/outputs TMPDIR_HOST=$MY/tmp \
MELT_QOS=acc_debug MELT_TIME=00:30:00 MELT_NODES=1 \
infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
  --config config/train/MA-VP3-v1.0.yaml \
  --trainer.max_steps 10 \
  --trainer.eval_on_start false \
  --trainer.eval_steps 5 --trainer.save_steps 10 \
  --data.train_ds.buffer_size 2000 \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name $EXP
```

The `buffer_size` override matters: the config's 50k would spend most of a
30-minute debug allocation just filling the buffer.

**The real run** — 2000 steps (~2.3 epochs of this mix) on `acc_ehpc`:

```bash
# [mn5] from ~/training
EXP=MA-VP3-de-es-fr-v1.0
MY=/gpfs/scratch/epor48/$USER

LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar-indexed \
OUTPUT_DIR=$MY/outputs TMPDIR_HOST=$MY/tmp \
MELT_QOS=acc_ehpc MELT_TIME=12:00:00 MELT_NODES=1 \
infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
  --config config/train/MA-VP3-v1.0.yaml \
  --trainer.max_steps 2000 \
  --trainer.eval_steps 500 --trainer.save_steps 500 \
  --trainer.save_total_limit 3 \
  --trainer.per_device_eval_batch_size 4 \
  --trainer.report_to wandb \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name $EXP
```

Results land in `$MY/outputs/$EXP` on the host. Cost at 1 node is 4 GPUh per
wall hour, so a 12 h allocation is at most 48 GPUh — and you are charged only
for elapsed time.

**Three things that will bite you:**

1. **`--trainer.output_dir` must be the CONTAINER path** (`/workspace/outputs/…`).
   The host `OUTPUT_DIR` is only the bind source; a host path here fails.
2. **Check `--trainer.per_device_eval_batch_size` is a positive number.**
   `per_device_train_batch_size` is `-1` ("Lhotse handles batching"), which the
   train path understands but the eval path does not. Every shipped config now
   sets the *eval* one to `4` explicitly, and since 0.5.2 the trainer refuses a
   negative eval batch size *at startup* when evaluation is enabled, instead of
   crashing at the first eval tens of minutes in. You only need to pass it if
   you write a config from scratch or override the train value by mistake. `4`
   is known-good; `16` OOMs on 64 GB H100s.
3. **Pick the QoS, wall time and node count with environment variables.**
   Do *not* edit the site file for these:

   | var | default | what it does |
   |---|---|---|
   | `MELT_QOS` | `acc_ehpc` | which QoS to charge |
   | `MELT_TIME` | `01:00:00` | wall limit, `HH:MM:SS` |
   | `MELT_NODES` | `1` | nodes; `bash/run_train.sh` derives world_size from SLURM |

   | QoS | wall limit | priority | use for |
   |---|---|---|---|
   | `acc_debug` | 2 h | 10000 | smoke tests — schedules almost immediately |
   | `acc_ehpc` | 3 days | 100 | real runs |

   ```bash
   MELT_QOS=acc_debug MELT_TIME=00:20:00 infra/runners/submit-container.sh mn5 …
   ```

   `SBATCH_ARGS` is a bash array rather than an env var, but its *entries* read
   these `MELT_*` variables, so everything you normally vary is settable from
   the command line. The **storage paths** in the same file are overridable the
   same way — see the next section.

   Accounting follows **elapsed** time, not requested wall time, so a generous
   `MELT_TIME` costs nothing by itself. The node is allocated whole, though —
   all 80 cores and 4 GPUs — however few you use.

**Always smoke-test a new config first:** `--trainer.max_steps 10
--trainer.eval_on_start false` on `acc_debug`. It catches config errors in
minutes instead of after a queue wait.

### Point the run at your own output directory

The project space is shared by several accounts, and a directory under it
belongs to whoever created it — with a default umask that means group-readable
but **owner-writable only**. The site default `OUTPUT_DIR`
(`/gpfs/scratch/epor48/outputs`) is owned by one account and is *not* group
writable, so it works for its owner and fails for everyone else:

```
PermissionError: [Errno 13] Permission denied: '/workspace/outputs/<EXP>'
```

That path is inside the container; the directory it actually failed on is the
bind *source*, `$OUTPUT_DIR` from `infra/runners/sites/mn5.sh`. Give yourself
your own, once:

```bash
# [mn5]
mkdir -p /gpfs/scratch/epor48/$USER/{outputs,tmp}
```

`/gpfs/scratch/epor48` is `drwxrws---` owned by group `epor48`, so any member
can create a directory there without anyone granting them anything — and the
setgid bit means everything you create inside inherits the `epor48` group
automatically. Use **scratch**, not projects: the projects quota is 93% full.

Then export both on every submit:

```bash
# [mn5] from ~/training
EXP=ablation-adapter-4layer
MY=/gpfs/scratch/epor48/$USER

OUTPUT_DIR=$MY/outputs TMPDIR_HOST=$MY/tmp \
infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
  --config config/train/my_config.yaml \
  ... \
  --trainer.output_dir /workspace/outputs/$EXP \
  --run.exp_name $EXP
```

Put that `OUTPUT_DIR=… TMPDIR_HOST=…` prefix in a shell alias or a two-line
submit wrapper of your own — it is the same on every run. Don't edit
`sites/mn5.sh` to hardcode your paths: `sync_repo.sh` refuses to push onto a
dirty remote tree, so a locally edited site file blocks your next code sync.

Notes on the five overridable variables:

| var | container path | who writes it | what to do |
|---|---|---|---|
| `OUTPUT_DIR` | `/workspace/outputs` | the run (checkpoints, wandb) | **set your own** |
| `TMPDIR_HOST` | `/workspace/tmp` | the run (triton/lhotse caches) | **set your own** |
| `LOCAL_DATASETS_DIR` | `/workspace/shar` | nobody, read-only | **point at `shar-indexed`** |
| `SINGULARITY_IMG` | — | nobody, read-only | default is correct; override only to pin |
| `HF_HOME` | `/workspace/hf_cache` | nobody (`HF_HUB_OFFLINE=1`) | shared, leave it |

Three of these need overriding on every submit, so the full prefix is:

```bash
LOCAL_DATASETS_DIR=/gpfs/projects/epor48/melt-data/shar-indexed \
OUTPUT_DIR=$MY/outputs TMPDIR_HOST=$MY/tmp \
```

`--trainer.output_dir` stays `/workspace/outputs/$EXP` regardless of where
`OUTPUT_DIR` points — the container path is fixed, only the bind source moves.
Your results then land in `$OUTPUT_DIR/$EXP` on the host, which is what §B5
copies back.

Alternatively the owner can open a directory up to the whole project:

```bash
# [mn5] as the owner of the directory
chgrp -R epor48 <dir> && chmod -R g+w <dir> && chmod g+s <dir>
```

The `g+s` matters: without it, subdirectories created inside keep inheriting the
creator's primary group and the problem comes back. Per-user directories are
still the better default — two people writing the same `$EXP` name into one
`outputs/` will overwrite each other's checkpoints.

## B4. Monitor

```bash
# [mn5]
squeue -u $USER
tail -f ~/training/logs/melt-train-container.<jobid>.out
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -X
```

### Expect a long silent start — it is not a hang

The sampler fills `data.train_ds.buffer_size` cuts **before it emits a first
batch**. On the campaign runs, with `buffer_size: 300000`, that took **35
minutes** from scratch and **91 minutes** on a resume. Throughout, the GPUs sit
at 0% while the CPUs saturate. That is the expected shape of a startup, not a
stall.

Two consequences:

- Don't diagnose by staring at the log. Check `nvidia-smi` and `ps` on the
  allocated node (`squeue -u $USER -o "%N"` gives you the node name).
- Size the buffer to the mix. A buffer larger than the dataset is pure waste —
  for a few-hundred-thousand-cut mix, 50k shuffles perfectly well and starts in
  a fraction of the time. Drop it further for smoke tests.

### "I don't see any log"

The log is `logs/<job-name>.<jobid>.out` **relative to the directory you
submitted from** — not to the repo root, and not to `--trainer.output_dir`.
The job name is `melt-train-container`, so from `~/training` it is
`~/training/logs/melt-train-container.<jobid>.out`.

Don't guess the path — ask SLURM, which knows it exactly:

```bash
# [mn5] while the job is queued or running
scontrol show job <jobid> | grep -E "StdOut|StdErr|WorkDir"

# after it has finished
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,WorkDir%60 -X
```

If there is **no file at all** and the job went to `FAILED` within seconds, the
cause is almost always that **`logs/` did not exist when you ran `sbatch`**.
SLURM will not create the `--output` directory; it kills the job before anything
runs, and because the log *is* the thing that failed, there is nowhere for it to
tell you so.

```bash
# [mn5] from the directory you submit from
mkdir -p logs
```

The `infra/runners/submit-*.sh` runners do this for you. Submitting
`bash/run_train_singularity.sbatch` with a bare `sbatch` does not — that is the
usual way to hit this. Prefer the runner (§B3).

If the file exists but looks empty, give it a moment: it is written by the
compute node and can lag the job entering `RUNNING`, especially behind a queue.

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
`eval_loss`, `eval_wer`, `eval_cer`, per-language `eval_wer_<lang>` /
`eval_cer_<lang>`, and per-task `eval_wer_asr` / `eval_wer_st` (same for
`cer`). Campaign configs also name their validation sources (`asr_<lang>`,
`st_<src>_<tgt>`), which splits eval into one reported set per name: each
metrics dict then carries an `eval_<name>_` prefix and a per-set
`eval_<name>_loss`. To watch them:

```bash
# [mn5]
grep -oE "'eval_(loss|wer|cer)[a-z_]*': [0-9.]+" ~/training/logs/melt-train-container.<jobid>.out
```

W&B runs **offline** (no internet), writing to `$OUTPUT_DIR/wandb/`. To see them
in the browser, sync from artemis (§B5).

**GPU memory:** `export GPU_MEM_MONITORING=1` before submitting to get
`logs/gpu_mem_<jobid>_node<N>.csv`. It samples every 30 s, so it *aliases* and
under-reports true peaks — treat its numbers as a lower bound.

## B4b. Resume a run

`acc_ehpc` caps at 3 days, so any long training is a chain of jobs. Point the
next one at the previous run's output directory and HF picks the highest
checkpoint in it:

```bash
# [mn5] from ~/training
--trainer.resume_from_checkpoint /workspace/outputs/$PREV_EXP
```

Container path, same rule as `--trainer.output_dir`.

Three constraints, all of which bite:

1. **The layout must be identical.** The sampler refuses to restore its state
   under a changed `(world_size, num_workers)` — so `MELT_NODES` × GPUs/node ×
   `data.train_ds.num_workers` must match the original job exactly. Changing
   any one of them silently costs you the data position.
2. **HF refuses a non-empty `output_dir`,** so the resumed leg needs its *own*
   `--trainer.output_dir` and `--run.exp_name`. Name them so the chain is
   obvious (`MA-VP3-legA`, `MA-VP3-legB`).
3. **Resume startup is slower than a cold start** — buffer fill plus
   fast-forward. Budget ~1.5× (see the previous section).

A healthy resume says so in the log:

```
Continuing training from global step 2500
```

and the progress bar opens at 2501 rather than 0. Verify both before walking
away. A clean seam shows the loss continuing the previous curve to within a few
hundredths, with any small offset decaying over ~100 steps as the optimiser
re-warms its momentum — that is normal, not a broken restore.

## B5. Retrieve results

### W&B metrics — `utils/sync_wandb.sh`

MN5 compute nodes have no internet, so W&B runs offline: training writes run
data to `$OUTPUT_DIR/wandb/wandb/` on MN5 and uploads nothing. Getting metrics
into the browser is a **two-step** job that has to happen from a machine with
network access — artemis:

1. copy the offline run directories from MN5 to artemis, then
2. `wandb sync` them up to wandb.ai.

`utils/sync_wandb.sh` does both, and handles the case that matters most in
practice: a run that is **still training**. Nothing in it is hardcoded to one
account.

#### Running it from artemis

```bash
# [artemis] from the repo root
utils/sync_wandb.sh \
  --remote-path /gpfs/scratch/epor48/<your-mn5-user>/outputs/wandb/wandb \
  --entity <the-shared-team> \
  --venv /mnt/scratch-artemis/$USER/venvs/<your-venv>/bin/activate
```

Three things to fill in:

- **`--remote-path`** is where *your* runs are on MN5. Your BSC username is
  usually not your local one: `ssh mn5 'echo $USER'`. It is required, on
  purpose — outputs are per-account, so any default would be wrong for
  everyone but one person.
- **`--entity`** decides which W&B account the runs land in. Read the next
  section before you pick it.
- **`--venv`** is only needed if `wandb` is not already on your `PATH`. It is
  the **activate script**, not the venv directory — the same convention as
  `VENV_PATH` everywhere else in this repo.

Preview first; it changes nothing and shows exactly what would transfer and how
each run would be classified:

```bash
# [artemis]
utils/sync_wandb.sh -r /gpfs/scratch/epor48/<your-mn5-user>/outputs/wandb/wandb --dry-run
```

Every option, with the environment variable that sets the same thing (the flag
wins):

| flag | env | default |
|---|---|---|
| `-r, --remote-path` | `WANDB_REMOTE_PATH` | **required** |
| `-l, --local-path` | `WANDB_LOCAL_PATH` | `/mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb` |
| `-H, --host` | `WANDB_REMOTE_HOST` | `mn5` |
| `-e, --entity` | `WANDB_ENTITY` | none — **warns**, see below |
| `-p, --project` | `WANDB_PROJECT` | whatever the run recorded (`melt`) |
| `-v, --venv` | `VENV_PATH` | none — uses the current environment |
| `-t, --threshold` | `ACTIVE_THRESHOLD_MINUTES` | `10` |
| `-n, --dry-run` | — | off |

Put the ones that never change in your shell profile and the command shortens
to `utils/sync_wandb.sh`:

```bash
# [artemis] ~/.bashrc or ~/.zshrc
export WANDB_REMOTE_PATH=/gpfs/scratch/epor48/<your-mn5-user>/outputs/wandb/wandb
export WANDB_ENTITY=<the-shared-team>
```

#### It is safe to run mid-training

A run whose `.wandb` file was touched within `--threshold` minutes (default 10)
is treated as **still training**: it is uploaded with `--append` and left
unmarked, so a later invocation picks up the rest of it. A run that has gone
quiet is treated as **finished**: it is finalised and gets a `.synced` marker,
and every later invocation skips it. So running it repeatedly during a long job
is not just safe, it is the intended use — each pass tops up the live run and
costs nothing for the ones already done.

A typical summary looks like:

```
  - Skipped (already synced): 73
  - Active runs synced: 1
  - Finished runs synced: 6
  - Failed: 0
```

#### Which W&B account do the runs land in?

**This is the part to agree on as a group, because the default is per-person
and silently splits the project's results across accounts.**

The training code never sets an entity — it calls `wandb.init(project=...)`
and nothing more. The entity is therefore decided at **sync time**, by whoever
runs `sync_wandb.sh`: without `--entity`, runs go to the personal account that
person happens to be logged in as, where nobody else can see them. The script
prints a warning when that is about to happen.

> **Current state:** there is no shared MELT team yet. Runs so far live under
> the personal entity `g8a9`, in project `melt`. The decision for now is to
> keep it that way.

Be aware of what that costs, because it is the reason to revisit it:

- A personal W&B entity has **no members**. A collaborator cannot sync into
  `g8a9` unless that project is opened up for it (W&B project visibility —
  check the project's settings), so by default *their* runs land under *their*
  own account.
- Results are then spread over several accounts and cannot be put on one chart,
  which is the main thing a shared tracker is for.

**When you want everyone's runs in one place**, the fix is a W&B *team*:
create one, invite the collaborators, and have everyone set

```bash
export WANDB_ENTITY=<team>
```

Nothing in the code has to change — see the note on tracker portability below.
Runs already uploaded elsewhere can be moved from the W&B UI.

#### Tracker portability

Nothing about the launcher is W&B-specific, so replacing it later (trackio,
MLflow, …) does not mean editing the training code:

- `train.py` only touches W&B inside an `if "wandb" in cfg.trainer.report_to`
  branch, and passes no entity — each library reads its own environment.
- `bash/run_train_singularity.sbatch` forwards tracker settings into the
  container **by prefix** (`WANDB_`, `MLFLOW_`, `TRACKIO_`, `COMET_`,
  `NEPTUNE_`, `TENSORBOARD_`), not by listing individual variable names. A new
  setting for the current tracker needs no change; a new tracker needs only its
  prefix added to that list.
- `bash/run_train.sh` logs whichever of those are set at startup (with anything
  that looks like a key or token filtered out), so the log answers "where did
  this run go".

Switching tracker is then: change `trainer.report_to`, and export that
tracker's own variables. `sync_wandb.sh` is W&B-specific by nature — offline
sync is a W&B concept — and would be replaced by whatever the new tracker uses.

Manually, what the script does is:

```bash
# [artemis]  (MN5 user, not local user, in the remote path)
rsync -avh mn5transfer:/gpfs/scratch/epor48/<your-mn5-user>/outputs/wandb/wandb/ \
           /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb/
wandb sync --entity <the-shared-team> \
           /mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb/offline-run-*
```

To push **one** run rather than every offline run sitting in that directory —
useful when the directory holds many and you only want the live one:

```bash
# [artemis]
RUN=offline-run-<timestamp>-<id>
rsync -ravh --append-verify --exclude='.synced' \
  "mn5:/gpfs/scratch/epor48/<your-mn5-user>/outputs/wandb/wandb/$RUN/" \
  "/mnt/scratch-artemis/$USER/melt-data/outputs/wandb/wandb/$RUN/"
wandb sync ".../wandb/$RUN" --entity <the-shared-team> --include-offline --append
```

Two harmless artefacts you will see: a `FileNotFoundError` uploading a stale
artifact staging file, and W&B reporting a run as `finished` while it is
plainly still training. Neither affects the metrics; a later manual sync marks
the run caught-up.

**Checkpoints** — pull via the transfer node:

```bash
# [artemis]
EXP=ablation-adapter-4layer
rsync -avh --partial --info=progress2 \
  mn5transfer:/gpfs/scratch/epor48/$USER/outputs/$EXP/ \
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

## Budget: know what a job costs before you submit it

The GPU-hour grant is shared, finite, and **expires 2026-08-31**. Check the
balance before planning a sweep.

```bash
# [mn5] official, but lags reality by up to a day
bsc_acct          # look for project epor48

# [mn5] current, per job
sacct -j <jobid> --format=JobID,Elapsed,AllocCPUS,CPUTimeRAW -X
```

**Get the divisor right — it is easy to be wrong by 2×.** An MN5 ACC node is 80
physical cores and 4 GPUs, so:

> **1 GPUh = 20 physical core-hours.**

`bsc_acct` reports khours of *physical* core-hours, so `1752 khours` of grant is
**87,600 GPUh**. But `sacct`'s `AllocCPUS` reports **160 per node** — it counts
SMT threads — so from `sacct` the conversion is:

```
GPUh = CPUTimeRAW / 3600 / 40
```

Sanity-check any formula you write against a job you know: a 20 h job on 8 GPUs
must come to 160 GPUh.

Rules of thumb: a 1-node job burns **4 GPUh per wall hour**, a 2-node job 8.
Accounting follows elapsed time, so a job that finishes early is only charged
for what it used — but a job that sits in a 20 h allocation doing nothing after
crashing at step 3 is charged for all of it. Check on long runs.

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

**A new YAML** — for changing dataset mixes or many fields at once. Copy an
existing one, edit, and point `--config` at it. Two places to put it, and the
choice matters:

| location | tracked by git? | use for |
|---|---|---|
| `config/train/` | **yes** | anything a collaborator should be able to run, review, or reproduce |
| `runs/` | no (gitignored, but rsynced by `sync_repo.sh`) | throwaway local variants |

Prefer `config/train/`. A config that only ever exists in `runs/` cannot be
shared, reviewed, or recovered — and since `runs/` reaches MN5 by rsync rather
than by commit, a run launched from one has no commit to trace its checkpoints
back to.

Existing configs to copy from: `MA-VP-only-v1.0.yaml` (4 VoxPopuli languages),
`MA-VP3-v1.0.yaml` (3 languages, indexed tree), `MA-v1.2.yaml`,
`SFT-v1.3.1.yaml`.

### What the knobs do

| knob | where | notes |
|---|---|---|
| `model.encoder.freeze` / `decoder.freeze` / `adapter.freeze` | YAML | defines the phase: **MA** freezes encoder+decoder and trains the adapter; **IFT** unfreezes the decoder |
| `model.decoder.name` | either | the LM backbone; **must be in `$HF_HOME`** (offline) |
| `model.adapter._type`, `num_adapter_layers`, `adapter_kernel_size`, `adapter_stride` | either | adapter architecture ablations |
| `model.ckpt` | either | start from a previous run — **this is how IFT consumes the MA checkpoint** |
| `data.train_ds.<...>.shar_path` | YAML | dataset mix; paths resolve under `/workspace/shar` |
| `data.train_ds.indexed` | YAML | `true` to require `.idx` sidecars (exact epochs, resumable). Set it with `LOCAL_DATASETS_DIR` pointing at `shar-indexed` |
| `data.train_ds.batch_duration` | either | **seconds of audio per batch** — the real batch-size lever for training (Lhotse dynamic bucketing), not `per_device_train_batch_size` |
| `data.train_ds.buffer_size` | either | shuffle buffer; smaller = faster startup for smoke tests. Never set it larger than the mix |
| `data.train_ds.max_tokens` / `max_tps` | either | **silently inert on any source without `custom.num_tokens`** — see the data note below |
| `data.train_ds.num_workers` | YAML | must stay fixed across a resume chain (§B4b) |
| `data.apply_chat_template` | YAML | declared at `data.`; since 0.5.1 `validation_ds` inherits it, so train and eval format text the same way |
| `data.prompt_template` | YAML | e.g. `"{audio_token}{lang}"` |
| `optimization.{encoder,decoder,adapter}_lr` | either | per-component LRs |
| `trainer.max_steps`, `eval_steps`, `save_steps`, `warmup_steps` | CLI | schedule |

### Know your data before you put it in a mix

Not every Shar source is equally healthy, and the defects are invisible until
they show up as bad metrics.

**VoxPopuli, specifically.** Only five of the sixteen languages — **de, en, es,
fr, it** — went through the dedicated `voxpopuli.py` converter. The other eleven
(`cs et fi hr hu lt nl pl ro sk sl`) went through the generic HF→Shar batch
converter and have **no `custom` block**, which means:

- no `num_tokens`, so `max_tokens` / `max_tps` are silently inert on them;
- positional fallback IDs (`chunk{N}_item{M}`) that **collide across corpora**;
- unfiltered empty transcripts — `hr` is **~47% textless** in every split.

Prefer the healthy five. If you need one of the others, know what you are
accepting.

**Across the whole tree** (793 leaves, 99.9M cuts): durations are 100% clean,
and genuinely textless cuts are only 8,598 (0.01%), almost all VoxPopuli. But
text can also be stored under a non-default key — `cv22_sidon` keeps it at
`custom.metadata.sentence`, not `text` — so a "missing text" report is often a
`text_field` misconfiguration rather than a data problem. Check before believing
it.

**`num_tokens` coverage is thin overall**: only 109 of 793 leaves carry it. By
cut count it looks fine (92%) because `yodas-granary` dominates, so the gap is a
long tail of small sources. Note that the VoxPopuli *validation* and *test*
splits carry no `num_tokens` at all, in any language — `max_tokens` in a
`validation_ds` is inert today.

### Running the two phases

MA and IFT are two sequential runs; the second points at the first's output:

```bash
# 1) MA — adapter only
--config config/train/MA.yaml --run.exp_name MA-v1 --trainer.output_dir /workspace/outputs/MA-v1

# 2) IFT — starts from the MA checkpoint
--config config/train/IFT.yaml --run.exp_name IFT-v1 --trainer.output_dir /workspace/outputs/IFT-v1 \
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
change, so it belongs in a YAML rather than on the CLI — copy a config in
`config/train/` and edit the `data.train_ds` source list. Keep the total hours
fixed across variants, otherwise task mix and data volume are confounded.

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
- **A shared project directory is not a writable one.** Being in `epor48` gets
  you read access; the directories under it are owner-writable. Run with your
  own `OUTPUT_DIR` and `TMPDIR_HOST` (§B3).
- **The image and the code must agree.** The default `.sif` is a symlink to the
  lhotse 2 image and matches `main`; if you pin an older one with
  `SINGULARITY_IMG`, check out matching code too.
- **A silent first 30–90 minutes is normal**, not a hang — the sampler is
  filling its shuffle buffer.
- **The epoch counter is wrong** by a config-dependent factor (measured 3.29× on
  one config). Harmless while `max_steps` governs the run, but it makes
  epoch-axis comparisons across configs meaningless. Govern with `max_steps`.
- **`gpfs_projects` is 93% full.** Write results to `gpfs_scratch`.

## Troubleshooting

| symptom | cause |
|---|---|
| `trainer.per_device_eval_batch_size is -1, but evaluation is enabled` | exactly what it says — pass `--trainer.per_device_eval_batch_size 4` or fix the config. Before 0.5.2 the same config instead crashed at the first eval with `Trying to create tensor with negative dimension -1` (or `batch_size should be a positive integer, but got -1`) |
| `False is not a valid SaveStrategy` (or `…EvalStrategy`) | you passed `--trainer.save_strategy no`. Overrides are parsed as YAML, so `no`/`off` become `false` and `yes`/`on` become `true`. Quote it: `--trainer.save_strategy "'no'"` |
| Job exits instantly, no log | `logs/` didn't exist, or a bad `--output` path — see §B4 |
| `` `use_bucketing` is retired `` | config predates `lhotse_sampler_type`; swap it as the message says |
| `PermissionError: … '/workspace/outputs/<EXP>'` | shared `OUTPUT_DIR` owned by someone else — set your own (§B3) |
| Permission denied under `/workspace/tmp` | same cause, `TMPDIR_HOST` — set your own (§B3) |
| `SINGULARITY_IMG not found` | image not shipped, or site-file path is stale |
| `ModuleNotFoundError: torchdata`, or a lhotse API error | running `main` against a pre-lhotse2 image — unset `SINGULARITY_IMG` to get the default, or point it at `melt_cuda126_lhotse2_td.sif` |
| Model load fails / tries to reach the Hub | weights not in `$HF_HOME` (§A3) |
| `CUDA out of memory` during eval | eval batch too large — first batches are worst-case |
| Output dir "not empty" | add `--trainer.overwrite_output_dir true`, or pick a new `EXP` |
| Push rejected by `sync_repo.sh` | remote working tree is dirty — commit/stash on MN5 |
| Build writes to host `/workspace` | built with `singularity` instead of `apptainer` |
| No output for 30–90 min, GPUs at 0% | normal: shuffle buffer filling (§B4). Confirm with `nvidia-smi` on the node |
| Job dies at a `--qos` / `--partition` error | submitted from a `glogin` node; use `alogin1` |
| Sampler state not restored on resume | `(nodes × GPUs × num_workers)` changed since the original run (§B4b) |
| `max_tokens` appears to do nothing | the source has no `custom.num_tokens` — inert by design, see the data note |

If something here does not match what you observe, check the repository's open
issues before debugging from scratch — known rough edges are tracked there.
