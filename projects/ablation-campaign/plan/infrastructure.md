# Infrastructure — what an agent needs to know before touching anything

Cross-project cluster facts also live in the private `g8a9/agents-info`
repository (`HPCs/`), which is the canonical place for environment
knowledge; this file keeps what the campaign needs day to day. Figures here
were measured unless marked; check `git log` and `arms.tsv` before trusting
anything time-sensitive.

## 1. Machines

### MN5 (MareNostrum 5, BSC) — where the campaign trains

- Project **`epor48`**. Grant 1,752 khours of physical core-hours;
  **1 GPU-h = 20 core-h**, so 87,600 GPU-h in total. 43% used on
  2026-09-15, ~49,700 GPU-h left. **Expires 2026-11-30, no renewal.**
  Official position: `bsc_acct` (lags ~1 day). Current usage: `sacct`, with
  `GPUh = CPUTimeRAW / 3600 / 40` (its `AllocCPUS` counts SMT threads).
- Partition `acc`: H100 **64 GB** (not 80), 4 GPUs per node, nodes are
  allocated whole. QOS `acc_ehpc`: up to 100 nodes, 3-day wall. QOS
  `acc_debug`: 8 nodes, 2 h, priority, **one job per user at a time**.
  Observed queue waits up to ~23 h for 2-node jobs; `sbatch --test-only`
  showed no scheduling penalty for asking for more nodes or time
  (2026-09-02), re-check under load.
- Submit from **`alogin1`** (the accelerated login pool), not `glogin1`.
- **No internet on compute or login nodes.** Stage HF models from nyx with
  `rsync` to `mn5transfer:/gpfs/scratch/epor48/hf_cache/hub/` (whole
  `models--org--name` directory); verify an offline load before spending an
  allocation. The repo cannot `git fetch` there: push to it with
  `infra/sync_repo.sh mn5`.
- Filesystems: `/gpfs/scratch/epor48` holds `.sif` images, `hf_cache`,
  `outputs` (write results here); `/gpfs/projects/epor48` holds `shar`,
  `shar-indexed`, `tmp` and was 93% full in August. `st_blksize` on GPFS is
  16 MiB, which is why unbounded open-file caches killed 2-node jobs at
  ~490 GB host RAM until lhotse's shard cache got an LRU (fixed 2026-08-31).
- Container: `/gpfs/scratch/epor48/melt_cuda126.sif` is a symlink to the
  current image (transformers 5.x, lhotse 2.0.0a3, torchdata, torch 2.9.1
  cu126). Promote a new image under its own name, prove it with one run,
  then `ln -sfn`; never overwrite a `.sif` a job may be reading.
- Always pass `MELT_GPUS_PER_NODE=4` (SLURM reports the node's full GPU
  count regardless of the request) and confirm `world_size` in the
  `[run_train] starting` log line; a resume must match the world size that
  wrote the checkpoint.
- W&B runs are created offline; sync with the user's `sync_wandb.sh`
  (marker-aware). Entity `g8a9/melt`, personal by decision; tag debug runs
  `WANDB_TAGS=debug` at creation.
- Rendered `ABL-*.yaml` configs are gitignored: **regenerate them on MN5**
  rather than syncing a copy (`build_campaign_config.py`, cache file kept).

### artemis (internal, SARDINE) — GPU jobs for eval and small runs

- **GPU work only through `sbatch`** (`infra/runners/submit-container.sh
  artemis …`), never interactively, even for a smoke test. CPU-only work
  (freezing eval sets, tests) may run directly.
- Never write anything sizeable under `/mnt/home/giuseppe` (GlusterFS);
  venvs, outputs, frozen sets, caches go under `/mnt/scratch-artemis/giuseppe`
  or `/mnt/data-artemis/giuseppe`. Scratch was 96% full in August; check
  `df` before writing audio.
- Its checkout `/mnt/home/giuseppe/melt-proj/training` is separate from
  nyx's; push to it over the `artemis-host` remote and check out the branch
  there before submitting. `sites/artemis.sh` default image is a symlink to
  the transformers-5 image; the previous `melt_cuda126_lhotse2_td.sif`
  (transformers 4.57.1) is still on disk.
- `gpu-h100` had a 2-GPU ceiling per user in August. A6000, H100 and
  possibly H200 are what remains after 2026-11-30.

### nyx (dev box) — code, tests, data inspection

- No GPU, no SLURM client. Reaches artemis through the `artemis` ssh alias
  and MN5 through `mn5` and `mn5transfer`.
- Tests run **inside the container** with a host-side pytest overlay. The
  image is `/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif`,
  bound with `--bind /mnt/scratch-nyx,/mnt/scratch-artemis`; inside it,
  activate `/workspace/venv`, prepend `/mnt/scratch-nyx/giuseppe/container-extras`
  to `PYTHONPATH` (that is where pytest lives), point `HF_HOME` at
  `/mnt/scratch-artemis/giuseppe/.cache/huggingface`, and run
  `python -m pytest tests/ -q`. `vm.overcommit_memory=2` makes intermittent
  "Cannot allocate memory" a host artefact, not a test failure.
- Shar tree: `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar`. HF cache:
  `/mnt/scratch-artemis/giuseppe/.cache/huggingface` (443 GB; never copy).
  CPU-heavy jobs (bucket-bin measurement, manifest scans) run here, not on
  an MN5 login node.

## 2. Repositories

**They are siblings under `/home/giuseppe/melt-proj/`** — `training` (this
one), `preprocessing`, `melt-eval` and `handoff-extras` sit next to each
other in that directory. That is the entire search space: this repo, those
siblings, and the checkouts named below. **Never `find /` or grep from `/`
or `$HOME` looking for a file** (`agent-protocol.md` §1); if something is
not where this section says, ask the PI and the answer gets added here.

| repo | role |
|---|---|
| `MELT-proj/training` (this) | model, trainer, lhotse data pipeline, launchers, the campaign under `projects/ablation-campaign/` |
| `MELT-proj/eval` (`melt-eval`, checkout `~/melt-proj/melt-eval`) | generative evaluation as an `inspect_ai` extension: `melteval freeze` builds zero-copy frozen sets (audio locators, not copies), `inspect eval` generates and scores corpus-level WER/CER/BLEU/chrF with per-language breakdowns, `melteval rescore` adds COMET/MetricX in a separate env. Prompt parity with training is enforced (`melteval/prompt.py`). Launchers for artemis work; the MN5 venv is scaffolded, not built |
| `MELT-proj/preprocessing` | shar building, the truecase/PNC pass, `verification/check_st_sources.py`, `check_shar_content.py`, chat-template checks |
| `g8a9/agents-info` (private) | cross-project infrastructure facts (`HPCs/`), for any agent |
| `deep-spin/wiki` (private) | SARDINE cluster wiki; drifts from the live cluster, verify with `sacctmgr`/`nvidia-smi` |

## 3. The campaign tooling (this repo)

- `projects/ablation-campaign/campaign.yaml` — the intent: one row per arm
  (stage, config, decoder, seed, per-arm batch/accum/checkpointing
  overrides). Shared constants stay in the base YAMLs
  (`ABL-MA-700-asr.yaml`, `ABL-IFT-700.yaml`).
- `campaign.py status | plan <arm> | run <arm> [--resume]` — status joins the
  grid against SLURM and the output dirs; `plan` prints the exact command;
  `run` submits and appends to `arms.tsv` (timestamp, `EXP_NAME`, job id,
  command). `arms.tsv` is the ledger and is committed; do not hand-edit.
- **The ledger and the sync fight each other, every time (2026-09-21).**
  `campaign.py run` appends to `arms.tsv` *on MN5*, and `arms.tsv` is a
  committed file, so every batch of submissions leaves the MN5 checkout
  dirty. `infra/sync_repo.sh mn5` then refuses to push — deliberately, so it
  can never clobber work done on the cluster. MN5 has no internet, so the
  cluster cannot push the rows out itself. The reconciliation is:

  1. Copy `arms.tsv` back from MN5 to a connected checkout.
  2. Commit and push it from there, so the rows are on `main`.
  3. On MN5, verify the working copy is **byte-identical** to what you just
     pushed, then discard MN5's copy (`git checkout --` it, or stash it) so
     the tree is clean.
  4. `infra/sync_repo.sh mn5` now succeeds and fast-forwards MN5 onto the
     commit that already contains those rows.

  Step 3 is the one that looks wrong and is not: you are discarding a file
  whose content you have just verified is already committed. Do not skip the
  byte-identity check, and do not `--dirty` your way around it — that pushes
  an unprovenanced tree over the cluster's ledger. If a guard blocks the
  `checkout --`, stash that one file with a unique tag and **drop the stash
  once the sync lands**; the worktrees share one stash stack, so a forgotten
  entry is another session's hazard.
- `EXP_NAME` grammar: `{STAGE}-{data}-{encoder}{F|T}-{decoder}{F|T}[-lora]-{adapter}{F|T}[-bdN][-gaN][-skN][-epN][-tt<template>]-{elr}-{dlr}-{lr}-s{seed}-{world}g`.
- `build_campaign_config.py` renders the data axis (budget × task) from the
  Italian-anchored corpus template; run it where the data is, keep the
  `--cache` file, never train on a `--sample-shards` render.
- `infra/compute_mix_weights.py` (**in this repo**, documented at
  `docs/mixture_weights.md`) computes the per-source mux sampling weights:
  two-tier balancing, corpora within a language by `alpha` first, then
  languages against each other by `beta`, `n(.)` measured in **hours of
  audio** and not utterance counts. Its output is a training config whose
  `train_ds.input_cfg` carries the weights. Read the doc before changing a
  mixture. This is the tooling `06-fondue.md` §3 means by "the config
  builder already implements it"; it is not in `preprocessing`.
- Effective batch = `batch_duration × gradient_accumulation_steps ×
  world_size` in audio seconds (with `quadratic_duration` unset). One epoch
  is derived from it; never pin `max_steps` per arm.
- `run.memory_preallocation: true` runs a worst-case forward/backward at
  `max_duration` before step 1; turn it on whenever what is trainable
  changes.
- `save_steps` on the launcher command line overrides the YAML; resume takes
  the **parent** run directory, never `checkpoint-N`.

## 4. Measured throughputs and costs (2026-08/09)

| what | value |
|---|---|
| MA, adapter-only, DDP, 8 GPUs, `batch_duration 150` | ~1,060 audio-s per wall-s; 700 h/lang (3,500 h) in ~3.3 h, ≈ 30 GPU-h |
| DDP vs FSDP2 for MA | 4.0× faster; use `config/accelerate/ddp.yaml` |
| IFT Llama-1B, 2 nodes, `grad_accum 4`, effective 3,840 s | 5.73 s/step; 6,310 steps ≈ 10.3 h, 82 GPU-h |
| IFT scaling | keep `grad_accum 4` and add nodes: GPU-h flat, wall halves per doubling; trading accum for nodes at fixed batch scales at 46–65% |
| IFT Qwen3.5-2B under DDP | needs `gradient_checkpointing: true` (OOM at step 1 otherwise); throughput at 2×4 **unmeasured** |
| MMS-1b encoder | matches w2v-BERT step time with `flash_attention_2`; 1.54× slower with sdpa |
| in-training generative eval | ~3 min per 5-set round at 200 utterances per set on 8 GPUs; `max_samples` is per named set |
| first step of any run | 9–20 min (dataloader build, `eval_on_start`); read steady-state s/it from the second-to-last tqdm line |
| MN5 → internal transfer | ~15–20 MB/s over `mn5transfer` |

## 5. Traps that have already cost days

- **Completed FSDP2 runs save no consolidated weights**; consolidate the last
  checkpoint in a CPU batch job (`gp`/`gp_ehpc`), never on a login node
  (killed at exit 137). DDP runs save real safetensors.
- **A checkpoint records no `attn_implementation`**; loading it defaulted to
  sdpa on cuDNN and made generation ~12× slower. Fixed on `main`; check the
  `Text decoder attention implementation from config:` log line.
- **Batched bf16 generation is not reproducible across batch compositions**;
  compare batched WER only at the same batch width, use
  `flash_attention_2`, sort by duration. fp32 agrees but costs 2×.
- **`--trainer.eval_strategy no`** becomes boolean `False` (YAML 1.1); push
  `eval_steps`/`save_steps` past `max_steps` instead, or quote `"'no'"`.
- **Qwen 3.x/3.5 emit an empty think block** that is part of the training
  target; eval must render the prompt with `enable_thinking=False` (melt-eval
  does). Llama 3.x injects a dated system message.
- **`text_field` falls back silently** to the supervision text when the
  configured field is missing (`strict_text_field` defaults to false).
  Set it true for any ST mix. The truecase/PNC pass rewrites words: ~3%
  WER against the audio on Italian MLS vs ~1% on English; accepted.
- **Length filters read `custom.num_tokens`**, present on MLS/FLEURS/Granary
  and on VoxPopuli only for de/en/es/fr/it; absent on every validation split.
- **FLEURS shard duplication** (78 leaves, af_za…fr_fr) was quarantined on
  nyx on 2026-08-11; check other copies of the tree before trusting hours.
- **Every ST source once had its direction backwards** (Granary `ast`
  labelled en→X with the transcript as target); fixed, and
  `verification/check_st_sources.py` guards it. Any ST result from before
  the fix is meaningless.
- **Bucket bins are a property of `max_duration`**; re-measure on nyx
  whenever it changes, or a wide top bucket pads a batch far past its budget.
- Two nodes plus indexed Shar can exhaust `ulimit -n`; the launcher raises
  it, but check on a new site.

## 6. Models and where they are

Staged on MN5 (`/gpfs/scratch/epor48/hf_cache`) as of 2026-09-16:
`facebook/w2v-bert-2.0`, `facebook/mms-1b`, `meta-llama/Llama-3.2-1B` and
`-Instruct`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-2B-Base`,
`utter-project/EuroLLM-1.7B` and `-Instruct` (offline load verified for the
last three, board entry 2026-09-16). **Not yet staged:** Whisper-large-v3,
mHuBERT-147 (both have artemis launchers; confirm they are on MN5 before the
crossing). Verify each with `HF_HUB_OFFLINE=1` inside the container.
