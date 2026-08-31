# Ablation campaign

Cross-language ablation of the MA (modality alignment) -> IFT (instruction
fine-tuning) pipeline. This file documents the "campaign grouping" launcher
layout (data / architecture / optimisation axes, one launcher per stage) --
for the data-mixture design itself, read `build_campaign_config.py`'s module
docstring.

## The three axes

The campaign varies three things. Each lives in exactly one place:

| axis | varies how | belongs in |
|---|---|---|
| **Data**: mixture, hours, task | a rendered config, ~550 lines | one YAML per budget x task: `ABL-MA-125-asr.yaml`, `ABL-MA-700-asr.yaml`, `ABL-IFT-125.yaml`, `ABL-IFT-700.yaml` |
| **Architecture**: adapter, encoder, decoder, freezing | 0-7 CLI overrides | `ADAPTER`, `ADAPTER_FREEZE`, `ENCODER`, `ENCODER_FREEZE`, `DECODER`, `DECODER_FREEZE` env vars |
| **Optimisation**: adapter LR | 0-1 CLI override | `ADAPTER_LR` env var |

There is deliberately **no YAML per arm**. A data axis change (a new budget or
task mix) is big enough, and shared enough across many arms, to earn its own
rendered config via `build_campaign_config.py`. Architecture and LR changes
are a handful of keys and belong on the command line, where they show up in
`resolved_config.json` and in `arms.tsv` (below) without anyone having to diff
two 550-line YAMLs to find them.

## Before the first launch (MN5)

MN5 compute nodes are air-gapped (`HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`,
set in `infra/runners/sites/mn5.sh`), so nothing can be fetched at run time.

1. **Sync the code.** MN5 cannot `git pull`; push to it over SSH instead.
   ```bash
   infra/sync_repo.sh mn5
   ```
2. **Pre-download every checkpoint** a config references into `HF_HOME`,
   listed in `infra/setup/download_hf_models.sh`. A Base-decoder arm that
   borrows a chat template (see below) needs *both* checkpoints even though
   only one trains.
   ```bash
   infra/setup/download_hf_models.sh
   ```
3. **Pick your own writable output dir** if you are not the owner of the
   shared default. `OUTPUT_DIR` and `TMPDIR_HOST` are written to; `HF_HOME`
   and `LOCAL_DATASETS_DIR` are read-only during a run and can stay shared.
   ```bash
   export OUTPUT_DIR=/gpfs/scratch/epor48/<you>/outputs
   ```
4. **Pre-launch config check**, catches most mixture/bucket-bin mistakes
   before a job burns an allocation:
   ```bash
   python3 infra/check_training_config.py \
     --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
     --datasets-root "$LOCAL_DATASETS_DIR"
   ```
   Known false positive: **B3 fails on validation bins that are correct**,
   and prints a `measured:` line byte-identical to the config's own. C4
   reports a top bin that is not the top bin in the list it prints. Tracked
   in issue #101 -- the bins are confirmed correct by
   `infra/estimate_bucket_bins.py`. Do not "fix" the config to satisfy B3.

On artemis, nothing needs pre-downloading (the box has internet access);
`infra/runners/sites/artemis.sh` runs with `WANDB_MODE=online` for the same
reason.

## Launching an arm

```bash
# stage 1 (MA), 125 h/language, default architecture (whatever
# ABL-MA-125-asr.yaml itself declares)
CONFIG=projects/ablation-campaign/ABL-MA-125-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh

# stage 1 (MA), 700 h/language, ablating the adapter LR
CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml ADAPTER_LR=1e-4 \
    bash projects/ablation-campaign/launch_MA.sh

# stage 2 (IFT)
CONFIG=projects/ablation-campaign/ABL-IFT-125.yaml \
    bash projects/ablation-campaign/launch_IFT.sh

# see the exact command without submitting anything
DRY_RUN=1 CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh

# a quick test on artemis (1 node, 2 H100s) before the real submission on mn5
SITE=artemis ACCELERATE_CONFIG=config/accelerate/ddp.yaml \
MELT_NODES=1 MELT_GPUS_PER_NODE=2 MELT_QOS=gpu-h100 MELT_PARTITION=h100 \
CONFIG=projects/ablation-campaign/ABL-MA-125-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh
```

`launch_MA.sh` and `launch_IFT.sh` are thin wrappers (set `STAGE` and a
per-stage default `CONFIG`) around the shared `launch_campaign.sh`, which does
the real work: compose `EXP_NAME`, derive `eval_steps`/`save_steps`, work out
which architecture overrides are actually needed, and call
`infra/runners/submit-container.sh`. MA and IFT differ only in which module is
frozen by convention (see below) and in their default `CONFIG` -- everything
else is identical, so one core script with a `STAGE` variable seemed better
than duplicating the whole launch/registry/dry-run mechanics twice.

**Every architecture/optimisation env var defaults to empty, meaning
"inherit whatever the chosen `CONFIG` already says."** No CLI override is
emitted, and `EXP_NAME` reflects the real, effective value either way. Set one
explicitly to run an ablation; it turns into a CLI override only if the
requested value actually differs from the config (which is also why the
acceptance-test parity check in the PR description works -- the two old,
hand-written launchers had no architecture overrides beyond the 125 h arm's
decoder-and-chat-template bundle, and this launcher reproduces exactly that
when asked for the same values).

This was a deliberate choice over a single "sensible default" set of freeze
flags: MA trains only the adapter (encoder and decoder frozen), while IFT
freezes the adapter stage 1 produced and trains the decoder on top of it (see
`ABL-IFT-125.yaml`'s own `model.adapter`/`model.decoder` comments). A
hardcoded MA-shaped default (encoder/decoder frozen, adapter trainable) would
have been right for stage 1 and silently backwards for stage 2 the first time
someone ran `launch_IFT.sh` without overriding it.

`SITE` selects `infra/runners/sites/<site>.sh` (`mn5` or `artemis`) and
defaults to `mn5`. Topology (`MELT_NODES`, `MELT_GPUS_PER_NODE`), `MELT_QOS`,
`MELT_PARTITION` and `SEED` are also overridable; the topology defaults (2
nodes x 4 GPUs, `acc_ehpc`) match every MN5 arm run so far and are wrong for
artemis -- pass `SITE=artemis` with `MELT_NODES=1 MELT_GPUS_PER_NODE=2
MELT_QOS=gpu-h100 MELT_PARTITION=h100` (artemis's own site defaults, see the
example above) for a quick test there before committing an MN5 allocation.
`MELT_GPUS_PER_NODE` must stay pinned, not autodetected -- MN5 allocates
`acc` nodes whole, so `SLURM_GPUS_ON_NODE` reports the node's full GPU count
regardless of what was requested (PR #90).

Any argument after the env vars is passed through as an extra override, e.g.:

```bash
CONFIG=... bash projects/ablation-campaign/launch_MA.sh --trainer.max_grad_norm 2.0
```

which is also how a resume is requested:

```bash
CONFIG=... bash projects/ablation-campaign/launch_MA.sh \
    --trainer.resume_from_checkpoint "/workspace/outputs/<EXP_NAME>"
```

Point it at the **run directory**, never at a `checkpoint-N` subdirectory --
`train.py` calls `get_last_checkpoint()` on whatever it is given. Keep
`MELT_GPUS_PER_NODE` pinned to the same value across the resume, or the run
dies on a `world_size` mismatch.

## Naming convention

`EXP_NAME` is composed, never typed by hand, from:

```
{STAGE}-{data tag}-{encoder}{F|T}-{decoder}{F|T}-{adapter}{F|T}-{lr tag}-s{seed}-{world_size}g
```

e.g. `MA-125asr-w2vbF-llama1bInsF-mlpT-lr2e4-s42-8g`. Trailing `F`/`T` marks a
module frozen/trainable. The **LR tag is new** in this refactor (adapter LR
is now an ablated axis, so it has to be visible in the name or two arms that
only differ in LR become indistinguishable in W&B and in `outputs/`); every
other segment matches the convention documented in the pre-refactor
`launch_MA_llama32-1b-instruct_mn5.sh` header. The data tag comes straight
from the config's own filename (`ABL-{STAGE}-<tag>.yaml` -> `<tag>` with
dashes stripped), not from parsing hours out of the YAML, since the filename
is already the source of truth for "which budget x task is this."

## Two rules that matter more than the file layout

**Anything that must be identical across arms belongs in the base config,
never the launcher.** `batch_duration`, `quadratic_duration`, `max_samples`
and similar knobs govern effective batch size and eval noise; if one of them
ends up as a launcher default instead of a YAML value, it is only a matter of
time before two arms drift apart on it without anyone deciding that on
purpose. `launch_campaign.sh` only ever overrides the three declared axes plus
the campaign-wide invariants (`num_train_epochs: 1`, `save_total_limit: 2`)
that apply to every arm identically by convention, not per-arm.

**Ablating data breaks step-comparability.** With `num_train_epochs: 1`, a
125 h arm and a 700 h arm derive different step counts by design (903 vs
2188) -- one epoch means different things at different budgets. Compare data
arms at equal **audio hours** (or wall-clock, or `train_hours/*` from W&B),
never at equal step. Architecture and LR arms, by contrast, share a `CONFIG`
and therefore a step count, so they stay directly comparable step-for-step.

## Derived eval_steps / save_steps

`eval_steps` and `save_steps` used to be hand-maintained per arm (100/200 for
the 125 h arm's 903 steps, 200/200 for the 700 h arm's 2188) and drifted
whenever `batch_duration` changed. They are now derived from the config's own
`total_hours`, `batch_duration`, `quadratic_duration` and
`trainer.gradient_accumulation_steps` at the actual world_size, targeting
~11 eval rounds per run:

```
steps = ceil(total_hours * 3600 / batch_duration * inflation / world_size / gradient_accumulation_steps)
eval_steps = save_steps = round(steps / 11)
```

`inflation` corrects for `quadratic_duration`: lhotse charges each cut
`d + d^2/quadratic_duration`, so a batch holds less real audio than
`batch_duration` seconds and `steps` would otherwise be understated (586 vs
the true 903 for `ABL-MA-125-asr.yaml`). `plan_arm.py`'s
`effective_duration_inflation` is a deliberate line-for-line duplicate of
`melt/training/data/audio/lhotse/dataloader.py`'s
`_effective_duration_inflation` -- duplicated rather than imported so this can
run in any dev shell with PyYAML, not just inside the training container
(that module pulls in torch/omegaconf). Verified against the two known-good
step counts: 2188 for `ABL-MA-700-asr.yaml` and 903 for `ABL-MA-125-asr.yaml`.

Note `save_steps == eval_steps` is a simplification versus the old arms (the
125 h launcher used `save_steps = 2 x eval_steps`); see the PR description for
the full before/after diff.

## Arms registry

Every real submission (not `DRY_RUN=1`) appends one line to `arms.tsv`:
UTC timestamp, `EXP_NAME`, SLURM job id, and the exact effective command. The
campaign's deliverable is a comparison table; reconstructing "what exactly did
arm X run" from shell history is how campaigns go wrong. `arms.tsv` is
checked into git as the source of truth -- do not hand-edit it.

## IFT and `max_steps`

`ABL-IFT-125.yaml` currently pins `trainer.max_steps: 6250` in the base
config, computed (per its own comment) at world_size 2. `launch_IFT.sh` does
NOT override `max_steps` -- see that script's header for why this is flagged
rather than silently fixed.

## Why DDP and not FSDP2 for an adapter-only (MA) arm

Only the adapter trains. FSDP shards *parameters*, but the state sharding
actually saves -- gradients and optimizer moments -- exists only for the
trainable adapter (6.3 M params on the Llama 1B arm, ~75 MB) and sharding the
frozen decoder on top saves a few GB per rank while charging an all-gather of
the whole model on every forward to get it. It also cannot be expressed in
FSDP2 at all: with `fsdp_version: 2` accelerate takes `reshard_after_forward`
as a bool and torch's FSDP2 has no `NO_SHARD`, so "one full replica per rank"
simply *is* DDP (`config/accelerate/ddp.yaml`).

Measured on the Llama 3.2 1B Instruct 700 h arm: **6.81 s/it** in steady
state on DDP against **209.7 s/it**-equivalent throughput... concretely a
**4.0x speedup** over the FSDP2 configuration, because FSDP2's activation
checkpointing (recomputing every activation to buy back memory this arm is
not short of) was the single biggest cost, and `Trainer.save_model` now
writes a real `model.safetensors` instead of FSDP2's sharded no-op (issue
#91), so a completed run needs no consolidation step. Use `ddp.yaml` (the
default `ACCELERATE_CONFIG` in `launch_campaign.sh`) for any arm that freezes
both encoder and decoder; reach for an FSDP config only once a stage trains
enough of the backbone that DDP no longer fits in memory (e.g. IFT's decoder
training, currently still on FSDP for the 125 h config -- see
`ABL-IFT-125.yaml`'s own comment on why its stage 1 needed
`utils/merge_fsdp_weight.py`).

## `max_duration` / `max_tokens`: a caution from a real incident

An earlier MA pass on the 700 h Llama arm filtered training and validation
data at an unintended `max_duration: 30` with no `max_tokens` set --
`batch_duration` still budgeted the full audio-hours, just drawn entirely
from shorter cuts, so the run silently trained on a different mixture than
the config claimed. It was discarded and re-run rather than patched in place
(patching after the Base arm had already diverged on the same config would
have made a Base-vs-Instruct comparison unattributable). Every current
`ABL-*` config runs at `max_duration: 60` / `max_tokens: 400`; double-check a
new or edited config's `data.train_ds` / `data.validation_ds` blocks agree
with that convention before trusting a step-count derivation or launching a
real arm. `run.memory_preallocation: true` (a forward+backward at
`max_duration` and `min_duration` before step 1, peak CUDA memory logged, OOM
caught rather than fatal) is cheap insurance against the same class of
surprise showing up as an OOM instead.

## Legacy hand-written 700 h/language MN5 launchers (Base arm, IFT-700)

Two arms of the Llama 3.2 1B campaign still run from dedicated,
hand-written scripts rather than `launch_MA.sh`/`launch_IFT.sh` -- they
predate this refactor and have not been folded in yet:

- `launch_MA_700_llama32-1b-base_mn5.sh` -- the **Base**-decoder half of the
  Instruct/Base pair described below.
- `launch_IFT_700_llama32-1b_mn5.sh` -- stage 2 on top of the (now
  parameterised) Instruct MA arm, config `ABL-IFT-700.yaml`.

The (now-superseded) **Instruct** MA arm these two used to sit alongside is
launched through the generic system instead:

```bash
CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh
```

### Base arm

```bash
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.decoder.name meta-llama/Llama-3.2-1B \
    --model.decoder.chat_template_from meta-llama/Llama-3.2-1B-Instruct \
    --run.exp_name "MA-700asr-w2vbF-llama1bBaseF-mlpT-s42-8g" \
    --trainer.output_dir "/workspace/outputs/MA-700asr-w2vbF-llama1bBaseF-mlpT-s42-8g" \
    --trainer.num_train_epochs 1 \
    --trainer.eval_steps 199 \
    --trainer.save_steps 199 \
    --trainer.save_total_limit 2 \
    --trainer.seed 42
```

with `MELT_NODES=2 MELT_GPUS_PER_NODE=4 MELT_QOS=acc_ehpc MELT_TIME=6:00:00
MELT_SEED=42`. Both the Base and Instruct arms run off the **same**
`ABL-MA-700-asr.yaml` -- deliberately, so they see the same data mixture,
bucket bins, step budget and rendered prompt, and a difference between them
is attributable to the backbone alone and nothing else.

**Why `chat_template_from`.** A base checkpoint ships no chat template
(`meta-llama/Llama-3.2-1B`'s tokenizer config has none;
`meta-llama/Llama-3.2-1B-Instruct`'s carries 3,827 bytes of Jinja), so
`data.apply_chat_template: true` against the Base checkpoint has nothing to
render and fails at startup. `--model.decoder.chat_template_from
meta-llama/Llama-3.2-1B-Instruct` copies *only* the template string (never
the tokenizer -- adopting a whole tokenizer would pair one vocabulary with
another checkpoint's weights). Verified on the real tokenizers: `input_ids`
and the post-masking label span are identical between the two arms,
`<|eot_id|>`/`<|finetune_right_pad_id|>` are already in the Base vocabulary
(128009/128004), so nothing resizes an embedding table. Do **not** set
`chat_template_from` on the Instruct arm -- it has its own template already,
and the run warns if you overwrite it with a copy of itself.

**Why the format can't move to the CLI.** `--data.prompt_template
'{audio_token}'` alone does not work from a shell: OmegaConf reads a leading
brace as YAML flow-mapping syntax, so `{audio_token}` arrives as the dict
`{audio_token: None}`, a hard error since PR #100 (issue #94). It can be
forced through with nested quoting (`"'{audio_token}'"`, double outside,
single inside -- which is what `plan_arm.py` does when `DECODER` triggers the
chat-template override bundle), but that is a trap to leave in shell history.
Prompt format belongs in the YAML wherever it can live there instead, which
is why both arms above declare it in `ABL-MA-700-asr.yaml` rather than as a
launcher override.

### Stage 2 -- IFT-700

Instruction fine-tuning on top of the Instruct MA arm's weights, mirroring
stage 1's freeze pattern (decoder trains, encoder and adapter do not):

```bash
bash projects/ablation-campaign/launch_IFT_700_llama32-1b_mn5.sh
```

which expands to:

```bash
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-IFT-700.yaml \
    --run.exp_name "IFT-700both-w2vbF-llama1bInsT-mlpF-s1337-8g" \
    --trainer.output_dir "/workspace/outputs/IFT-700both-w2vbF-llama1bInsT-mlpF-s1337-8g" \
    --trainer.num_train_epochs 1 \
    --trainer.eval_steps 500 \
    --trainer.save_steps 500 \
    --trainer.save_total_limit 2 \
    --trainer.seed 1337
```

with `MELT_NODES=2 MELT_GPUS_PER_NODE=4 MELT_SEED=1337 MELT_TIME=6:00:00`.
Seed 1337 (not stage 1's 42) matters because `run_train.sh` feeds
`MELT_SEED` to `--data.*.shard_seed`; reusing 42 would replay the very cuts
stage 1 already streamed.

| | Stage 1 (MA) | Stage 2 (IFT) |
|---|---|---|
| Encoder | frozen | frozen |
| Adapter | **trainable** (6.3 M) | frozen |
| Decoder | frozen | **trainable** (~1.24 B) |
| Init | `meta-llama/Llama-3.2-1B-Instruct` | stage 1's output dir |
| Data | 5 ASR langs, 3500 h | 5 ASR + 5 ST, **6,729.9 h** |
| Prompt | `"{audio_token}"`, no instruction | per-task instruction |
| `batch_duration` | 150 s | 120 s |
| Seed | 42 | 1337 |
| LR | `adapter_lr: 2e-5` | `decoder_lr: 2e-5` |

The prompt difference is the stage's whole point -- stage 1 trains on a bare
`{audio_token}` (no task to instruct); stage 2 introduces per-task
instructions (`"{audio_token} Transcribe this audio in {lang}."` for ASR,
`"{audio_token} Translate this audio to {lang}."` for ST, `{lang}` resolving
to the **target** language).

`model.ckpt` points at stage 1's **run directory** (DDP wrote consolidated
`model-0000N-of-00002.safetensors`, not FSDP's sharded state -- no
`utils/merge_fsdp_weight.py` needed, unlike `ABL-IFT-125.yaml`'s FSDP-trained
stage 1, #91). `model.decoder.attn_implementation` is still read from the
YAML and pushed onto the checkpoint's sub-config, since a checkpoint's
`config.json` records no attention implementation and falling back to sdpa
costs ~12x on generation (#86).

Stage 2's `batch_duration` drops to 120 (from stage 1's 150/180) because a
trainable 1.24 B decoder adds fp32 gradients (~4.9 GB) and AdamW's two
moments (~9.9 GB) on top of the resident model -- resident state that does
not depend on batch size at all, so the transient (activation) budget has to
shrink to keep the same peak-memory margin. `run.memory_preallocation: true`
reports the true worst-case peak (both `max_duration` and `min_duration`
passes) before step 1 rather than leaving it to surface as an OOM later.
Checkpoints are much larger here too (AdamW's moments are saved with the
weights): budget ~22 GB each, ~45 GB for the two `save_total_limit` keeps.

6310 steps (the derived one-epoch count at `batch_duration: 120`) does not
fit in one 6 h allocation. Resume with:

```bash
bash projects/ablation-campaign/launch_IFT_700_llama32-1b_mn5.sh \
    --trainer.resume_from_checkpoint True
```

`True` (a bool, not a path) makes HF scan `output_dir` for the last
checkpoint itself. Keep `MELT_GPUS_PER_NODE` pinned across the resume or the
run dies on a `world_size` mismatch.
