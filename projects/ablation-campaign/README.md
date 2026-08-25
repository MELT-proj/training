# Ablation campaign — the 700 h/language arms on MN5

How to run the campaign's two stages on MN5 with Llama 3.2 1B, on DDP:

- **Stage 1 — MA** (modality alignment): adapter trains, everything else
  frozen. Two arms, **Base** and **Instruct**. Everything up to
  *Pre-launch check* is about this stage.
- **Stage 2 — IFT** (instruction fine-tuning): decoder trains, everything
  else frozen, initialised from stage 1's weights. See
  [Stage 2 — IFT](#stage-2--ift) at the end.

Both arms run from **one** config, `ABL-MA-700-asr.yaml`. The Base arm is two
command-line overrides on top of it, not a second file. That is deliberate: it
is the only way to guarantee the two arms see the same data mixture, the same
bucket bins, the same step budget and the same rendered prompt, so a difference
between them is the backbone and nothing else.

## TL;DR

From the repo root **on MN5**:

```bash
# Instruct
bash projects/ablation-campaign/launch_MA_700_llama32-1b_mn5.sh

# Base
bash projects/ablation-campaign/launch_MA_700_llama32-1b-base_mn5.sh
```

Everything below explains what those two lines expand to and what has to be
true before they work.

## Before the first launch

MN5 compute nodes are air-gapped (`HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`,
set in `infra/runners/sites/mn5.sh`), so nothing can be fetched at run time.

1. **Sync the code.** MN5 cannot `git pull`; push to it over SSH instead.
   ```bash
   infra/sync_repo.sh mn5
   ```
2. **Pre-download both checkpoints** into `HF_HOME`. Both are already listed in
   `infra/setup/download_hf_models.sh`. The Base arm needs *both* of them —
   the Base weights and the Instruct tokenizer it borrows a chat template from
   (see below) — so neither is optional even if you only run one arm.
   ```bash
   infra/setup/download_hf_models.sh
   ```
3. **Pick your own writable output dir** if you are not the owner of the shared
   default. `OUTPUT_DIR` and `TMPDIR_HOST` are written to; `HF_HOME` and
   `LOCAL_DATASETS_DIR` are read-only during a run and can stay shared.
   ```bash
   export OUTPUT_DIR=/gpfs/scratch/epor48/<you>/outputs
   ```

## The two commands, in full

Both launchers wrap `infra/runners/submit-container.sh`, which submits an
`sbatch` job into the container. Written out:

### Instruct

```bash
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --run.exp_name "MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g" \
    --trainer.output_dir "/workspace/outputs/MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g" \
    --trainer.num_train_epochs 1 \
    --trainer.eval_steps 200 \
    --trainer.save_steps 200 \
    --trainer.save_total_limit 2 \
    --trainer.seed 42
```

The decoder, its tokens and the whole chat-template block come from the YAML.
Nothing about the prompt format is overridden.

### Base

```bash
infra/runners/submit-container.sh mn5 config/accelerate/ddp.yaml \
    --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
    --model.decoder.name meta-llama/Llama-3.2-1B \
    --model.decoder.chat_template_from meta-llama/Llama-3.2-1B-Instruct \
    --run.exp_name "MA-700asr-w2vbF-llama1bBaseF-mlpT-s42-8g" \
    --trainer.output_dir "/workspace/outputs/MA-700asr-w2vbF-llama1bBaseF-mlpT-s42-8g" \
    --trainer.num_train_epochs 1 \
    --trainer.eval_steps 200 \
    --trainer.save_steps 200 \
    --trainer.save_total_limit 2 \
    --trainer.seed 42
```

Exactly two lines differ. Both are plain strings, which is what makes running
the Base arm off the shared config possible at all — see *Why the format cannot
be overridden on the command line* below.

Set the topology through the environment, the same for both arms:

```bash
export MELT_NODES=2
export MELT_GPUS_PER_NODE=4        # -> world_size 8
export MELT_QOS=acc_ehpc
export MELT_TIME=12:00:00
export MELT_SEED=42
```

`MELT_GPUS_PER_NODE` is **pinned, not autodetected**. MN5 allocates `acc` nodes
whole, so `SLURM_GPUS_ON_NODE` reports the node's full GPU count whatever
`--gpus-per-node` asked for; pinning it is what keeps `world_size` stable across
a resume (PR #90).

## `chat_template_from`: why the Base arm needs it

Both arms train under the chat template, with an empty task instruction — the
user turn carries only the audio (`prompt_template: "{audio_token}"`).

A **base** checkpoint ships no chat template. Verified against the real
tokenizer configs:

| checkpoint | `chat_template` |
|---|---|
| `meta-llama/Llama-3.2-1B` | absent |
| `meta-llama/Llama-3.2-1B-Instruct` | present, 3,827 bytes of Jinja |

So `data.apply_chat_template: true` against the Base checkpoint has nothing to
render with, and the run fails at startup:

```
ValueError: chat_template_config is 'llama3' and apply_chat_template is on, but
this tokenizer has no chat template, so nothing can render.
```

`--model.decoder.chat_template_from meta-llama/Llama-3.2-1B-Instruct` copies the
Instruct template — and *only* the template string, never the tokenizer, since
adopting a whole tokenizer would pair one vocabulary with another checkpoint's
weights. With it set, the two arms render identically. Verified on the real
tokenizers, same cut through the real `SpeechToTextDataset`:

- `input_ids` — **identical**
- the label span that survives masking — **identical**
- `<|eot_id|>` lands inside the loss in both, so both learn to stop
- `add_special_tokens` adds **0** new tokens to either vocabulary:
  `<|eot_id|>` (128009) and `<|finetune_right_pad_id|>` (128004) are already in
  the Base vocabulary, so nothing resizes an embedding table

Do **not** set `chat_template_from` on the Instruct arm. It already has its own
template, and setting it would only overwrite it with a copy of itself — the
run warns if you do.

## What the two arms share

Everything except `model.decoder.name`:

| | |
|---|---|
| Data | 5 languages × 700 h = **3500 h**, ASR only, matched domain mix |
| Encoder | `facebook/w2v-bert-2.0`, frozen |
| Adapter | `mlp`, **trainable** — 6.3 M of ~1.8 B params |
| Prompt | chat template, `llama3` boundaries, `prompt_template: "{audio_token}"` |
| Tokens | eos `<\|eot_id\|>`, pad `<\|finetune_right_pad_id\|>` |
| Batching | `batch_duration: 180` s of audio, `gradient_accumulation_steps: 4` |
| Schedule | 1 epoch, `max_steps: -1`, `adapter_lr: 2e-5`, `warmup_steps: 20` |
| Eval | `predict_with_generate`, `generation_max_length: 256`, 200 samples **per named eval set** (5 sets ⇒ ~1000 utterances/round) |

One epoch is **2188 steps**:

```
ceil(3500 h × 3600 s/h / 180 s / 8 ranks / 4 accum) = 2188
```

`quadratic_duration` is unset in this config, so `batch_duration` budgets real
audio seconds and the step count is exact rather than an estimate.

## Why DDP and not FSDP2

Only the adapter trains. FSDP shards *parameters*, but the state sharding
actually saves — gradients and optimizer moments — exists only for those 6.3 M
params and comes to ~75 MB. Sharding the frozen 1.8 B saves ~3.2 GB per rank on
a 64 GB H100 (~5% of the card) and charges an all-gather of the whole model on
every forward to get it.

It also cannot be expressed in FSDP2: with `fsdp_version: 2` accelerate takes
`reshard_after_forward` as a bool and torch's FSDP2 has no `NO_SHARD`, so "one
full replica per rank" simply *is* DDP.

Two consequences:

- **No activation checkpointing.** `fsdp2.yaml` had it on, i.e. it recomputed
  every activation to buy back memory this arm is not short of. This is the
  single biggest speed difference.
- **Saving works normally.** `Trainer.save_model` writes a real
  `model.safetensors` rather than the sharded no-op of issue #91, so a completed
  run needs no consolidation step.

Measured over the completed Instruct arm: **6.81 s/it at
`batch_duration: 180`** in steady state = 846 audio-s per wall-s, against
209.7 for the FSDP2 configuration — a **4.0× speedup**. Peak GPU was
32.0 GB of 64 at `batch_duration: 120` (7.29 GB resident + 24.68 GB
transient).

> An earlier revision of this file quoted 12.05 s/it. That number was taken
> from a window that still included the one-time 8 min 50 s startup
> (dataloader construction and `eval_on_start`), so it described the first
> few minutes rather than the run. The steady-state figure is 6.81.

## Resuming

At 6.81 s/it, one epoch of 2188 steps is **~4.1 h of training** plus ~9 min
of startup, so stage 1 *does* fit in a single 12 h allocation — an earlier
revision of this file said it did not, on the strength of the inflated
s/it above. A 6 h request is the better ask: it fits with room to spare and
MN5's backfill scheduler starts it sooner (all three stage-1 allocations
queued ~5 min at that length).

Resume anyway, because infrastructure will interrupt you — the Instruct arm
needed three allocations for its one epoch, losing the first to a
`NODE_FAIL` at step 894 and the second to an NCCL collective timeout at
step 1862. Submit, and if it dies, resubmit with:

```bash
    --trainer.resume_from_checkpoint "/workspace/outputs/<EXP_NAME>"
```

Point it at the **run directory**, never at a `checkpoint-N` subdirectory —
`train.py` calls `get_last_checkpoint()` on whatever it is given.

Keep `MELT_GPUS_PER_NODE` pinned to the same value across the resume, or the
run dies on a `world_size` mismatch.

## Why the format cannot be overridden on the command line

`--data.prompt_template '{audio_token}'` does not work, and this is worth
knowing before anyone tries to make a third arm out of a CLI flag.

OmegaConf reads a leading brace as YAML flow-mapping syntax, so the shell's
`{audio_token}` arrives as the dict `{audio_token: None}`. Since PR #100 that is
a hard error rather than a silent misparse (issue #94):

```
ValueError: prompt_template parsed as a dict with no values
```

and `'{audio_token}{t}'` does not even parse as YAML. It *can* be forced through
with nested quoting — `"'{audio_token}'"`, double outside, single inside — but
that is a trap to leave in a shell history. **Prompt format belongs in the
YAML.** Both Base-arm overrides above are plain strings precisely so that this
never comes up.

## Pre-launch check

```bash
python3 infra/check_training_config.py \
  --config projects/ablation-campaign/ABL-MA-700-asr.yaml \
  --datasets-root "$LOCAL_DATASETS_DIR"
```

Known false positive: **B3 fails on validation bins that are correct**, and
prints a `measured:` line byte-identical to the config's own. C4 reports a
top bin that is not the top bin in the list it prints. Tracked in issue #101 —
the bins are confirmed correct by `infra/estimate_bucket_bins.py`. Do not
"fix" the config to satisfy B3.


---

# Stage 2 — IFT

Instruction fine-tuning, initialised from stage 1's weights. **The freeze
pattern is the mirror image of stage 1**: the decoder trains, the encoder and
adapter do not.

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

with `MELT_NODES=2`, `MELT_GPUS_PER_NODE=4`, `MELT_SEED=1337`,
`MELT_TIME=6:00:00`. Note the seed: `run_train.sh` feeds `MELT_SEED` to
`--data.*.shard_seed`, so leaving it at stage 1's 42 would replay against the
model the very cuts it has already seen.

## What changes between the stages

| | Stage 1 (MA) | Stage 2 (IFT) |
|---|---|---|
| Encoder | frozen | frozen |
| Adapter | **trainable** (6.3 M) | frozen |
| Decoder | frozen | **trainable** (~1.24 B) |
| Init | `meta-llama/Llama-3.2-1B-Instruct` | stage 1's output dir |
| Data | 5 ASR langs, 3500 h | 5 ASR + 5 ST, **6,729.9 h** |
| Prompt | `"{audio_token}"`, no instruction | per-task instruction |
| `batch_duration` | 180 s | 120 s |
| `max_duration` | 30 s *(see below)* | 90 s |
| `max_tokens` | unset *(see below)* | 400 |
| Seed | 42 | 1337 |
| LR | `adapter_lr: 2e-5` | `decoder_lr: 2e-5` |
| One epoch | 2188 steps | **6310 steps** |

The prompt difference is the stage's whole point. Stage 1 trains on a bare
`{audio_token}` because modality alignment has one task and no instruction to
give; stage 2 introduces the instructions:

```yaml
prompt_template_selection: custom
prompt_template:
  asr: "{audio_token} Transcribe this audio in {lang}."
  st:  "{audio_token} Translate this audio to {lang}."
```

`{lang}` resolves to the **target** language for `st` (`get_tags_from_cut`
collapses `lang` to `tgt_lang`), so the X→en directions read "Translate this
audio to English."

## No consolidation step

`model.ckpt` points at stage 1's **run directory**, not a `checkpoint-N/`
subdirectory. Stage 1 ran on DDP, so `Trainer.save_model` wrote consolidated
`model-0000N-of-00002.safetensors` alongside `config.json`,
`preprocessor_config.json`, the tokenizer and `chat_template.jinja` — exactly
what `MELTForCausalLM.from_pretrained` needs.

This is the difference from `ABL-IFT-125.yaml`, whose stage 1 ran on FSDP and
left sharded state that had to be merged with `utils/merge_fsdp_weight.py`
first (#91).

`model.decoder.attn_implementation` is still read from the YAML and pushed onto
the checkpoint's sub-config: a checkpoint's `config.json` records no attention
implementation, and letting it fall back to sdpa costs ~12× on generation (#86).

## Memory: why `batch_duration` drops 180 → 120

Stage 2 adds resident state that stage 1 never carried. For the 1.24 B
trainable decoder:

| | |
|---|---|
| resident model (1.82 B params, fp32) | 7.29 GB |
| **+ fp32 gradients** | ~4.9 GB |
| **+ AdamW's two moments** | ~9.9 GB |
| transient activations @ 120 s | 24.68 GB *(measured, job 44991596)* |
| **projected peak** | **~46.8 GB of 64** |

Activations barely change between the stages — stage 1 already backpropagated
through every decoder layer to reach the adapter, so those tensors were already
being kept. What is new is ~14.8 GB of resident state that does not depend on
batch size at all.

Holding `batch_duration` at 120 means the transient term is the one that was
*measured*, so the projection extrapolates nothing. 180 would put the peak near
59 GB and leave a batch of long cuts nowhere to go.

`run.memory_preallocation: true` is set in this config: before step 1 it runs a
forward+backward at `max_duration` and another at `min_duration`, logging peak
CUDA memory for each. An OOM there is caught and warned about rather than fatal.
It turns "we might OOM at step 4000" into a number printed in the first minute:

```
[Preallocation/max_duration] rank=0 — pass complete. Peak CUDA memory: X → Y GB
```

## Step budget

```
6729.85 h × 3600 s/h / (120 s × 8 ranks × 4 accum) = 6310 steps
```

each carrying 3840 audio-seconds. `quadratic_duration` is null, so
`batch_duration` budgets *real* audio seconds and this is exact rather than an
estimate (#96). `max_steps` stays `-1` and the trainer derives it — pinning it
per arm is how two arms end up trained for different amounts.

Checkpoints are much larger here: a trainable decoder means AdamW's moments are
saved with the weights, so budget **~22 GB each**, ~45 GB for the two kept.

## Known divergence from stage 1: `max_duration` and `max_tokens`

Stage 1 was **intended** to run with `max_duration: 90` and a `max_tokens`
filter, and did not. MN5 job 45013277's `resolved_config.json` records:

```
train.max_duration = 30.0      train.max_tokens = None
  val.max_duration = 30.0        val.max_tokens = None
```

So the stage-1 model was aligned, and validated, on a strictly shorter slice of
each source than stage 2 sees. Two things follow:

1. **The practical loss is small.** That mixture's measured bucket bins top out
   at 15.9 s, so cuts beyond 30 s are a thin tail, and the hours actually
   consumed still add up to the intended 3500 — drawn from shorter cuts.
2. **The Base arm has not run yet.** If `ABL-MA-700-asr.yaml` is corrected to
   90 now, the two stage-1 arms differ in their data filter and a Base-vs-Instruct
   difference stops being attributable to the backbone. Either both arms stay at
   30, or the Instruct arm is re-run at 90 (~4.1 h). **This is an open
   decision** — the MA config has deliberately not been changed.

`ABL-IFT-700.yaml` applies both settings as intended, which is why the table
above shows them differing across the stages.

Nothing about the frozen modules objects to the longer cuts: the `mlp` adapter
maps frames pointwise and carries no length dependence, and w2v-bert chunks at
`max_audio_seq_len=1500` frames (30 s) via `_unfold_tensor`, so encoder
attention never sees a longer window. The decoder runs `flash_attention_2`,
whose footprint is linear in sequence length, so a 90 s cut costs ~3× a 30 s cut
in decoder memory rather than ~9×. Compute still grows quadratically, which is a
throughput cost rather than an OOM risk.

## Resuming stage 2

6310 steps does not fit in one 6 h allocation. The launcher forwards extra
arguments, so:

```bash
bash projects/ablation-campaign/launch_IFT_700_llama32-1b_mn5.sh \
    --trainer.resume_from_checkpoint True
```

`True` (a bool) makes HF scan `output_dir` for the last checkpoint itself.
Keep `MELT_GPUS_PER_NODE` pinned across the resume or the run dies on a
`world_size` mismatch.
