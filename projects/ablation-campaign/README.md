# Ablation campaign — launching MA at 700 h/language

How to run stage 1 (MA, modality alignment) of the campaign on MN5, with
Llama 3.2 1B in **both** variants — Base and Instruct — on DDP.

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

Measured: **12.05 s/it at `batch_duration: 180`** = 478 audio-s per wall-s,
against 209.7 for the FSDP2 configuration — a 2.28× speedup. Peak GPU was
32.0 GB of 64 at `batch_duration: 120`.

## Resuming

Neither arm fits in one 12 h allocation. Submit, let it hit the wall clock,
then resubmit with:

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
