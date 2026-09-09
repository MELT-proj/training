# Generation-based evaluation

During training, `eval_wer` / `eval_cer` are computed from text the model
**generates**, not from the argmax of a teacher-forced forward pass.

Teacher forcing measures the wrong thing here. Every token is predicted given
all *ground-truth* previous tokens, so it structurally cannot observe the
failure modes an audio-LLM adapter actually exhibits — exposure bias, looping,
early EOS, language drift — and the number it reports is not reproducible at
inference on any sample. Generation also removes, by construction, an
off-by-one that lived in the old path: it compared `argmax(logits[j])` against
`labels[j]` while `logits[j]` predicts token `j+1`, which cost a *perfect* model
a WER of 0.4 on a five-token reference.

`eval_loss` is unchanged. It still comes from a teacher-forced forward pass over
the full inputs, so losses stay comparable across the switch. WER and CER do
not: numbers from before this change are not comparable with numbers after it.

## How it works

`MELTTrainer` derives from `Seq2SeqTrainer`. Per eval batch:

1. `MELTDataCollator` (built with `is_train=False`) emits **two** input sets
   from one featurisation of the audio:
   - `input_ids` / `attention_mask` / `labels` — audio placeholder **and**
     transcript, as before;
   - `prompt_input_ids` / `prompt_attention_mask` — everything up to where the
     transcript starts, left-padded.
2. `MELTTrainer.prediction_step` generates from the prompt pair plus the shared
   audio features, then runs a `no_grad` forward on the full inputs for the loss.
3. `TrainingEvaluator` decodes both the generated ids and the labels and scores
   them with jiwer, keeping the per-language and per-task breakdowns (each
   emitted only when the eval set actually spans more than one bucket).

Generating from `inputs_embeds` returns only the newly generated tokens, so
there is no prompt to strip.

## Configuration

```yaml
trainer:
  predict_with_generate: true   # default; MELTTrainer refuses WER/CER without it
  generation_max_length: 256    # see the note below — this is a *new-token* budget
  generation_num_beams: 1       # greedy

data:
  validation_ds:
    max_samples: 200            # required; applied PER NAMED eval set

evaluation:
  log_num_samples: 10           # REF/HYP pairs printed and sent to W&B; 0 disables
```

### `generation_max_length` counts new tokens

`MELTForCausalLM.generate` merges the audio embeddings and delegates to the text
decoder with `inputs_embeds=`. For that input form,
`GenerationMixin._prepare_generated_length` subtracts the prompt length from
`max_length` — and the prompt here is ~1,200 audio frames, so a `max_length` of
256 comes out negative and generation returns before emitting a token.
`MELTTrainer._generation_kwargs` therefore translates `max_length` into
`max_new_tokens`, which is also the only reading that makes sense when the
prompt is audio.

### `max_samples` is not optional, and it is *per named eval set*

Generation costs roughly one sequential decoder step per output token per
sample, against a single forward before. The unbounded validation set (28,815
cuts / 186 h in the ablation configs) was already the dominant cost under
teacher forcing and does not finish under generation.
`materialize_cuts_for_eval` applies `max_samples` with a seeded shuffle, so
every run scores the same subset. `infra/check_training_config.py` flags a
config that omits it (check **C6**).

The cap applies to each named validation source separately. A config whose
`validation_ds` entries carry `name: asr_en`, `asr_de`, … is split into one
eval set per name, and each one draws up to `max_samples` cuts — so
`max_samples: 200` across five named sets evaluates **1,000** utterances, not
200. The shipped configs set the per-set number so the total lands near 1,000.

**Budget it against `eval_steps` before raising either number.** Measured on
one H100 (Qwen3-1.7B + w2v-bert-2.0, batch 4):

| | |
|---|---|
| `generation_max_length: 64` | ~2.2 s per utterance |
| 1,000 utterances at 64 tokens | ~35 min per eval |
| 1,000 utterances at 256 tokens | ~2 h per eval (cost is ~linear in the budget) |

The linearity is not incidental — see the next section. `eval_steps: 100`
predates generation-based eval and is not viable with it.

## Reading the generations

Each eval prints a compact block on the global master:

```
[eval_asr_de] step 400 — 10 sample generation(s):
  [0] lang=de task=asr
      REF: guten tag wie geht es ihnen
      HYP: guten tag wie geht es ihn
```

and logs the same rows to W&B as `<prefix>/samples`, with both the normalised
and the raw text. The raw text is what the checkpoint would hand a user; the
normaliser rewrites case and punctuation, so a run that looks broken usually
looks broken in the raw column first.

## Known issue: nothing teaches the model to stop

In the non-chat-template path the training target is `f"{audio_token}{text}"` —
no EOS is appended, and the Qwen tokenizer does not add one. A model trained
that way has no stopping signal, so generation runs to the full `max_new_tokens`
budget on every sample and the hypothesis carries a long tail of continuation
past the transcript. That inflates WER and makes eval as slow as the budget
allows, regardless of utterance length.

This is a property of the training data, not of the eval path, and fixing it
means changing what the model is trained on (and retraining). Until then, keep
`generation_max_length` tight: it is not a ceiling that is rarely reached, it
is the per-sample cost of every eval, which is why the table above scales
linearly with it. A model that stopped at its own EOS would decode a 20-token
transcript in 20 steps instead of 256.

## Batched generation

`tests/integration/inference/run_inference.py --batch-size N` decodes N samples
per `generate()` call. Prompts are left-padded by the processor and the merged
audio embeddings are left-padded by `_inject_tensor`, so every sequence's last
real position lines up.

**A batch does not reproduce `--batch-size 1`.** It used to say here that it
must, and that a difference meant padding was leaking into attention. Neither
is true, and #118 checked it stage by stage on a real checkpoint: the
per-sample features are bit-identical, the audio encoder is byte-identical when
its input is padded to the batch's length, the merged embeddings and mask agree
to 1e-16 in float64, and `generate()` already derives correct per-row position
ids from the mask. What remains is arithmetic. A different batch shape means a
different reduction order, and in bfloat16 that flips greedy decisions on the
utterances where the top two logits are close. So a batched number is only
comparable with another batched number at the same width.

It is the **dtype**, not the attention implementation. Batch 16 against batch 1
over the same 16 utterances, with the first-step logit perturbation the batch
introduces and the top-two margin it has to cross to flip a greedy choice:

| configuration | transcripts differing | median \|Δlogit\| | median top-2 margin |
|---|---|---|---|
| bfloat16, sdpa as shipped | 2 of 16 | 1.6e-01 | 4.16 |
| bfloat16, sdpa on deterministic backends | 4 of 16 | 1.8e-01 | 4.13 |
| bfloat16, eager | 1 of 16 | 1.9e-01 | 4.09 |
| bfloat16, flash_attention_2 | 2 of 16 | 1.6e-01 | 4.13 |
| float32, eager | **0 of 16** | 1.6e-04 | 4.07 |
| float32, sdpa on deterministic backends | **0 of 16** | 1.8e-04 | 4.07 |

Every bfloat16 configuration perturbs the logits by about the same 0.2, and the
implementation only shuffles which rows land on the wrong side of a close call.
float32 shrinks the perturbation by roughly a thousand and every transcript
then agrees. That is not a guarantee — the perturbation is smaller, not zero,
and a genuinely tied logit pair would still flip — but at this margin
distribution nothing comes close. float32 costs about twice the memory and
throughput, so it is a way to settle an argument about a specific number, not a
default for a campaign.

**A batch must at least repeat itself, and until #118 it did not.** Five
identical `generate()` calls on the same mixed-length batch of 16 returned five
different transcripts, while batch size 1 always repeated. Mixed lengths make
the merged mask non-trivial; flash attention refuses a non-null mask, so SDPA
fell through to the cuDNN backend, whose decode step is not deterministic. That
is the check to run first when batched eval WER looks wrong: run the same batch
twice. `torch.use_deterministic_algorithms` does not catch it and issues no
warning.

The fix was to let the model be loaded with an attention implementation at all.
`from_pretrained(attn_implementation=...)` used to set the flag on the composite
config and leave both backbones on sdpa, and `flash_attention_2` was refused
outright. Pass it explicitly for anything that decodes:

```python
MELTForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16,
                                attn_implementation="flash_attention_2")
```

Measured on one H100 at batch 16, five repeats of the same call:

| attention | distinct results | s/run |
|---|---|---|
| sdpa, backend chosen by torch | 4 of 5 | 1.416 |
| sdpa pinned to the memory-efficient backend | 1 of 5 | 1.611 |
| eager | 1 of 5 | 2.351 |
| flash_attention_2 | 1 of 5 | 1.333 |

**Sort by duration before batching.** The processor pads every row to the
batch's longest audio, so a batch drawn in dataset order costs its longest
utterance times its width. On librispeech clean/test that is two thirds of the
batch spent on padding, and it is why throughput stops improving past about 16:

| batch | dataset order, s/sample | sorted by duration, s/sample |
|---|---|---|
| 8 | 0.0573 | 0.0370 |
| 16 | 0.0431 | 0.0210 |
| 32 | 0.0372 | 0.0171 |
| 64 | 0.0340 | 0.0339 |

`run_inference.py` sorts for you whenever `--batch-size` is above 1. Anything
else that batches MELT should do the same.
