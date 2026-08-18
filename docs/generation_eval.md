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
   them with jiwer, keeping the per-language and per-task breakdowns.

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
    max_samples: 1000           # required in practice

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

### `max_samples` is not optional

Generation costs roughly one sequential decoder step per output token per
sample, against a single forward before. The unbounded validation set (28,815
cuts / 186 h in the ablation configs) was already the dominant cost under
teacher forcing and does not finish under generation.
`materialize_cuts_for_eval` applies `max_samples` with a seeded shuffle, so
every run scores the same subset. `infra/check_training_config.py` flags a
config that omits it (check **C6**).

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
`generation_max_length` tight.

## Batched generation

`tests/integration/inference/run_inference.py --batch-size N` decodes N samples
per `generate()` call. Prompts are left-padded by the processor and the merged
audio embeddings are left-padded by `_inject_tensor`, so every sequence's last
real position lines up and a batch must produce the same hypotheses as
`--batch-size 1`. If it does not, padding is leaking into attention — that is
the check to run first when batched eval WER looks wrong.
