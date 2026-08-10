# The MELT ablation campaign

A guide to running the campaign end to end. It assumes you know your way around
a SLURM cluster but have not been part of the design discussions, so it explains
*why* each choice was made — you will need that when something looks wrong and
you have to decide whether it matters.

Read [`docs/hpc_runbook.md`](hpc_runbook.md) for cluster mechanics and
[`docs/run_training.md`](run_training.md) for the launcher. This document covers
only what is specific to the campaign.

---

## 1. What we are testing

We want to know which parts of the MELT recipe actually matter. The method is
the one used in *The Art of Scaling Reinforcement Learning Compute for LLMs*:
fix one defensible baseline, vary a single component per arm, and check that the
conclusions survive a change of scale.

MELT is audio encoder → adapter → text decoder, trained in two stages:

| stage | what trains | what is frozen | chat template |
| --- | --- | --- | --- |
| **MA** (modality alignment) | the adapter | encoder, decoder | on for instruct models, with a task-free prompt |
| **IFT** (instruction tuning) | the decoder | encoder, adapter | on |

Every variant runs **MA then IFT**, because the two stages can rank variants
differently. At MA the decoder is frozen, so the stage measures how *alignable*
a frozen embedding space is; an instruct model's actual advantage — following
instructions — cannot show up until IFT. Report the stages separately.

The axes, in the order they should run:

1. **Backbone**: base vs instruct, across three families.
2. **Audio stack**: encoder and projector variants, on the winning backbone.
3. **Task composition**: ASR-only / ST-only / ASR+ST during MA.
4. **Prior language knowledge**: text-only perplexity vs audio-task score.

---

## 2. The data design

### 2.1 Languages

**Trained on: `en, de, fr, es, it`. Held out: `nl, pt, pl, uk`.**

The training five are the only languages in the collection that share all five
corpora. That matters more than it sounds. If Italian came mostly from
audiobooks and Russian only from YouTube, an Italian-vs-Russian gap would be
partly a domain gap wearing a language label. Dutch is effectively MLS-only and
Russian is YODAS-only, so both are held out rather than trained.

Holding out these four buys a contrast that a single held-out language would
not:

- **`nl` and `pt`** are unseen languages inside a *seen* family (Germanic,
  Romance) — they test transfer from relatives.
- **`pl` and `uk`** are Slavic, and no Slavic language is trained on at all —
  they test transfer to an unseen family.

Two notes. `cv22_sidon/uk` exists on disk with a full train/test/validation
split, so Ukrainian is a better probe than the 10 h of YODAS in the older
configs suggests. And FLEURS carries `sr_rs` (Serbian) and `mt_mt` (Maltese),
both with test splits — far too small to train on, but usable as extra zero-shot
targets if you want them.

### 2.2 Hours, and the domain template

**709 h per language, identical corpus proportions in every language.**

Italian is the scarcest of the five, and 709 h is *all* of it. Taking Italian's
natural mix as the template gives:

| corpus | share | h at 709 |
| --- | --- | --- |
| cv22_sidon | 35.9% | 254.6 |
| mls_sidon | 34.9% | 247.4 |
| yodas-granary | 16.9% | 120.2 |
| voxpopuli | 11.0% | 78.1 |
| fleurs | 1.3% | 9.0 |

Every other training language has enough in *every one* of those corpora to
match Italian at 709 h, so "equal hours, matched domain, five languages" is
exactly achievable — and 709 h is the ceiling for it.

**This leaves Italian with zero headroom**, which is worth understanding before
you launch. Sampling is stochastic, so at exactly 709 h some Italian sources
will wrap and repeat a little. If you want a strict no-repetition guarantee,
build the anchor at ~650 h instead and note the change. Otherwise expect
Italian's realized epoch count to sit at roughly 1.0 and check it in the
exposure audit.

### 2.3 The scaling ladder

Run the baseline, and one base-vs-instruct pair, at **175 / 350 / 709 h**.

The point is not to find the best budget. It is that a ranking measured at one
budget is only interesting if it holds at another. If base beats instruct at
350 h and loses at 709 h, no single-budget conclusion in this campaign is
trustworthy, and that is a result worth reporting on its own.

### 2.4 Speech translation

**Main arm X→en; `en→de` as a probe.**

This surprises people, so: genuine en→X barely exists here. CoVoST2 on disk has
en→{ar, ca, cy, de, et, fa, id, ja, lv, mn, sl, sv, ta, tr, zh} — no en_es,
en_fr, en_it. Among the training languages the only real en→X direction is
**en→de at 430 h**.

X→en, by contrast, is abundant: the YODAS Granary `ast` sets hold an English
translation at `custom.translation_en`, giving es→en 25,833 h, fr→en 11,512 h,
de→en 7,429 h, it→en 6,025 h. Every held-out language has one too (nl→en
1,328 h, pt→en 1,641 h, pl→en 664 h, uk→en 596 h), so the zero-shot probes cover
translation as well as transcription.

So the ST arm is 709 h for each of de/es/fr/it→en, plus the 430 h CoVoST2
`en→de` to check the model can translate *out* of English at all. All targets
being English also means one BLEU tokenizer across the main arm.

**ASR and ST use disjoint audio.** `asr_only` and `ast` were measured for
overlap: across 20,000-cut samples in four languages, **zero cuts are shared**.
They do share source videos (roughly 35-50% of `ast` videos also appear in
`asr_only`), so the two pools are correlated in speaker and topic, but no clip
is ever seen twice. That is the property the "no repetition" claim needs.

### 2.5 Totals

| | hours |
| --- | --- |
| ASR, 5 languages × 709 h | 3,546 |
| ST, 4 directions × 709 h | 2,836 |
| ST probe, en→de | 430 |
| **total per full-budget run** | **6,812** |

For calibration, `config/train/MA-v1.2.yaml` already trained on 3,720 h with
this stack, so this is a known-workable scale.

---

## 3. Building the training config

```bash
python3 infra/build_campaign_config.py \
    --template      config/train/MA-v1.2.yaml \
    --datasets-root /gpfs/projects/epor48/melt-data/shar \
    --budget-hours  709 \
    --tasks         both \
    --cache         campaign_hours.json \
    --out           config/train/ABL-MA-709.yaml
```

It measures every source, derives the domain template from Italian, checks each
language can supply its share, and writes a config with equal per-language
weights and template-share per-corpus weights.

**Run it where the data is.** It reads every cut manifest; a full pass over this
collection is hours of I/O. Keep `--cache` — a re-run after changing one thing
then costs a minute.

**Do not ship a config built with `--sample-shards`.** It extrapolates from a few
shards per source, and the error is large enough to move the domain template
itself. In testing, 2 shards per source put YODAS at 24.5% against a true 16.9%.
The script prints a loud warning; heed it. Use the flag only to check the script
runs.

### Hours are not enforced by the manifests

This is the one structural thing to understand. A Shar source pairs
`cuts.NNNNNN.jsonl` **positionally** with `recording.NNNNNN.tar`. Subset the cut
manifest and it no longer lines up with its audio; rewriting the tars would mean
copying audio, which the disk budget cannot absorb.

So hours come from **sampling weights plus a step budget**:

```
max_steps = total_hours * 3600 / (batch_duration * world_size * grad_accum)
```

and the realized figure is **measured afterwards** by the exposure audit (§6.3).
Do not use `num_train_epochs`: epoch length is under-reported by a
config-dependent factor (measured 3.29× — issue #46), which makes epoch-axis
comparisons across configs meaningless.

---

## 4. The run matrix

### Phase A — baseline and ladder

| run | budget | purpose |
| --- | --- | --- |
| `ABL-base-709` | 709 h | the anchor |
| `ABL-base-350` | 350 h | ladder rung |
| `ABL-base-175` | 175 h | ladder rung |

Baseline stack: `facebook/w2v-bert-2.0` (frozen) + conformer adapter
(2 layers, stride 2) + the decoder under test. MA then IFT for each.

### Phase B — backbone axis

Six variants, each MA + IFT, on the fixed baseline stack:

| family | base | instruct |
| --- | --- | --- |
| Qwen 3.5 | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-2B` |
| EuroLLM | `utter-project/EuroLLM-1.7B` | `utter-project/EuroLLM-1.7B-Instruct` |
| Llama 3.2 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` |

> Confirm the EuroLLM and Llama ids against the Hub before downloading — they
> were chosen by the campaign design, not verified against a release page.

Add one of these pairs to the ladder (§2.3). The winner fixes the decoder for
Phase C.

**Each backbone needs the right `chat_template_config`** (§7.1). Get this wrong
and nothing raises — you simply train on the wrong tokens.

| backbone family | `chat_template_config` |
| --- | --- |
| Qwen 2.5, EuroLLM | `chatml` |
| Qwen 3, Qwen 3.5 | `qwen3` |
| Llama 3.x | `llama3` |

### Phase C — audio stack

Fixed decoder = Phase B winner. Ordered by readiness:

| variant | state |
| --- | --- |
| conformer, 2 layer configs | ready, config only |
| MLP | ready, config only — the no-compression control |
| Q-Former | **not working.** `MELTQFormerAdapter` calls `AutoModel.from_config` on a `melt_adapter` config, which does not resolve |
| Whisper encoder | **not built** — needs an encoder-only load path and fixed 30 s log-mel handling |
| HuBERT / mHuBERT | **not built** — consumes raw waveform as `input_values`, not `input_features` |
| NoEnc (Mel-LLM) | **not built** — see below |

Everything below the first two rows needs implementation work; ask Giuseppe to
schedule it rather than improvising. There is also a shared prerequisite: the
conformer projector is hard-wired to a w2v-BERT-shaped encoder config
(`MELTConformerAdapter` builds `Wav2Vec2BertAdapterLayer(encoder_config)`), so it
must be decoupled *before* any encoder swap. Otherwise the alternative encoders
cannot use the baseline projector, and the encoder comparison is confounded by a
projector change.

**NoEnc** follows [arXiv 2606.10231](https://arxiv.org/html/2606.10231v1)
("Mel-LLM"): 80-dim log-Mel → mean-variance normalization from precomputed
training statistics → convolutional downsampling at time-reduction `r=8` for ASR
→ a single linear projection into the decoder's embedding space. That is
12.5 Hz against the current stack's 50 Hz, so this arm varies *compression* as
much as it varies "encoder or not" — keep that in the interpretation. The paper
freezes the LLM and trains LoRA (r=320, α=640); MELT already supports LoRA.

### Phase D — task composition

Three arms — ASR-only, ST-only, ASR+ST during MA — all followed by the same
IFT. Hold total hours fixed so the variable is composition, not budget.

### Phase E — prior language knowledge

Text-only perplexity of each backbone on the **exact** reference transcripts and
translations of the eval sets, plotted against that backbone's WER/BLEU. No
audio, no training, so it is cheap; it turns Phase B from a ranking into an
explanation. Use the same text the audio task targets, or the correlation means
nothing.

---

## 5. Launching

```bash
infra/runners/submit-container.sh mn5 config/accelerate/fsdp2.yaml \
    --config config/train/ABL-MA-709.yaml \
    --run.exp_name ABL-base-709-MA \
    --trainer.output_dir /workspace/outputs/ABL-base-709-MA \
    --trainer.max_steps <computed> \
    --model.decoder.name Qwen/Qwen3.5-2B \
    --data.chat_template_config qwen3 \
    --data.strict_text_field true
```

### MA formatting for instruct backbones

MA applies the chat template for instruct models, but with **no task
instruction** — the user turn carries only the audio. That is config-only:

```bash
    --data.apply_chat_template true \
    --data.prompt_template_selection custom \
    --data.prompt_template '{audio_token}'
```

Verified to render correctly for all three families; the user turn becomes just
`<|audio|>`, the assistant span stays locatable, and nothing sits between the
boundary and the target. Base backbones run MA unformatted
(`--data.apply_chat_template false`) unless §6.1 leads you to decide otherwise.

At IFT both arms use the chat template with real task prompts.

### Environment

Environment comes from `infra/runners/sites/mn5.sh`: `HF_HOME`, `OUTPUT_DIR`,
`LOCAL_DATASETS_DIR`, `SINGULARITY_IMG`, plus `MELT_NODES`, `MELT_QOS`
(`acc_debug` for ≤2 h, `acc_ehpc` otherwise) and `MELT_TIME`.

Four things that bite every time:

- **Paths inside overrides are container paths** (`/workspace/outputs/...`), not
  host paths. The host `OUTPUT_DIR` is only the bind source.
- **HF refuses a non-empty `output_dir`**, so every phase needs its own.
- **Resume requires unchanged topology.** The sampler refuses to restore under a
  changed `(world_size, num_workers)`, so a resumed job must keep the same
  nodes × GPUs × workers. Pass `--trainer.resume_from_checkpoint <run dir>` and
  HF picks the highest checkpoint.
- **MN5 compute nodes are offline.** Every model must be in `HF_HOME` first:
  `infra/setup/download_hf_models.sh` lists the six backbones.

Prefer CLI dot-overrides to new YAML files. The config then stays fixed and the
difference between two runs is visible in your shell history and in
`resolved_config.json`.

---

## 6. Pre-flight checklist

Run these before the first job of each phase. They are quick, and each one
catches a failure that is otherwise silent.

### 6.1 Chat templates

```bash
HF_HUB_OFFLINE=1 python3 infra/check_chat_templates.py
```

Reports, per backbone, whether it ships a chat template and which
`chat_template_config` matches. **Read the base-model warnings.** Qwen 2.5 base
ships a chat template complete with a "You are a helpful assistant." system
turn; if that is silently applied, your base-vs-instruct arm compares formatting
rather than instruction tuning. Decide which format each arm uses and write the
decision down.

If a model reports `NONE MATCH`, stop. A new entry is needed in
`CHAT_TEMPLATE_CONFIGS` before that backbone can be trained.

### 6.2 ST source labels

```bash
python3 infra/audit_st_sources.py \
    --config config/train/ABL-MA-709.yaml \
    --datasets-root $LOCAL_DATASETS_DIR \
    --sample-cuts 500
```

Every ST source must pass. This is not a formality — see §7.3.

### 6.3 Exposure

Set `MELT_EXPOSURE_DIR` on every training run:

```bash
--trainer.output_dir /workspace/outputs/$EXP
# and in the environment:
MELT_EXPOSURE_DIR=/workspace/outputs/$EXP/exposure
```

Then afterwards:

```bash
python3 infra/exposure_audit.py \
    --dir outputs/$EXP/exposure --expected-hours 709 --tolerance 0.05
```

It prints realized hours per language and per direction and exits non-zero if
any deviates by more than the tolerance. **A run that fails this cannot support
a cross-language claim** — report the realized hours alongside the result, or
rebalance and rerun.

### 6.4 Dry run, then smoke

```bash
# config resolves, data loads, nothing trains
... --run.dry_run true

# ~30 min on 1 node
... --trainer.max_steps 30 --trainer.eval_steps 10
```

Confirm each language appears as its own `eval_<name>_loss` curve and all of
them fall below the `eval_on_start` baseline.

---

## 7. Known traps

These are all real: each one was hit, or found by inspection, while preparing
this campaign.

### 7.1 A chat-template mismatch is silent

Label masking finds `assistant_start` / `assistant_end` as **literal strings**.
If they are not in the rendered text, the span is simply never located: nothing
raises, and the run trains on the wrong tokens while reporting a plausible loss.
A validator now fails the run instead (`validate_chat_template_config`), but the
underlying fragility is worth knowing.

The live case: **Qwen 3 and 3.5 open the assistant turn with an empty `<think>`
block**, and `enable_thinking=False` does *not* remove it. Under the plain
`chatml` boundary the mask opens before that block, so the model is trained to
emit `<think>\n\n</think>\n\n` before every transcript. Four shipped configs had
this; they now pin `chat_template_config: qwen3`.

### 7.2 Llama's system preamble

Llama 3.x injects a system turn with a "Cutting Knowledge Date / Today Date"
preamble that Qwen and EuroLLM do not. The Llama arm's input is therefore
structurally different. It does not invalidate the comparison, but mention it
when reporting, and do not be surprised by a Llama-specific offset.

### 7.3 An ST label error looks exactly like a healthy run

Every `yodas-granary/<Language>/ast` source shipped in `SFT-v1.3.0.yaml` was
tagged `src_lang: en, tgt_lang: X` with the default `text_field`. In fact the
audio is X-language, the supervision holds the X transcript, and the English
translation lives at `custom.translation_en`. So the direction was inverted
*and* the training target was the transcript: roughly **64,000 h of "ST" data
was ASR data wearing an ST label**. The loss fell the whole time.

Fixed in the template config, and `infra/audit_st_sources.py` now checks for it.
Treat any ST result from a pre-fix checkpoint as unusable.

### 7.4 `text_field` falls back silently

`get_text_from_cut` falls back to the supervision text when the configured field
resolves to nothing. For an ST source that means a cut with a null translation
quietly becomes an ASR pair — the same bug, reintroduced one row at a time.
**Set `data.strict_text_field: true` for any mix containing ST sources**, which
turns that into an error.

### 7.5 `max_tokens` / `max_tps` are inert on most sources

Issue #59: they depend on `custom.num_tokens`, which only 109 of 793 sources
carry. YODAS has it and dominates by cut count, so the gap is a long tail of
smaller sources — meaning length filtering behaves differently across corpora
*within* one language's mix. Keep it in mind when a per-language result looks
odd.

### 7.6 Italian has no headroom at 709 h

See §2.2. At the anchor budget Italian is drawn at essentially 100%, so expect
around one epoch of it and some wrap-around. Check it in the exposure audit
rather than assuming.

### 7.7 ST evaluation is one run per target language

`corpus_bleu`'s tokenizer is chosen per target language, so a frozen set
spanning several targets raises by design rather than silently picking one. Pass
`-T lang=<code>` per run.

---

## 8. Evaluation

Scoring is **generative**, at several checkpoints per variant, through
`melt-eval` (https://github.com/MELT-proj/eval). Per-language eval loss stays on
during training as a health check, but it is teacher-forced and is not the
decision metric — it can rank variants differently from generation, which is
exactly the failure this campaign is trying to avoid.

- **In-domain**: held-out splits of the training corpora, per training language.
- **Zero-shot**: `nl, pt, pl, uk` — all four have CV22 and FLEURS test splits,
  and nl/pt/pl also MLS/VoxPopuli, so the probe can be domain-matched too.
- **ASR**: WER and CER, corpus-level (not the mean of per-sample scores).
- **ST**: BLEU and chrF, one run per target language.

**Prompt and format parity is a requirement, not a detail.** The repo has
already shipped a bug where training and eval formatted differently (fixed in
PR #61). Whatever `melt-eval` renders must match the format the checkpoint was
trained in — including the `chat_template_config` chosen in §6.1.

---

## 9. What to record

For each arm, keep:

1. The `resolved_config.json` from the run directory — the authoritative record
   of what actually ran.
2. The exposure audit output, including realized hours per language.
3. The pre-flight chat-template output, especially the base-model decisions.
4. Generative scores per language and task, per checkpoint, and the checkpoint
   step each came from.
5. Anything that surprised you. A ranking that flips between MA and IFT, or
   between ladder rungs, is a finding rather than a mistake — it is arguably the
   most interesting thing this campaign can produce.

An arm is done when it has a generative score on both the in-domain and
zero-shot sets, and a passing exposure audit. Without the second, the number
cannot support a cross-language comparison.
