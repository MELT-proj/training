# 01 — The interface recipe: step 0 on LibriSpeech and the five-language screen

**Settled (2026-09-14/15):** the August MA baseline is a failed alignment and
is not a lower bound; recipe work precedes every comparison; MA-stage
generative WER is the selection metric for MA-stage hyperparameters; the
`stack_factor` option is the one code change scheduled before the screen;
the August failure is a coarse-feature plateau, not ignored audio (measured
2026-09-15, §1a).
**Open:** the recipe values themselves (decided by the runs below).
**Owner:** PI decides the recipe at the week-2 gate.

## 1. Diagnosis

The MA arms run in August (five languages, 700 h each, w2v-BERT 2.0 frozen,
MLP adapter trainable, decoder frozen, adapter LR 2e-5, effective batch
4800 s, one epoch ≈ 2,600 optimizer steps) ended with:

- eval loss 2.6–3.1 nats per token, which is 1.1–1.5 nats *below* the
  measured no-audio floor of 4.0–4.3 (§1a): the audio was used, but only
  coarsely;
- generative WER 1.10–1.16, insertions outnumbering reference words;
- hypotheses fluent and in-domain but unrelated to the audio, sometimes in
  the wrong language.

The adapter learned coarse audio-conditioned information (language
identity, register, utterance length, perhaps partial lexical content),
worth 1.1–1.5 nats, and stopped short of the frame-to-token alignment that
transcription needs: a coarse-feature plateau.
A 10× longer run (MoE adapter, verbatim prompt, same LR) left that plateau
only after ~9,000 h seen, dropping from ~3.6 to ~2.6: the alignment
transition, arriving very late.

Reference point: SLAM-ASR (Ma et al., 2024) trains only a linear projector
between a frozen encoder and a frozen 7B LLM on LibriSpeech 960 h and reaches
about 2% WER, reporting the same plateau-then-drop dynamics. So an adapter-only
MA that cannot transcribe after 3,500 h is a recipe problem, not a
data-quantity problem.

Suspects, in order of the evidence:

| suspect | current value | why it is suspect |
|---|---|---|
| adapter LR | 2e-5 | the repo's own IFT default is 2e-4; SLAM-ASR uses 1e-4, LLaVA's projector stage 1e-3. With Adam a weight moves ≈ LR per step, so 2,600 steps cap total movement near 0.05 |
| optimizer steps | ~2,600 per epoch | effective batch 4800 s ≈ 1.3 h of audio per update; a 6.3M-parameter module has a small critical batch, so most of each batch averages redundant gradients |
| frame rate | 50 Hz, no stacking | 3,000 decoder positions per 60 s cut for ~150 target tokens; SLAM-ASR stacks to 10 Hz, most speech LLMs sit at 6–25 Hz |
| adapter output scale | LayerNorm × learnable gain initialised at 0.1 | plausible but second-order; the gain is learnable |

Two controls settle the diagnosis without training:

1. **The no-audio floor.** Score the frozen backbone on the eval transcripts
   with the same chat prompt and no audio. **Measured 2026-09-15 (§1a): the
   floor is 1.1–1.5 nats above the MA loss, so audio was not ignored.** What
   the runs below must show is fine alignment, which loss alone cannot.
2. **The text prior per language** (`02-backbones.md` §3), which also gives
   the ST upper bound through the cascade oracle.

### 1a. No-audio floor — measured, and it does not match the assumption above

`no_audio_floor.py` (2026-09-15, MN5 job 45888659) scores frozen
Llama-3.2-1B-Instruct on the exact 200-cut-per-language eval subset
`MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g-md60` uses (same seed, same
chat-templated `"{audio_token}"` prompt, same assistant-span label masking),
with `input_features=None` so no audio is read at all. Compared against that
arm's own final (step 2600) `eval_<lang>_loss`:

| lang | no-audio floor (nats/token) | MA eval_loss with audio | floor − with-audio |
|---|---|---|---|
| en | 4.073 | 2.948 | +1.125 |
| de | 4.312 | 3.123 | +1.189 |
| fr | 3.968 | 2.739 | +1.229 |
| es | 4.040 | 2.659 | +1.381 |
| it | 4.154 | 2.674 | +1.480 |
| overall (token-weighted) | 4.133 | — | — |

**This contradicts the line above it and the framing in `00-status.md`**
("eval loss 2.6–3.1 ... roughly what a 1B text LM scores ... with no
audio"): the floor is *not* close to the observed loss, it is 1.1–1.5
nats/token *above* it, consistently across all five languages. Audio was
not ignored — conditioning on it lowers the loss by a factor of
e^1.1..1.5 ≈ 3–4.4x in per-token perplexity versus no audio at all. That
signal just is not the right signal for correct tokens: WER stayed at
1.10–1.16 and hypotheses were fluent but unrelated to the audio. Read as:
the adapter learned something coarse (e.g. "audio present -> respond in
this register/language") that measurably helps next-token prediction
without helping transcription. This does not by itself say the recipe is
fine -- WER is still catastrophic and the suspects table above still holds
end to end -- but "the adapter did nothing" is no longer the failure mode to
assume; whatever it did learn should factor into interpreting the
LibriSpeech screen below (e.g. a run that drops the loss further without
moving WER would be repeating this same coarse-signal pattern, not
progress).

**Action needed (flagged, not resolved here):** PI review -- this changes
the interpretation the week-2 gate reads section 1 through. Board entry
with the full numbers is at the top of `board.md`.

*Strategy session, 2026-09-15:* §1's diagnosis, the Settled block and the
step-0 interpretation rules were updated to this reading (a coarse-feature
plateau; loss is read together with WER). Confirmed by the PI on
2026-09-15.

## 2. Step 0 — LibriSpeech

Everything as in the campaign except the data: w2v-BERT 2.0 frozen, MLP
adapter, Llama-3.2-1B-Instruct frozen, llama3 chat template, audio-only
prompt `{audio_token}`, `max_duration 60`, `max_tokens 400`. Train on the
three LibriSpeech train splits (≈960 h, already in the shar tree as
`librispeech/clean/train.100`, `clean/train.360`, `other/train.500`).
In-training generative eval on `clean/validation` and `other/validation`;
final numbers on `clean/test` and `other/test` through melt-eval. At the
measured MA rate (~1,000 audio-seconds per wall-second on 8 GPUs) an epoch is
under an hour, ≈ 7 GPU-h.

| run | adapter LR | effective batch | isolates |
|---|---|---|---|
| L1 | 2e-5 | 4800 s | the August recipe, as control |
| L2 | 2e-4 | 4800 s | LR alone |
| L3 | 2e-4 | 1200 s | LR plus 4× more optimizer steps |
| L4 | 1e-3 | 1200 s | upper end of the LR range |
| L5 | second seed of the best of L2–L4 | | noise |

Effective batch 1200 s is `batch_duration 150`, `grad_accum 1`, 8 GPUs
(or 300 / 1 / 4 GPUs). Warmup moves from 20 steps to a ratio of ~3% so it
stays comparable across step budgets. Everything else stays at the campaign
values.

Read for each run: dev WER at the end, the step at which the loss leaves the
plateau, the gap between eval loss and the no-audio floor, and the
hypothesis/reference length ratio (so "does not stop" is distinguishable from
"wrong words").

**Interpretation rules, fixed before the runs:**

- L1 already transcribes → the recipe is fine and the problem is the
  multilingual data or its text; go to the five-language screen with the
  data hypotheses (truecasing, corpus mix, text normalisation) in front.
- L1 fails and L2 works → LR alone. L2 fails and L3 works → steps matter as
  much as LR. L4 better than L3 → keep going up in week 2.
- Success bar: under 10% WER on test-clean and a transition inside the
  epoch. SLAM-ASR's 2% used an ASR-fine-tuned encoder and a 7B decoder; a
  self-supervised w2v-BERT into a 1B decoder will land higher, and that is
  fine.
- Loss is read together with WER, never alone. A working frozen-LLM ASR
  should reach an eval loss well under 1 nat per token on dev-clean,
  plausibly 0.2–0.5. A run whose loss drops while WER stays above 0.5 is
  repeating the August coarse-signal plateau (§1a), not fixing the recipe.

## 3. The five-language interface screen (week 2)

Same five languages and corpus mix as the campaign, ASR-only MA, rendered at
`--budget-hours 125` (625 h total, ≈ 35 min per run on 8 GPUs). Full grid,
not a fraction, because MA-stage runs are cheap and the grant is not the
constraint:

| factor | levels |
|---|---|
| adapter LR | 2e-4, 1e-3 (2e-5 as a single control run, not a grid level) |
| effective batch | 1200 s, 4800 s |
| `stack_factor` | 1 (50 Hz), 4 (12.5 Hz) |
| MA prompt | audio only; plus one verbatim run at the best corner |

Eight grid runs, one verbatim run, one 2e-5 control, two extra seeds at the
best corner: twelve runs, ≈ 60 GPU-h. Metrics: MA-stage generative WER and
CER per language from the in-training eval (200 utterances per set), the loss
transition step, and FLEURS-24 zero-shot CER from melt-eval on the final
checkpoint, which gives a first 24-language signal for free.

The screen decides four campaign constants for MA: adapter LR, effective
batch, stack factor, prompt. IFT keeps its own effective batch (3840 s at
2 nodes / grad_accum 4, measured to hide the all-reduce) and decoder LR 2e-5
until `04-regime.md` says otherwise.

Why a 4× stack and not 5: 4 keeps the 60 s cut at 750 positions and divides
the w2v-BERT frame count evenly; 5 would match SLAM-ASR. Either is fine; 4 is
the default so that 2 and 8 are one halving away if the crossing needs them.

## 4. What `stack_factor` has to do

Concatenate k consecutive encoder frames along the feature axis before
`fc1` (input width k × encoder hidden size), pad the frame count to a
multiple of k, subsample the attention mask by k, and update the adapter's
`_get_output_features_shape` so the length bookkeeping used by the model and
the bucket bins stay consistent. It applies to the MLP; the Conformer already
has a stride of 2 and the Q-Former downsamples by 5 by design, so for the
audio-stack crossing (`03-audio-stack.md`) every adapter is configured to the
same output rate.

## 5. Results

Fill in as the runs land. Quote experiment names, not job ids.

| run | exp_name | dev-clean WER | dev-other WER | transition step | notes |
|---|---|---|---|---|---|
| L1 | | | | | |
| L2 | | | | | |
| L3 | | | | | |
| L4 | | | | | |
| L5 | | | | | |
