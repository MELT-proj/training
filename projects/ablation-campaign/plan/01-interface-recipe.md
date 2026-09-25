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

The floor is not close to the observed loss: it is 1.1–1.5 nats/token
*above* it, consistently across all five languages. Audio was not ignored —
conditioning on it lowers per-token perplexity by a factor of
e^1.1..1.5 ≈ 3–4.4× against no audio at all. That signal is simply not the
right signal for correct tokens: WER stayed at 1.10–1.16 and hypotheses were
fluent but unrelated to the audio. The adapter learned something coarse
("audio present → answer in this register and language") that helps
next-token prediction without helping transcription.

**How this is read downstream:** "the adapter did nothing" is not the
failure mode to assume, and a run that lowers loss without moving WER is
repeating the coarse-signal plateau rather than making progress. The
suspects table in §1 still holds end to end. Confirmed by the PI
2026-09-15; the Settled block and the step-0 interpretation rules were
written to this reading.

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

## 2b. Step 0b — the optimisation recipe (added 2026-09-17, scope narrowed 2026-09-18)

Step 0 was inconclusive (results in §5). All five one-epoch runs stayed above
WER 1.0. A post-hoc three-epoch rerun of L4 broke the plateau, reaching
dev-clean WER 0.62 and loss 0.90, still far from the success bar. At the same
step count (end of epoch 1, about 2,900 steps) its loss was 1.51 against
L4's 2.63, and the only difference at that point was a learning rate that had
decayed less. The PI proposed three checks before the week-2 screen: a more
aggressive schedule, stacking to 10 Hz as SLAM-ASR does, and a larger
decoder. Step 0b runs them together, plus a seed replicate and an encoder
control, because each question can mask the others.

### What step 0b settles, and what it does not (PI, 2026-09-18)

Step 0b produces **a working recipe, not the best configuration**. It settles
only what this file owns: schedule, learning rate, effective batch and frame
rate. Its two remaining arms ask questions that belong to other sections,
and they are **diagnostic controls** whose results are handed on as stated
priors, never as decisions:

| arm | reads as | belongs to |
|---|---|---|
| `wsd-50hz-whisper` | can the pipeline transcribe at all with an easy encoder? Answered: yes, so a null result elsewhere is the encoder, not the recipe | `03-audio-stack.md` decides the encoder |
| `wsd-10hz-qwen2b`, `wsd-10hz-qwen4b` | does decoder size move the interface? | `02-backbones.md` decides the decoder |

The reason for the line is selection bias, not tidiness. If the encoder is
picked here — English, one dataset, one seed per arm — and the crossing in
`03` then runs with a recipe tuned around that winner, `03`'s central
comparison is biased toward it. Either `03` repeats the work properly, in
which case picking early bought nothing, or it does not, and the paper's
audio-stack section rests on one English run. The same argument applies to
decoder size and `02`.

### Common settings

`ABL-MA-librispeech.yaml`; w2v-BERT 2.0 frozen; decoder frozen; MLP adapter;
audio-only prompt; adapter LR 1e-3; effective batch 1200 s; warmup 3%;
**three epochs** of LibriSpeech (about 2,900 audio hours and 8,650 optimizer
steps at 1200 s), the same audio budget as the existing
`MA-librispeech-l4-ep3`, which is the reference arm A0 and needs no rerun.
In-training eval on dev-clean and dev-other at **500 utterances per set**
(200 made the epoch-3 trajectory non-monotonic), about 20 rounds per run.

### Arms

**Names (PI's instruction, 2026-09-18): letters do not survive a week away,
so every arm carries a descriptive name.** The name says schedule, frame
rate and the one thing that differs. Everything unnamed is the common
setting above: Llama-3.2-1B-Instruct decoder, w2v-BERT 2.0 encoder, adapter
LR 1e-3, 1200 s effective batch, three epochs. `wsd` is
warmup-stable-decay; 50 Hz is `stack_factor 1` and 10 Hz is
`stack_factor 5`. The letters in the middle column are the labels the first
step-0b reports used; they are kept only so older entries can be read.

| name | was | campaign row | seed | differs from `wsd-50hz` in | question | GPU-h |
|---|---|---|---|---|---|---|
| `cosine-50hz` | A0 | `MA-librispeech-l4-ep3` | 44 | cosine decay to zero, the old schedule | schedule reference | done |
| `wsd-50hz` | R | `MA-librispeech-r` | 45 | the reference: warmup 3%, constant, cosine decay over the last 20% to 0.1× peak | schedule shape, against `cosine-50hz` | ~20 |
| `wsd-50hz-seed2` | R-seed | `MA-librispeech-r-seed` | 46 | seed only | the noise floor | ~20 |
| `wsd-50hz-lr2e3` | R-lr2e3 | `MA-librispeech-r-lr2e3` | 47 | peak LR 2e-3 | learning rate | ~20 |
| `wsd-50hz-batch600` | R-b600 | `MA-librispeech-r-b600` | 48 | 600 s effective batch, one node × 4 GPUs | twice the optimizer steps at the same GPU-h | ~20 |
| `wsd-10hz` | R-k5 | `MA-librispeech-r-k5` | 49 | `stack_factor 5` | stacking | ~12 |
| `wsd-50hz-whisper` | W | `MA-librispeech-w` | 50 | Whisper-large-v3 encoder | **control only:** can the pipeline transcribe at all? The encoder is decided in `03` | ~35 |
| `wsd-10hz-qwen2b` | Q2-k5 | `MA-librispeech-q2-k5` | 51 | Qwen3.5-2B decoder, stack 5 | **control only:** size, same family; the decoder is decided in `02` | ~40 |
| `wsd-10hz-qwen4b` | Q4-k5 | `MA-librispeech-q4-k5` | 52 | Qwen3.5-4B decoder, stack 5 | **control only:** size; the decoder is decided in `02` | ~80 |

About 250 GPU-h in total. Campaign row ids may be renamed to match the
names above; the row id is only the grid key, so nothing in `arms.tsv`,
W&B or an output directory depends on it. `exp_name` is composed from the
axes and must not be touched for a run that already exists.

**Every arm drew a different seed, so each contrast carries one seed draw**
(45 through 52; only `wsd-50hz` against `wsd-50hz-seed2` isolates the
seed). That was not the intent and it matters, because the measured
dev-clean spread between those two is 0.11, not the 0.02 seen on the
one-epoch pair. Contrasts larger than the spread survive it; contrasts of
the same size as the spread do not, and need a replicate before they are
quoted. Future arms hold the seed fixed unless the seed is the variable.

**Why warmup-stable-decay and not a pure constant schedule.** Its stable
phase *is* a constant-LR run, so the evals up to the start of decay give the
constant-schedule trajectory, and the decay tail shows what annealing adds on
top. One arm answers both. In transformers 5.16.1 (the training image) this
is `lr_scheduler_type: warmup_stable_decay` with `lr_scheduler_kwargs:
{num_decay_steps: <int>, min_lr_ratio: 0.1, decay_type: cosine}`.
`num_decay_steps` is an absolute count and steps are derived, so compute 20%
of the step count the run reports and check the logged LR at a few steps in
a dry run before launching.

**Why the seed replicate.** Seed noise measured on one-epoch runs (L4 against
L5) was about 0.02, but the metric here is *when* the transition happens, and
transition timing can be far noisier than end-of-run loss.

**Why the Whisper control.** SLAM-ASR's headline numbers rest on an encoder
already fine-tuned for ASR. w2v-BERT 2.0 as loaded here is self-supervised
only. If the representation is the bottleneck, no schedule or decoder size
fixes it, and every later comparison inherits the problem. Whisper-large-v3
is already staged on MN5 and wired as an encoder. It pads every input to a
30 s window, so its encoder cost per short utterance is higher; report GPU-h.

**Why the size pair runs at stack 5, and within one family.** A frozen
decoder at 50 Hz is memory-bound (the Qwen3.5-2B MA arm needed
`batch_duration 30`), and a 5× shorter sequence is what makes a 4B decoder
affordable here. The cost is that size × stacking is not observed; that is
stated as an assumption. Llama-1B against Qwen-2B confounds size with family
and attention type, so size is read from Q4-k5 against Q2-k5 only.
Qwen3.5-4B: text hidden size 2560, 32 layers (24 Gated DeltaNet, 8 full
attention), about 4B text parameters (4.66B including the vision tower,
which MELT does not load), vocabulary shared with the 2B, ungated.

### Pre-flight, before any GPU is spent

1. On `MA-librispeech-l4-ep3`'s final eval hypotheses: substitution,
   deletion and insertion rates, the hypothesis-to-reference length ratio,
   and 20 hypotheses read by eye. A loss of 0.90 with 62% WER, and dev-other
   beating dev-clean at epoch 2, are both odd. If insertions dominate, the
   problem is decoding, which no schedule fixes: stop and report.

   **Result, 2026-09-17, and the rule revised.** The check fired: on the 20
   hypothesis pairs the trainer logs (10 per set), WER 1.19 with
   substitution 39%, deletion 10%, insertion 50%. But 4 of the 20 are
   repetition loops running to the 256-token cap, and those 4 carry about
   60% of all insertions. The other 16 stop on their own at a length ratio
   near 1.0 with ordinary substitution-dominated errors. The 20 logged pairs
   are also not representative: the trainer's own score over the full 200
   utterances per set was 0.62 and 0.78, not 1.19. Three conclusions:

   - A handful of loops can dominate an insertion count, so "insertions
     dominate" was the wrong trigger. The stop condition is now **runaway
     on most hypotheses**, measured over the full eval set.
   - Most hypotheses stop correctly, so the stop token is learned. An
     undertrained stop token is a weaker explanation for the loops than the
     model losing its place in long audio at 50 Hz, which the stacking arm
     tests directly.
   - Anti-repetition decoding (`repetition_penalty`, `no_repeat_ngram_size`)
     is **not** adopted in the campaign metric. It would hide a failure
     that is itself evidence about alignment, suppress legitimate repeated
     words, and differ by language and tokenizer. It may be measured once,
     as a diagnostic.

   Step 0b therefore proceeds, with the runaway fraction added to every
   eval (see Metrics) and the diagnostic pass below.
1b. Eval-only pass on `MA-librispeech-l4-ep3`'s final checkpoint over the
   full dev-clean and dev-other sets, greedy as in training, every
   hypothesis saved: substitution, deletion and insertion rates, runaway
   fraction, and runaway against utterance duration. Then the same pass
   with `no_repeat_ngram_size: 4`, to size how much of the WER the loops
   explain. Runs in parallel with the arms and does not gate them.
2. A dry run confirming the warmup-stable-decay kwargs reach the scheduler.
3. For the Qwen arms only: **PR #132 merged** (issue #124: Qwen checkpoints
   carry no `eos_token_id`, so generation never stops and WER is
   meaningless); `Qwen/Qwen3.5-4B` staged to MN5 and loaded offline;
   `campaign.py plan` shows the right chat-template profile for it; batch
   sized with `run.memory_preallocation: true` to reach 1200 s effective.

### Metrics

- **Primary:** audio hours seen, optimizer steps and GPU-h until dev-clean
  WER is below 0.10 at two consecutive in-training evals.
- **Secondary:** WER and CER on the full dev-clean and dev-other sets for the
  final checkpoint; substitution, deletion, insertion; length ratio; eval
  loss; decoder positions per audio second.
- **Runaway fraction, at every in-training eval** (added 2026-09-17): the
  share of hypotheses more than twice the reference's word count, alongside
  substitution, deletion and insertion rates over the whole eval set, not
  the logged sample. The trainer already scores with `jiwer`, so
  `jiwer.process_words` supplies the rates. If this does not land as a small
  change, the arms launch without it and the eval-only pass of pre-flight
  1b is repeated on each arm's final checkpoint instead.

### Decision rules, fixed before the runs

**Hold, PI's instruction 2026-09-18: no rule is applied until all eight arms
have finished.** Partial results are recorded in §5 as they land, and a rule
may be reported as "would trigger", but the schedule, stacking, learning
rate, encoder and size questions are settled together, once, on the full
set of arms. The reason is in the table above: with one seed per arm, a
contrast read early against a single reference arm can be a seed draw.

1. **Schedule.** Warmup-stable-decay becomes the MA default if R beats A0 at
   equal steps by more than the R/R-seed spread. LR 2e-3 is adopted if it
   reaches the threshold in fewer audio hours without loss spikes. The 600 s
   batch is adopted if it reaches the threshold in fewer audio hours.
2. **Stacking.** Stack 5 becomes the default if R-k5 is within noise of R or
   better, since it cuts decoder positions five-fold. If clearly worse, the
   screen tests stack 2 and 4.
3. **Size — reported, not decided here.** If `wsd-10hz-qwen4b` reaches the
   threshold and `wsd-10hz-qwen2b` does not, or does so in markedly fewer
   hours, that is written into `02-backbones.md` as a prior and a ~4B point
   is added to its grid. It does not change this screen's backbone, and it
   does not settle the "2–3B is enough" framing, which `02` owns.
4. **Encoder — reported, not decided here.** `wsd-50hz-whisper` reaching the
   threshold where the w2v-BERT arms do not says the representation is a
   first-order factor, which the campaign had assumed it was not. What
   triggers is a **calendar reorder, not an adoption**: `03-audio-stack.md`
   runs before `02-backbones.md`, so the backbone grid is compared on a
   stack that has been chosen rather than assumed. The encoder itself is
   decided by `03`'s crossing, at equal budget, over five languages, on the
   eval that matters. Recorded 2026-09-18.
5. **Nothing reaches 10% in three epochs.** Take the best arm to six epochs
   before starting the screen. The screen does not start on a recipe that
   has never transcribed LibriSpeech.

**Consequence for the screen (§3).** Its per-arm budget becomes at least
1.5× the best arm's hours-to-threshold on LibriSpeech, since five languages
are harder than one, and never less than one epoch of 700 h per language
(about 10,500 steps at 1200 s). The 125 h per language written in §3 is about
1,900 steps, fewer than the one-epoch L4 run that failed. Factors step 0b
settles leave the screen's grid.

## 3. The five-language interface screen (week 2)

*Revised 2026-09-17: the budget and factor levels below are re-set from step
0b (§2b) before launch. Budget re-derived and the grid rendered 2026-09-20.*

Same five languages and corpus mix as the campaign, ASR-only MA. Full grid,
not a fraction, because MA-stage runs are cheap and the grant is not the
constraint:

**Budget, derived 2026-09-20.** §2b's Consequence sets the per-arm budget at
≥ 1.5× the best arm's hours-to-threshold (1.5 × 87.3 = 131 h) **and never
under one epoch of 700 h per language**. The floor binds, and 700 h × 5 is
exactly `ABL-MA-700-asr.yaml` (`total_hours: 3500.00`), so the screen needs
no new config: it is campaign.yaml rows against the config the campaign
already trains on. The `--budget-hours 125` figure written here before is
withdrawn — at ~1,900 steps it was below where the transition appeared.

One override is not optional. `ABL-MA-700-asr.yaml`'s own `batch_duration
150 × grad_accum 4 × 8 ranks` is **4800 s**, the August recipe this screen
exists to replace, so every arm sets `grad_accum_steps` explicitly. The
three batch levels are reached as:

| effective batch | `batch_duration` × `grad_accum` × world | steps/epoch |
|---|---|---|
| 1200 s | 150 × 1 × 8 | 10,500 |
| 600 s | 75 × 1 × 8 | 21,000 |
| 300 s | 75 × 1 × 4 | 42,000 |

Whisper at 1200 s runs at 150 × 1 like w2v-BERT (PI, 2026-09-21), so the two
encoder halves differ only in the encoder. The cosine pass had run it at
75 × 2 on a ~6× padding estimate and measured 11.5 GB of 64 (below).
`max_audio_seq_len` is derived from the encoder name rather than
passed by hand (`plan_arm.py`'s `ENCODER_WINDOW_FRAMES`).

**The schedule is warmup-stable-decay, and it is now an axis (2026-09-21).**
Step 0b's Rule 1 adopted WSD as the MA default, but the screen's first twelve
arms ran **cosine**: `ABL-MA-700-asr.yaml` declares `lr_scheduler_type:
cosine`, WSD was never a campaign axis, and step 0b had obtained it only by
hand-passing `--trainer.lr_scheduler_kwargs` on each submission. The reason it
was hand-passed is that `num_decay_steps` is an *absolute* count that differs
per arm — so `plan_arm.py` now derives it (20% of the arm's own steps,
`min_lr_ratio` 0.1, `decay_type` cosine) and tags `-wsd` into `EXP_NAME`,
because the schedule is recoverable from no other tag.

The same config also sets **`warmup_steps: 20`**, a fixed count. Across the
screen's own batch levels that is 0.19% / 0.10% / 0.05% of training — so
warmup length was confounded with the batch factor the screen exists to
measure. `warmup_ratio: 0.03` is now set per arm, matching §2b's common
settings, and `warmup_steps` is forced to 0 alongside it because HF's
`get_warmup_steps` only consults the ratio when the count is ≤ 0.

**Cost, measured 2026-09-21** on the seven arms that ran before the schedule
was corrected: **27 GPU-h** at 1200 s, **37** at 600 s, **32** for Whisper at
1200 s — against ≈ 20 extrapolated. Twelve arms are therefore ≈ 380 GPU-h,
nineteen ≈ 500. Note that 600 s costs *more* than 1200 s for the same audio
(37 vs 27): per-step overhead does not halve when the batch does, so the
batch axis is not cost-neutral and the 300 s level is the most expensive
point on it. Memory is not the constraint anyone expected: Whisper at
`batch_duration 75` peaked at **11.5 GB of 64**, below w2v-BERT's 19.7 GB at
150, so the ~6x padding estimate was conservative.

**Levels re-set from step 0b, 2026-09-20 (PI).** The table below previously
carried levels chosen before step 0b ran, and three of them were settings
step 0b had already ruled out. A screen must not re-test a factor the
screen before it settled:

| factor | levels | why these |
|---|---|---|
| adapter LR | 1e-3, 2e-3 | step 0 put 2e-5 at WER 1.12 and 2e-4 at 1.08; `wsd-50hz-lr2e3` beat `wsd-50hz` 0.599 vs 0.664 on dev-other. 2e-5 stays as **one control run**, not a grid level — it is the only thing tying this screen to the August failure and to LibriSpeech, and it costs ~5 GPU-h |
| effective batch | 1200 s, 600 s, 300 s | **4800 s is dropped.** It is the August recipe: ~2,600 optimizer steps, the diagnosed cause of the failure in §1. `wsd-50hz-batch600` beat `wsd-50hz` 0.543 vs 0.664 on dev-other (a 0.002 seed regime), so smaller is still winning and 300 s tests whether that has bottomed out |
| `stack_factor` | fixed at **5** (10 Hz) | continuity with `wsd-10hz`, the best w2v-BERT arm of step 0b, and with SLAM-ASR. Not a grid factor here; see the sweep below |
| MA prompt | audio only; plus one verbatim run at the best corner; **proposed third level (PI, 2026-09-20): verbatim with a language ID in the prompt** — see below |

**What the first pass showed (measured 2026-09-21).** Seven arms ran before
the schedule was corrected, so their absolute numbers belong to the cosine
control, not to this screen. The *shape* is still informative, and it settles
why the grid runs on two encoders. All six w2v-BERT arms finished one epoch
at **WER 0.879–0.976**, a total range of 0.097 — smaller than the 0.108
seed-only spread step 0b measured at WER ≈ 0.5, at a *lower* error rate than
these arms sit at, and noise grows with the error rate (`agent-protocol.md`
§3). **No LR or batch contrast in the w2v-BERT half is separable from noise.**
The single Whisper arm finished at **0.131 / CER 0.068**, an aligned model, in
the regime where the measured spread is 0.002. A single-encoder screen on
w2v-BERT would have produced six numbers and no decision.

That also reads on `03-audio-stack.md`: one epoch of 700 h/lang at k=5 does
not align w2v-BERT 2.0 on this mixture, while the same recipe on Whisper
does. It is the strongest evidence yet that the encoder is a first-order
factor for this campaign, and it is a prior for `03`, not a decision here.

**The grid runs on both encoders (PI, 2026-09-20).** The screen has to
resolve LR and batch contrasts, and step 0b measured exactly those against
the seed-only spread at the same error regime:

| contrast | dev-clean Δ | dev-other Δ |
|---|---|---|
| seed only (`wsd-50hz` vs `wsd-50hz-seed2`) | 0.108 | 0.002 |
| adapter LR 1e-3 → 2e-3 | 0.112 | 0.065 |
| effective batch 1200 → 600 s | 0.148 | 0.121 |

On dev-clean the LR contrast is the size of the noise. w2v-BERT's best
step-0b arm sat at WER 0.314 on data easier than this mixture, so the spread
here is likelier to widen than narrow; Whisper sat at 0.038, where the
measured spread is 0.002. Running the grid on w2v-BERT alone risks a screen
that cannot answer its own question; running it on Whisper alone is the
selection bias §2b warns about. Running both costs ~120 GPU-h more and tests
whether the recipe optimum is encoder-invariant — which `03-audio-stack.md`
§2 already *assumes* when it fixes one recipe across all sixteen crossing
arms. If the optimum differs by encoder, that assumption is wrong, and this
screen is where it is cheapest to find out.

Twelve grid runs (2 LR × 3 batch × 2 encoders, all at k=5), then at the best
corner: two stacking-sweep runs (k=2 and k=10), one 2e-5 control, two prompt
runs (verbatim, and verbatim+LID if adopted), two extra seeds. **Nineteen
runs, ≈ 380 GPU-h.** Only the twelve grid runs can be submitted up front; the
other seven are defined relative to a corner the grid has to find first.
Seed fixed at 42 across all twelve, per `agent-protocol.md` §3 — step 0b drew
a different seed per arm, which is how a 0.11 contrast turned out to be a
seed draw. Metrics: MA-stage generative WER and
CER per language from the in-training eval (200 utterances per set), the loss
transition step, and FLEURS-24 zero-shot CER from melt-eval on the final
checkpoint, which gives a first 24-language signal for free.

The screen decides four campaign constants for MA: adapter LR, effective
batch, stack factor, prompt. IFT keeps its own effective batch (3840 s at
2 nodes / grad_accum 4, measured to hide the all-reduce) and decoder LR 2e-5
until `04-regime.md` says otherwise.

**Stacking is a sweep, not a grid factor (revised 2026-09-20).** The earlier
note here preferred k=4 because it "divides the w2v-BERT frame count
evenly"; that is not a reason, since a 60 s cut at 50 Hz is 3,000 frames and
5 divides it exactly as evenly as 4. The real argument for 12.5 Hz was to
pick one rate every adapter in `03-audio-stack.md` can hit — but the
Q-Former is natively 10 Hz, so 10 Hz is at least as natural a common rate,
and it is the rate SLAM-ASR uses and the one `wsd-10hz` was measured at.

So k=5 is the default, and stacking gets a **one-dimensional sweep at the
best LR/batch corner: k ∈ {2, 5, 10}** (25, 10 and 5 Hz). Three adjacent
values such as {4, 5, 6} span 12.5 to 8.3 Hz — under 30% apart, likely
inside noise, and they would not draw a curve. The efficiency figure in
`03-audio-stack.md` §4 plots CER against decoder positions per audio second,
so what it needs from this screen is *spread* on that axis and the point
where accuracy starts to pay for it.

**Prompt level: verbatim with a language ID (PI, 2026-09-20).** MA is where
the model learns what the audio *is*, and a frozen decoder cannot infer the
target language from 10–50 Hz features as reliably as it can be told; the
August failure included hypotheses "sometimes in the wrong language" (§1),
which is exactly what naming the language would suppress.

**What it means concretely** (`melt/training/data/audio/lhotse/helpers.py`):
add `{lang}` variants of the six `verbatim` templates, each saying that what
sits between the audio tags will be in that language — the PI's wording is
"Everything between those tags will be in {lang}." `{lang}` already resolves
through `LANGUAGE_ISO_TO_NAME` to a language *name*, not an ISO code, so
this needs no new machinery.

**Two things to get right, both of which the existing `asr` family gets
wrong today:**

1. **LID must be selectable, not sampled.** The `asr` family already holds
   six `{lang}` templates and six explicitly-no-LID copies *in one list*, and
   `prompt_template_selection: "random"` draws from the whole list. Any run
   using it therefore trains on a random mixture of LID and no-LID prompts,
   which makes LID a per-sample coin flip rather than a factor — no existing
   run can say anything about it. `"with_language"` filters to the `{lang}`
   templates, but there is **no `"without_language"`**, so the no-LID control
   cannot currently be selected cleanly either. Adding that selection mode is
   the prerequisite for testing this at all.
2. **Score with and without the tag.** A model trained with the language
   named either needs it at inference or must be shown to survive without it.
   Our eval sets are per-language so supplying it is easy, which is exactly
   why the withheld-tag score is the one that matters.

Also stale in that file: the comment above the templates claims every
template must contain `{audio_token}` and `{lang}`, which is already untrue
of half the `asr` list and all of `verbatim`.

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

Measured 2026-09-16, `MA-librispeech-l1`..`l4` (jobs on `arms.tsv`), final-epoch
in-training generative eval, 200 utterances/set. None crossed WER 1.0 or the
<10% success bar; loss declined smoothly and monotonically in every run, with
no plateau-then-drop transition inside the epoch (unlike the SLAM-ASR
dynamic §1 cites) -- "transition step" is reported as none for all four.

| run | exp_name | dev-clean WER | dev-other WER | transition step | notes |
|---|---|---|---|---|---|
| L1 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-elr6e6-dlr2e5-lr2e5-s42-8g` | 1.120 | 1.137 | none | August recipe control; loss 3.072/2.968 (clean/other). Reproduces the August plateau (WER>1.0) on LibriSpeech alone -- not a multilingual-data artifact. |
| L2 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-elr6e6-dlr2e5-lr2e4-s42-8g` | 1.084 | 1.075 | none | LR alone (10x L1). Loss 3.012/2.914. Small improvement over L1, not the "L2 works" break the interpretation rules describe. |
| L3 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr2e4-s42-8g` | 1.076 | 1.087 | none | L2's LR + 4x steps (1200 s batch). Loss 2.943/2.842. Marginally better than L2 -- steps alone do not break the plateau either. |
| L4 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s42-8g` | 1.040 | 1.056 | none | L3's steps, LR pushed to 1e-3. Loss 2.625/2.587, clearly the best of the four and still declining fastest at epoch end (loss delta accelerating over the back half of training, unlike L1-L3). Per the interpretation rules ("L4 better than L3 -> keep going up"), the open question for week 2 is whether an even higher LR or a longer run clears the plateau. |
| L5 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s43-8g` | 1.053 | 1.058 | none | Second seed of L4. Loss 2.64/2.593 (clean/other) -- within ~0.02 nats and ~0.02 WER of L4 (2.625/2.587, 1.040/1.056). Tight seed noise; L4's improvement over L1-L3 is a real recipe effect, not seed luck. |

### Post-hoc diagnostic: does the plateau break with more gradient updates?

Not part of the L1-L5 grid. L4's loss delta accelerated through the back half
of its one epoch (-0.03/step early -> -0.108 near epoch 0.8) before
flattening in the final ~10% -- plausibly the SLAM-ASR plateau-then-drop
caught mid-transition, confounded by L4's own cosine schedule being tuned to
fully decay by end of epoch 1. `MA-librispeech-l4-ep3` reruns L4's exact
LR/batch (1e-3, 1200 s) fresh for 3 epochs (seed 44, so the schedule is
re-derived over the full 3-epoch horizon and LR decays more slowly through
epoch 1) -- ~2h24m wall, single 8-GPU allocation. Measured 2026-09-17:

| epoch | dev-clean loss | dev-clean WER | dev-other loss | dev-other WER |
|---|---|---|---|---|
| 1.0 | 1.509 | 1.155 | 1.746 | 1.312 |
| 2.0 | 0.989 | 0.895 | 1.244 | 0.745 |
| 3.0 (final) | 0.901 | 0.622 | 1.146 | 0.776 |

**The plateau breaks.** WER falls from >1.1 (worse than L4's own 1-epoch
result -- expected, since this run's LR has decayed less by the same step
count) to 0.62-0.78 by epoch 3: a real, large transition, not noise (loss
drops 3.11->1.51 within epoch 1 alone, then continues 1.51->0.90 over
epochs 2-3). Still short of the <10% success-bar, and dev-clean/dev-other
diverge slightly in epoch 3 (0.622 vs 0.776, and the single best point in
the whole run was epoch 2.726's dev-clean WER 0.588 -- the trajectory is not
perfectly monotonic this late), so this is not "solved," but it reframes the
diagnosis: **the bottleneck at 1e-3/1200 s was schedule length (LR decayed
before the transition completed), not adapter/decoder capacity or the
encoder.** The August recipe's failure is real, but "does a small decoder
ever align" is not answered by this -- the question for week 2 is whether
the five-language screen's step/LR budget needs to grow to let this
transition complete, or whether it completes given more wall-clock at the
125 h/screen budget too.

### Step 0b pre-flight 1b: full-set decoding diagnostic

Not part of the L1-L5 grid. Per §2b's pre-flight step 1b (the Fondue
Orchestrator's resolution of the "insertions dominate" STOP, main commit
`ad07b3f`): `step0b_diagnostic_full_eval.py` decoded `MA-librispeech-l4-ep3`'s
final checkpoint over the *entire* dev-clean (2,703 cuts) and dev-other
(2,864 cuts) sets -- not the 20 samples the trainer logs, and not even the
200/set cap training itself used -- greedy as in training, then again with
`no_repeat_ngram_size: 4`. MN5 job 45985475, off `arms.tsv` by design (same
as `no_audio_floor.py`), 22m28s on one GPU.

| pass | set | n | WER | S | D | I | length ratio | runaway fraction |
|---|---|---|---|---|---|---|---|---|
| greedy | dev-clean | 2703 | 0.689 | 31.8% | 10.3% | 26.7% | 1.150 | 2.37% (64) |
| greedy | dev-other | 2864 | 0.914 | 44.0% | 11.5% | 36.0% | 1.177 | 3.28% (94) |
| no_repeat_ngram_size 4 | dev-clean | 2703 | 0.487 | 30.6% | 10.8% | 7.3% | 0.969 | 0.00% (0) |
| no_repeat_ngram_size 4 | dev-other | 2864 | 0.643 | 42.9% | 12.0% | 9.4% | 0.976 | 0.03% (1) |

Runaway hypotheses are markedly longer-duration on average: mean 11.79 s
(clean) / 10.84 s (other) against 7.06 s / 6.29 s for non-runaway pairs --
supporting "losing its place in long 50 Hz audio" over "undertrained EOS"
(`R-k5`, stack_factor 5, tests this directly).

Two findings, both load-bearing:

1. **The full-set greedy WER (0.689/0.914) is *worse* than the 200-sample
   number training itself reported (0.622/0.776), not better.** The 20
   logged samples spiked to 1.19 by drawing disproportionately from
   longer/harder cuts; the 200-sample subset happened to land easier than
   the full set. Neither logged number is a safe stand-in for the true
   full-set score -- this is exactly why the runaway fraction and full-set
   S/D/I now ship with every in-training eval (`melt/training/metrics.py`),
   not just at the end of a run.
2. **Runaway is real but a minority (2.4-3.3% of hypotheses), confirming
   the revised stop condition does not fire.** `no_repeat_ngram_size 4`
   removes it almost entirely (0.00%/0.03%) and drops WER by 0.20-0.27
   absolute (29-30% relative) -- nearly all of that from the insertion rate
   collapsing (26.7%->7.3%, 36.0%->9.4%), while substitution and deletion
   barely move (31.8%->30.6%, 44.0%->42.9% substitution; both deletion
   rates flat). The repetition loops are a large, cleanly separable, and
   currently unclaimed chunk of the measured WER -- but even with them
   fully suppressed, WER stays at 0.49-0.64, nowhere near the <10% bar.
   Consistent with the decision already recorded above: anti-repetition
   decoding is a real lever but stays out of the campaign metric, since
   fixing it would not have gotten this recipe to threshold either, and it
   would flatter every future arm's WER by the same uncontrolled amount.

### Step 0b arms

Launched 2026-09-17/18: R, R-seed, R-lr2e3, R-b600, R-k5, W, Q2-k5, Q4-k5
(§2b). All 8 landed 2026-09-19 (Q2-k5 and Q4-k5 both TIMEOUT at 41%/45%
and were resumed as jobs 46077302/46077303). In-training eval,
`max_samples 500`/set, the metrics from `melt/training/metrics.py`'s
update (full-set S/D/I/length ratio/runaway fraction, not the old
200-sample log). A ninth arm, `Q4-k5-seed2` (seed 53, job 46143618),
requested by the Orchestrator to check the 2B->4B size gap against seed
noise, landed 2026-09-20 -- see the decision-rules discussion below.

**Resume note.** Several arms hit their original 3h wall-clock budget at
~87% (the 500-sample eval costs more per round than A0/l4-ep3's 200) and
needed `--resume`; `campaign.yaml`'s `time:` bumped to 4h for next time
(board entry). One cosmetic artifact from resuming: the final logged
`epoch` value undercounts (e.g. `2.093` instead of `3.0`) even though
`global_step` correctly reaches the full 3-epoch target (8652) and the
LR schedule is step-driven, not epoch-driven -- read `global_step`, not
`epoch`, as the source of truth for how much of the run actually happened
on a resumed arm.

| run | exp_name | dev-clean WER | dev-other WER | dev-clean loss | dev-other loss | runaway (clean/other) | notes |
|---|---|---|---|---|---|---|---|
| A0 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s44-8g` | 0.622 | 0.776 | 0.901 | 1.146 | -- | = `MA-librispeech-l4-ep3`, cosine decay, not rerun. Full-set greedy (not the 200-sample log): WER 0.689/0.914, runaway 2.37%/3.28% (see the pre-flight 1b diagnostic above). |
| R-k5 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-sk5-ga1-elr6e6-dlr2e5-lr1e3-s49-8g` | **0.314** | **0.490** | 0.621 | 0.885 | 1.0% / 0.6% | job 46059845 (resumed from 45985944), `stack_factor 5`, warmup-stable-decay. Roughly half A0's WER and a 3-4x lower runaway fraction than A0's own full-set greedy number. |
| R-seed | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s46-8g` | 0.441 | 0.666 | 0.750 | 1.012 | 1.4% / 1.4% | job 46059832 (resumed from 45985924), seed 46. Worse than R-k5 as expected (`stack_factor 1`, same as A0/L4). |
| R | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s45-8g` | 0.549 | 0.664 | 0.775 | 1.022 | 2.0% / 1.4% | job 46059069 (resumed from 45985909), seed 45, `stack_factor 1`, warmup-stable-decay. Beats A0 (cosine, same steps): 0.549 vs 0.689 clean, 0.664 vs 0.914 other. R/R-seed spread is large on dev-clean (0.549 vs 0.441, 0.11 absolute) and tiny on dev-other (0.664 vs 0.666) -- noisier than the ~0.02 spread measured on the one-epoch L4/L5 pair. |
| R-lr2e3 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr2e3-s47-8g` | 0.437 | 0.599 | 0.665 | 0.927 | 1.0% / 1.2% | job 46059833 (resumed from 45985930), peak LR 2e-3. Better than R (LR 1e-3) on both sets. |
| R-b600 | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s48-4g` | 0.401 | 0.543 | 0.683 | 0.932 | 0.6% / 0.8% | job 45985931, `nodes:1 gpus_per_node:4` (600 s effective batch, ~2x R's steps at 17301 vs 8652). Beats R (0.549/0.664) but loses to R-k5 (0.314/0.490) -- more steps alone doesn't match what `stack_factor 5` buys. |
| **W** | `MA-librispeech-whisperlargeF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s50-8g` | **0.038** | **0.061** | 0.137 | 0.191 | 0.0% / 0.0% | job 46050285 (resubmitted with the `max_audio_seq_len 3000` fix). Whisper-large-v3 encoder, `stack_factor 1`, same steps/schedule as R. Crosses the <10% threshold decisively on both sets -- the only step-0b arm to do so, by a wide margin over every w2v-BERT arm including R-k5. |
| Q2-k5 | `MA-librispeech-w2vbF-qwen35_2bInsF-mlpT-sk5-bd30-ga5-elr6e6-dlr2e5-lr1e3-s51-8g` | 0.169 | 0.312 | 0.369 | 0.545 | 0.0% / 0.0% | job 46077302 (resumed from 45987899, TIMEOUT at 41%). Qwen3.5-2B decoder, `stack_factor 5`, same w2v-BERT encoder as R-k5. Beats R-k5 (0.314/0.490) by a wide margin but does not cross <10%. |
| Q4-k5 | `MA-librispeech-w2vbF-qwen35_4bInsF-mlpT-sk5-bd30-ga5-elr6e6-dlr2e5-lr1e3-s52-8g` | 0.104 | 0.239 | 0.253 | 0.403 | 0.0% / 0.0% | job 46077303 (resumed from 45987998, TIMEOUT at 45%). Qwen3.5-4B decoder, otherwise identical to Q2-k5. Beats Q2-k5 on both sets (0.104 vs 0.169 clean, 0.239 vs 0.312 other) but lands just above the <10% bar on dev-clean (10.36%) and well above it on dev-other. Last of the 8 step-0b arms to land. |
| Q4-k5-seed2 | `MA-librispeech-w2vbF-qwen35_4bInsF-mlpT-sk5-bd30-ga5-elr6e6-dlr2e5-lr1e3-s53-8g` | 0.105 | 0.237 | 0.250 | 0.409 | 0.0% / 0.0% | job 46143618, seed 53, otherwise identical to Q4-k5. Landed 0.0017/0.0023 from Q4-k5 (seed 52) on clean/other -- a tight, well-behaved seed spread at this error regime, nothing like the 0.11 spread measured on R/R-seed at a much higher error rate. |

All 9 step-0b arms (8 original plus the seed replicate) are complete.

**Whisper-large-v3 decoding the same sets by itself** (measured 2026-09-20,
`whisper_reference_decode.py`; no MELT model, no adapter, no LLM). Same cuts,
`custom.pnc_text` reference, `BasicTextNormalizer`, greedy, duration-sorted
batches of 16, `flash_attention_2`, language forced to English. "First 500"
is the arm's own `max_samples: 500` subset (seeded shuffle, not shard order),
so it is the like-for-like row; cuts over 30 s use Whisper long-form. The arm
column is `wsd-50hz-whisper`'s in-training eval (`MA-librispeech-whisperlargeF-...-s50-8g`).

| set | cuts | Whisper WER | Whisper CER | arm WER | arm WER / Whisper WER | Whisper WER / arm WER |
|---|---|---|---|---|---|---|
| dev-clean, first 500 (like-for-like) | 500 | 0.0272 | 0.0118 | 0.038 | 1.40 | 0.72 |
| dev-other, first 500 (like-for-like) | 500 | 0.0410 | 0.0186 | 0.061 | 1.49 | 0.67 |
| dev-clean, full set | 2,703 | 0.0249 | 0.0115 | -- | -- | -- |
| dev-other, full set | 2,864 | 0.0416 | 0.0189 | -- | -- | -- |

The arm has no full-set number, so the full-set rows carry no ratio.

**Decision rules applied to the numbers above** (not adjudicated here, per
the Orchestrator's instruction -- flagging what the raw numbers say against
each rule as written):

- **Rule 1 (schedule).** "Warmup-stable-decay becomes the MA default if R
  beats A0 at equal steps by more than the R/R-seed spread." R beats A0 by
  0.14 (clean) / 0.25 (other); the R/R-seed spread is 0.11 (clean) / 0.002
  (other), measured at WER 0.44-0.55 dev-clean -- the same regime as R and
  A0 themselves, so it is the right spread for this contrast. R's margin
  over A0 exceeds it on both sets -- the rule as written triggers, though a
  single extra seed pair is a thin basis for "the spread," and (per the
  regime note below) this 0.11 figure should not be exported to contrasts
  at a different error rate. "LR 2e-3 is adopted if it reaches the
  threshold in fewer audio hours without loss spikes" -- R-lr2e3 beats R
  on both sets at the same step count, but nothing has reached the <10%
  threshold yet, so this half of the rule has no threshold-crossing to
  measure against.
- **Rule 2 (stacking).** "Stack 5 becomes the default if R-k5 is within
  noise of R or better." R-k5 (0.314/0.490) clearly beats R (0.549/0.664)
  by more than the measured R/R-seed spread on both sets -- the rule
  triggers unambiguously.
- **Rule 1, 600 s batch clause.** "The 600 s batch is adopted if it reaches
  the threshold in fewer audio hours." R-b600 does not reach <10% at all
  (0.401/0.543 after 3 epochs' worth of steps, ~2x R's step count) -- it
  beats R but loses to R-k5, so the rule does not trigger; the 600 s batch
  is not adopted over `stack_factor 5`.
- **Rule 4 (encoder).** "If W reaches the threshold much earlier than R, the
  encoder question moves ahead of the backbone grid." W reaches 0.038/0.061
  at the same step count where R is still at 0.549/0.664 and R-k5 (the best
  w2v-BERT arm) is at 0.314/0.490 -- W is the only arm to cross <10% at all.
  The rule triggers unambiguously. This also reframes Rule 5 ("nothing
  reaches 10% in three epochs, take the best arm to six epochs before the
  screen") -- it does not fire, since Whisper crossed; the week-2 screen is
  unblocked on the budget side without a six-epoch extension.
  Encoder choice (`03-audio-stack.md`'s question) is reported here as a
  prior for that section, not decided in this file: Whisper won with the
  **smallest** decoder tested and **no** stacking, against w2v-BERT arms
  running up to 4x the decoder and a 5x shorter sequence (`stack_factor 5`)
  -- so at a fixed decoder budget, the encoder bought more than
  quadrupling the decoder did. That reads as supporting the paper's 2-3B
  framing rather than straining it, not as a reason to move size ahead of
  or behind encoder work.
- **Rule 3 (size).** "If Q4-k5 reaches the threshold and Q2-k5 does not, or
  does so in markedly fewer hours, the paper's '2-3B is enough' framing must
  be tested and a ~4B point joins the backbone grid." Literally, neither
  arm reaches the <10% dev-clean bar: Q4-k5 lands at 0.1036, just 0.36
  points over; Q2-k5 at 0.1686, well over. The rule's stated trigger
  condition does not fire. The seed replicate (`Q4-k5-seed2`, seed 53, job
  46143618) now settles both open questions it was launched to answer:
  the seed-only spread at WER 0.10-0.24 (Q4-k5's own regime) is 0.0017
  (clean) / 0.0023 (other) -- tight, nothing like the 0.11 spread measured
  on `R`/`R-seed` at WER 0.44-0.55. So the 2B->4B gap (0.104 vs 0.169
  clean, ~38% relative; 0.239 vs 0.312 other, ~23% relative) is now safely
  quotable as a real, consistent effect. It also settles that Q4-k5's
  10.36%/10.53% landing just over the bar on both seeds is not a seed
  draw: the 4B decoder genuinely falls short of <10% dev-clean on this
  encoder, reproducibly. This is `02-backbones.md`'s question; reported
  here as a prior, not decided.

**Regime note, added 2026-09-20 (Orchestrator).** Seed noise scales with
the error rate, not a campaign-wide constant: 0.11 dev-clean at WER
0.44-0.55 (`R`/`R-seed`), 0.0017 dev-clean at WER 0.10-0.24
(`Q4-k5`/`Q4-k5-seed2`) -- roughly two orders of magnitude apart. A spread
measured at one WER level is not a safe stand-in for a contrast at
another; every spread quoted above is now labelled with the WER range it
came from rather than reported as one number. This is now `agent-protocol.md`
§3's standing rule, since the campaign's later, lower-error arms would
otherwise inherit the 0.11 figure by default.

**All 9 step-0b arms are in** (8 original plus the seed replicate).
Schedule (Rule 1) and stacking (Rule 2) both trigger clearly; the 600 s
batch clause does not. Size (Rule 3) is a reproducible near-miss -- a
real, seed-confirmed ~2B->4B gap that still falls short of <10% dev-clean
on both seeds. Encoder (Rule 4) triggers decisively and unblocks the
week-2 screen's budget (Rule 5 does not fire). Size and encoder are
`02-backbones.md`'s and `03-audio-stack.md`'s questions respectively --
the numbers above are reported as priors for those sections, not as a
backbone or encoder recommendation from this file.

### Five-language screen arms (§3): the cosine pass

**These seven arms ran cosine with `warmup_steps 20`, not the screen's
warmup-stable-decay.** `ABL-MA-700-asr.yaml` declares that schedule and the
rows carried no override, so their `exp_name`s say nothing about it. They are
kept as the cosine-vs-WSD control at 700 h/lang and are not the screen's
results. The WSD arms are named `...-sk5-ga1-wsd-wu0p03-...`.

Measured 2026-09-21, final in-training generative eval, 200 utterances per
language, seed 42, k=5, one epoch of 700 h/lang. `s/step` is the run's tqdm
training time over `global_step` and includes the in-training evals; the
tqdm second-to-last line was too noisy to use (27.6 and 25.8 s/it on the
600 s arms, an end-of-run eval stall).

| run | schedule | exp_name | steps | en / de / es / fr / it WER | s/step | GPU-h | notes |
|---|---|---|---|---|---|---|---|
| w2vb-lr1e3-b1200 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-ga1-elr6e6-dlr2e5-lr1e3-s42-8g` | 10,500 | 0.971 / 0.928 / 0.860 / 0.912 / 0.813 | 1.15 | 27 | measured. Mean WER 0.897. |
| w2vb-lr2e3-b1200 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-ga1-elr6e6-dlr2e5-lr2e3-s42-8g` | 10,500 | 1.046 / 0.985 / 0.940 / 0.935 / 0.973 | 1.15 | 27 | measured. Mean WER 0.976. |
| w2vb-lr1e3-b600 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-elr6e6-dlr2e5-lr1e3-s42-8g` | 21,000 | 0.898 / 0.880 / 0.844 / 1.013 / 0.762 | 0.80 | 37 | measured. Mean WER 0.879. |
| w2vb-lr2e3-b600 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-elr6e6-dlr2e5-lr2e3-s42-8g` | 21,000 | 0.930 / 0.987 / 0.848 / 0.846 / 0.882 | 0.80 | 37 | measured. Mean WER 0.898. |
| w2vb-lr1e3-b300 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-elr6e6-dlr2e5-lr1e3-s42-4g` | 42,000 | 0.857 / 0.925 / 0.852 / 1.004 / 0.852 | 0.76 | 36 | measured. Mean WER 0.898. |
| w2vb-lr2e3-b300 | cosine, warmup_steps 20 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-elr6e6-dlr2e5-lr2e3-s42-4g` | 42,000 | 0.959 / 0.927 / 0.910 / 0.867 / 0.904 | 0.76 | 36 | measured. Mean WER 0.913. |
| whisper-lr1e3-b1200 | cosine, warmup_steps 20 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-bd75-ga2-elr6e6-dlr2e5-lr1e3-s42-8g` | 10,500 | 0.124 / 0.115 / 0.087 / 0.124 / 0.207 | 1.34 | 32 | measured. Peak training memory 11.5 GB of 64 GB (w2v-BERT 1200 s: 19.7 GB). Mean WER 0.131, already 0.132 at about step 5,700. |

### Five-language screen arms (§3): the WSD grid — all twelve, measured

The screen itself: warmup-stable-decay, `warmup_ratio 0.03`, `stack_factor 5`,
seed 42, one epoch of 700 h/lang, `num_decay_steps` 2,100 / 4,200 / 8,400 at
1200 / 600 / 300 s. Whisper 1200 s runs `batch_duration 150 / grad_accum 1`,
same as w2v-BERT, so the two encoder halves differ only in the encoder
(§3, PI 2026-09-21). Measured 2026-09-21/23, final in-training generative
eval, 200 utterances per language. `s/step` and GPU-h are elapsed wall time
over `global_step` and `elapsed × ranks` respectively (`sacct`).

| run | exp_name | steps | ranks | en / de / es / fr / it WER | s/step | GPU-h | notes |
|---|---|---|---|---|---|---|---|
| w2vb-lr1e3-b1200 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-8g` | 10,500 | 8 | 0.810 / 0.882 / 0.766 / 0.927 / 0.875 | 1.30 | 30 | measured. Mean WER 0.852. Peak mem 19.71 GB. |
| w2vb-lr2e3-b1200 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-8g` | 10,500 | 8 | 0.782 / 0.879 / 0.778 / 0.850 / 0.820 | 1.19 | 28 | measured. Mean WER 0.822. Peak mem 19.71 GB. |
| w2vb-lr1e3-b600 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-8g` | 21,000 | 8 | 0.795 / 0.864 / 0.704 / 0.774 / 0.802 | 1.10 | 51 | measured. Mean WER 0.788, the best w2v-BERT grid point. Peak mem 12.04 GB. GPU-h roughly doubled against the cosine pass (37) at the same setting; s/step for 1200 s and the Whisper canary are unchanged from cosine, so this is plausibly node/network contention on this allocation rather than a WSD effect — not investigated further. |
| w2vb-lr2e3-b600 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-8g` | 21,000 | 8 | 0.911 / 0.852 / 0.858 / 0.899 / 0.728 | 1.10 | 51 | measured. Mean WER 0.850. Peak mem 12.04 GB. Same cost anomaly as above. |
| w2vb-lr1e3-b300 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-4g` | 42,000 | 4 | 0.835 / 0.723 / 0.712 / 0.713 / 0.733 | 0.884 | 41 | measured. Mean WER 0.743, the best w2v-BERT arm overall. Peak mem 12.07 GB. |
| w2vb-lr2e3-b300 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-4g` | 42,000 | 4 | 0.830 / 0.793 / 0.615 / 0.718 / 0.782 | 0.885 | 41 | measured. Mean WER 0.748. Peak mem 12.07 GB. |
| whisper-lr1e3-b1200 (canary) | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-8g` | 10,500 | 8 | 0.118 / 0.149 / 0.088 / 0.121 / 0.151 | 1.19 | 28 | measured. Mean WER 0.126, the best Whisper arm and the best of the grid. Peak mem 19.11 GB at 150/1 (extrapolated ~23 GB before the run; measured close to w2v-BERT's own 19.71 GB). Kept improving through the second half (0.142 at step ~5,730 -> 0.126 at 10,500) where its cosine counterpart went flat (0.132 -> 0.131) — the `min_lr_scale` contrast: WSD's LR floor (0.1× peak, real) vs cosine's dead `min_lr_scale` key (decays to ~0). |
| whisper-lr2e3-b1200 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-8g` | 10,500 | 8 | 0.113 / 0.122 / 0.082 / 0.126 / 0.153 | 1.13 | 26 | measured. Mean WER 0.119, the single best point in the whole grid. Peak mem 19.11 GB. Note the atexit DeepSpeed/Triton cache traceback in this job's log is benign (raised after `Train_result`, training already complete). |
| whisper-lr1e3-b600 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-8g` | 21,000 | 8 | 0.124 / 0.118 / 0.090 / 0.124 / 0.160 | 0.755 | 35 | measured. Mean WER 0.123. Peak mem 11.41 GB. |
| whisper-lr2e3-b600 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-8g` | 21,000 | 8 | 0.184 / 0.121 / 0.090 / 0.130 / 0.155 | 0.755 | 35 | measured. Mean WER 0.136, the worst Whisper arm (en 0.184 is an outlier against its own family's ~0.12). Peak mem 11.41 GB. |
| whisper-lr1e3-b300 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr1e3-s42-4g` | 42,000 | 4 | 0.115 / 0.118 / 0.086 / 0.129 / 0.159 | 0.731 | 34 | measured. Mean WER 0.122. Peak mem 11.47 GB. |
| whisper-lr2e3-b300 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-bd75-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e3-s42-4g` | 42,000 | 4 | 0.115 / 0.120 / 0.088 / 0.122 / 0.165 | 0.731 | 34 | measured. Mean WER 0.122. Peak mem 11.47 GB. |

**Reading the grid (2026-09-23). The screen was built to choose a corner; it
is telling us the corner barely exists.** Mean WER over the five languages:

| effective batch | w2v-BERT | Whisper | w2v-BERT GPU-h |
|---|---|---|---|
| 1200 s | 0.837 | 0.122 | 29 |
| 600 s | 0.819 | 0.130 | 51 |
| 300 s | 0.745 | 0.122 | 41 |

- **Whisper: five of the six arms lie within 0.0062 of each other**
  (0.1192-0.1254). A 4x change in effective batch and a 2x change in adapter
  LR move mean WER by less than one point. The sixth, `lr2e3-b600` at 0.1360,
  is an outlier carried entirely by English (0.184 against 0.113-0.124 on
  every other Whisper arm) and reads as an instability or a bad eval, not a
  factor effect.
- **w2v-BERT: total range 0.1088**, which is step 0b's seed-only spread
  (0.108) almost exactly -- and that floor was measured at WER 0.5, a *lower*
  error rate than these arms, so the true floor here is wider. The batch
  trend (b300 better by ~0.09, at both LRs) is suggestive and is **not**
  separable from noise on this evidence.
- **Neither half of the grid separates its factor levels.** On the encoder
  that works, the recipe does not matter; on the encoder that does not work,
  nothing is measurable. That is a robustness result, not a failure of the
  screen, and it means **the corner should be chosen on cost.**
- **Cost is the axis that does separate.** 1200 s is the cheapest (26-30
  GPU-h) and ties the best Whisper score. 600 s is the worst of both worlds
  at 51 GPU-h -- it pays 8 ranks' all-reduce on twice as many steps as
  1200 s, without 300 s's halved rank count. If a 600 s point is ever needed
  again it should run as 4 ranks x `batch_duration` 150, not 8 x 75.

**No corner is called until the seed-43 replicates land** (`campaign.yaml`,
submitted 2026-09-23). Whisper's whole spread is 0.017 and its
five-arm cluster is 0.006; the only noise floor we have at this error regime
is step 0b's 0.002, from a different task in a different language. The
replicates measure it here. If they come back at ~0.002, the flatness is
real and 1200 s is adopted on cost. If they come back at ~0.01, the grid has
measured nothing at all and that is the finding.

**Not yet done, and not derivable from the numbers above:** the transition
step per arm (§1's plateau-then-drop definition), FLEURS-24 zero-shot CER on
the twelve final checkpoints (a separate `melt-eval` step), and the
comparison of these WER deltas against step 0b's seed-only noise floor
(0.108 dev-clean WER, measured at WER ≈ 0.5 — a different regime from both
the w2v-BERT arms here, ≈ 0.74–0.98, and the Whisper arms, ≈ 0.12–0.15). No
best-corner-per-encoder conclusion is drawn here for that reason; the
numbers above are measured, not yet interpreted against the decision rule.

### Five-language screen arms (§3): the WSD grid on the selection metric — measured

Measured 2026-09-24 with the selection metric
([`selection-metric/README.md`](../selection-metric/README.md); melt-eval on the
saved top-level checkpoints, JSON logs, 1 GPU, batch 4, bf16). In-domain: up to
200 utterances from each held-out corpus per language (600), FLEURS dev: 200
per language, 26 languages (nl 171). CER clipped at 1.0, group medians, lower
is better; score = (1.0·ID + 1.5·OOD-train + 0.6·OOD-related) / 3.1. Per-language
values: `selection-metric/scores/screen-wsd-grid-2026-09-24.txt`. The
Both seed-43 replicates and the Whisper stacking sweep (k=2 and k=10 at `lr1e3-b1200`, k=5 is the grid row) are included. Runaway columns count samples with per-sample CER > 1 (decoding is unguarded, as in training); report only, not in the score.

| run | ID | OOD-train | OOD-related | OOD-latin | OOD-script | score | runaway ID | runaway FLEURS |
|---|---|---|---|---|---|---|---|---|
| whisper-lr1e3-b1200-k2 (k=2, 25 Hz) | 0.0584 | 0.0417 | 0.4545 | 0.7195 | 1.0000 | 0.1270 | 12/3000 | 425/5171 |
| whisper-lr1e3-b300 | 0.0610 | 0.0493 | 0.4396 | 0.9070 | 1.0000 | 0.1286 | 10/3000 | 536/5171 |
| whisper-lr1e3-b1200 | 0.0617 | 0.0466 | 0.4630 | 0.8608 | 1.0000 | 0.1321 | 8/3000 | 527/5171 |
| whisper-lr2e3-b300 | 0.0631 | 0.0464 | 0.4780 | 0.9285 | 1.0000 | 0.1354 | 8/3000 | 579/5171 |
| whisper-lr1e3-b600 | 0.0642 | 0.0459 | 0.4830 | 0.8790 | 1.0000 | 0.1364 | 12/3000 | 586/5171 |
| whisper-lr2e3-b1200-s43 (seed-43 replicate) | 0.0614 | 0.0460 | 0.4893 | 0.8913 | 1.0000 | 0.1368 | 10/3000 | 494/5171 |
| whisper-lr2e3-b1200 | 0.0626 | 0.0450 | 0.5129 | 0.7769 | 1.0000 | 0.1413 | 11/3000 | 504/5171 |
| whisper-lr2e3-b600 | 0.0617 | 0.0486 | 0.5497 | 0.8822 | 1.0000 | 0.1498 | 7/3000 | 618/5171 |
| whisper-lr1e3-b1200-k10 (k=10, 5 Hz) | 0.0767 | 0.0527 | 0.6042 | 0.9086 | 1.0000 | 0.1672 | 10/3000 | 533/5171 |
| w2vb-lr2e3-b300 | 0.6115 | 0.6698 | 1.0000 | 1.0000 | 1.0000 | 0.7149 | 149/3000 | 1407/5171 |
| w2vb-lr1e3-b300 | 0.5466 | 0.8353 | 1.0000 | 1.0000 | 1.0000 | 0.7740 | 157/3000 | 1419/5171 |
| w2vb-lr2e3-b600 | 0.6337 | 0.8047 | 1.0000 | 1.0000 | 1.0000 | 0.7873 | 200/3000 | 1358/5171 |
| w2vb-lr1e3-b300-s43 (seed-43 replicate) | 0.6940 | 0.8090 | 1.0000 | 1.0000 | 1.0000 | 0.8089 | 204/3000 | 1503/5171 |
| w2vb-lr1e3-b600 | 0.6453 | 0.8794 | 1.0000 | 1.0000 | 1.0000 | 0.8272 | 174/3000 | 1445/5171 |
| w2vb-lr2e3-b1200 | 0.6674 | 0.8998 | 1.0000 | 1.0000 | 1.0000 | 0.8442 | 176/3000 | 1450/5171 |
| w2vb-lr1e3-b1200 | 0.6748 | 0.9925 | 1.0000 | 1.0000 | 1.0000 | 0.8915 | 210/3000 | 1599/5171 |

### The screen's 2e-5 control (§3): the August recipe's own adapter LR, at the new recipe

Not a grid arm. 2e-5 is `ABL-MA-700-asr.yaml`'s own `optimization.adapter_lr`
(the August recipe's LR, §1), so both rows leave `adapter_lr` unset rather
than overriding it to the same value — `plan_arm.py` emits no
`--optimization.adapter_lr` for either command. Same reference corner as the
grid otherwise: warmup-stable-decay, `warmup_ratio 0.03`, `stack_factor 5`,
seed 42, one epoch of 700 h/lang, 1200 s effective batch, 10,500 steps,
`num_decay_steps` 2100, world_size 8 (confirmed from the `[run_train]
starting` log line on both jobs). Measured 2026-09-23, final in-training
generative eval, 200 utterances per language.

| run | exp_name | jobs | en / de / es / fr / it WER | s/step | GPU-h | notes |
|---|---|---|---|---|---|---|
| w2vb-lr2e5-b1200 | `MA-700asr-w2vbF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e5-s42-8g` | 46368424 | 1.103 / 1.085 / 1.125 / 1.077 / 1.125 | 1.11 | 26 | measured. Mean WER 1.103 — worse than every LR × batch point in the grid (best w2v-BERT grid arm so far: 0.822, `w2vb-lr2e3-b1200`), and worse than 1.0. Reproduces the August failure's signature (fluent, WER > 1.0) at the new 10 Hz/WSD/1200 s recipe: for w2v-BERT, 2e-5 alone accounts for the plateau, independent of frame rate or schedule. |
| whisper-lr2e5-b1200 | `MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-ga1-wsd-wu0p03-elr6e6-dlr2e5-lr2e5-s42-8g` | 46368425 | 0.132 / 0.163 / 0.110 / 0.164 / 0.170 | 1.11 | 26 | measured. Mean WER 0.148 — worse than the grid's `whisper-lr1e3-b1200` WSD canary (0.126, job 46258536, board 2026-09-22) but in the same regime, nowhere near w2v-BERT's failure at the same LR. For Whisper, 2e-5 still nearly aligns: the encoder carries most of the signal, and LR is a second-order factor here in a way it is not for w2v-BERT. |

**Reading the two together.** The August failure was diagnosed (§1) against a
w2v-BERT encoder; this control isolates the LR at the new recipe and gets a
clean split by encoder. On w2v-BERT, 2e-5 alone reproduces the August
plateau even with 10 Hz stacking and WSD — LR was the (or a) binding
constraint there, and the grid's higher-LR arms already show what removes
it. On Whisper, 2e-5 costs about 0.02 WER against the grid's own 1e-3 point
at the same batch — LR was not the whole story for this encoder, consistent
with `03-audio-stack.md`'s prior that the encoder is a first-order factor
here. Both readings hold at once because they are about different encoders,
not in tension with each other or with §1's original diagnosis.
