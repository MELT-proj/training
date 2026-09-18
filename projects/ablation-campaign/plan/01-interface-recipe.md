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

## 2b. Step 0b — schedule, stacking, encoder and decoder size (added 2026-09-17)

Step 0 was inconclusive (results in §5). All five one-epoch runs stayed above
WER 1.0. A post-hoc three-epoch rerun of L4 broke the plateau, reaching
dev-clean WER 0.62 and loss 0.90, still far from the success bar. At the same
step count (end of epoch 1, about 2,900 steps) its loss was 1.51 against
L4's 2.63, and the only difference at that point was a learning rate that had
decayed less. The PI proposed three checks before the week-2 screen: a more
aggressive schedule, stacking to 10 Hz as SLAM-ASR does, and a larger
decoder. Step 0b runs them together, plus a seed replicate and an encoder
control, because each question can mask the others.

**A dead config key found while designing this.** `optimization.min_lr_scale:
0.1` appears in every campaign and SFT config, but no code reads it
(`git grep` on main, 2026-09-17). Every cosine run so far decayed to zero,
not to 10% of the peak. It does not invalidate comparisons between arms,
which all shared it, but the configs misstate what ran. Step 0b sets its
floor through `lr_scheduler_kwargs` instead. Wiring or deleting the key is a
PI decision.

### Common settings

`ABL-MA-librispeech.yaml`; w2v-BERT 2.0 frozen; decoder frozen; MLP adapter;
audio-only prompt; adapter LR 1e-3; effective batch 1200 s; warmup 3%;
**three epochs** of LibriSpeech (about 2,900 audio hours and 8,650 optimizer
steps at 1200 s), the same audio budget as the existing
`MA-librispeech-l4-ep3`, which is the reference arm A0 and needs no rerun.
In-training eval on dev-clean and dev-other at **500 utterances per set**
(200 made the epoch-3 trajectory non-monotonic), about 20 rounds per run.

### Arms

| arm | differs from R in | question | GPU-h, extrapolated |
|---|---|---|---|
| A0 | cosine decay to zero, seed 44: the existing `MA-librispeech-l4-ep3` | schedule reference | done |
| R | warmup-stable-decay: warmup 3%, constant, cosine decay over the last 20% to 0.1× peak | schedule shape, against A0 | ~20 |
| R-seed | seed 43 | noise on hours-to-threshold | ~20 |
| R-lr2e3 | peak LR 2e-3 | learning rate | ~20 |
| R-b600 | effective batch 600 s: one node × 4 GPUs, `batch_duration 150`, `grad_accum 1` | twice the optimizer steps at the same GPU-h, twice the wall clock | ~20 |
| R-k5 | `stack_factor 5`, 50 Hz to 10 Hz | stacking | ~12 |
| W | Whisper-large-v3 encoder, stack 1 | supervised-ASR encoder against self-supervised w2v-BERT | ~35 |
| Q2-k5 | Qwen3.5-2B (instruct) decoder, stack 5 | size control, same family | ~40 |
| Q4-k5 | Qwen3.5-4B (instruct) decoder, stack 5 | decoder size | ~80 |

About 250 GPU-h in total.

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

1. **Schedule.** Warmup-stable-decay becomes the MA default if R beats A0 at
   equal steps by more than the R/R-seed spread. LR 2e-3 is adopted if it
   reaches the threshold in fewer audio hours without loss spikes. The 600 s
   batch is adopted if it reaches the threshold in fewer audio hours.
2. **Stacking.** Stack 5 becomes the default if R-k5 is within noise of R or
   better, since it cuts decoder positions five-fold. If clearly worse, the
   screen tests stack 2 and 4.
3. **Size.** If Q4-k5 reaches the threshold and Q2-k5 does not, or does so in
   markedly fewer hours, the paper's "2–3B is enough" framing must be tested
   and a ~4B point joins the backbone grid. It does not change the screen's
   backbone by itself.
4. **Encoder.** If W reaches the threshold much earlier than R, the encoder
   question moves ahead of the backbone grid.
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
0b (§2b) before launch.*

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
(§2b). Fill in as each lands. In-training eval, `max_samples 500`/set, the
metrics from `melt/training/metrics.py`'s update (full-set S/D/I/length
ratio/runaway fraction, not the old 200-sample log).

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
| R-seed | `MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s46-8g` | 0.441 | 0.666 | 0.750 | 1.012 | 1.4% / 1.4% | job 46059832 (resumed from 45985924), seed 46 -- the noise check against `R` (seed 45), still in progress. Worse than R-k5, as expected (`stack_factor 1`, same as A0/L4). |
