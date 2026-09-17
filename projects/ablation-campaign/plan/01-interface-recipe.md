# 01 — The interface recipe: step 0 on LibriSpeech and the five-language screen

**Settled (2026-09-14/15):** the August MA baseline is a failed alignment and
is not a lower bound; recipe work precedes every comparison; MA-stage
generative WER is the selection metric for MA-stage hyperparameters; the
`stack_factor` option is the one code change scheduled before the screen.
**Open:** the recipe values themselves (decided by the runs below).
**Owner:** PI decides the recipe at the week-2 gate.

## 1. Diagnosis

The MA arms run in August (five languages, 700 h each, w2v-BERT 2.0 frozen,
MLP adapter trainable, decoder frozen, adapter LR 2e-5, effective batch
4800 s, one epoch ≈ 2,600 optimizer steps) ended with:

- eval loss 2.6–3.1 nats per token, roughly what a 1B text LM scores on
  lowercase unpunctuated transcripts with no audio;
- generative WER 1.10–1.16, insertions outnumbering reference words;
- hypotheses fluent and in-domain but unrelated to the audio, sometimes in
  the wrong language.

The adapter learned "emit fluent text in the right register" and stopped.
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
   with the same chat prompt and no audio (or shuffled audio). If MA loss
   equals the floor, audio was ignored.
2. **The text prior per language** (`02-backbones.md` §3), which also gives
   the ST upper bound through the cascade oracle.

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
