# 04 — Training regime: an IFT-level half fraction, and the MA:IFT ratio

**Settled (2026-09-15/16):** four two-level factors, eight IFT runs, on the
provisional backbone from the confirmed MA checkpoint; "adapter frozen at
IFT" is treated as an ablation level, not the default, pending this result;
the decoder-frozen regime (only the adapter trains at IFT, under instruction
templates) joins the comparison as two runs outside the fraction; the
MA:IFT hours ratio is a named study (§6), run after the regime decision,
splitting ASR hours only while ST hours stay fixed.
**Open:** the exact levels for "MA epochs" (1 vs 3 is the default proposal);
the ratio points if the decoder-frozen regime wins (then the single-stage
comparison becomes the headline).
**Owner:** PI decides the regime at the week-5 gate and reads the ratio
sweep before the Fondue freeze.

## 1. Why a fraction here and not at the MA stage

An IFT run costs 80–300 GPU-h and one to two days of wall clock. Sixteen
runs would take a month of queue; eight take a week. At the MA stage the
same factors are cheap enough for the full grid, which is why
`01-interface-recipe.md` and `03-audio-stack.md` do not use fractions.

## 2. Factors

| factor | low (−) | high (+) | why it matters |
|---|---|---|---|
| A: adapter during IFT | frozen | trainable | freezing a weak adapter and training only the decoder cannot repair the interface; most published recipes keep the projector trainable |
| B: decoder update | full fine-tune | LoRA (r=16, alpha 32) | text-ability retention and cost |
| C: MA data | ASR only | ASR + ST | whether translation supervision during alignment helps or hurts |
| D: MA length | 1 epoch | 3 epochs | strong vs light alignment |

## 3. The design

A 2^(4−1) half fraction with D = ABC. Each main effect is aliased only with a
three-factor interaction (assumed negligible); two-factor interactions are
aliased in pairs (AB with CD, AC with BD, AD with BC).

| run | A adapter@IFT | B decoder | C MA data | D MA epochs |
|---|---|---|---|---|
| R1 | frozen | full | ASR | 1 |
| R2 | trainable | full | ASR | 3 |
| R3 | frozen | LoRA | ASR | 3 |
| R4 | trainable | LoRA | ASR | 1 |
| R5 | frozen | full | ASR+ST | 3 |
| R6 | trainable | full | ASR+ST | 1 |
| R7 | frozen | LoRA | ASR+ST | 1 |
| R8 | trainable | LoRA | ASR+ST | 3 |

Reading: the main effect of A is the mean of the four "trainable" runs minus
the mean of the four "frozen" runs; the other factors are balanced within
both halves. Factors C and D need their own MA checkpoints (ASR vs ASR+ST,
1 vs 3 epochs): four MA runs, cheap.

**The third regime, outside the fraction.** Factor B compares full
fine-tuning with LoRA. The decoder-frozen regime, in which IFT trains only
the adapter under the instruction templates, is a different animal: the
decoder never changes, so text ability is preserved by construction, IFT
costs the same per hour as MA rather than several times more, and ST
depends entirely on the frozen decoder translating from adapter embeddings.
It is added as two runs:

| run | adapter@IFT | decoder | MA data | MA epochs | compare with |
|---|---|---|---|---|---|
| R9 | trainable | frozen | ASR | 1 | R4 (LoRA, same A/C/D) |
| R10 | trainable | frozen | ASR+ST | 1 | R6 (full, same A/C/D) |

Adapter-trainable is the only sensible level for this regime, so it does
not enter the fraction's aliasing; R9/R10 are read as direct pairs.

## 4. Metrics and decision rule

In-domain CER and FLEURS-24 CER after IFT, X→en chrF/COMET, text-ability
retention (a small text benchmark or held-out text perplexity before vs
after IFT), and GPU-h. The regime for Fondue is the combination whose main
effects are favourable and whose aliased interactions are not large; if an
interaction pair is large, the two candidate explanations are separated by
one extra run.

## 5. Results

| run | CER in-domain | CER FLEURS-24 | chrF X→en | text retention | GPU-h |
|---|---|---|---|---|---|
| R1 | | | | | |
| … | | | | | |
| R9 | | | | | |
| R10 | | | | | |

## 6. The MA:IFT hours ratio (added 2026-09-16)

Every arm so far uses 700 h per language of ASR in MA and 700 h per
language of ASR plus ST in IFT. Nobody chose that split; it fell out of the
data design. What the split *means* changes with the regime, which is why
this study runs after the week-5 decision rather than beside it:

| regime at IFT | what the split means | expectation, to be tested |
|---|---|---|
| decoder frozen, adapter trainable | both stages train the same 6.3M parameters at the same cost per hour; the split is a pure curriculum question, and a single stage with instructions from the start is the honest comparison | the split matters little; single-stage may match two-stage; ST is capacity-limited by the adapter |
| full fine-tune | IFT can repair a weak interface but costs several times more per hour and erodes text ability; MA front-loads the cheap work | a flat optimum somewhere around 20–50% of the ASR hours in MA; the no-MA point is the baseline that matters |
| LoRA | limited repair capacity, less forgetting | between the two |

Factor A changes the meaning again: with the adapter trainable at IFT, MA
is a warm start rather than the adapter's only training, and the optimum
shifts towards less MA.

**Design rules.**

- **Split ASR hours only; hold ST fixed.** Only IFT carries ST, so at a
  fixed total a larger MA share would silently cut ST exposure and confound
  the effect with a data-quantity change. The sweep keeps the campaign's ST
  hours in IFT (700 h per direction) and splits a fixed 700 h per language
  of ASR between the two stages.
- **Separate MA runs per budget, never intermediate checkpoints.** A
  checkpoint at 100 h of a 700 h cosine run has not decayed its learning
  rate and is not a 100 h run. MA is cheap; render and run each budget with
  its own full schedule.
- **One seed per point**, two at the campaign split (the week-3
  confirmation run and its replicate already provide them).
- **Metrics:** in-domain CER, FLEURS-24 CER, X→en chrF/COMET (with ST
  exposure fixed, an ST change reflects alignment quality rather than
  exposure), text retention, and GPU-h. Read the same table two ways: CER
  against MA share at fixed hours (the data-limited view, which is the
  ladder's tail languages) and CER against GPU-h using each stage's
  measured per-hour cost under that regime (the compute-limited view, which
  is Fondue).

**Points, under the winning regime.** MA ASR hours per language against IFT
ASR hours per language, with IFT ST fixed at 700 h per direction:

| point | MA | IFT ASR | note |
|---|---|---|---|
| P0 | 0 | 700 | no MA: the adapter starts from random init inside IFT (adapter trainable). Under the decoder-frozen regime this *is* the single-stage model |
| P1 | 100 | 600 | |
| P3 | 300 | 400 | |
| P7 | 700 | 0 | ST-only IFT: ASR then has to survive as instruction-following with no ASR instruction data |
| campaign | 700 | 700 | the existing arm; no trade-off, the reference point |
| zero-IFT probe | 700 | none | evaluation only, instruct backbones: score the MA checkpoint with the IFT prompts for ASR and ST |

Then P0 and P3 again under the runner-up regime as the interaction check.
If the two curves have the same shape, the split is a property of the
pipeline and the paper reports one number; if not, it reports one per
regime, which is itself the more interesting result.

**What it decides.** Fondue's "MA data budget and stage split" row in
`06-fondue.md` §3: whether MA runs over the full ASR pool, a subset, or (if
the decoder-frozen regime wins and P0 matches the campaign point) not at
all. Default if the sweep misses the 2026-10-25 freeze: full MA, as today.

**Cost.** MA at 100 and 300 h/lang is about 5 and 15 GPU-h. The IFT side
scales with the measured per-arm cost of the winning backbone: Llama-1B is
80–100 GPU-h per arm, Qwen3.5-2B is **306 GPU-h** measured on a full
production arm (job noted in `arms.tsv`; see `06-fondue.md` §2). So the
four-point sweep plus two runner-up points is roughly 450 GPU-h for a
Llama-class winner and roughly 1,800 for a Qwen-class one. Both fit, but a
Qwen winner makes this the most expensive study outside Fondue and the
runner-up points become the first thing to cut.

**Needs:** per-task budgets in `build_campaign_config.py`, so an IFT render
can carry ASR at 600/400/0 h per language while ST stays at 700. Small
change, scheduled for week 5 Track B.

### 6.1 Results

| point | regime | CER in-domain | CER FLEURS-24 | chrF X→en | text retention | GPU-h |
|---|---|---|---|---|---|---|
| P0 | | | | | | |
| P1 | | | | | | |
| P3 | | | | | | |
| P7 | | | | | | |
| campaign | | | | | | |
| zero-IFT probe | | | | | | |
| P0 runner-up | | | | | | |
| P3 runner-up | | | | | | |
