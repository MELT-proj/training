# 04 — Training regime: an IFT-level half fraction

**Settled (2026-09-15):** four two-level factors, eight IFT runs, on the
provisional backbone from the confirmed MA checkpoint; "adapter frozen at
IFT" is treated as an ablation level, not the default, pending this result.
**Open:** the exact levels for "MA epochs" (1 vs 3 is the default proposal).
**Owner:** PI decides the regime at the week-5 gate.

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
