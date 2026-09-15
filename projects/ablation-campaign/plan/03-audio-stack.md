# 03 — Audio stack: encoders × adapters, and the cost axis

**Settled (2026-09-15):** this is a paper section; the full crossing runs at
MA-stage cost on the provisional backbone; every stack is configured to the
same decoder frame rate; the Q-Former stays in and the PI fixes it in week 4;
the efficiency figure is CER against decoder positions per audio second.
**Open:** which frame rate the crossing standardises on (12.5 Hz is the
default; the screen in `01-interface-recipe.md` may move it); whether Whisper
runs as encoder-only or with its own conv frontend.
**Owner:** PI.

## 1. What is wired

| encoder | params | input | native rate | notes |
|---|---|---|---|---|
| `facebook/w2v-bert-2.0` | 580M | precomputed features | 50 Hz | the baseline; no flash attention (relative-position bias) |
| `facebook/mms-1b` | 962M | raw waveform | 50 Hz | flash-eligible; matches w2v-BERT step time with flash, 1.54× slower with sdpa; ships `apply_spec_augment: true`, which must be set to match the others |
| Whisper-large-v3 encoder | ~635M | log-Mel | 50 Hz after conv | supervised ASR pretraining, the odd one out |
| mHuBERT-147 | 95M | raw waveform | 50 Hz | the small one; 147 languages |

| adapter | params (w2v-BERT → 2048-wide decoder) | output rate | state |
|---|---|---|---|
| MLP (2-layer, GELU, LayerNorm × gain) | 6.30M | 50 Hz, or 50/k with `stack_factor` k | baseline |
| Conformer, 1 layer | 27.28M | 25 Hz (stride 2) | ready |
| MoE, 8 SwiGLU experts, top-2, load-balancing aux loss | 33.57M total, 8.39M active | 50 Hz | on a branch; used by the running 10-epoch verbatim arm |
| Q-Former, window 15, 3 queries | not instantiable today | 10 Hz by design | broken; PI fixes in week 4 |

## 2. The crossing

Four encoders × four adapters = 16 MA arms on the provisional backbone
(Llama-3.2-1B-Instruct) with the recipe from `01-interface-recipe.md`, five
languages at 700 h, ASR-only, one epoch. ≈ 30 GPU-h each, all in the queue
at once in week 4. Scored on MA-stage generative WER/CER in-domain and
FLEURS-24 zero-shot CER from melt-eval.

Then IFT for the top three or four stacks (week 5), and one confirmation of
the winning stack on the winning backbone (week 6). The encoder × backbone
interaction is assumed small and is stated as an assumption.

## 3. Confounds, and how each is handled

| confound | handling |
|---|---|
| frame rate differs by adapter (MLP 50, Conformer 25, Q-Former 10) | configure every stack to the same output rate: MLP with `stack_factor`, Conformer with stride, Q-Former with window/queries. The frame-rate effect itself comes from the screen |
| adapter parameter count (6M / 27M / 34M) | report; add an MLP with a wider hidden layer if capacity needs isolating |
| supervised vs self-supervised encoder pretraining (Whisper vs the rest) | report; it is part of what the section measures |
| spec-augment default differs (MMS on, others off) | pin it explicitly in every arm |
| waveform vs feature input, and encoder compute | goes into the cost axis, not hidden |
| MoE aux-loss weight and router LR | the MoE's router barely moves at 2e-5; the crossing runs it at the recipe LR, and its aux loss is logged |

The running 10-epoch MoE arm at 2e-5 does not count as MoE evidence: it
changes adapter, training length and prompt at once.

## 4. The cost axis and the figure

Per stack, from `resolved_config.json` and SLURM accounting:

- encoder + adapter parameters (and active parameters for the MoE);
- decoder positions per audio second (the frame rate after the adapter);
- training GPU-h per 1,000 audio hours for MA and for IFT;
- eval throughput in utterances per second at batch 16 sorted by duration.

The figure that sells the pipeline: CER (in-domain mean, and FLEURS-24 mean
as a second panel) against decoder positions per audio second, one point per
stack, size of the marker by encoder+adapter parameters. Stacking is the
lever that moves along the x-axis.

## 5. Results

| encoder | adapter | rate | MA-stage CER (5-lang) | FLEURS-24 CER | GPU-h / 1K h | notes |
|---|---|---|---|---|---|---|
| | | | | | | |
