# 03 — Audio stack: encoders × adapters, and the cost axis

**Settled (2026-09-15):** this is a paper section; the full crossing runs at
MA-stage cost on the provisional backbone; every stack is configured to the
same decoder frame rate; the Q-Former stays in and the PI fixes it in week 4;
the efficiency figure is CER against decoder positions per audio second.
**Open:** which frame rate the crossing standardises on (12.5 Hz is the
default; the screen in `01-interface-recipe.md` may move it); whether Whisper
runs as encoder-only or with its own conv frontend.
**Owner:** PI.
**Order (2026-09-18):** this section now runs **before** `02-backbones.md`.
The reason is in §0.

## 0. What step 0b handed over, as a prior

Step 0b ran a Whisper arm as a diagnostic control, not as an encoder
decision (`01-interface-recipe.md` §2b). On LibriSpeech, MA stage, frozen
Llama-3.2-1B-Instruct, identical recipe and step count:

| encoder | dev-clean WER | dev-other WER |
|---|---|---|
| Whisper-large-v3, 50 Hz | 0.038 | 0.061 |
| w2v-BERT 2.0, 10 Hz (best self-supervised arm) | 0.314 | 0.490 |
| w2v-BERT 2.0, 50 Hz | 0.549 | 0.664 |

**This is a hypothesis for the crossing to test, not a result it may
assume.** One dataset, English only, one seed per arm, and Whisper is
supervised on exactly this kind of read speech. What it does establish is
that the pipeline transcribes when the representation is easy, so a null
result here is about encoders and not about the recipe.

Two consequences for how the crossing is run:

- **The recipe is fixed by `01` and identical across all sixteen arms.** It
  is not re-tuned per encoder. Tuning around a provisional winner and then
  comparing against it is how a crossing produces the answer it started with.
- **The order changed.** The backbone grid compares six decoders on one
  audio stack; if the encoder is worth an order of magnitude and the stack
  is the wrong one, every backbone sits against the same encoder-imposed
  floor and their differences compress into noise. So the stack is chosen
  first, then `02` runs on it. The encoder × backbone interaction is still
  assumed small, and that assumption is now doing less work than before.

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
at once in **week 3**. Scored on MA-stage generative WER/CER in-domain and
FLEURS-24 zero-shot CER from melt-eval.

**The encoder is read on FLEURS-24, not on the in-domain five.** The five
training languages are all high-resource, which is where a supervised
encoder's coverage is best; the question the campaign actually needs
answered is what happens on the low-resource end of the EU-24. Report the
FLEURS-24 CER split into a high-resource and a low-resource half, and treat
a stack that wins in-domain while losing the low-resource half as unproven,
not as the winner.

Then IFT for the top three or four stacks (week 4), and one confirmation of
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
| Whisper's fixed 30 s input window | every utterance costs a full window of encoder compute whatever its duration (`encoder_specs.py` `window_frames: 3000`; `processing_melt.py::_extract_windowed` pads the tail waveform). The *decoder* side is unaffected — the mask keeps only real frames, so positions per audio second stay 50 Hz as for w2v-BERT. Measure the padding ratio (`30 s / mean utterance duration`) per corpus and put it in the cost axis; it is larger on Common Voice and FLEURS than on LibriSpeech, and duration-sorted batching does not recover it because the window is per utterance |
| MoE aux-loss weight and router LR | the MoE's router barely moves at 2e-5; the crossing runs it at the recipe LR, and its aux loss is logged |

The running 10-epoch MoE arm at 2e-5 does not count as MoE evidence: it
changes adapter, training length and prompt at once.

## 4. The cost axis and the figure

Per stack, from `resolved_config.json` and SLURM accounting:

- encoder + adapter parameters (and active parameters for the MoE);
- decoder positions per audio second (the frame rate after the adapter);
- training GPU-h per 1,000 audio hours for MA and for IFT, with the
  window-padding ratio reported alongside it for any fixed-window encoder,
  since that cost is corpus-dependent and does not appear in the parameter
  count (see §3);
- eval throughput in utterances per second at batch 16 sorted by duration.

The figure that sells the pipeline: CER (in-domain mean, and FLEURS-24 mean
as a second panel) against decoder positions per audio second, one point per
stack, size of the marker by encoder+adapter parameters. Stacking is the
lever that moves along the x-axis.

## 5. Results

| encoder | adapter | rate | MA-stage CER (5-lang) | FLEURS-24 CER | GPU-h / 1K h | notes |
|---|---|---|---|---|---|---|
| | | | | | | |
