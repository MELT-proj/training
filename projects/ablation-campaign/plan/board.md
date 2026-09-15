# Message board

Append-only, newest first. Anyone (person or agent) posts here: findings,
doubts, proposals, things that surprised you. Fold the settled part into the
numbered files afterwards and leave the entry.

Entry template:

```
## YYYY-MM-DD — <who> — <one-line title>
Context: what you were doing.
Finding / proposal: the substance, with numbers and experiment names.
Action needed: who should do what, or "none".
```

---

## 2026-09-15 — Claude (worker session mlp-adapter-stack-factor-13e900) — stack_factor PR opened

Context: follow-up to the entry directly below, after the PI merged the
plan folder into `main`.

Finding / proposal: rebased the branch onto the updated `main`, re-ran the
full suite in the nyx container (unchanged: only the two pre-existing
failures noted below), and opened
[PR #126](https://github.com/MELT-proj/training/pull/126) against `main`.

Action needed: PI review/merge.

---

## 2026-09-15 — Claude (worker session mlp-adapter-stack-factor-13e900) — stack_factor implemented for the MLP adapter

Context: week 1 Track B, `01-interface-recipe.md` §4. Branch
`claude/mlp-adapter-stack-factor-13e900`, commit `5822129`, 12 files, local
only until the PI decides on a PR.

Finding: `MELTMLPAdapter` concatenates k consecutive encoder frames along the
feature axis before `fc1` (input width k × encoder hidden size), pads the
frame count to a multiple of k, and subsamples the attention mask by k in
`_get_output_features_shape` with the same prefix-mask assumption the
Conformer adapter uses. New `model.adapter.stack_factor` field, default 1,
byte-identical to the previous adapter. Conformer and Q-Former ignore it.
Wired as a `STACK_FACTOR` campaign axis (env var, `ArmAxes.stack_factor`,
`--model.adapter.stack_factor`), tagged into `EXP_NAME` as `-skN` only when
it differs from the base config, so no existing arm is renamed. Tests added
for fc1 width, exact and non-multiple downsampling, mask subsampling, the
no-mask case, composition with an encoder's own downsampling, config parsing,
and `plan_arm.py` tag composition (its first tests). Full suite in the nyx
container: 452 passed.

Two pre-existing failures found, unrelated to this change and reproduced on
a clean checkout: `TestAttnImplementationPropagation::test_sdpa_reaches_an_encoder_that_has_no_flash_kernel`
(sdpa not propagated into `Wav2Vec2BertConfig`, likely transformers version
skew) and every test in `test_processing_melt.py` (tokenizer fixture's
`add_special_tokens` against Qwen2.5-1.5B). Background tasks queued for both.

Action needed: PI decides push/PR for the branch (12 files, so a PR against
`main` per the protocol). Reviewed at plan level by the strategy session:
padding, reshape and ceil(valid/k) prefix mask look right; end-to-end
validation comes from the LibriSpeech screen's stack=4 runs.

## 2026-09-15 — Claude (Fable 5.1, research-buddy session) — Plan folder created; the August baseline is a failed alignment

Context: three-turn strategy session with the PI on how to organise the
campaign, then two added commitments (24 EU languages, the Fondue run).

Findings:
- MA eval loss 2.6–3.1 and WER > 1.0 after one epoch means the adapter never
  conditioned the decoder on audio. Adapter LR 2e-5 (10× below the repo's
  own IFT default), ~2,600 optimizer steps per epoch at a 4,800 s batch, and
  a 50 Hz frame rate with no stacking are the suspects, in that order. The
  10-epoch MoE/verbatim run's late loss drop is the alignment transition
  arriving late, not MoE evidence.
- MA-stage generative WER is a valid selection metric for anything on the
  audio side (SLAM-ASR shows a frozen LLM with a good projector transcribes),
  so encoders, adapters and frame rate are screened at ~30 GPU-h per arm.
- Qwen3.5-2B is dense with hybrid linear attention, not an MoE. Llama vs
  Qwen is also full vs hybrid attention.
- The spreadsheet's ST column mixes X→en and en→X; real X→en exists for 13
  EU languages, none for sl/lv/mt/ga. Tiers 10/30/100/300/700 h are reached
  in full by 24/17/13/8/8 languages.
- Budget: ~49,700 GPU-h remain until 2026-11-30, no renewal. Compute is not
  the constraint; queue wait, wall clock and serial dependencies are.

Action needed: PI reviews `timeline.md` week 1; an implementation session
lands `stack_factor` and the LibriSpeech step-0 configs.
