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

## 2026-09-17 — Claude (worker session librispeech-step0-l1-l4) — LibriSpeech step 0: recipe fails at 1 epoch, but the plateau breaks with more gradient updates

Context: week 1 Track A, `01-interface-recipe.md` §2 ("is the failed
alignment a recipe problem or a multilingual-data problem?"). Branch
`claude/librispeech-step0-l1-l4-60d122`. New base config
`ABL-MA-librispeech.yaml` (hand-written, not `build_campaign_config.py`
-- LibriSpeech is one corpus/language, not the campaign's N-language
reference-matched mixture; its three train splits are weighted by their
own measured hours, not the alpha/beta corpus-balancing policy --
documented in the config so nobody re-emits it). `campaign.yaml` gained
`MA-librispeech-l1`..`l5` plus a post-hoc diagnostic arm,
`MA-librispeech-l4-ep3`.

Finding: L1-L4 (adapter LR 2e-5/2e-4/2e-4/1e-3, effective batch
4800/4800/1200/1200 s, one epoch of LibriSpeech's ~961 h) all reproduce
the August plateau -- none crossed WER 1.0, loss declined smoothly with
no plateau-then-drop transition. This rules out multilingual data as the
sole cause (LibriSpeech alone fails the same way) and confirms LR+steps
help monotonically: L4 (loss 2.625/2.587 clean/other, WER 1.040/1.056)
is clearly best. L5 (second seed of L4) replicated within ~0.02
nats/WER, so this is a real recipe effect, not seed noise.

But L4's loss delta was still accelerating at epoch end (-0.03/step
early -> -0.108 near epoch 0.8), so I ran a diagnostic outside the
designed grid: `MA-librispeech-l4-ep3`, L4's exact LR/batch (1e-3,
1200 s) fresh for 3 epochs (a fresh run, not a resume, so the cosine
schedule re-derives over the full 3-epoch horizon and decays much more
slowly through what was epoch 1). **The plateau breaks**: dev-clean/
dev-other WER goes 1.155/1.312 (epoch 1) -> 0.895/0.745 (epoch 2) ->
0.622/0.776 (epoch 3, final). Loss falls 3.11->1.51 within epoch 1 alone
(a schedule effect: this run's LR has decayed far less by that step
count than L4's own 1-epoch-tuned schedule had), then keeps falling
1.51->0.90 over epochs 2-3. Not monotonic in the last ~10% of training
(best single point was epoch 2.726's dev-clean WER 0.588; dev-other got
slightly worse from epoch 2 to 3) and still short of the <10%
success-bar, but this is a real, large transition, not noise.

Reframing: the August recipe's failure at 1 epoch is real and
reproduces on English-only data, but the bottleneck looks like schedule
length (LR decaying before the transition completes at a 1B decoder),
not a hard adapter/decoder/encoder capacity ceiling. `03-audio-stack.md`
and `02-backbones.md` still have open questions this doesn't touch
(w2v-BERT vs Whisper; SLAM-ASR's 7B decoder vs our 1-3B target class),
but they're no longer the only explanation on the table for "why didn't
it transcribe."

Action needed: PI decision for the week-2 gate (`01-interface-recipe.md`
§3, five-language screen) -- does the screen's step/LR budget need to
grow to let this transition complete at 125 h/language, or is more
wall-clock at that budget enough? Full trajectories and job ids in
`01-interface-recipe.md` §5 and `arms.tsv`.

---

## 2026-09-15 — Claude (worker session fleurs-24-asr-frozen-sets) — FLEURS-24 ASR frozen sets built

Context: week 1 Track B, `05-language-ladder.md` §3 / `timeline.md` week 1
("FLEURS-24 ASR frozen sets in melt-eval"). Branch
`claude/fleurs24-asr-frozen-sets` in `melt-eval`,
[PR #10](https://github.com/MELT-proj/eval/pull/10).

Finding / proposal: added `configs/fleurs24-asr-test.yaml` (full FLEURS
`test`, all 24 EU languages) and `configs/fleurs24-asr-dev.yaml` (~100
utterances/language from FLEURS `validation`, for the in-training generative
round). Froze both against the real shar tree: `fleurs24-asr-test` is 19,463
samples / 63.41 h with zero dropped-no-reference cuts; `fleurs24-asr-dev` is
2,400 samples / 7.41 h. Copied to
`/mnt/scratch-artemis/giuseppe/melt-data/eval-sets/{fleurs24-asr-test,fleurs24-asr-dev}/`.

Checked `custom.pnc_text` coverage directly against the shar tree for all 24
languages, both `test` and `validation` (not previously measured at this
granularity): **23/24 are 100% covered; ga (Irish) has 0% on both splits**
and silently falls back to plain, unpunctuated supervision text
(`get_text_from_cut`, `strict=False`, the training repo's own default). Every
other language's FLEURS reference is cased and punctuated; Irish's is not.
Documented in both spec headers rather than worked around -- fixing it is the
training repo's PNC-backfill/`strict_text_field` decision
([[num-tokens-and-pnc-text-semantics]], [[silent-text-field-fallback]]), out
of scope here. Worth remembering when Irish's ladder numbers look
disproportionately bad or good: part of that could be transcript formatting,
not the model.

Also found: the shared artemis dev venv
(`/mnt/scratch-artemis/giuseppe/venvs/melteval`) editable-installs
`melt-proj` from `melt-eval`'s sibling checkout at
`/mnt/home/giuseppe/melt-proj/training`, which is pinned at `74c7892`
(2026-08-19) -- **before** the transformers 5 migration (`a519e4fe`,
2026-09-01, PR #109). Importing `melt.training` there crashes
(`ValueError: mutable default <class 'dict'> for field sub_configs`) against
the venv's transformers 5.16.1. I froze the sets from nyx instead (training
repo's own `.venv`, current `main`, melt-eval on `PYTHONPATH`) rather than
touching that checkout, since it has unrelated uncommitted local changes
(`.github/workflows/ci.yml`, `.gitignore`, `AGENTS.md`, `README.md`) and its
scratch-side sibling (`/mnt/scratch-artemis/giuseppe/melt-proj-src/melt-eval`,
on `claude/air-bench-support`) has unrelated in-progress work from another
session. Did not touch either.

Action needed: PI review/merge PR #10. Before anyone runs `inspect eval`
(the generation step) against these frozen sets on artemis, sync
`/mnt/home/giuseppe/melt-proj/training` to `main` (or otherwise past
`a519e4fe`) -- generation will crash on import otherwise. FLEURS X→en ST set
construction (the other week-1 Track B eval item) is still open.

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
