# Status — living snapshot

Update this file whenever something starts, finishes, or blocks. Keep it
short; the reasoning goes to `board.md`, the plan to the numbered files.

**Last updated:** 2026-09-17 (LibriSpeech step 0 complete: L1-L5 plus an
epoch-extension diagnostic).
**Current week:** week 1 of `timeline.md` (2026-09-14 to 2026-09-20).

## Running on MN5

- `MA-700asr-w2vbF-llama1bInsF-moeT-ep10-ttverbatim-…` — 10-epoch MA with the
  MoE adapter and the verbatim prompt, adapter LR 2e-5. Submitted 2026-09-13.
  Observed: loss plateau then a smooth drop from ~3.6 to ~2.6 after ~9,000 h
  seen. Read as an LR-limited alignment transition, not as MoE evidence
  (`01-interface-recipe.md` §1). Let it finish; do not build on it.
- IFT arms for Llama-1B-Instruct and Qwen3.5-2B-Instruct at 700 h/lang
  completed on the August recipe; in-domain WER ≈ 20%, X→en BLEU ≈ 4–5 on
  MLS and CoVoST2. These are the "old baseline" numbers; they will be
  superseded in week 3.

## Done

- LibriSpeech step 0 (`01-interface-recipe.md` §2): L1-L4 all reproduce the
  August plateau (WER>1.0, no transition inside one epoch) -- rules out
  multilingual data as the sole cause. L4 (LR 1e-3, 1200 s effective batch)
  is clearly best (WER 1.040/1.056 clean/other); L5 (second seed) replicates
  within ~0.02 WER, so it's a real recipe effect. Diagnostic
  `MA-librispeech-l4-ep3` (fresh 3-epoch run at L4's LR/batch) shows the
  plateau DOES break with more gradient updates: WER falls to 0.622/0.776
  by epoch 3 -- still short of <10%, but a real, large transition. Bottleneck
  looks like schedule length (cosine decay cutting the transition off at 1
  epoch), not a hard capacity ceiling. Full numbers in
  `01-interface-recipe.md` §5; branch `claude/librispeech-step0-l1-l4-60d122`.
- Campaign tooling: `campaign.yaml` grid, `campaign.py plan/run/status`,
  `arms.tsv` ledger, parameterised launchers, DDP for MA, host-RAM wall fixed.
- MA-700 Llama-Instruct (August recipe): WER 1.10–1.16, eval loss 2.6–3.1.
- Data audit for 24 EU languages plus ru/uk/ca: `data/hours_by_language.csv`.
- Adapter sizes measured: MLP 6.30M, Conformer 27.28M, MoE 33.57M (8.39M active).
- `stack_factor` for the MLP adapter, with tests and the `-skN` EXP_NAME tag:
  branch `claude/mlp-adapter-stack-factor-13e900`, PR #126 open against
  `main`. See the board entry.
- FLEURS-24 ASR frozen sets in `melt-eval`: full `test` (19,463 samples,
  63.41 h, all 24 EU languages) and a 100/lang `validation` dev subset
  (2,400 samples, 7.41 h). Branch `claude/fleurs24-asr-frozen-sets` in
  `melt-eval`, PR #10 open against `main`. See the board entry -- ga
  (Irish) has no `pnc_text` on either split and falls back to plain text.

## Blocked / waiting

- Q-Former adapter is broken; the PI fixes it in week 4.
- `Qwen/Qwen3.5-2B-Base` and both EuroLLM checkpoints need staging to MN5.
- FLEURS X→en ST eval set does not exist yet (preprocessing task, week 1).
- The shared artemis melt-eval venv can't currently run generation:
  its sibling `training` checkout (`/mnt/home/giuseppe/melt-proj/training`)
  is pinned before the transformers 5 migration (`a519e4fe`); needs a sync
  to `main` before `inspect eval` will import there. See the board entry.
- Two pre-existing test failures on `main` (sdpa propagation into
  `Wav2Vec2BertConfig`; all of `test_processing_melt.py`), likely
  transformers version skew; background tasks queued, not blocking week 1.
- PR #126 (stack_factor) awaiting review/merge.
- PR #10 (FLEURS-24 frozen sets, melt-eval) awaiting review/merge.

## Next decisions, in order

1. Week 1 gate: recipe vs data, from LibriSpeech step 0 -- answered (recipe,
   not data; see Done above). Open follow-up: does the five-language screen
   (week 2, 125 h/language) need a longer schedule/more steps than its
   current budget to let the same transition complete, or does more
   wall-clock at that budget suffice? PI decides before the screen launches.
2. Week 2 gate: the interface recipe.
3. Week 5 gate: backbone and regime.
4. 2026-10-25: Fondue freeze.

## Open questions parked here

- EuroLLM checkpoint ids on the Hub are unconfirmed (`utter-project/EuroLLM-1.7B`,
  `-Instruct` assumed).
- Whether Fondue's MA stage uses the full 247K h ASR pool or a subset; decided
  by where the ladder's MA-stage WER saturates.
