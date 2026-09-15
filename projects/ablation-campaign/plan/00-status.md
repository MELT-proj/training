# Status — living snapshot

Update this file whenever something starts, finishes, or blocks. Keep it
short; the reasoning goes to `board.md`, the plan to the numbered files.

**Last updated:** 2026-09-15, evening (Qwen3.5-2B IFT throughput measured; a completed-but-unrecorded Qwen MA+IFT pair discovered on MN5).
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

- Campaign tooling: `campaign.yaml` grid, `campaign.py plan/run/status`,
  `arms.tsv` ledger, parameterised launchers, DDP for MA, host-RAM wall fixed.
- MA-700 Llama-Instruct (August recipe): WER 1.10–1.16, eval loss 2.6–3.1.
- **No-audio floor measured** (`01-interface-recipe.md` §1a, MN5 job
  45888659): frozen-backbone floor is 4.0–4.3 nats/token per language, 1.1–1.5
  *above* the MA eval loss above, not equal to it as assumed. Audio was not
  ignored; whatever the adapter learned lowers loss without fixing WER. Needs
  PI review (see Blocked/waiting).
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
- **Qwen3.5-2B IFT throughput measured** (`06-fondue.md` §2, MN5 job
  45894977, acc_debug, 30 steps, eval/save off): steady state ~31 s/step
  at 2 nodes x 4 GPUs, DDP, gradient_checkpointing. Cross-checked against a
  full production `IFT-700-qwen35-2b-ins` run (job 45685241) that had
  already completed 2026-09-12, unrecorded: 27.3 s/step whole-epoch
  average, ~306 GPU-h for the 6,729.85 h arm. Replaces the "46 h budget
  implies 25,000+" guess with ~15,000-17,000 GPU-h at 2-node topology
  (extrapolated, no node-scaling correction to Fondue's 8-node topology --
  see the board entry for what that still leaves open).

## Blocked / waiting

- Q-Former adapter is broken; the PI fixes it in week 4.
- `Qwen/Qwen3.5-2B-Base` and both EuroLLM checkpoints need staging to MN5.
- FLEURS X→en ST eval set does not exist yet; spec in
  `05-language-ladder.md` §3.1, home is melt-eval (week 1 Track B).
- The shared artemis melt-eval venv can't currently run generation:
  its sibling `training` checkout (`/mnt/home/giuseppe/melt-proj/training`)
  is pinned before the transformers 5 migration (`a519e4fe`); needs a sync
  to `main` before `inspect eval` will import there. See the board entry.
- Two pre-existing test failures on `main` (sdpa propagation into
  `Wav2Vec2BertConfig`; all of `test_processing_melt.py`), likely
  transformers version skew; background tasks queued, not blocking week 1.
- PR #126 (stack_factor) awaiting review/merge.
- PR #10 (FLEURS-24 frozen sets, melt-eval) awaiting review/merge.
- **No-audio floor result needs PI review**: contradicts the "audio was
  ignored" reading in `01-interface-recipe.md` §1 (see §1a and the
  2026-09-15 board entry). Does not block the LibriSpeech step-0 runs, but
  should factor into how their results get read.
- **A completed Qwen MA-700 + IFT-700 pair on MN5 is off the ledger**: the
  current-recipe `MA-700-qwen35-2b-ins` (done 2026-09-05) and
  `IFT-700-qwen35-2b-ins` (done 2026-09-12, full eval scores in its
  `trainer_state.json`) arms both finished but are in neither `arms.tsv`
  nor this file. Needs a session to backfill `arms.tsv`, fold the eval
  numbers into `02-backbones.md` §5 once week 3's recipe-confirmation
  methodology is settled, and reconcile with the "old baseline" note above
  (this may already be the new-recipe number, not the August one). See the
  board entry.
- ~~`infra/sync_repo.sh` cannot target an alternate `REMOTE_REPO` on MN5~~ (fixed
  this session): `infra/runners/sites/mn5.sh` hardcoded
  `REMOTE_REPO=training` (no `:-` fallback, unlike every other var in that
  file), so an env override was silently ignored and a push landed on the
  shared checkout. Caused a near-miss (see the board entry); now uses the
  same `${VAR:-default}` pattern as the rest of the file.

## Next decisions, in order

1. Week 1 gate: recipe vs data, from LibriSpeech step 0.
2. Week 2 gate: the interface recipe.
3. Week 5 gate: backbone and regime.
4. 2026-10-25: Fondue freeze.

## Open questions parked here

- EuroLLM checkpoint ids on the Hub are unconfirmed (`utter-project/EuroLLM-1.7B`,
  `-Instruct` assumed).
- Whether Fondue's MA stage uses the full 247K h ASR pool or a subset; decided
  by where the ladder's MA-stage WER saturates.
