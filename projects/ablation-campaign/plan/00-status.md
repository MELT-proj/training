# Status — living snapshot

Update this file whenever something starts, finishes, or blocks. Keep it
short; the reasoning goes to `board.md`, the plan to the numbered files.

**Last updated:** 2026-09-19 (all 8 step 0b arms complete: R/R-seed/R-lr2e3/
R-b600/R-k5/W/Q2-k5/Q4-k5. Headline result -- encoder choice dominates every
other factor tested: W (Whisper-large-v3) reaches dev-clean/dev-other WER
0.038/0.061, 3-6x better than the best w2v-BERT arm on any other axis
(decoder size, stacking, schedule/LR/batch); decision rule 4 (encoder)
triggers unambiguously. Rules 1 and 2 (schedule, stacking) also trigger;
rule 3 (size) is a near-miss -- Q4-k5 lands at dev-clean 10.36%, just over
the bar, beating Q2-k5 by a consistent margin but not literally crossing
the threshold. Full rule-by-rule writeup and table in
`01-interface-recipe.md` §5, board entries 2026-09-18/19 -- prior updates:
step 0b pre-flight cleared 2026-09-17; the "insertions dominate" pre-flight
STOP was reviewed and reversed (main commit `ad07b3f`); the full-set
decoding diagnostic (job 45985475) found runaway is real but only 2.4-3.3%
of hypotheses, and that none of the 20-sample/200-sample/full-set eval
numbers safely stood in for each other -- see the board entry;
LibriSpeech step 0 complete, L1-L5 plus the epoch-extension diagnostic; step
0b designed with the PI, `01-interface-recipe.md`
§2b -- prior update 2026-09-16: text-prior tool built and run on all six
backbones on artemis, PR #14 -- Qwen3.5 leads overall, EuroLLM wins on
Maltese, instruct beats base in every family; a live `LANGUAGE_ISO_TO_NAME`
gap for Irish found and fixed; ru/uk added to the FLEURS melt-eval configs,
PR #12 + PR #13; PR #11 merged; Qwen3.5-2B-Base and both EuroLLM
checkpoints staged to MN5, offline loads verified; Qwen IFT throughput
measured at 2, 8 and 16 nodes -- scaling is flat).
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

- Step 0b (`01-interface-recipe.md` §2b): all 8 arms complete --
  R/R-seed/R-lr2e3/R-b600/R-k5/W/Q2-k5/Q4-k5. Encoder (W, Whisper-large-v3)
  dominates: dev-clean/dev-other WER 0.038/0.061, the only arm to cross the
  <10% bar, 3-6x better than the best w2v-BERT arm on any other axis. Rules
  1/2 (schedule, stacking) trigger; rule 3 (size) is a near-miss (Q4-k5
  10.36% dev-clean, just over). Full writeup in `01-interface-recipe.md`
  §5, board entries 2026-09-18/19.
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
  `melt-eval`, PR #10, **merged** 2026-09-15. ga (Irish) has no `pnc_text`
  on either split and falls back to plain text.
- FLEURS X→en ST frozen sets in `melt-eval` (`05-language-ladder.md` §3.1):
  `fleurs24-st-xen-test` (18,816 samples, 61.63 h, 23 locales, zero
  dropped) and `fleurs24-st-xen-dev` (2,300 samples, 7.14 h, 100/lang).
  New `reference_map` reader option joins in the English FLEURS text by
  sentence id. Branch `claude/fleurs-x-to-en-st` in `melt-eval`,
  [PR #11](https://github.com/MELT-proj/eval/pull/11), **merged**
  2026-09-16. See the board entry -- one `cs_cz` sentence id has a
  PNC-pass leak (unrelated to this set's correctness), flagged for the
  preprocessing repo.
- **ru/uk added to the FLEURS melt-eval configs** (both ASR and ST X→en,
  test and dev): closes the gap the text-prior tool spec surfaced
  (`02-backbones.md` §3.5). Checked directly first (both locales carry
  `custom.pnc_text`, unlike `ga`; the ST `reference_map` covers their
  sentence ids with zero drops, same as the other 23), then re-froze all
  four configs to confirm: ASR test 19,463→20,988 samples (63.41h→68.17h),
  ASR dev 2,400→2,600 (7.41h→8.01h), ST test 18,816→20,341
  (61.63h→66.39h), ST dev 2,300→2,500 (7.14h→7.75h).
  [melt-eval PR #12](https://github.com/MELT-proj/eval/pull/12) (ASR) and
  [melt-eval PR #13](https://github.com/MELT-proj/eval/pull/13) (ST), both
  open against `main`. Does not redeploy the production frozen-set copies
  on artemis scratch -- whoever next consumes the 26-language set should
  re-run `melteval freeze` and copy over.
- **Qwen3.5-2B IFT throughput measured** (`06-fondue.md` §2) at 2, 8 and
  16 nodes: ~31, ~32.3 and ~33 s/step respectively -- essentially flat
  scaling across the whole range. Fondue's own 8-node topology costs
  ~17,800 GPU-h for the ~330K h IFT pool; 16 nodes is ~18,150, same
  ballpark, so above 8 nodes it's a wall-clock lever, not a GPU-h cost.
  Replaces the old unmeasured "46 h budget implies 25,000+" guess.
  Cross-checked against a full production `IFT-700-qwen35-2b-ins` run (job
  45685241) that had already completed 2026-09-12, unrecorded: 27.3 s/step
  whole-epoch average, ~306 GPU-h for the 6,729.85 h arm.
- **Text-prior tool spec drafted** (`02-backbones.md` §3.1–3.6): two
  mechanisms, not one -- a new standalone `melteval text-prior` CLI command
  for teacher-forced NLL/BPC/fertility (inspect_ai's Task/Solver/Scorer
  triad has no logprob-of-a-given-continuation path, confirmed against
  `scorers.py`), and a new solver on the existing `st` task for the cascade
  oracle. Chat-template-per-backbone verified directly, not assumed:
  `Qwen/Qwen3.5-2B-Base` ships its own (no borrowing); EuroLLM base does not
  (checked its `tokenizer_config.json`) and has no `DECODER_PROFILES` entry
  yet -- see Blocked/waiting. Also surfaced a ru/uk gap in the FLEURS
  melt-eval configs, closed same-day (see Done).
- **Text-prior tool built and tested on all six backbones**
  (`02-backbones.md` §3.7, [melt-eval PR #14](https://github.com/MELT-proj/eval/pull/14)):
  rows 1-4 only (teacher-forced NLL/BPC/fertility), not the cascade
  oracle. 12 runs on artemis (`dionysus`, h100/gpu-h100) -- all six
  backbones × `fleurs24-asr-dev` (24 langs) + `fleurs24-st-xen-dev`
  (23 locales) -- all completed clean. Two infra snags fixed along the
  way, neither a tool bug: `LANGUAGE_ISO_TO_NAME` had no entry for Irish
  (fixed, `main` `b5f4962`); a shared HF cache held tokenizer-only or
  config-only entries for three checkpoints, not full weights (retried
  against complete ones). **Qwen3.5 leads overall** (conditioned ASR BPC
  1.428 instruct / 1.651 base), **EuroLLM wins specifically on Maltese**
  (1.911 vs Qwen's 2.002), and **instruct beats base in every family on
  both tasks with no exception** -- a direct measurement of the
  "template-naive base" confound (`02-backbones.md` §2). Full table in
  the board entry. Still to go: ru/uk, the `-test` splits, the cascade
  oracle.
- **Backbone checkpoints staged to MN5**: `Qwen/Qwen3.5-2B-Base` (4.3G) and
  both `utter-project/EuroLLM-1.7B` checkpoints (3.1G each) downloaded on nyx,
  rsynced to `mn5transfer:/gpfs/scratch/epor48/hf_cache/hub/`, and offline
  load verified inside the container on `alogin1` (`HF_HUB_OFFLINE=1`) --
  `qwen3_5`/1.88B params and `llama`/1.66B params respectively. The EuroLLM
  Hub ids are now confirmed, not assumed. See the board entry.

## Blocked / waiting

- Q-Former adapter is broken; the PI fixes it in week 4.
- PR #14 (`melteval text-prior`, the tool itself) awaiting review/merge.
- PR #12 and PR #13 (ru/uk added to the FLEURS ASR and ST melt-eval
  configs) awaiting review/merge.
- **EuroLLM has no `DECODER_PROFILES` entry** (`plan_arm.py`): `chatml`,
  `chat_template_from: utter-project/EuroLLM-1.7B-Instruct` for the base
  checkpoint, verified directly against both checkpoints'
  `tokenizer_config.json` while drafting the text-prior tool spec
  (`02-backbones.md` §3.1) but not yet added to the dict. Needed for the
  text-prior tool and week 3's EuroLLM MA arms; tracked as a `timeline.md`
  week-1 item. (The ru/uk frozen-set gap the same spec surfaced is now
  closed, see Done.)
- The shared artemis melt-eval venv can't currently run generation:
  its sibling `training` checkout (`/mnt/home/giuseppe/melt-proj/training`)
  is pinned before the transformers 5 migration (`a519e4fe`); needs a sync
  to `main` before `inspect eval` will import there. See the board entry.
- Two pre-existing test failures on `main` (sdpa propagation into
  `Wav2Vec2BertConfig`; all of `test_processing_melt.py`), likely
  transformers version skew; background tasks queued, not blocking week 1.
- PR #126 (stack_factor) awaiting review/merge.
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
- **`optimization.min_lr_scale` is a dead config key** (found 2026-09-17):
  set to 0.1 in every campaign and SFT config, read by no code, so every
  cosine run decayed to zero. Comparisons between arms stand; the configs
  misstate what ran. PI decides whether to wire it or delete it.

## Next decisions, in order

1. Week 1 gate: recipe vs data, from LibriSpeech step 0 -- answered (recipe,
   not data). Step 0 itself stayed inconclusive on the recipe's own success
   bar (all five one-epoch arms stayed above WER 1.0), so step 0b
   (`01-interface-recipe.md` §2b) now decides schedule, stacking, encoder
   and decoder size before the week-2 screen launches -- **running**: six
   Llama/Whisper arms launched 2026-09-17, Qwen decoder-size control queued
   behind the Qwen3.5-4B transfer. Apply §2b's decision rules once results
   land.
2. Week 2 gate: the interface recipe.
3. Week 5 gate: backbone and regime.
4. 2026-10-25: Fondue freeze.

## Open questions parked here

- Whether Fondue's MA stage uses the full 247K h ASR pool, a subset, or none
  at all (a single stage): decided by the MA:IFT ratio sweep
  (`04-regime.md` §6, week 6) together with the ladder's MA-stage curves.
  Default is full MA if the sweep misses the 2026-10-25 freeze.
