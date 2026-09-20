# Timeline — 2026-09-14 to 2026-11-30, then December

Weeks run Monday to Sunday. The MN5 allocation (`epor48`) expires on
**Monday 2026-11-30** with no renewal; after that only internal GPUs
(A6000, H100, possibly H200) are available, for evaluation and small
follow-ups, not for training Fondue.

Two tracks run in parallel every week:

- **Track A — GPU (MN5).** Training arms. Submitted through
  `campaign.py run`, recorded in `arms.tsv`.
- **Track B — preparation.** Code, configs, data, evaluation sets, text-only
  measurements. CPU work on nyx, GPU work on artemis via `sbatch`.

Budget frame, measured 2026-09-19 (`bsc_acct` for the official position,
`sacct` for the live one; 1 GPU-h = 20 physical core-hours on ACC):

| | GPU-h |
|---|---|
| `epor48` grant (1,752 khours ACC) | 87,600 |
| consumed, whole project, all users | ~38,500 (44%) |
| **remaining** | **~49,100** |
| of the consumed total: before 2026-08-01, i.e. pre-campaign | ~35,000 (92%) |
| of the consumed total: this ablation campaign (Aug + Sep) | ~2,900 |

`bsc_acct` lags about a day, so the remaining figure is a slight
overstatement. Everything in this timeline except Fondue is ~8K GPU-h;
Fondue is 10–30K depending on the backbone, and Raclette is ~1.4K
(Llama-class) to ~4K (Qwen-class). Spend the rest on seed replicates and
ladder points; unspent hours vanish on 11-30.

Cost anchors used below (measured unless marked): MA arm at 700 h/lang
≈ 30 GPU-h; IFT arm at 700 h/lang, Llama-1B ≈ 80–100 GPU-h; Qwen-2B IFT
unmeasured at the real topology, expected several times more.

**This list is owned by the orchestrator.** Sessions tick the boxes they
complete and nothing else: no adding, deleting, rewording or moving items
between weeks. If work needs to change, post it on `board.md` and it is
folded in from there. A box you could not finish stays unticked, with a
board entry saying why.

---

## Blocked / waiting

Carried here 2026-09-18 when `00-status.md` was retired. Findings live on
`board.md`; this list is only what is holding something up.

- **Q-Former adapter is not instantiable.** PI fix, week 2 Track B. It gates
  the four Q-Former arms of the week-3 audio-stack crossing; the other
  twelve arms do not wait for it.
- **melt-eval PRs awaiting review/merge:** #14 (the `text-prior` command),
  #12 and #13 (ru/uk added to the FLEURS ASR and ST configs). The ru/uk
  frozen sets are also not yet redeployed to the production copies on
  artemis scratch — whoever next consumes the 26-language set re-runs
  `melteval freeze` and copies over.
- **The shared artemis melt-eval venv cannot run generation.** Its sibling
  `training` checkout is pinned before the transformers 5 migration, so
  `inspect eval` will not import there until it is synced to `main`.
- **Two pre-existing test failures on `main`** (sdpa propagation into
  `Wav2Vec2BertConfig`; all of `test_processing_melt.py`), likely
  transformers version skew. Not blocking any week.
- **`optimization.min_lr_scale` is a dead config key** — set in every
  campaign and SFT config, read by no code, so every cosine run decayed to
  zero rather than to 10% of peak. Comparisons between arms are unaffected;
  the configs misstate what ran. **PI decides** whether to wire it or delete
  it.

---

## Week 1 — Mon 2026-09-14 to Sun 2026-09-20

**Gate at end of week:** is the failed alignment a recipe problem (LR, batch,
frame rate) or a multilingual-data problem? Decided by LibriSpeech step 0.

### Track A — GPU
- [x] **No-audio floor** (done 2026-09-15, MN5 job 45888659). Text-only NLL
      of the frozen backbone (Llama-3.2-1B-Instruct) on the campaign
      validation transcripts with the MA chat prompt and no audio.
      *Outcome:* floor is 4.0–4.3 nats/token per language, **1.1–1.5 above**
      the August MA eval loss (2.6–3.1), not matching it — audio was not
      ignored. `01-interface-recipe.md` §1a.
- [x] **LibriSpeech step 0**, five MA runs (`01-interface-recipe.md` §2):
      L1 current recipe (adapter LR 2e-5, batch 4800 s), L2 LR 2e-4 / 4800 s,
      L3 LR 2e-4 / 1200 s, L4 LR 1e-3 / 1200 s, plus a second seed of the
      best. ~7 GPU-h each. Done 2026-09-16.
      *Outcome:* inconclusive — all five stayed above WER 1.0 and none showed
      a transition inside the epoch, so step 0b was added below. Results in
      `01-interface-recipe.md` §5.
- [x] **Qwen3.5-2B IFT throughput** at 2 nodes × 4 GPUs, 20–50 steps,
      gradient checkpointing on, eval and saves off (`acc_debug`). Done
      2026-09-15, MN5 job 45894977.
      *Outcome:* ~31 s/step steady state, cross-checked against a completed
      production arm at 27.3 s/step and ~306 GPU-h. Scaling is flat from 2 to
      16 nodes. `06-fondue.md` §2.
- [x] **Step 0b** (`01-interface-recipe.md` §2b), added 2026-09-17 because
      step 0 was inconclusive. Pre-flight first, no GPU: substitution,
      deletion and insertion rates on the three-epoch L4 run's hypotheses,
      and a dry run of the warmup-stable-decay scheduler. Then eight
      LibriSpeech MA arms at three epochs: warmup-stable-decay, its seed
      replicate, LR 2e-3, 600 s batch, stack 5, Whisper encoder, and
      Qwen3.5-2B against Qwen3.5-4B at stack 5. ~250 GPU-h.
      *Outcome:* audio hours to 10% dev-clean WER per arm, which sets the
      week-2 screen's budget and removes settled factors from its grid.
      *State 2026-09-18:* six of eight landed, results in
      `01-interface-recipe.md` §5; the two Qwen arms are queued. No decision
      rule is applied until all eight are in. Scope narrowed the same day:
      step 0b settles the recipe only, and its encoder and size arms are
      controls handed to `03` and `02` (§2b).

### Track B — preparation
- [x] **`stack_factor` for the MLP adapter** (done 2026-09-15, [PR #126](https://github.com/MELT-proj/training/pull/126)) (concatenate k consecutive
      encoder frames; `fc1` input becomes k × encoder width; masks and length
      bookkeeping follow). Unit tests. Tag into `EXP_NAME`.
      *Outcome:* frame rate becomes an axis available to every adapter.
- [x] **FLEURS-24 ASR frozen sets** in melt-eval: test split for all 24 EU
      languages, plus a small dev subset (≈100 utterances per language) for
      in-training generative eval. Done 2026-09-15
      ([PR #10](https://github.com/MELT-proj/eval/pull/10)): 19,463 samples
      / 63.41 h (test), 2,400 samples / 7.41 h (dev). ga has no `pnc_text`
      on either split -- see the board entry.
- [x] **FLEURS X→en ST frozen sets** in melt-eval, per
      `05-language-ladder.md` §3.1 (done 2026-09-15,
      [PR #11](https://github.com/MELT-proj/eval/pull/11)): join each
      language's test audio with the English FLEURS text of the same
      sentence id, reference from the original English transcription,
      zero-copy. `fleurs24-st-xen-test` 18,816 samples/61.63 h (23 locales,
      zero dropped), `fleurs24-st-xen-dev` 2,300 samples/7.14 h (100/lang).
      CoVoST2 covers ten of the 24 for cross-corpus comparison. See the
      board entry.
- [x] **Stage models on MN5** (done 2026-09-16) over `mn5transfer`:
      `Qwen/Qwen3.5-2B-Base`, `utter-project/EuroLLM-1.7B`,
      `utter-project/EuroLLM-1.7B-Instruct`. Hub ids confirmed before
      downloading; offline load verified inside the container on `alogin1`
      for all three. See the board entry.
- [x] **Text-prior tool spec** agreed (`02-backbones.md` §3). Done
      2026-09-16: two mechanisms (standalone teacher-forced scorer for
      NLL/BPC/fertility; a new solver on the existing `st` task for the
      cascade oracle), chat-template-per-backbone verified directly. Surfaced
      two prerequisites for week 2's build; the ru/uk one is closed below,
      the `DECODER_PROFILES` one is its own item below -- see the board
      entry and Blocked / waiting above.
- [x] **ru/uk added to the FLEURS melt-eval configs** (done 2026-09-16,
      closing the gap the text-prior tool spec surfaced above):
      [melt-eval#12](https://github.com/MELT-proj/eval/pull/12) (ASR test +
      dev) and [melt-eval#13](https://github.com/MELT-proj/eval/pull/13) (ST
      X→en test + dev), both open against `main`. Verified by re-freezing:
      zero dropped cuts for either locale on any of the four sets.
- [x] **Add EuroLLM to `DECODER_PROFILES`** (`plan_arm.py`): `chatml`,
      `chat_template_from: utter-project/EuroLLM-1.7B-Instruct` for the base
      checkpoint -- verified directly against both checkpoints'
      `tokenizer_config.json` while drafting the text-prior tool spec
      (`02-backbones.md` §3.1, 2026-09-16), not yet added to the dict itself.
      Needed for the text-prior tool and for week 3's EuroLLM MA arms.
- [ ] **Eyeball 20 Qwen hypotheses** from the running IFT arm's eval tables
      for a leaked think block.
- [x] **`wsd-50hz-whisper`'s threshold-crossing step**, from its existing
      in-training eval history: the first `global_step` and audio-hour count
      at which each dev set went below WER 0.10. No GPU. It is the only
      hours-to-threshold number step 0b will produce, and the week-2 screen
      budget is derived from it (`01-interface-recipe.md` §2b, Consequence).
- [ ] **Whisper-large-v3's own WER** on the same normalised dev-clean and
      dev-other, as a reference line for the arm above. A decode, no
      training; artemis preferred so it does not queue behind MN5 work.
      *Outcome:* `wsd-50hz-whisper`'s 0.038/0.061 restated as the fraction
      of the encoder's own ability the projector recovers, which is the
      form `03-audio-stack.md` §0 needs.
- [x] **Backfill `arms.tsv`** with the completed-but-unrecorded Qwen pair
      (`MA-700-qwen35-2b-ins`, done 2026-09-05, and `IFT-700-qwen35-2b-ins`,
      done 2026-09-12, whose full eval scores sit in its `trainer_state.json`).
      Fold its numbers into `02-backbones.md` §5 once week 3's
      recipe-confirmation methodology is settled.

---

## Week 2 — Mon 2026-09-21 to Sun 2026-09-27

**Gate:** the interface recipe (adapter LR, effective batch, stack factor,
MA prompt) is chosen by Sunday and written into `01-interface-recipe.md` as
Settled.

### Track A — GPU
- [ ] **Five-language interface screen**, MA only, ASR-only mixture rendered
      at `--budget-hours 125` (625 h total, ≈35 min per run on 8 GPUs):
      full grid adapter LR {2e-4, 1e-3} × effective batch {1200, 4800} s ×
      stack {1, 4} = 8 runs; one verbatim-prompt run at the best corner; two
      extra seeds at the best corner; one 2e-5 control. Twelve runs, ~60 GPU-h.
      *Outcome:* MA-stage generative WER per language, loss transition step,
      per-run FLEURS-24 zero-shot CER from melt-eval on the final checkpoint.
      *Revised 2026-09-17:* waits for step 0b. Budget becomes at least 1.5×
      the best LibriSpeech arm's hours-to-threshold and never under one
      epoch of 700 h per language; 125 h per language is ~1,900 steps, below
      where the transition appeared. Settled factors leave the grid. Launch
      moves to about Tue 2026-09-22; the gate stays Sun 2026-09-27 if the
      queue allows.
- [ ] If step 0 was ambiguous, re-run the deciding LibriSpeech pair with the
      second seed before the screen. *Superseded by step 0b (week 1).*

### Track B — preparation
- [x] **Text-prior tool built** in melt-eval (`02-backbones.md` §3): bits per
      character and tokens per word on the FLEURS-24 references, with and
      without the IFT instruction, for ASR transcripts and English ST
      references. Run on all six backbones on an internal GPU.
      *Outcome:* the 6 × 24 prior table, a first ranking of backbones by
      language coverage before any training. Table in `02-backbones.md`
      §3.7, reading on the board.
      *Done 2026-09-16,* [melt-eval #14](https://github.com/MELT-proj/eval/pull/14),
      twelve runs. Still open on this item: ru/uk, the cascade oracle
      (§3.2) and the `-test` splits.
- [ ] **Q-Former fix** (PI), moved up from week 4: the audio-stack crossing
      now runs in week 3, so its four Q-Former arms need the adapter
      instantiable by then. If it slips, the crossing launches without them
      and they join as a late addition at the same recipe.
- [ ] **MoE adapter branch merged to `main`** with its aux-loss logging,
      moved up from week 4 for the same reason: the MoE is one of the four
      adapters in the week-3 crossing, so it must be on `main` before the
      sixteen arms are rendered.
- [x] **Fondue config drafted** (`06-fondue.md` §3): language set incl. ru/uk,
      two-tier mixture weights (alpha/beta), filters, eval subset. Not frozen.
- [x] **Raclette config drafted**: same mixture at 25K h, big-run batch,
      three LR points.
- [ ] melt-eval MN5 venv built and a smoke eval run there, so end-of-run
      evaluation can happen on MN5 when queues allow.
- [ ] **Re-derive the screen's budget and render its configs**, now that
      step 0b has produced an hours-to-threshold number: `wsd-50hz-whisper`
      crossed at `global_step` 262, ~87.3 audio-h. `01-interface-recipe.md`
      §2b's Consequence sets the per-arm budget at ≥1.5× that and never
      under one epoch of 700 h/lang. Settled factors (schedule, stacking)
      leave the grid. Nothing to launch — this produces the rendered configs
      and the arm list the screen starts from.
      *Blocks the whole of week 2 Track A.*
- [x] **LID prompt templates and a `without_language` selection mode**
      (`01-interface-recipe.md` §3): `{lang}` variants of the six `verbatim`
      templates, plus the selection mode that makes LID a controllable
      factor instead of a per-sample coin flip. Prerequisite for the
      screen's prompt runs; additive only, no change to `random`'s current
      behaviour. No GPU.
- [ ] **Quantify FLEURS in or out of the Fondue pool** (`06-fondue.md` §3,
      new row 2026-09-18): per language, how many ASR hours excluding
      `asr_fleurs` would remove, from `data/hours_by_language.csv`, with the
      tail languages called out separately. The campaign excludes FLEURS
      (`--exclude-corpus fleurs`) and FLEURS-24 is both the zero-shot
      benchmark and Fondue's drafted in-training eval set, so this decides
      whether Fondue can keep that benchmark clean. Data only, no GPU.

---

## Week 3 — Mon 2026-09-28 to Sun 2026-10-04

**Gate:** does the recipe carry to 700 h per language and through IFT, and
which audio stack does the campaign build on? *Revised 2026-09-18: the
audio-stack crossing moves here from week 4, and the backbone MA arms move
to week 4, so the six decoders are compared on a chosen stack rather than an
assumed one (`03-audio-stack.md` §0).*

### Track A — GPU
- [ ] **Recipe confirmation at 700 h/lang**: MA then IFT on
      Llama-3.2-1B-Instruct with the chosen recipe, current regime (adapter
      frozen at IFT). Evaluate in-domain and FLEURS-24, run the cascade oracle.
      *Outcome:* the new baseline numbers, replacing the August ones.
- [ ] **Audio-stack MA crossing** (`03-audio-stack.md` §2), moved from week 4:
      4 encoders × 4 adapters at MA-stage cost on Llama-3.2-1B-Instruct, one
      recipe for all sixteen, all at the same frame rate. ~16 × 30 GPU-h,
      all in the queue at once. Q-Former arms wait for its fix.
      *Outcome:* the audio stack — encoder, adapter, frame rate — read on
      FLEURS-24 split into high- and low-resource halves, not on the
      in-domain five alone.

### Track B — preparation
- [ ] **Fondue dry run at full scale on MN5**: config resolves, dataloader
      builds, bucket bins re-measured on the full distribution, startup time,
      exposure audit output, host-RAM trace. Write findings to `06-fondue.md`.
- [ ] **Ladder tier configs** (`05-language-ladder.md` §2): the config builder
      needs a "min(tier, available)" per-language budget; implement and render
      tiers 10/30/100/300/700.
- [ ] **Whisper's window-padding ratio per corpus** (`03-audio-stack.md` §3,
      §4): a fixed-window encoder spends a full 30 s of encoder compute on
      every utterance however short, so its cost per audio hour is
      `30 s / mean utterance duration` and varies by corpus. Measure the mean
      duration per corpus from the shar manifests and report the ratio
      alongside GPU-h per 1,000 audio hours. Needed before the crossing's
      cost axis means anything, since Whisper is one of its four encoders.
      Data only, no GPU.
- [ ] Efficiency instrumentation: log decoder positions per audio second and
      GPU-h per 1,000 audio hours for every arm (from `resolved_config.json`
      and SLURM accounting).

---

## Week 4 — Mon 2026-10-05 to Sun 2026-10-11

**Gate:** none; this is the heavy submission week. Queue everything, in
parallel, in this order of priority: backbone MA arms, IFT for the leading
audio stacks, regime fraction.

### Track A — GPU
- [ ] **MA arms for all six backbones** (`02-backbones.md`), moved from
      week 3, now on the stack chosen at the week-3 gate. ~30 GPU-h each,
      all six in the queue at once.
      *Outcome:* MA-stage WER per backbone, in-domain and FLEURS-24.
- [ ] **IFT for the top three or four audio stacks** from the week-3
      crossing (`03-audio-stack.md` §2), moved from week 5.
- [ ] **Regime half fraction** (`04-regime.md`): 8 IFT runs on
      Llama-3.2-1B-Instruct from the confirmed MA checkpoint. Runs on the
      week-3 confirmation checkpoint, so on the provisional stack if the
      crossing moved it; regime factors are assumed independent of the
      encoder, and that assumption is stated in `04-regime.md`.

### Track B — preparation
- [ ] Ladder eval pipeline: melt-eval configs for FLEURS-24 ASR, FLEURS X→en,
      CV22 test, CoVoST2 X→en where it exists; COMET rescoring environment on
      an internal GPU.

---

## Week 5 — Mon 2026-10-12 to Sun 2026-10-18

**Gate (Sunday 10-18): backbone decision and regime decision.** Criteria in
`02-backbones.md` §4 and `04-regime.md` §4.

### Track A — GPU
- [ ] **IFT for the six backbones** (`02-backbones.md`), moved from week 4.
      Llama ~100 GPU-h each, Qwen and EuroLLM per their measured rates,
      20–46 h wall each. *This week is tight:* if the queue will not turn
      six IFTs around before Sunday, IFT only the three leading backbones on
      the MA-stage signal and carry the rest into week 6, rather than
      shortening the runs.
- [ ] Evaluate the backbone IFTs: in-domain, FLEURS-24, cascade oracle,
      text-ability retention. Second seed on the two leading backbones.
- [ ] **MA runs at 100 and 300 h/lang** for the ratio study (`04-regime.md`
      §6), full schedules each, not checkpoints of the 700 h run. ~20 GPU-h,
      and independent of the regime decision so they can go early.
- [ ] **R9 and R10**, the decoder-frozen regime runs (`04-regime.md` §3),
      if not already queued in week 4.
- [ ] Q-Former arms of the week-3 crossing, if the fix landed late.

### Track B — preparation
- [ ] **Per-task budgets in `build_campaign_config.py`** and the ratio IFT
      renders: ASR at 600, 400 and 0 h/lang with ST fixed at 700
      (`04-regime.md` §6).
- [ ] Ladder configs final; mixed-tier, repetition and Russian probe configs.
- [ ] Raclette config final (batch on the order of one audio hour per step,
      three LR points, big-run topology).
- [ ] Off-boarding plan written (`06-fondue.md` §7): what leaves MN5, where it
      lands, who verifies.

---

## Week 6 — Mon 2026-10-19 to Sun 2026-10-25

**Gate (Sunday 10-25): Fondue configuration frozen.** After this date findings
go into the paper as ablations, not into Fondue.

### Track A — GPU
- [ ] **Winning stack × winning backbone** confirmation, MA plus IFT.
- [ ] **Raclette** (`06-fondue.md` §4): three LR points at ~25K h on the
      big-run batch and topology. Choose LR on loss at matched steps and
      FLEURS-24 dev CER.
- [ ] Seed replicates for the headline backbone pair.
- [ ] **MA:IFT ratio sweep under the winning regime** (`04-regime.md` §6):
      IFT from the 0/100/300/700 h MA points, plus the zero-IFT evaluation
      probe. Queue on Monday; feeds Fondue's "MA data budget and stage
      split" row before the freeze if the queue allows, otherwise the
      default holds.

### Track B — preparation
- [ ] Fill the decision table in `06-fondue.md` §3 and mark it frozen.
- [ ] Fondue launch scripts and resume chain (`--dependency=afterany`) tested
      on a short run.

---

## Week 7 — Mon 2026-10-26 to Sun 2026-11-01

### Track A — GPU
- [ ] **Fondue MA starts** (Monday). ~5 days on 8 nodes at the measured MA
      rate if MA uses the full ASR pool; less if the ladder showed MA
      saturating earlier, or not at all if the ratio sweep says so
      (`06-fondue.md` §3, decision "MA data budget and stage split").
- [ ] **Ratio sweep, runner-up regime**, the two points P0 and P3
      (`04-regime.md` §6): the interaction check. First thing to cut if the
      queue is tight and the winning backbone is Qwen-class.
- [ ] **Ladder tiers 10, 30, 100**: MA plus IFT each, on the winning
      backbone and stack (`05-language-ladder.md`).

### Track B — preparation
- [ ] Copy finished arms' weights, logs and eval outputs off MN5 as they
      complete (rolling, not at the end).
- [ ] Ladder analysis notebook: curve fitting, family pooling.

---

## Week 8 — Mon 2026-11-02 to Sun 2026-11-08

### Track A — GPU
- [ ] **Fondue IFT starts** as soon as MA finishes.
- [ ] **Ladder tiers 300 and 700**, the two mixed-tier runs, the repetition
      probe (two runs), the Russian probe (two runs).

### Track B — preparation
- [ ] First ladder fits with tiers 10–100; check the functional form.
- [ ] Draft the efficiency figure from the audio-stack results.

---

## Weeks 9–10 — Mon 2026-11-09 to Sun 2026-11-22

### Track A — GPU
- [ ] Fondue IFT continues; Llama-class finishes around 11-10, Qwen-class
      would not finish (see the contingency in `06-fondue.md` §6).
- [ ] Seed replicates and any missing ladder points, filling spare capacity.
- [ ] In-training evals of every arm complete on MN5.

### Track B — preparation
- [ ] Off-board everything not yet copied. Sync all offline W&B runs.
- [ ] Ladder fits final; Fondue's intermediate checkpoints evaluated on the
      FLEURS-24 dev subset.

---

## Week 11 — Mon 2026-11-23 to Sun 2026-11-29, and Mon 2026-11-30

- [ ] Nothing new submitted after Wednesday 11-25.
- [ ] Fondue final checkpoint consolidated and copied off MN5; copy verified
      by checksum.
- [ ] Last W&B sync; `arms.tsv` closed out and the board's final entry written.
- **2026-11-30:** allocation ends.

---

## December 2026 – January 2027 — internal clusters

- [ ] melt-eval over every frozen set for every headline arm and Fondue
      checkpoint; COMET rescoring.
- [ ] Text prior vs adaptability figure (`02-backbones.md` §3).
- [ ] Ladder curves with the Fondue point per language, predicted vs achieved.
- [ ] Writing. Paper deadline is more than four months from 2026-09-15.
