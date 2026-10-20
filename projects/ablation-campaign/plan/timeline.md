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

**Position on 2026-10-19:** 950 of 1,752 khours used (54%), **~40,000 GPU-h
remaining**. Fondue plus Raclette need ~10K (Llama) to ~26K (Qwen).

**Position on 2026-10-14:** 868 of 1,752 khours used (50%), **~44,000 GPU-h
remaining**. The re-planned weeks below add up to ~19K GPU-h with a
Llama-class Fondue and ~34K with a Qwen-class one. Compute is not the
constraint any more; wall clock and queue waits are.

---

## Re-plan of 2026-10-14

The attention-adapter divergence (weeks 3–4) and the PI's travel put the
campaign 1–2 weeks behind the original weeks 4–6. Nothing is dropped. The
slip is absorbed by these changes, which are the reason weeks 5 onward look
different from the original:

1. **The encoder is decided on the crossing as it stands.** Whisper scores
   0.14–0.16 against mHuBERT's 0.26 and worse (board 2026-10-13), far
   outside the seed noise. The non-Whisper QK-norm cells finish for the
   paper's table only.
2. **The adapter is chosen provisionally and confirmed in parallel.** The
   Whisper cells are within 0.02 of each other on one seed each. Seed
   replicates of the top two run alongside the backbone arms (data seed now
   really changes, `fea165a`); the final adapter is called at the 10-23 gate.
3. **The backbone and regime studies run together**, as week 4 already
   assumed (regime factors independent of the stack).
4. **Runs are reused, not repeated.** The Llama-1B-Instruct backbone MA arm
   is also the recipe confirmation and the regime study's base MA
   checkpoint. The 2,100 h crossing cell with the chosen Whisper adapter is
   the regime study's "long MA, ASR only" checkpoint.
5. **The freeze is split in two.** **MA freeze Thu 10-29** (everything Fondue's
   MA reads: backbone, stack, prompt, language set, mixture, MA budget, MA
   batch and LR, filters, FLEURS in/out). **IFT freeze Wed 11-04** (regime,
   decoder LR from Raclette, ASR:ST mix, IFT mixture, cadence). The IFT
   decisions are made while Fondue's MA runs; none of them changes MA.
6. **Evaluation and short runs go to artemis** (2 jobs per user per QOS):
   selection-metric and IFT evals, COMET, the ratio study's 100 h and 300 h
   MA runs, and testing the bf16 fix.

What it costs: Fondue's IFT starts ~11-05 instead of ~10-31 and ends ~11-15
(Llama, 8 nodes) or ~11-19 (Qwen, 16 nodes, with queue gaps between the
3-day resume segments), against the 11-25 submission cut-off. A Qwen-class
Fondue has ~6 days of buffer, so `06-fondue.md` §6's "stop at the last
checkpoint" becomes a likely outcome, not a remote one. **A second week of
slip is not recoverable** without dropping Raclette or the ratio study.

**PI decisions of 10-14:** all five defaults below are accepted. Encoder
Whisper-large-v3; provisional adapter **MoE** (the Conformer stays open
until its QK-norm behaviour is known; final call on 10-23). `min_lr_scale`
is wired (PR #154), not deleted.

**Defaults** (the weeks below are written on them):
(a) Whisper-large-v3 is the encoder; the provisional adapter is the PI's
pick on 10-14. (b) Backbone MA arms at 700 h/lang, as `02-backbones.md` §1
specifies. (c) The two-part freeze above. (d) The crossing cell is reused as
the regime study's long-MA checkpoint. (e) `min_lr_scale` (wire or delete)
and FLEURS in/out of the Fondue pool are decided by the MA freeze.

Cost anchors used below (measured unless marked): MA arm at 700 h/lang
≈ 30 GPU-h; IFT arm at 700 h/lang, Llama-1B ≈ 80–100 GPU-h; Qwen-2B IFT
unmeasured at the real topology, expected several times more.

**This list is owned by the orchestrator.** Sessions tick the boxes they
complete and nothing else: no adding, deleting, rewording or moving items
between weeks. If work needs to change, post it on `board.md` and it is
folded in from there. A box you could not finish stays unticked, with a
board entry saying why.

---

## Status and proposed cut, 2026-10-19

**PI decisions of 10-19:** the **MA freeze moves to Fri 10-23**. The cut
below is undecided. The PI is weighing a WER-based data filter or
curriculum instead: cuts with WER in [0.1, 0.4] for warmup and the stable
phase, and [0, 0.1] for the decay. No per-cut WER exists in the training
repo yet. If the filter applies to Fondue's MA, it must be specified by
10-23; if it applies only to IFT, by the IFT freeze. The adapter
(MoE or Conformer) is still open; the analysis is on the board, 10-19.

**Where the critical path stands.** The backbone is decidable around Wed
10-21, from the six IFTs (the EuroLLM pair depends on its queue wait). The
adapter is decidable now. Nothing else that week 6 needs before Raclette
has started: the regime study (R1-R10), the ratio and prompt sweeps, the
Fondue dry run, the resume-chain test, re-emitting the Fondue and Raclette
drafts with per-cut weights, and the FLEURS in/out numbers. The X->en ST
eval of the IFTs failed at start: `~/eval` lacks `speech_prompt`. Text
retention and the cascade oracle have no tool for trained checkpoints.

**Proposed cut (PI to decide).** Freeze the IFT regime as the backbone IFTs
ran it: full decoder fine-tune, adapter trainable at 2e-4, 1-pass ASR-only
MA, decoder LR 2e-5, the per-cut 700 h ASR+ST mix. That removes the
dependency in `06` §4 that holds Raclette behind the regime decision.
Dropped from MN5: the regime fraction R1-R10 and its MA checkpoints; the
MA:IFT ratio sweep and its 100/300 h MAs; the MA prompt sweep; the
second-seed backbone IFTs; and the IFT of the top two audio stacks. The
backbone decision rule keeps rules 1 and 4 (CER and feasibility); X->en
joins once the eval is fixed. Text retention and the cascade oracle move to
December. The ladder (`05`) stays as filler beside Fondue.

| | as planned (10-14) | with the cut |
|---|---|---|
| backbone + adapter call | Fri 10-23 | Wed 10-21 or Thu 10-22 |
| MA freeze, Fondue MA submitted | Thu 10-29 | Fri 10-23 (if the dry run passes) |
| Raclette | Tue 10-27 to ~11-03 | ~10-23 to ~10-28, from the winning backbone's MA arm |
| IFT freeze | Wed 11-04 | ~Thu 10-29 |
| Fondue IFT ends (Llama / Qwen) | ~11-15 / ~11-19 | ~11-07 / ~11-11 |
| buffer before 11-25 | ~6-10 days | ~2-2.5 weeks |

---

## Regime restructured, ladder postponed, 2026-10-20

**PI decisions of 10-20.**
- **Regime (`04`, rewritten).** Fondue is a full fine-tune, so the regime
  study now asks one thing: is MA worth running before it, with which
  objective, and for how many hours. Retired: R1-R10 and their MA
  checkpoints, the MA:IFT ratio sweep with its 100/300 h MAs, the ASR+ST MA
  config and the per-task budgets.
- **Language ladder (`05`).** It leaves the path to Raclette and Fondue and
  becomes a paper experiment, run on MN5 capacity Fondue does not need, or
  in December.

**What each tier answers (`04` §3).** Fondue fine-tunes everything, so
MA counts only if its effect survives the fine-tune. Every point therefore
gets the same EU24 fine-tune on the same batches and is compared at equal
fine-tune steps (718 and 2,872).
- **Tier 1 (reuse, now)** answers three questions:
  - *Is MA needed?* No MA against MA, with seed replicates for the noise.
  - *Roughly how much?* CE at 0 / 120 / 360 / 1,000 / 3,700 h.
  - *Which format?* At 3,700 h: ASR only, teacher ST, SIFT (gold ST in MA dropped 10-20).

  Limits: below 1,000 h it covers only en/de/fr/es/it, and every point
  above 1,000 h is built on the same 1,000 h CE start. So it gives a firm
  yes or no and a format ranking, but only a rough number of hours.
- **Tier 2 (only if MA matters)** redoes the hours curve the way Fondue
  would run MA: from scratch, on all 24 languages, for each format still
  in the running, at 120 / 360 / 1,000 h. That gives the hours to reserve.

**Plan (`04` §3, Tier 1).** The self-distillation study's MA starts are
fine-tuned with its step-5 EU24 recipe, from the frozen clone, on matched
batches, and read at steps 718-2,872:
- no MA;
- CE at 120 h, 360 h (two decay branches of GOLDW) and 1,000 h (exists);
- CE + SD-mix at 1,050 h (exists);
- CE, SD-mix and SIFT at 3,700 h;
- replicates of no-MA and 1,000 h.

That is seven new fine-tunes, ~650 GPU-h, submitted Wed 10-21. On top
comes one no-MA IFT on the Fondue stack, once the backbone is called.
Tier 2 (EU24 from scratch, ~500 GPU-h) runs only if an MA effect survives
step 718.

**What moves.**
- The 10-23 MA freeze holds for every row except "MA data budget and stage
  split", which is read **Mon 10-26** (`04` §5).
- Fondue's MA, if any, is submitted 10-26/27. It is WSD, so its length is
  set when the decay is triggered.
- Raclette no longer waits for a ratio sweep.

**Also decided 10-20 (PI):**
- ~~Gold ST inside MA is tested: T6 in Tier 1~~ dropped later on 10-20 (PI: not tested for now).
- The MA prompt sweep becomes a later paper experiment.
- The ASR:ST ratio is not studied; IFT uses Canary's weighting scheme by
  language and corpus.

---

## Blocked / waiting

Carried here 2026-09-18 when `00-status.md` was retired. Findings live on
`board.md`; this list is only what is holding something up.

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
- **bf16 fix merged** (PR #153, `6fc8228`, 10-14): `model.master_dtype`
  loads `model.ckpt` in fp32. IFTs must run from a clone at or after it.
  Still check `optimizer.pt` (8 bytes per trainable parameter, not 4) on the
  first checkpoint of the first IFT.
- **Crossing: all 16 cells finished** (mms1b-conformer last, 10-14; all
  five QK-norm cells stable end to end). 15 scored, mms1b-conformer scoring
  submitted. The seed-43 replicates of whisper-conformer and whisper-moe
  read the same 2,100 h data, so until **they** end (resumes included):
  - ~~Don't merge `yodas3-rest/`~~ hold lifted 10-17: no crossing job reads
    it (the backbone arms use no yodas3). The merge itself is on the PI's word. de, fr and it copies are complete; es restarted
    10-10. Verify all four against nyx before the merge.
- **Frozen clones on MN5:** `~/training-crossing` @ e6cc269,
  `~/training-lr1e3` @ 55e3a6a, `~/training-qf1e3` @ ecccf89,
  `~/training-qkn` @ 93209ca, `~/training-eval` @ ad9171c, `~/eval` (melt-eval
  `selection-metric-sets` @ fa5a961), and the self-distillation line's
  `~/training-sd-sft`, `~/training-sd24-qwen` @ 07f84c2 and
  `~/training-sd24-eurollm` @ 29247d4, `~/training-sd24-sift` @ 89c4413,
  the crossing replicates' `~/training-s43` @ 5912102, the QK-norm
  Conformer cells' `~/training-qkc` @ 715b426, and the backbone arms'
  `~/training-backbones` and `~/training-backbones-eb` @ aa63c11 (until
  the Qwen resumes end). Queued jobs bind
  their clone when they *start*: no sync, branch switch or edit until their
  jobs end. `~/training` is free.
- **Seed replicates before 2026-10-13 never changed the data order**: the
  container wrapper did not forward `MELT_SEED`, so every "seed 43..53" run
  read data order 42 (fixed on `main`, `fea165a`). The screen's noise floors
  (Whisper 0.0045, w2v-BERT 0.035) and step 0b's 0.002 are numerical noise
  only. New replicates must be submitted from a clone at or after `fea165a`.
- **Multi-node start from a checkpoint** failed on the second node (output
  directory already written); fixed on `main` (`b37823b`), not yet run on a
  real multi-node job. Raclette and Fondue's IFT depend on it: the resume
  chain test (week 6) is that first run.
- **Known and accepted, for the paper's methods section:** training targets
  are lowercased (`dataset.py:237`), so "punctuated text" keeps punctuation
  but not case; Llama's chat template stamps the render date into the system
  message (not pinned), so it drifts across days.
- **FLEURS shard duplication on MN5's indexed tree** (78 leaves, af_za..fr_fr).
  The PI has a dry-run-by-default `~/fleurs_dedupe.sh`, not yet run.
  `fleurs24-asr-dev` on MN5 was frozen from the doubled tree; re-freeze it
  after cleanup. Check the selection-metric FLEURS set as well.
- **`optimization.min_lr_scale` is being wired** (PR #154, PI decision
  10-14): every decaying schedule ends at scale x peak; cosine runs become
  `cosine_with_min_lr`. New backbone/regime MA and all IFTs run with it; the
  crossing seed replicates must not (they keep the originals' schedule).
  History: it was a dead config key — set in every
  campaign and SFT config, read by no code, so every cosine run decayed to
  zero rather than to 10% of peak. Comparisons between arms are unaffected;
  the configs misstate what ran. *Measured since:* every cosine arm logged
  end-of-run LR at 1e-11..1e-9, and the cosine Whisper arm went flat over
  its second half where its WSD twin kept improving (board 2026-09-22).
  Its value (0.1 today) is still settled by the MA freeze (10-29).

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
- [x] **Whisper-large-v3's own WER** on the same normalised dev-clean and
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
- [x] **Five-language interface screen — the LR × batch grid** (twelve arms,
      `01-interface-recipe.md` §3, rows `MA-700-screen-*` in `campaign.yaml`,
      added 2026-09-20). Adapter LR {1e-3, 2e-3} × effective batch {1200,
      600, 300} s × encoder {w2v-BERT 2.0, Whisper-large-v3}, all at k=5, on
      `ABL-MA-700-asr.yaml` (one epoch of 700 h/lang), seed 42 throughout.
      ≈ 240 GPU-h. All twelve can go into the queue at once — nothing in the
      grid depends on another arm's result.
      *Outcome:* MA-stage generative WER and CER per language, the loss
      transition step, FLEURS-24 zero-shot CER on the final checkpoint, and
      the best LR/batch corner per encoder — which also answers whether the
      recipe optimum is encoder-invariant, the assumption `03`'s crossing
      rests on.
- [x] **Read the screen** (done 2026-09-24, PR #143). All eighteen
      checkpoints scored on the selection metric (`01` §3, "the WSD grid on
      the selection metric"). Seed shift at one corner per encoder:
      Whisper 0.0045, w2v-BERT 0.035. The PI picked sweep corners per
      encoder (Whisper `lr1e3-b1200`, w2v-BERT `lr2e3-b300`). The
      transition-step column was not formalised; the trajectories are on the
      board (2026-09-23). Original wording below.
      Needs
      three things none of which are derivable from the WER table: the
      seed-43 replicates (jobs 46396669/46396670, submitted 2026-09-23) to
      give a noise floor *at this error regime*; FLEURS-24 zero-shot CER on
      the twelve final checkpoints (**landed 2026-09-23**, jobs
      46398587-46398598, numbers on the board, not yet folded into §3); and the
      transition step per arm (§1's plateau-then-drop definition). **No
      LR/batch corner is called until the replicates land** — Whisper's whole
      grid spans 0.017 and five of its six arms span 0.006, and the only
      noise floor we have at that regime is step 0b's 0.002 from another task
      in another language. Read the replicate gap against the grid's own
      eval-only noise (0.002-0.010 on Whisper, board 2026-09-23), since the
      replicates carry both. `01-interface-recipe.md` §3.
- [x] **Screen follow-ups at the best corner** (closed 2026-09-26: five of
      seven ran; the two prompt runs moved to week 6, PI).
      (seven arms, ≈ 140 GPU-h).
      Blocked on the grid above, because every one of them is defined
      relative to a corner the grid has to find: stacking sweep k=2 and
      k=10; the 2e-5 control; two prompt runs (verbatim, and verbatim+LID);
      two extra seeds. Rows are not in `campaign.yaml` yet — they are added
      once the corner is known.
      *State 2026-09-23:* the 2e-5 control landed (`01` §5, it never needed
      the corner) and the two seeds are the seed-43 replicates under "Read
      the screen". Still owed: k=2, k=10, and the two prompt runs.
      *State 2026-09-26:* k=2 and k=10 landed on both encoders (`01` §3).
      k=10 is clearly worse on both; k=2 is inside the noise of k=5 on both.
      The two prompt runs (verbatim, verbatim+LID) are **moved to week 6**
      (PI, 2026-09-26): they are orthogonal to the recipe and the audio stack,
      and are swept on the winning combination instead.
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
- [x] **Q-Former fix** (**done 2026-09-23, PR #138**), moved up from week 4: the audio-stack crossing
      now runs in week 3, so its four Q-Former arms need the adapter
      instantiable by then. If it slips, the crossing launches without them
      and they join as a late addition at the same recipe.
      *Closed 2026-09-23:* `qformer-smoke-ma` rebased and merged as PR #138.
- [x] **MoE adapter branch merged to `main`** (**done 2026-09-23, PR #139**) with its aux-loss logging,
      moved up from week 4 for the same reason: the MoE is one of the four
      adapters in the week-3 crossing, so it must be on `main` before the
      sixteen arms are rendered.
      *Closed 2026-09-23:* merged as PR #139. Two conflicts in
      `modeling_melt.py` needed decisions, both recorded on the PR: the MLP
      adapter keeps **both** the optional `attention_mask` and main's
      `_stack_frames` call, and `MELTAudioAdapter.forward` keeps the uniform
      call with the `(output, aux_loss)` tuple, because the MoE's
      load-balancing loss needs the mask. **Still open and not a merge
      question:** the MoE has no defined `stack_factor` answer, which §2's
      "one output rate for every adapter" requires.
- [x] **Fondue config drafted** (`06-fondue.md` §3): language set incl. ru/uk,
      two-tier mixture weights (alpha/beta), filters, eval subset. Not frozen.
- [x] **Raclette config drafted**: same mixture at 25K h, big-run batch,
      three LR points.
- [x] melt-eval MN5 venv built and a smoke eval run there, so end-of-run
      evaluation can happen on MN5 when queues allow.
- [x] **Re-derive the screen's budget and render its configs** (done
      2026-09-20). `wsd-50hz-whisper` crossed at `global_step` 262, ~87.3
      audio-h; 1.5× that is 131 h, so §2b's other floor — one epoch of 700 h
      per language — binds, and that is exactly `ABL-MA-700-asr.yaml`
      (`total_hours: 3500.00`). **No new config was needed**; the screen is
      twelve `MA-700-screen-*` rows against the config the campaign already
      trains on, with `grad_accum_steps` set explicitly on every one because
      that config's own 150 × 4 × 8 is the 4800 s August recipe. Cost
      re-derived from ≈65 to ≈380 GPU-h over nineteen runs. See
      `01-interface-recipe.md` §3 and the board entry.
- [x] **LID prompt templates and a `without_language` selection mode**
      (`01-interface-recipe.md` §3): `{lang}` variants of the six `verbatim`
      templates, plus the selection mode that makes LID a controllable
      factor instead of a per-sample coin flip. Prerequisite for the
      screen's prompt runs; additive only, no change to `random`'s current
      behaviour. No GPU.
- [x] **MoE adapter crossing-ready** (added 2026-09-23; blocks the week-3
      crossing render). Verified on `main`: `MELTMoEAdapter` has no
      `stack_factor`, so it runs at 50 Hz against the crossing's common
      10 Hz, and its load-balancing loss is folded into the training loss
      but never logged. Two parts, one PR: (a) reach the common 10 Hz by
      stacking frames before the router, same `stack_factor` semantics and
      tag as the MLP (design confirmed by the PI 2026-09-23); (b) log the
      aux loss on its own, router entropy, and per-language expert usage
      (`03-audio-stack.md` §0.1). Unit tests. If it slips past the render,
      the crossing launches without the MoE arms.
- [ ] **Whisper's window-padding ratio per corpus** (`03-audio-stack.md` §3,
      §4), **moved up from week 3 on 2026-09-20**: a fixed-window encoder
      spends a full 30 s of encoder compute on every utterance however short,
      so its cost per audio hour is `30 s / mean utterance duration` and
      varies by corpus. Measure the mean duration per corpus from the shar
      manifests and report the ratio alongside GPU-h per 1,000 audio hours.
      Still needed before the week-3 crossing's cost axis means anything, but
      now also for this week: six Whisper arms joined the screen's grid, and
      their `batch_duration` was set to 75 on an *estimated* ~6x inflation
      (~5 s mean utterance) against the ~2.5x `MA-librispeech-w` saw on
      LibriSpeech. If the real ratio is materially higher, the 1200 s Whisper
      arms are the ones that will OOM. Data only, no GPU.
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

*Re-plan 2026-10-14: the open boxes of weeks 1-4 are carried into the weeks
below, where they are ticked; they stay here unticked as the record.*

### Track A — GPU
- [ ] **Recipe confirmation at 700 h/lang**: MA then IFT on
      Llama-3.2-1B-Instruct with the chosen recipe, current regime (adapter
      frozen at IFT). Evaluate in-domain and FLEURS-24, run the cascade oracle.
      *Outcome:* the new baseline numbers, replacing the August ones.
- [ ] **Audio-stack MA crossing** (`03-audio-stack.md` §2), moved from week 4:
      4 encoders × 4 adapters at MA-stage cost on Llama-3.2-1B-Instruct, one
      recipe for all sixteen, all at the same frame rate, **2,100 h per
      language, 10,500 h per arm** (PI, 2026-10-02, replacing five epochs;
      `03` §1b). Explicit `max_steps`, 1.25 × the
      estimate (~80% of each step's nominal audio is real), ≈ 2,800 GPU-h for
      sixteen, 12-43 h wall per arm, all in the queue at once. Rows rendered
      (726ca28). **All 16 arms submitted
      2026-10-08** (12 at d91c0e1, the 4 Conformer cells at fd347d6, all with
      warmup), ~2,830 GPU-h, 25 nodes if all start at once. **Three cells
      re-run at adapter LR 1e-3** (w2vb-conformer, mms1b-conformer,
      mms1b-qformer; PI, 2026-10-09). whisper-conformer was already at 1e-3.
      The 1e-3 re-runs did not fix the attention cells; **the Q-Former column
      and mms1b-conformer re-ran with QK-norm at 2e-3** (PI, 2026-10-12,
      `~/training-qkn` @ 93209ca; #149 merged as ad9171c). **15 of 16 have
      finished**; mms1b-conformer ends ~10-15. Eleven are scored (board
      2026-10-13): Whisper 0.136-0.157, mHuBERT 0.26-0.39, MMS-1b 0.43-0.44,
      w2v-BERT 0.52-0.92. Report each cell's adapter LR and `qk_norm`: the
      Conformer column is not one recipe.
      *Outcome:* the audio stack — encoder, adapter, frame rate — ranked by
      the selection metric (`selection-metric/README.md`, PI 2026-09-23),
      which replaces the high/low-resource FLEURS split named here before.

### Track B — preparation
- [x] **Crossing prep** (added 2026-09-26; the crossing's critical path).
      Make all sixteen encoder × adapter cells renderable and proven before
      the rows go in: `plan_arm.py` support for the Conformer's route to
      10 Hz and for the MMS/mHuBERT encoders (tags, input length, MMS
      spec-augment pinned off); an offline load of MMS-1b and mHuBERT-147 on
      MN5; a short `acc_debug` smoke per new encoder and per non-MLP adapter;
      one selection-metric eval on a smoke checkpoint for each untested
      encoder; then sixteen `campaign.yaml` rows at the PI's recipe.
      Cold-agent prompt handed to the PI 2026-09-26.
- [x] **Attention-adapter divergence: diagnose and fix** (added 2026-09-29;
      reopened and widened 2026-10-09, board 4182f9a). Warmup is not enough:
      with it, w2vb-conformer diverged at ~15.6k steps and mms1b-conformer
      at ~6.2k (`grad_norm` jumps to 1e11, then nan, loss 0).
      mhubert-conformer and whisper-conformer (LR 1e-3) are alive. **The
      Q-Former diverges too**: all 4 crossing runs, and the earlier 700 h
      ones, have unbounded `grad_norm` after the LR peak. w2vb and mms1b
      don't learn (WER 1.1-1.3), mhubert finished unusable, whisper learns
      (WER ~0.20). Every MLP and MoE cell is stable. The common factor is an
      attention adapter at adapter LR 2e-3. Candidate fixes are QK-norm on
      both attention adapters (#149, plus a new Q-Former patch, validated
      first), a lower LR, or dropping both columns. **PI decision 2026-10-09:
      test the LR first** (board b483fc6). w2vb-conformer, mms1b-conformer and
      mms1b-qformer re-run at adapter LR 1e-3 (55e3a6a), same seed, data,
      steps and schedule. QK-norm is held back (#149, plus a local Q-Former
      patch, since added to #149). The Q-Former column was widened to 1e-3 on
      2026-10-10. Still open: re-run mhubert-conformer (healthy at 2e-3) at
      1e-3, or keep it. **Before #149 can merge** it needs a
      `model.adapter.qk_norm` switch, default off: as it stands the patch
      makes old Conformer/Q-Former checkpoints, the crossing's included,
      load with strict=False and compute a different function. Note
      that the crossing is no longer one recipe for all sixteen: report each
      cell's LR.
      *Closed 2026-10-14:* LR 1e-3 fixed no Q-Former cell and one Conformer
      cell blew up at it (board 2026-10-11). PI, 2026-10-12: QK-norm at the
      recipe LR, no further LR tests. #149 merged with `model.adapter.qk_norm`
      off by default (ad9171c). The five QK-norm cells passed every earlier
      failure point without a `grad_norm` above 100. The trainer still has no
      non-finite-gradient abort; the worker's watcher covers it.
- [ ] **Re-check every hour-weighted config for per-cut mux weights**
      (added 2026-10-08, after #151). The ladder renders (`05`), the IFT
      config and the Fondue/raclette drafts were weighted as hour shares
      under the old `compute_mix_weights.py` default. Re-emitting them
      changes their weights, and their per-language hours may be off the
      same way the crossing's were. Re-render, predict the hours share as
      weight × mean cut duration, and confirm before anyone reuses the
      numbers.
- [ ] **Fondue dry run at full scale on MN5**: config resolves, dataloader
      builds, bucket bins re-measured on the full distribution, startup time,
      exposure audit output, host-RAM trace. Write findings to `06-fondue.md`.
- [ ] **Ladder tier configs** (`05-language-ladder.md` §2): the config builder
      needs a "min(tier, available)" per-language budget; implement and render
      tiers 10/30/100/300/700.
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
- [ ] ~~**Regime half fraction**~~ retired 10-20 (PI: `04` restructured). (`04-regime.md`): 8 IFT runs on
      Llama-3.2-1B-Instruct from the confirmed MA checkpoint. Runs on the
      week-3 confirmation checkpoint, so on the provisional stack if the
      crossing moved it; regime factors are assumed independent of the
      encoder, and that assumption is stated in `04-regime.md`.

### Track B — preparation
- [ ] ~~**Whisper size ladder prep**~~ dropped 10-15 (PI). (`03` §2b): stage whisper-tiny/-base/
      -small/-medium in MN5's hf_cache via `mn5transfer`; add tiny and base
      to `plan_arm.py` `ENCODER_WINDOW_FRAMES`; one offline-load `acc_debug`
      smoke per size; render the four rows once the crossing names its
      adapter.
- [ ] Ladder eval pipeline: melt-eval configs for FLEURS-24 ASR, FLEURS X→en,
      CV22 test, CoVoST2 X→en where it exists; COMET rescoring environment on
      an internal GPU.

---

## Week 5 — Mon 2026-10-12 to Sun 2026-10-18 (re-planned 2026-10-14)

**Decisions this week:** encoder and provisional adapter (Wed 10-14, PI).
No gate on Sunday: the backbone and regime gates move to Fri 10-23.

### Track A — GPU
- [x] **Score the five QK-norm cells** (done 10-15): Q-Former whisper 0.177,
      mhubert 0.423, w2vb 0.827, mms1b 0.912; mms1b-conformer 0.254, the
      best MMS-1b cell.
- [x] **QK-norm Conformer on Whisper, mHuBERT, w2v-BERT** at 2e-3 (PI
      10-15), done and scored 10-17: Whisper 0.151, mHuBERT 0.376 (0.385
      without), w2v-BERT 0.629 (0.922 without); all four QK-norm Conformers
      trained clean. Its rank against MLP/MoE flips with the encoder.
- [x] **Stack call, provisional** (PI, Wed 10-14): Whisper-large-v3 + **MoE**
      (MLP 0.157, MoE 0.143, Conformer at 1e-3 0.136, Q-Former with QK-norm
      0.177). Conformer left open for 10-23. Recorded in `03` §2.
- [x] **Seed replicates of the top two Whisper cells** at 2,100 h/lang
      from a clone at or after `fea165a` so the data order really changes.
      Done and scored Fri 10-16 (seed 43, pre-#154 schedule): Conformer
      0.1526 (seed 42: 0.1364), MoE 0.1596 (0.1429); 286 GPU-h for the pair.
      A second seed costs both ~0.016 and keeps Conformer 0.007 ahead, so on
      Whisper the metric does not separate Conformer, MoE and MLP. They give the crossing its
      first real noise floor and settle the adapter on 10-23.
- [x] **MA arms for all six backbones** (`02-backbones.md`), carried from
      week 4: 700 h/lang on Whisper + the provisional adapter, all six in the
      queue at once on Thu 10-15. The `llama1b-ins` arm is also the recipe
      confirmation (week 3) and the regime study's base MA checkpoint.
      *Outcome:* MA-stage WER per backbone, in-domain and FLEURS-24.
      **Submitted 10-15** (Whisper + MoE, `~/training-backbones`). 10-16:
      both Llama arms timed out 360 steps short, resumes queued; EuroLLM-Base
      restarted after a tokenizer fix (#155); Qwen (~2.5 s/step) needs a
      ~11 h resume each; EuroLLM-Instruct done. 10-17: five arms finished
      and their scoring is submitted; **EuroLLM-Base timed out at 47.7k with
      ~90% runaway generations, cause not found** (PI: diagnose first or
      resume as is).
      10-19, scored (selection metric, lower is better): Qwen-ins .122,
      Qwen-base .131, EuroLLM-ins .141, Llama-ins .148, Llama-base 1.0
      (runaway, ~85% of clips); EuroLLM-Base not scored (runaway). Both base
      runaways are frozen-decoder template artefacts; IFT repairs them.
- [ ] ~~**Regime MA checkpoints**~~ retired 10-20 (PI: `04` restructured). The three the backbone
      arm does not give: ASR+ST at 700 h/lang and at 2,100 h/lang (new), and
      ASR at 2,100 h/lang (the crossing cell with the provisional adapter,
      reused). Submit Thu 10-15.
- [ ] ~~**MA runs at 100 and 300 h/lang**~~ retired 10-20 (PI: `04` restructured). For the ratio study (`04-regime.md`
      §6), full schedules each. On artemis (H100 or H200), Thu 10-15.
- [ ] **First IFTs**, as soon as the bf16 fix is on `main` and their MA
      checkpoints land (from Sat 10-17): the six backbone IFTs and the ten
      regime runs R1-R10, all on Llama-3.2-1B-Instruct for the regime study.
      *10-20:* the R1-R10 part is retired (PI: `04` restructured).
      Check `optimizer.pt` size on the first checkpoint of the first one.
      If the queue will not turn six backbone IFTs around by 10-22, IFT the
      three leading backbones on the MA-stage signal first; never shorten
      the runs.
- [ ] ~~**Whisper size ladder**~~ dropped 10-15 (PI: not essential now, frees < ~750 GPU-h). (`03-audio-stack.md` §2b), tiny/base/small/
      medium with the provisional adapter, on the crossing recipe, one seed
      each (< ~750 GPU-h). Last in priority this week; submit Fri 10-16 if
      its prep is done. If it lands by 10-29 it can still inform Fondue's
      encoder cost.

### Track B — preparation
- [x] **bf16 fix on `main`** (merged 10-14, PR #153): `dtype=float32` at
      `from_pretrained` on the `model.ckpt` path, tested on artemis. Blocks
      every IFT.
- [ ] **Re-check every hour-weighted config for per-cut mux weights**,
      carried from week 3: the IFT config first (needed by Sat 10-17), then
      the ladder renders and the Fondue/raclette drafts (by 10-23).
- [ ] ~~**MA config on ASR+ST**~~ retired 10-20 (PI: `04` restructured). For the regime study's factor C, at 700 and
      2,100 h/lang. Needed Thu 10-15.
- [ ] ~~**Whisper size ladder prep**~~ dropped 10-15. Carried from week 4: stage whisper-tiny/
      -base/-small/-medium on MN5 via `mn5transfer`; add tiny and base to
      `plan_arm.py` `ENCODER_WINDOW_FRAMES`; one offline-load smoke per size.
- [ ] **Evaluation on artemis works with current code**: melt-eval generation
      with the training checkout first on `PYTHONPATH` (the shared venv's
      `melt` is stale), the selection metric and the IFT eval battery
      (in-domain, FLEURS-24, X→en, cascade oracle, text retention). From here
      on IFT evals run on artemis by default.
- [ ] **Eyeball 20 Qwen hypotheses** for a leaked think block, carried from
      week 1. Needed only if Qwen leads after the backbone MAs.

---

## Week 6 — Mon 2026-10-19 to Sun 2026-10-25

**Gate (Fri 10-23): backbone, final adapter, and the MA freeze
(moved from 10-29 by the PI on 10-19), except the MA budget and objective,
read Mon 10-26 from `04` Tier 1 (PI, 10-20).** Criteria in
`02-backbones.md` §4 and `04-regime.md` §4; the adapter on the replicate pair
plus the Whisper QK-norm Conformer cell.

### Track A — GPU
- [ ] Remaining backbone and regime IFTs in the queue by Mon 10-19.
      *10-19:* all six backbone IFTs submitted 10-18 (Whisper + MoE,
      `~/training-ift` @ d0c8151). Llama-base and -ins done and scored:
      .061 and .085. Qwen pair 2,296/6,310, ETA Tue 10-20 ~20:00Z.
      EuroLLM pair OOM'd at 4,040 (fp32 logits). Both were resumed from
      3,608 at bd 50 x ga 24 and are pending, needing ~20 h once started.
      IFT repaired EuroLLM-Base's ChatML rows (en WER .113 at step 902). **No
      regime IFT exists:** none of R1-R10, nor their MA checkpoints, was
      prepared or submitted.
- [ ] **Evaluate the backbone and regime IFTs** on artemis as they land:
      in-domain, FLEURS-24, X→en, cascade oracle, text-ability retention.
- [ ] **Second seed on the two leading backbones** (IFT), submitted on the
      MA-stage signal by Tue 10-20 so it lands before the gate.
- [ ] **IFT for the top two audio stacks** (the replicate pair), carried from
      week 4: does the MA ranking survive IFT. ~200 GPU-h.
- [ ] **MA utility study, Tier 1** (`04-regime.md` §3)
      on MN5 from `~/training-ma-utility` (2a26f33 + PR #159): fine-tunes from no MA (to the
      end), CE 120 h and 360 h, CE24, SD24 and SD-SIFT24, plus replicates of
      no-MA and A (to step 2,872). ~644 GPU-h. **Submitted 10-20 18:50Z** (PI's go): T0 (42 h wall), T0', A', T3-T5 (11 h walls); T1 and T2 follow their decay branches.
      T6 dropped (PI 10-20). Decay branches on hades (T1 running 10-20). Scored on artemis as each
      checkpoint lands. Read Mon 10-26.
- [ ] **No-MA IFT on the Fondue stack** (`04` §3): the winning backbone's IFT
      config without its MA checkpoint, submitted when the backbone is called.
- [ ] ~~**MA:IFT ratio sweep under the winning regime**~~ retired 10-20 (PI: `04` restructured). (`04-regime.md` §6),
      submitted Sat 10-24: IFT from the 0/100/300/700 h MA points, plus the
      zero-IFT probe. Read Tue 10-27.
- [ ] ~~**MA prompt sweep on the winning combination**~~ postponed 10-20 (PI: a later paper experiment). (verbatim, verbatim+LID;
      `01-interface-recipe.md` §3), submitted Sat 10-24. MA at 700 h is
      ~13 h wall; read by Tue 10-27, before the MA freeze.
- [ ] ~~**Ladder MA tiers 10, 30, 100**~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue). On the winning backbone and stack,
      submitted Sat 10-24 (they need only the backbone and the stack).
- [ ] **Winning stack × winning backbone confirmation**, MA plus IFT. If the
      adapter called on 10-23 is the provisional one, the winning backbone's
      own arms are the confirmation and this box is ticked on the gate.
      Otherwise submit it Sat 10-24 so it lands before the MA freeze.

### Track B — preparation
- [ ] **Fondue dry run at full scale on MN5**, carried from week 3, by Fri
      10-23: config resolves, dataloader builds, bucket bins on the full
      distribution, startup time, exposure audit, host-RAM trace. A failure
      here moves the MA freeze.
- [ ] **MA throughput at 8 vs 16 nodes** on the Fondue MA config (short
      runs). If flat, Fondue's MA runs on 16 nodes. Prior from the crossing
      (board 10-15): no wall-time gain from 1->2 or 2->4 nodes, but with a
      different adapter per node count, so not a clean test; something
      shared (data path suspected) may set the pace.
- [ ] **Resume chain tested on a short 2-node run from a checkpoint**
      (`--dependency=afterany`): also the first real test of `b37823b`.
      Needed before Raclette (10-27).
- [ ] **Quantify FLEURS in or out of the Fondue pool**, carried from week 2,
      so the PI can make that call by 10-29.
- [ ] **Tier-1 starts ready by Wed 10-21** (`04` §3, checklist in §6): the two GOLDW decay
      branches on artemis `hades`, 2 GPUs as GOLDW (check the resume takes the `max_steps` /
      `num_decay_steps` overrides), `launch_sft.sh` building a no-MA start,
      every start on MN5 via `mn5transfer`, and the step-5 scoring watcher
      extended to the new runs.
- [ ] ~~**Per-task budgets in `build_campaign_config.py`**~~ retired 10-20 (PI: `04` restructured). And the ratio IFT
      renders (ASR at 600/400/0 h/lang, ST fixed at 700; `04-regime.md` §6).
      Needed by Sat 10-24.
- [ ] **melt-eval: score the LID prompt with and without the tag** at
      inference. Needed by 10-26.
- [ ] ~~**Ladder tier configs**~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue). (min(tier, available) per language), carried
      from week 3; mixed-tier, repetition and Russian probe configs.
- [ ] Raclette config final once the backbone and regime are known (batch on
      the order of one audio hour per step, three LR points, topology).

---

## Week 7 — Mon 2026-10-26 to Sun 2026-11-01

**Gate (moved to Fri 10-23 by the PI on 10-19): MA freeze.** The MA rows of `06-fondue.md` §3 are filled
and frozen: backbone, stack, prompt, language set, mixture, MA data budget
and stage split, MA batch and LR, filters, FLEURS in/out, `min_lr_scale`.

### Track A — GPU
- [ ] **Raclette** (`06-fondue.md` §4). *10-20:* it no longer waits for a ratio sweep (retired). Submitted Tue 10-27 after the ratio
      sweep is read: three LR points at ~25K h on the Fondue IFT batch and
      topology (24 nodes if all three run at once). Read by Tue 11-03.
- [ ] **Fondue MA submitted Thu 10-29** (*10-20:* Mon 10-26 to Tue 10-27, if `04` keeps MA; its length is set by the decay trigger): ~5 days on 8 nodes, less on 16 if
      the scaling check was flat. Chain of 3-day segments.
- [ ] **MA utility study, Tier 2**, only if Tier 1 shows an effect that
      survives step 718 (`04` §3): CE and SD-mix from scratch on EU24 at
      120 / 360 / 1,000 h, plus their fine-tunes. ~500 GPU-h.
- [ ] ~~**Ladder IFT tiers 10, 30, 100**~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue).
- [ ] ~~**Ratio sweep, runner-up regime**~~ retired 10-20 (PI: `04` restructured). (P0 and P3, `04-regime.md` §6). First
      to drop if the queue is tight and the backbone is Qwen-class.

### Track B — preparation
- [ ] Fill the MA rows of `06-fondue.md` §3 and mark them frozen (10-29).
- [ ] **Off-boarding plan** (`06-fondue.md` §7), carried from week 5: what
      leaves MN5, where it lands, who verifies. Copying starts this week,
      rolling, as arms finish.
- [ ] ~~Ladder analysis notebook: curve fitting, family pooling.~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue).

---

## Week 8 — Mon 2026-11-02 to Sun 2026-11-08

**Gate (Wed 11-04): IFT freeze.** The IFT rows of `06-fondue.md` §3: regime,
decoder LR (Raclette), ASR:ST allocation, IFT mixture, checkpoint and eval
cadence, topology.

### Track A — GPU
- [ ] **Fondue IFT starts** when MA ends (~11-05). Llama-class: ~8 days on 8
      nodes. Qwen-class: 16 nodes from the start (~11.5 days).
- [ ] ~~**Ladder tiers 300 and 700**~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue). The two mixed-tier runs, the repetition
      probe (two runs), the Russian probe (two runs).

### Track B — preparation
- [ ] Fill the IFT rows of `06-fondue.md` §3 and mark them frozen (11-04).
- [ ] ~~First ladder fits with tiers 10-100; check the functional form.~~ postponed 10-20 (PI: the ladder is a paper experiment, off the path to Fondue).
- [ ] Draft the efficiency figure from the audio-stack results, with the
      Whisper window-padding ratio per corpus (carried from week 2) and the
      efficiency instrumentation (week 3, PR #135).

---

## Weeks 9–10 — Mon 2026-11-09 to Sun 2026-11-22

### Track A — GPU
- [ ] Fondue IFT continues: Llama-class ends ~11-15, Qwen-class ~11-19
      (resume queue gaps included). If it is not done by 11-23, apply
      `06-fondue.md` §6: stop at the last checkpoint.
- [ ] Seed replicates and ladder points (the ladder runs here on spare capacity, PI 10-20), filling spare capacity
      without delaying a Fondue resume.
- [ ] In-training evals of every arm complete on MN5.

### Track B — preparation
- [ ] Off-board everything not yet copied. Sync all offline W&B runs.
- [ ] Ladder fits (paper, PI 10-20); Fondue's intermediate checkpoints evaluated on the
      FLEURS-24 dev subset (artemis).

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
