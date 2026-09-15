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

Budget frame: ~49,700 GPU-h remained on 2026-09-15. Everything below except
Fondue is ~8K GPU-h; Fondue is 10–30K depending on the backbone. Spend the
rest on seed replicates and ladder points; unspent hours vanish on 11-30.

Cost anchors used below (measured unless marked): MA arm at 700 h/lang
≈ 30 GPU-h; IFT arm at 700 h/lang, Llama-1B ≈ 80–100 GPU-h; Qwen-2B IFT
unmeasured at the real topology, expected several times more.

---

## Week 1 — Mon 2026-09-14 to Sun 2026-09-20

**Gate at end of week:** is the failed alignment a recipe problem (LR, batch,
frame rate) or a multilingual-data problem? Decided by LibriSpeech step 0.

### Track A — GPU
- [ ] **No-audio floor.** Text-only NLL of the frozen backbone
      (Llama-3.2-1B-Instruct) on the campaign validation transcripts with the
      MA chat prompt and no audio (or shuffled audio). One short job.
      *Outcome:* a per-language floor to set against the August MA eval loss
      (2.6–3.1). If they match, audio was ignored.
- [ ] **LibriSpeech step 0**, five MA runs (`01-interface-recipe.md` §2):
      L1 current recipe (adapter LR 2e-5, batch 4800 s), L2 LR 2e-4 / 4800 s,
      L3 LR 2e-4 / 1200 s, L4 LR 1e-3 / 1200 s, plus a second seed of the
      best. ~7 GPU-h each.
      *Outcome:* dev-clean/dev-other generative WER per run, step at which the
      loss leaves the plateau, gap to the floor. Success bar: under 10% WER on
      test-clean and a transition inside the epoch.
- [ ] **Qwen3.5-2B IFT throughput** at 2 nodes × 4 GPUs, 20–50 steps,
      gradient checkpointing on, eval and saves off (`acc_debug`).
      *Outcome:* a measured s/step to replace the unverified 46 h budget;
      feeds the Fondue feasibility table in `06-fondue.md`.

### Track B — preparation
- [x] **`stack_factor` for the MLP adapter** (done 2026-09-15, [PR #126](https://github.com/MELT-proj/training/pull/126)) (concatenate k consecutive
      encoder frames; `fc1` input becomes k × encoder width; masks and length
      bookkeeping follow). Unit tests. Tag into `EXP_NAME`.
      *Outcome:* frame rate becomes an axis available to every adapter.
- [ ] **FLEURS-24 ASR frozen sets** in melt-eval: test split for all 24 EU
      languages, plus a small dev subset (≈100 utterances per language) for
      in-training generative eval.
- [ ] **FLEURS X→en ST set construction** started in the preprocessing repo:
      join each language's test audio with the English FLEURS text of the same
      sentence id. Needed for ST on all 24; CoVoST2 covers only ten.
- [ ] **Stage models on MN5** over `mn5transfer`: `Qwen/Qwen3.5-2B-Base`
      (absent from the artemis cache on 2026-09-15), `utter-project/EuroLLM-1.7B`,
      `utter-project/EuroLLM-1.7B-Instruct`. Confirm the EuroLLM ids on the Hub
      first; verify offline loads before any allocation.
- [ ] **Text-prior tool spec** agreed (`02-backbones.md` §3).
- [ ] **Eyeball 20 Qwen hypotheses** from the running IFT arm's eval tables
      for a leaked think block.

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
- [ ] If step 0 was ambiguous, re-run the deciding LibriSpeech pair with the
      second seed before the screen.

### Track B — preparation
- [ ] **Text-prior tool built** in melt-eval (`02-backbones.md` §3): bits per
      character and tokens per word on the FLEURS-24 references, with and
      without the IFT instruction, for ASR transcripts and English ST
      references. Run on all six backbones on an internal GPU.
      *Outcome:* the 6 × 24 prior table, a first ranking of backbones by
      language coverage before any training.
- [ ] **Fondue config drafted** (`06-fondue.md` §3): language set incl. ru/uk,
      two-tier mixture weights (alpha/beta), filters, eval subset. Not frozen.
- [ ] **Raclette config drafted**: same mixture at 25K h, big-run batch,
      three LR points.
- [ ] melt-eval MN5 venv built and a smoke eval run there, so end-of-run
      evaluation can happen on MN5 when queues allow.

---

## Week 3 — Mon 2026-09-28 to Sun 2026-10-04

**Gate:** does the recipe carry to 700 h per language and through IFT? One
confirmation pair answers it. MA-stage WER for all six backbones is the first
backbone signal.

### Track A — GPU
- [ ] **Recipe confirmation at 700 h/lang**: MA then IFT on
      Llama-3.2-1B-Instruct with the chosen recipe, current regime (adapter
      frozen at IFT). Evaluate in-domain and FLEURS-24, run the cascade oracle.
      *Outcome:* the new baseline numbers, replacing the August ones.
- [ ] **MA arms for all six backbones** with the recipe (`02-backbones.md`).
      ~30 GPU-h each, all six in the queue at once.
      *Outcome:* MA-stage WER per backbone, in-domain and FLEURS-24.

### Track B — preparation
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
parallel, in this order of priority: backbone IFTs, audio-stack MA crossing,
regime fraction.

### Track A — GPU
- [ ] **IFT for the six backbones** (`02-backbones.md`). Llama ~100 GPU-h
      each, Qwen and EuroLLM per their measured rates. 20–46 h wall each.
- [ ] **Audio-stack MA crossing** (`03-audio-stack.md`): 4 encoders ×
      4 adapters at MA-stage cost on Llama-3.2-1B-Instruct with the recipe,
      all at the same frame rate. ~16 × 30 GPU-h. Q-Former arms wait for its fix.
- [ ] **Regime half fraction** (`04-regime.md`): 8 IFT runs on
      Llama-3.2-1B-Instruct from the confirmed MA checkpoint.

### Track B — preparation
- [ ] **Q-Former fix** (PI) so its four arms can join the crossing in week 5.
- [ ] MoE adapter branch merged to `main` with its aux-loss logging.
- [ ] Ladder eval pipeline: melt-eval configs for FLEURS-24 ASR, FLEURS X→en,
      CV22 test, CoVoST2 X→en where it exists; COMET rescoring environment on
      an internal GPU.

---

## Week 5 — Mon 2026-10-12 to Sun 2026-10-18

**Gate (Sunday 10-18): backbone decision and regime decision.** Criteria in
`02-backbones.md` §4 and `04-regime.md` §4.

### Track A — GPU
- [ ] Evaluate the six backbone IFTs: in-domain, FLEURS-24, cascade oracle,
      text-ability retention. Second seed on the two leading backbones.
- [ ] **IFT for the top three or four audio stacks** from the MA crossing.
- [ ] Q-Former arms of the crossing, if fixed.

### Track B — preparation
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

### Track B — preparation
- [ ] Fill the decision table in `06-fondue.md` §3 and mark it frozen.
- [ ] Fondue launch scripts and resume chain (`--dependency=afterany`) tested
      on a short run.

---

## Week 7 — Mon 2026-10-26 to Sun 2026-11-01

### Track A — GPU
- [ ] **Fondue MA starts** (Monday). ~5 days on 8 nodes at the measured MA
      rate if MA uses the full ASR pool; less if the ladder showed MA
      saturating earlier (`06-fondue.md` §3, decision "MA data budget").
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
- [ ] Last W&B sync; `arms.tsv` and `00-status.md` closed out.
- **2026-11-30:** allocation ends.

---

## December 2026 – January 2027 — internal clusters

- [ ] melt-eval over every frozen set for every headline arm and Fondue
      checkpoint; COMET rescoring.
- [ ] Text prior vs adaptability figure (`02-backbones.md` §3).
- [ ] Ladder curves with the Fondue point per language, predicted vs achieved.
- [ ] Writing. Paper deadline is more than four months from 2026-09-15.
