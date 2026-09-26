# 06 — Fondue: the big run

Fondue is the final model: everything melts in one pot, the 24 EU languages
are the ingredients. Its pilot is **Raclette**.

**Settled (2026-09-15):** codename; the configuration freezes on
**2026-10-25**; Russian and Ukrainian are in the data; Fondue trains only on
MN5 and must finish, be consolidated and be copied off MN5 by 2026-11-30;
December's evaluation runs on internal GPUs.
**Open:** every row of the decision table in §3, until 2026-10-25.
**Owner:** PI.

## 1. The data

From `data/hours_by_language.csv` (unique train hours, de-duplicated leaves):

| pool | hours | shape |
|---|---|---|
| ASR, all languages in the collection | ~247,000 | English 152K (62%); es 28K, ru 20K, fr 14K, de 11K, it 6.7K, nl 3.0K, pt 2.1K, pl 0.9K; the other 15 EU languages together ≈ 1,100 h |
| ST X→en | ~82,000 | es 26K, ru 18K, fr 12K, de 7.6K, it 6.1K, pt 1.7K, nl 1.3K, pl 0.7K, uk 0.6K, then under 200 h each |
| ST en→X | 5 × 430 | de, et, lv, sl, sv (CoVoST2) |

So the "500K hours" is ~247K ASR + ~82K ST, and the run is English-dominated
by construction. What the tail languages see is decided by the mixture
weights and by repetition, not by the pool.

## 2. Cost and feasibility

Measured throughputs, extrapolated to 8 nodes with `grad_accum 4`, which the
scaling test showed keeps GPU-hours flat while halving wall clock per node
doubling:

| stage, one epoch over the pool | GPU-h | wall on 32 GPUs |
|---|---|---|
| MA over 247K h ASR, adapter only (~1,000 audio-s per wall-s on 8 GPUs) | ~2,000 (arithmetic corrected 2026-09-25; was 3,800) | ~2.6 days |
| MA over the 235K h dry-run pool, frozen Whisper-large-v3 + Qwen3.5-2B, stack 5, `batch_duration 30 x accum 4`, 8 x 4, **measured 6.15 s/step**, 0.896 audio-h per step (Fondue-MA-dryrun, job 46618634) | ~12,100 for the derived 220,492 steps (84% of the pool); ~14,400 for the whole pool — **extrapolated** | 15.7 to 18.7 days |
| same, `batch_duration 120 x accum 1` (same 3,840 s step), **measured 1.57 s/step**, 0.962 audio-h per step, 31.9 GB peak (Fondue-MA-dryrun-bd120, job 46649405) | ~3,100 for the derived 220,492 steps (90% of the pool); ~3,400 for the whole pool — **extrapolated** | 4.0 to 4.4 days |
| IFT over ~330K h, Llama-3.2-1B (5.73 s/step at 3,840 s effective) | ~6,000 | ~8 days |
| IFT, Qwen3.5-2B with gradient checkpointing, 8 nodes x 4 GPUs, DDP, `batch_duration 30`/`grad_accum 20` (effective batch 19,200 audio-s/step) | **measured** 32.3 s/step at 8 nodes (see below) -> ~17,800 | ~23.1 days |

**Qwen, measured 2026-09-15/16 (week 1, `timeline.md`, plus a same-session
follow-up):** four data points across three topologies, all well under the
old "46 h budget implies 25,000+" back-of-envelope, which was never a
measurement -- `46:00:00` was a conservative SLURM wall-time *request*
sized to cover a whole one-shot run, not an observed rate.
- **Clean acc_debug probe** (job 45894977, `IFT-700-qwen35-2b-throughput-debug`,
  30 steps, eval/save off, `--trainer.max_steps 30`): steady state ~31 s/step,
  converged by step ~15 (tqdm's own closing line: `30/30 [28:02<00:00,
  30.82s/it]`; delta between step 10 and step 30 gives 31.5 s/step).
- **Cross-check against the real arm**: the full `IFT-700-qwen35-2b-ins`
  production run (exp_name
  `IFT-700-w2vbF-qwen35_2bInsT-mlpF-bd30-ga20-elr6e6-dlr2e5-lr2e4-s1337-8g`,
  job 45685241) turned out to have **already completed** on MN5 on
  2026-09-12 -- 5,048 steps (one full epoch over the 6,729.85 h mix), closing
  average 27.3 s/step (includes `eval_on_start`, 7 eval rounds and 7
  checkpoint saves, so it is the *upper* bound the "never trust the closing
  average" rule warns about, yet it still reads faster than the clean debug
  probe -- most likely sample variance in which cuts a 30-step debug window
  happens to draw, not a real effect). Wall 38h19m at world_size 8 -> **~306
  GPU-h for the 6,729.85 h arm**. This run and its MA-stage parent
  (`MA-700asr-w2vbF-qwen35_2bInsF-mlpT-bd30-ga20-elr6e6-dlr2e5-lr2e5-s42-8g`,
  done 2026-09-05) were not recorded in `arms.tsv`; the backfill is a week-1
  item in `timeline.md`.
- **8 nodes x 4 GPUs, measured 2026-09-15/16** (job 45902184, `acc_debug`,
  same recipe with `grad_accum 20` held fixed so effective batch scales
  with world_size to 19,200 audio-s/step): steady state ~32.3 s/step,
  essentially flat versus the 2-node rate above (31 s/step) -- GPU-h stays
  ~constant per the "keep `grad_accum` and add nodes" regime
  (`ift700-scaling-measured`), same as Llama's own scaling. This IS
  Fondue's planned topology, so it replaces the earlier same-topology
  extrapolation: 330,000 / 6,729.85 ~= 49x the data, at 5.333 audio-h/step
  (19,200 audio-s / 3600) -> ~61,900 steps -> wall ~555 h (~23.1 days) on
  32 GPUs -> ~17,800 GPU-h, a directly measured-topology number, not an
  extrapolation. About 3x Llama's 6,000 GPU-h Fondue estimate, but far
  under the old unmeasured "25,000+" guess.
- **16 nodes x 4 GPUs, measured 2026-09-16** (job 45902185, `acc_ehpc`,
  same recipe with `grad_accum 20` held fixed, effective batch 38,400
  audio-s/step): started sooner than SLURM's own estimate (17:19, not
  ~22:00), completed cleanly in 28m28s. Steady state ~33 s/step --
  essentially the same as 2 and 8 nodes. **Scaling is flat from 2 to 16
  nodes**: GPU-h for the ~330K h IFT pool comes out to ~18,150 at 16 nodes,
  the same ballpark as ~17,800 at 8 nodes and ~15,000-17,000 extrapolated
  at 2 nodes. Node count above 8 is therefore a pure wall-clock-vs-node-
  count choice for Qwen IFT, not a GPU-h efficiency tradeoff (unlike
  Llama's measured 46-65% penalty for the fixed-effective-batch regime --
  this campaign's Qwen arms use the fixed-`grad_accum` regime instead,
  which is why it scales cleanly here).

The Qwen line was the week-1 planning risk; it is now measured at Fondue's
own planned topology (8 nodes) and its 16-node contingency, both flat with
2 nodes, which settles the number that matters for the decision table in
§3: node count is a wall-clock lever, not a GPU-h risk, for at least this
range. If Qwen wins the backbone comparison, 16 nodes is a straightforward
way to halve Fondue's wall clock at roughly the same total cost -- no LR
retune from Raclette is forced by this measurement alone, Raclette still
owns the actual LR choice, this only says every topology in range is
affordable.

MN5 caps one job at three days, so Fondue is a chain of resumes
(`--dependency=afterany`, same topology, `MELT_GPUS_PER_NODE` pinned).
Checkpoint cadence bounds a failure's cost; keep two checkpoints.

## 3. Decision table — freeze on 2026-10-25

| decision | options | informed by | status |
|---|---|---|---|
| backbone | week-5 winner | `02-backbones.md` §4, plus the Qwen throughput | open |
| audio stack, frame rate | week-6 confirmation | `03-audio-stack.md` | open |
| regime (adapter at IFT, full vs LoRA, MA data, MA length) | week-5 winner | `04-regime.md` | open |
| language set | EU-24 + ru + uk (+ ca?) | Russian probe, `05-language-ladder.md` | ru, uk in (settled); **ca in (PI, 2026-09-25)**: 1,747 h of Common Voice, Iberian family |
| mixture weights | two-tier alpha/beta (the config builder already implements it); values to choose | ladder and repetition probe | **drafted 2026-09-18:** alpha=0.5/beta=0.5 for MA (the paper's own pretraining setting, arXiv:2509.14128 §3.3.1); alpha=0.2/beta=0.5 for IFT (the paper's named fine-tuning-stage alpha). Reasoning and the concrete boost/repetition numbers this implies (e.g. mt's ASR tail at ~53× its 12.3 unique h under MA's weights) are in `fondue-ma-draft.yaml` and `fondue-ift-draft.yaml`. Not frozen — the repetition probe and ladder may move it. |
| epochs for the tail | 1–4× repetition of languages under 100 h | repetition probe | open |
| MA data budget and stage split | full 247K h of ASR in MA, a subset, or — if the decoder-frozen regime wins and the no-MA point matches — a single stage with instructions from the start | the MA:IFT ratio sweep (`04-regime.md` §6, week 6), ladder MA-stage curves, step-0 transition; default if the sweep misses the freeze: full MA, as today | open |
| effective batch and LR | batch ~ one audio hour per step; LR from Raclette | Raclette | open |
| topology | 8 nodes × 4 GPUs; 16 nodes as contingency. **`grad_accum` follows the backbone, not the node count:** the Llama IFT arms run 4, every Qwen arm in `campaign.yaml` runs 20, and the two differ 5× in effective batch. §2's Llama row is a `grad_accum 4` extrapolation and its Qwen rows are `grad_accum 20` measurements; do not read one against the other | scaling test, and the backbone row above | open |
| FLEURS in the training pool | FLEURS **train** in or out; FLEURS **test + validation** stay the benchmark either way | see §3.1 | **IN (PI, 2026-09-25)**: the tail keeps its only data. Transcript-level audit on nyx, 27 locales: 0 normalised sentences shared between FLEURS train and test+validation |
| checkpoint and eval cadence | checkpoints every ~6 h of wall clock; in-training generative eval on a FLEURS-24 dev subset of 100 utterances per language, matching the frozen set | eval cost | open — the **subset itself** is drafted: the frozen `fleurs24-asr-dev` set (melt-eval PR #10, 24 EU languages, 100/lang, plus ru/uk once PR #12/13 redeploy). The **cadence** (eval_steps/save_steps) stays open, gated on the batch/topology rows above. |
| filters | `max_duration 60`, `max_tokens 400`, as the campaign | settled |
| PNC trust rule | train on `custom.pnc_text` only for a leaf whose content-edit rate (added/dropped words plus non-spelling substitutions, `pnc_edit_rate.py`) is below 1%; otherwise the original transcript | measured 2026-09-25 on 44 leaves | **settled (PI, 2026-09-25)**: raw for MLS nl and pl, both People's Speech leaves and FLEURS ga; VoxPopuli outside de/en/es/fr/it is raw |

### 3.1 FLEURS in the training pool — the numbers behind the row

The PI's instruction on 2026-09-20 was to keep FLEURS out of training so it
stays an eval set. Two facts found while implementing it argue the decision
should be taken again, deliberately, rather than inherited:

**The splits are disjoint.** `asr_fleurs` counts the FLEURS **train** split
(`data/README.md`); every frozen benchmark is built from FLEURS **test** and
**validation**. Training on train and evaluating on test is how FLEURS is
meant to be used and does not contaminate anything by itself. Note also that
the campaign's own `--exclude-corpus fleurs` was never a contamination
decision: `build_campaign_config.py` excludes FLEURS because it cannot
supply its ~1% share at 709 h per language in every language ("en and es
fall 0.2–1.5 h short"), and says in terms that FLEURS "stays comparable if
this decision is revisited."

**Excluding it costs almost nothing globally and almost everything in the
tail.** FLEURS is 239.3 h of the 242,102 h ASR pool — 0.10%. But per
language:

| language | FLEURS h | total ASR h | share | net of FLEURS |
|---|---|---|---|---|
| ga (Irish) | 12.1 | 12.7 | **95.3%** | **0.6** |
| mt (Maltese) | 9.9 | 12.3 | **80.5%** | **2.4** |
| sl (Slovenian) | 7.8 | 15.3 | 51.0% | 7.5 |
| el | 10.0 | 27.0 | 37.0% | 17.0 |
| lt | 9.8 | 28.4 | 34.5% | 18.6 |
| et | 7.3 | 24.3 | 30.0% | 17.0 |
| bg | 9.5 | 35.2 | 27.0% | 25.7 |
| da | 7.5 | 29.0 | 25.9% | 21.5 |
| lv | 6.5 | 30.5 | 21.3% | 24.0 |
| hr | 11.8 | 56.0 | 21.1% | 44.2 |

Excluding FLEURS leaves Irish with **36 minutes** of training audio and
Maltese with 2.4 h, below the language ladder's lowest 10 h tier
(`05-language-ladder.md` §1). Three of the 24 EU languages would become
untrainable rather than low-resource, which collides with the coverage
commitment the paper rests on.

**The real risk is utterance overlap, not corpus identity**, and it is
checkable: audit sentence-id overlap between the FLEURS train shards and the
frozen test/validation sets. If it is zero, training on FLEURS train is
sound and the tail keeps its data; if it is not, exclude and say so. That
audit is cheap and turns a worry into a fact.

**Options, PI's call:**

| | consequence |
|---|---|
| exclude FLEURS train everywhere (as instructed) | benchmark unimpeachable; ga/mt/sl effectively have no training data and are reported as zero-shot languages — defensible, but it must be stated as a deliberate choice, not a data accident |
| include FLEURS train, keep test+validation frozen, after a zero-overlap audit | standard practice; the tail keeps its only data; needs the audit and one sentence in the paper |
| exclude only for the languages where it is a trivial share | incoherent — not recommended |

## 4. Raclette — the pilot

The winning configuration transfers in its architectural choices, not in
its optimizer settings. A 1,200 s effective batch would mean 1.5 million
steps over Fondue's pool, so Fondue needs a batch on the order of one audio
hour per step, and the learning rate at that batch is an extrapolation.
Raclette sets it: the Fondue mixture at ~25K h (5–10% of the pool), the
Fondue batch and topology, three LR points, chosen on loss at matched steps
and FLEURS-24 dev CER. Runs in week 6.

**Cost, corrected 2026-09-19.** "About a tenth of Fondue's cost" is the cost
of *one* Raclette arm: 25K h against the ~330K h IFT pool is 7.6%. Raclette
is three LR points, so it is three times that — **about 23% of Fondue's IFT
cost**, which is ~1,400 GPU-h at the Llama rate (18.2 GPU-h per 1K h) and
~4,000 at the measured Qwen rate (53.9 GPU-h per 1K h). Budget it as a
four-figure line item, not a rounding error, and note it scales with the
backbone decision.

**The LR points are not decidable yet (PI, 2026-09-20).** The week-2 draft
proposed 2e-5 / 5e-5 / 1.2e-4 on `optimization.decoder_lr`, scaled from the
campaign's own IFT arms. That is premature, and withdrawn, for three
reasons the PI gave:

1. **We may not be training the decoder at all.** If the decoder-frozen
   regime wins (`04-regime.md` §3, R9/R10) there is no `decoder_lr` to
   pilot and Raclette pilots the adapter LR instead — a different
   parameter, a different scale, a different grid.
2. **LoRA changes the answer.** A LoRA arm's effective LR is not comparable
   to a full fine-tune's, so the grid cannot be chosen before
   `04-regime.md`'s week-5 decision.
3. **The ASR:ST mix has to come first.** Task over-fitting
   ([arXiv:2310.13289](https://arxiv.org/abs/2310.13289), `04-regime.md`
   §6) means an ASR-heavy mixture and a balanced one want different
   learning rates, so allocating ASR against ST hours precedes piloting the
   LR rather than following it.

**Order, therefore:** regime decision (week 5) → ASR:ST allocation and the
MA:IFT ratio sweep (week 6, `04-regime.md` §6) → *then* Raclette's grid,
chosen against whatever parameter is actually trainable. The draft's
arithmetic (anchor at the campaign's measured 2e-5, sqrt-scaled to Fondue's
batch) is kept in `raclette-draft.yaml` as a method, not as values.

**Drafted 2026-09-18** (`raclette-draft.yaml`): the mixture is
`fondue-ift-draft.yaml`'s own weighted pool (alpha=0.2/beta=0.5) with
`total_hours` lowered to ~25,000 (≈7.6% of the ~330K h IFT pool) — the
sampling *proportions* don't change, only the epoch length. Batch and
topology stay open, gated on the backbone (week 5); the draft names the two
already-measured candidates it will slot into (8 nodes → 19,200 audio-s/step,
16 nodes → 38,400, both `grad_accum 20`, both flat with 2 nodes per §2).

Three LR points for `optimization.decoder_lr`, extrapolated from the one
number every completed IFT arm in this campaign has actually used —
**2e-5 at 4,800 audio-s/step**, common to both backbones — via the standard
√(batch-ratio) rule for Adam-family optimizers:

| LR | reasoning |
|---|---|
| 2e-5 | null hypothesis: the larger batch alone doesn't force a change from what's already run |
| 5e-5 | central estimate: inside the √-scaled range for the 8-node (4e-5) to 16-node (5.7e-5) candidates |
| 1.2e-4 | ~2.4× above the central estimate, in case a full-decoder fine-tune at this batch wants steeper-than-√ scaling |

This assumes Raclette pilots the **IFT decoder LR**, not the MA adapter LR
(a separate, already-running, PI-held question in `01-interface-recipe.md`
that this draft does not touch or duplicate) — flagged on `board.md` as
worth a PI confirmation, since if Raclette is meant to cover both, it needs
a fourth axis rather than three points on one parameter.

## 5. Preparation checklist

- [x] Fondue config drafted (week 2, `fondue-ma-draft.yaml` +
      `fondue-ift-draft.yaml`) — the decidable parts only (language
      set, mixture mechanism and weights, filters, eval subset); every
      gated field is an explicit placeholder. Dry-run at full scale on MN5
      (week 3) still owed: dataloader build time, bucket bins on the full
      distribution, exposure audit, host-RAM trace over the first hour, and
      confirming the ~40 unverified corpus paths the draft flags.
- [x] Fondue MA dry run at full scale (week 3, 2026-09-25, job 46618634; numbers on the board and in the §2
      table). Settled by it: bucket bins are measured on the mixture-weighted pool; `total_cuts` must be set;
      People's Speech (6,886 h) is out until its tars are rebuilt without pax headers (lhotse's indexed reader
      cannot read them); VoxPopuli's textless cuts are dropped from the weights (`compute_mix_weights.py
      --skip-cut`); warmup is an integer `warmup_steps` (transformers 5 has no `warmup_ratio`). Open: the
      weights act on cuts, not hours (hours share = weight x mean cut duration); `batch_duration 30` fills 84% of
      its budget and uses 12 of 64 GB.
- [x] Raclette config drafted (week 2, `raclette-draft.yaml`: mixture,
      three LR points and their reasoning). Final (week 5, once batch/
      topology are set) and run (week 6) still owed.
- [ ] Launch and resume chain tested on a short run (week 6).
- [ ] FLEURS-24 dev subset frozen for in-training eval (week 1–2).
- [ ] Off-boarding plan (§7) written by week 5.

## 6. Contingencies

| risk | signal | response |
|---|---|---|
| Qwen throughput too low | week-1 measurement | 16 nodes with an LR point in Raclette, or a Llama-class Fondue |
| queue wait for 8-node jobs | `sbatch --test-only` and observed waits | submit the chain early; keep `time:` at 3 days; smaller node count with longer wall as fallback |
| host memory or dataloader failure at scale | dry run in week 3 | fixed before Raclette or Fondue does not start |
| IFT not finished by 11-25 | progress vs plan | stop at the last checkpoint; the paper reports the hours actually seen |

## 7. Off-boarding before 2026-11-30

Everything on `/gpfs/scratch/epor48` is unreachable after the allocation
ends. Copy, and verify by checksum, to internal storage
(`/mnt/scratch-artemis` or `/mnt/data-artemis`): consolidated weights of
Fondue and of every headline arm, `resolved_config.json` and exposure audits,
training logs, in-training eval tables, offline W&B run directories (then
sync). Do it rolling from week 7, not in the last week. The `mn5transfer`
host sees the same `/gpfs/scratch` and moves ~15–20 MB/s; a 7 GB model is
~7 minutes, a Fondue checkpoint set with optimizer state is hours.
