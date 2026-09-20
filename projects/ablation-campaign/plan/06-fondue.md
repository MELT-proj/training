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
| MA over 247K h ASR, adapter only (~1,000 audio-s per wall-s on 8 GPUs) | ~3,800 | ~5 days |
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
| language set | EU-24 + ru + uk (+ ca?) | Russian probe, `05-language-ladder.md` | ru, uk in (settled); ca open |
| mixture weights | two-tier alpha/beta (the config builder already implements it); values to choose | ladder and repetition probe | open |
| epochs for the tail | 1–4× repetition of languages under 100 h | repetition probe | open |
| MA data budget and stage split | full 247K h of ASR in MA, a subset, or — if the decoder-frozen regime wins and the no-MA point matches — a single stage with instructions from the start | the MA:IFT ratio sweep (`04-regime.md` §6, week 6), ladder MA-stage curves, step-0 transition; default if the sweep misses the freeze: full MA, as today | open |
| effective batch and LR | batch ~ one audio hour per step; LR from Raclette | Raclette | open |
| topology | 8 nodes × 4 GPUs, `grad_accum 4`; 16 nodes as contingency | scaling test | open |
| checkpoint and eval cadence | checkpoints every ~6 h of wall clock; in-training generative eval on a FLEURS-24 dev subset of ~50 utterances per language | eval cost | open |
| filters | `max_duration 60`, `max_tokens 400`, as the campaign | settled |

## 4. Raclette — the pilot

The winning configuration transfers in its architectural choices, not in
its optimizer settings. A 1,200 s effective batch would mean 1.5 million
steps over Fondue's pool, so Fondue needs a batch on the order of one audio
hour per step, and the learning rate at that batch is an extrapolation.
Raclette sets it: the Fondue mixture at ~25K h (5–10% of the pool), the
Fondue batch and topology, three LR points, chosen on loss at matched steps
and FLEURS-24 dev CER. About a tenth of Fondue's cost; runs in week 6.

## 5. Preparation checklist

- [ ] Fondue config drafted (week 2) and dry-run at full scale on MN5 (week
      3): dataloader build time, bucket bins on the full distribution,
      exposure audit, host-RAM trace over the first hour.
- [ ] Raclette config (week 2), final (week 5), run (week 6).
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
