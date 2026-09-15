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
| IFT, Qwen3.5-2B with gradient checkpointing, 2 nodes x 4 GPUs, DDP, `batch_duration 30`/`grad_accum 20` (effective batch 4,800 audio-s/step) | measured 27.3-31 s/step (see below); ~15,000-17,000 at 2-node topology, **extrapolated**, no node-scaling correction | ~78 days on 2 nodes, **extrapolated** |

**Qwen, measured 2026-09-15 (week 1, `timeline.md`):** two independent
numbers at the same topology (2 nodes x 4 GPUs = world_size 8), both well
under the old "46 h budget implies 25,000+" back-of-envelope, which was
never a measurement -- `46:00:00` was a conservative SLURM wall-time
*request* sized to cover a whole one-shot run, not an observed rate.
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
  done 2026-09-05) were not recorded in `arms.tsv` or `00-status.md` -- see
  the board entry.
- **Extrapolation to Fondue's ~330K h IFT pool** (330,000 / 6,729.85 ~= 49x),
  holding the *same 2-node topology* (no correction for Fondue's actual
  8-node x 4-GPU topology): ~15,000 GPU-h using the 27.3 s/step production
  rate, ~17,000 using the 31 s/step clean rate. This replaces "25,000+" as
  a floor, but it is **not** the Fondue-topology number: Llama's own
  6,000 GPU-h Fondue estimate already bakes in a measured 46-65% node-scaling
  penalty for trading `grad_accum` for more nodes
  (`ift700-scaling-measured`); no equivalent measurement exists for Qwen at
  8 nodes yet. A same-shaped follow-up (this same acc_debug method, at 8
  nodes) is needed before the Qwen Fondue-topology GPU-h can be trusted.

The Qwen line was the week-1 planning risk; it is now measured, though the
node-scaling question above is still open. If Qwen wins the backbone
comparison, the options are still more nodes (16 nodes at `grad_accum 4`
doubles the effective batch again and needs its own LR point in Raclette),
a smaller data budget, or a Llama-class Fondue with Qwen reported at
campaign scale -- this measurement narrows how urgent that choice is but
does not settle it.

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
| MA data budget | full 247K h vs a subset where MA-stage WER saturates | ladder MA-stage curves, step-0 transition | open |
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
