# 08 — Raclette: were Fondue's learning rates right?

2026-11-09, stage 1 (plan, no GPU spend). Nothing is submitted, and nothing will be
until the PI says go. **Supersedes `06-fondue.md` §4 and `raclette-draft.yaml`**, which
assumed Raclette would set Fondue's LR before Fondue started. Fondue is queued at its
current LRs and does not wait.

Labels: **measured** (read from a log, a file or SLURM today), **derived** (arithmetic on
measured numbers), **estimate** (extrapolated or assumed).

## 0. Three things to know before reading on

1. **Raclette is a slow instrument for this question.** Reading from MN5 on 11-09: Fondue's
   first job (48754959, 16 nodes) has been pending 16 h with no start estimate;
   `sbatch --test-only` puts a *new* job of any shape (1, 2, 4, 8, 16 nodes) at
   **11-17 18:55** (estimate, backfill, unreliable). History is kinder: the two 16-node
   Fondue jobs of 11-05 waited 26 h and 61 h, and this user's `acc_ehpc` jobs since 10-01
   waited a median 8.0 h (2-4 nodes, n=40) and 15.5 h (8-32 nodes, n=4; max 61 h). The
   earlier plans assumed ~12 h. Plan for 1-2.5 days, carry 8 days as the bad case.
2. **A config change cannot move the LR of a resumed run.** Measured on CPU with
   transformers 5.16.1 (`AdamW` + `warmup_stable_decay`, save, rebuild with 2x LRs, load):
   the optimizer's `lr`/`initial_lr` and the scheduler's `base_lrs` come back from the
   checkpoint, and the new values are discarded. Every Fondue job after the first, and the
   anneal, therefore inherits 6e-6 / 2e-4 / 2e-5 (times the schedule) from the checkpoint.
   Changing it needs an offline patch of `optimizer.pt` and `scheduler.pt` in a *copy* of a
   checkpoint, plus replacement chain jobs from a new clone (§1). Untested on a real
   Fondue checkpoint.
3. **A full restart of Fondue is not realistically available after Raclette reads.** See §1.

## 1. What the result can change

Outcomes, from the grid in §3 (x = multiplier on all three LRs, "x1" = Fondue's own):

| outcome at the pre-registered reads (§5) | what it triggers |
|---|---|
| **A.** x1 is within noise of the best arm | Nothing. Fondue's LRs are "not distinguishable from the best of a 4-point grid at 5% of the horizon". Paper: the grid and the noise. |
| **B.** a higher arm beats x1 by more than noise, less than the action margin | Nothing for the run. Paper only. |
| **C.** a higher arm beats x1 by the action margin, consistently | Change the LR for the rest of Fondue if the dates allow (table below); else paper only. |
| **D.** a lower arm (x0.5) beats x1 by the action margin | Same as C. A decrease at a resume is the safer change. |
| **E.** x2 or x4 diverges or shows grad-norm blow-ups | Tells us the headroom above x1 is limited. Not a restart trigger. The 336-step warmup (§4) is ~10x steeper than Fondue's, so divergence at x4 is not evidence that Fondue at x4 would diverge. |
| **F.** x1 itself looks unhealthy against Fondue's live log (loss, grad norm, router stats) | Alarm to the PI at once, whatever the grid says. |

**Latest date per action.** Two Fondue calendars (derived; 8 s/step per the brief, +8% for
evals and checkpoints, 24 min start-up per job, 3-day walls, chain gaps at the measured
median 16 h / worst 61 h). A: Fondue's job 1 starts 11-11 ~04:00 (the 61 h precedent).
B: it starts at the `--test-only` date, 11-17 19:00.

| | job 1 ends (~30K steps) | job 2 reaches step 55,167 | anneal ends |
|---|---|---|---|
| A, median gaps | Sat 11-14 | Tue 11-17 | **Thu 11-19** |
| A, worst gaps | Sat 11-14 | Fri 11-19 | Mon 11-23 |
| B, median gaps | Fri 11-20 | Wed 11-23 | Thu 11-26 |
| B, worst gaps | Fri 11-20 | Wed 11-25 | **Sun 11-29: past the copy-off margin** |

At 5.5 s/step (the 2-node E24 smoke) scenario A ends about 11-17. Neither speed is
measured at 16 nodes.

| action | what it needs | latest date (A, median) | can Raclette serve it? |
|---|---|---|---|
| Full restart, new LRs | 4.8-6.9 d of steps + first wait + 2-3 chain gaps, done by 11-29 12:00 less 0.5 d for the copy-off | submit by ~11-20 (median queue), ~11-22 (5.5 s/step); not at all with worst gaps | **No.** Raclette's earliest read is ~11-15, leaving 5 days for a 9-day job. Only a *shortened* Fondue could restart, and that is a different scientific trade. |
| LR change on the plateau, at the job 1 -> 2 resume | cancel the queued jobs 2-4, patch a copy of the newest checkpoint, resubmit from a new clone with `afterany` on job 1 (so no extra wait) | before job 1 ends, **~11-14** | **Not realistically.** The read would have to land by ~11-13, i.e. Raclette submitted before Fondue starts, which §7 forbids. |
| LR change for the anneal only | patch a copy of `checkpoint-55167` in `Fondue-ck55167/` (it is a symlink today), hold the anneal job | before job 2 ends, **~11-17** | **Only in the good case** (read by ~11-16). The effect is small: the anneal is 20% of the schedule and decays from whatever peak it inherits. |
| Paper | nothing | any time before the paper | **Yes.** This is the realistic product: *the LRs were carried over from 6-14x smaller batches; here is a 4-point grid at Fondue's batch.* |

My honest summary for the PI: Raclette will not rescue a wrong LR in time; it will tell the
paper whether Fondue's LR was defensible, and it might inform the anneal's peak. If that
is not worth ~2,600 GPU-h and a place in the queue next to Fondue, the cheaper version is
4 arms (~2,100 GPU-h), or none. The PI's call (question 1).

The mid-run patch is an operator-side script outside `melt/` (load `optimizer.pt`, scale
`param_groups[*].lr` and `initial_lr`; load `scheduler.pt`, scale `base_lrs` and `_last_lr`;
write to a new directory). Moments of Adam are scale-free and absorb a step change within
~20 steps at beta2 0.95, but a step change is not a ramp and is untested here, so any change
should be small (<= x2). I will write and test it only if the PI asks (question 9).

## 2. Base recipe: Fondue, shrunk

Each arm is the live `Fondue.yaml` (identical to `origin/fondue-config` @ 3e60f65 in
`/gpfs/scratch/epor48/outputs/Fondue-configs/`, checked 11-09, `diff` empty) with only this changed:

| knob | Fondue | Raclette | why |
|---|---|---|---|
| `optimization.*_lr` | 6e-6 / 2e-4 / 2e-5 | the arm table (§3) | the object of the study |
| hours drawn / `max_steps` | 515,874 h / 68,958 | **25,136 h / 3,360** (3,360 x 26,931 audio-s) | ~5% of the draw; 3,360 = 2.5 x 1,344 puts the plateau end and the final step on the save grid |
| `num_decay_steps` | 13,791 (last fifth) | 672 (last fifth), decay starts at step 2,688 | same WSD shape, cosine to 0.1x |
| `warmup_steps` | 0.05 = 3,448 | **336** (10%) | §4 |
| `run.stop_after_step` | 55,167 | removed | one phase |
| `trainer.gradient_accumulation_steps` | 16 on 16 nodes | 64 on 4 nodes (PI, 11-09) | §6; 1,024 micro-batches per step either way |
| `data.train_ds.snapshot_every_n_steps` | 512 | 512, asserted > 34 | the 30 s/step data-loader stall |
| `save_steps` / `save_total_limit` / `save_only_model` | 1000 / 2 / unset | 1,344 / none / true | §5, disk; the final step also saves |
| `eval_on_start` / `eval_steps` | true / 500 | false / 1,344 | in-trainer eval is not a result; saves ~20 min |
| `trainer.seed`, `MELT_SEED` | 1337, 42 | same (replicate: 1338, 43) | same init and same data stream across arms |

Batch, filters, mixture weights, `max_grad_norm` 1.0, betas 0.9/0.95, `min_lr_scale` 0.1, `router_aux_loss_coef` 0.01 and `master_dtype` float32 are Fondue's.
Batch per step: 30 s x fill 0.877 (26.3 s) x 16 x 64 GPUs = **26,931 audio-s = 7.48 h** (derived). Only the *product* nodes x 4 x ga = 1,024 is held.

## 3. Arms

| arm | encoder | adapter | decoder | note |
|---|---|---|---|---|
| `Raclette-x0p5` | 3e-6 | 1e-4 | 1e-5 | brackets from below |
| **`Raclette-x1`** | 6e-6 | 2e-4 | 2e-5 | Fondue's values, the reference |
| `Raclette-x1-s43` | 6e-6 | 2e-4 | 2e-5 | MELT_SEED 43, trainer.seed 1338: the noise floor at x1 |
| `Raclette-x2` | 1.2e-5 | 4e-4 | 4e-5 | |
| `Raclette-x4` | 2.4e-5 | 8e-4 | 8e-5 | |

**Parameterisation.** One global multiplier on all three groups (keeps the ratios the backbone arms used). No adapter-only arm (PI, 11-09).

**Arithmetic** (derived). Fondue's batch is 26,931 audio-s per step.

| anchor | batch (audio-s/step) | ratio | sqrt rule | linear rule |
|---|---|---|---|---|
| backbone IFT and older campaign arms | 4,800 | 5.61 | x2.37 -> decoder 4.7e-5 | x5.6 -> 1.1e-4 |
| SFT-EU24, T0 (bd60 x ga4 x 8 GPUs, **nominal** 1,920; fill not checked) | 1,920 | 14.0 | x3.75 -> 7.5e-5 | x14 -> 2.8e-4 |

Fondue's LRs *are* the anchors' LRs, unscaled, so Fondue sits at x1 while the sqrt rule asks
for x2.4-3.7. x2 is just under the first sqrt prediction, x4 just over the second; the
interval between them is the "sqrt says here" band. The linear rule (x5.6-14) is not in the
grid: Adam at that batch with clipping 1.0 is beyond anything this campaign has run, and a
monotone trend to x4 would trigger a contingent x8 arm, not a larger first wave. The
withdrawn draft's 2e-5 / 5e-5 / 1.2e-4 used a stale batch; its 5e-5 happens to match the
anchor-A sqrt prediction (4.7e-5).

**x0.5 earns its cost?** The prior that it wins is low (a bigger batch never lowers the
optimal LR by itself; a 14x longer horizon can, mildly). Its job is to bracket: without it,
"x2 and x4 lose" cannot distinguish "x1 is right" from "x1 is already too high". It is the
first arm to drop if cost matters.

## 4. Size and schedule

3,360 steps (25,136 h) is shorter than Fondue's own warmup (3,448). A proportional 5% warmup
would be 168 steps; I choose **336 (10%)**: the plateau (steps 336-2,688, 2,352 steps) stays
long, and the ramp is still ~10x steeper than Fondue's. Adam with beta2 0.95 averages the
second moment over ~20 steps, so a short ramp is less dangerous than the usual case; the
random-init router is the real early risk (the E24 smoke's grad norm was 24 at step 1, 4-5
by step 20, with clipping at 1.0).

*What it transfers:* the ranking of plateau LRs at a constant LR and at Fondue's batch and
mixture. *What it does not:* Fondue's first 3,448 steps (an arm at step 3,360 has integrated
about twice Fondue's LR at the same step), the warmup stability margin, and anything past
5% of the horizon.

**Short-horizon bias.** A short run favours LRs that learn fast. The design limits it with
three reads per arm (step 1,344 early, 2,688 at the end of the plateau, 3,360 after the
decay), a loss-slope check on the plateau (loss against log step; a leader with a flatter
slope is flagged), and the rule that the ranking has to hold at both 2,688 and 3,360.
What the grid can honestly claim: *the best plateau LR for the first 5% of an Fondue-sized
run, at this batch, one seed per point except x1*. It cannot claim the best LR at step
68,000. A longer run (x1 against the winner at 4x the length, ~13.4K steps, ~2,100 GPU-h
each) is the only fix; I do not propose it unless the PI wants the claim (question 4).
Fondue's own checkpoints, scored with the same metric as they appear, give the x1 trajectory
out to the long horizon for free.

**Which action each design serves** (PI correction, 11-09). The from-scratch grid is the only
design that shows what a *different peak LR since step 0* does, which is what a restart would
be. A branch probe (Fondue's checkpoint at step N resumed with a patched LR) tests only a
*step change at step N* on a model shaped by x1; it cannot say what x2 from the start would
give. It is the right test only for the mid-run-change action, and I have not added it to this
plan; the PI has not asked for it.

**Anneal.** Single phase, decay on the constant mixture, as in `07`. I agree. Fondue's real
anneal switches the mixture; copying that would confound the LR comparison with a data change.

## 5. Reads, retention and the decision rule (fixed before any result)

**Headline.** `sel(ASR+ST)`, the joint score of `selection-metric/README.md` (default since
#168; lower is better), with its ASR half, ST half and per-tier CER/WER and chrF++ columns,
at matched steps **1,344 / 2,688 / 3,360** from the saved checkpoints with melt-eval. The
in-trainer eval and the training loss are health checks. Loss is plotted as cross-entropy:
`loss - 0.01 x train_router/aux_loss`, aux loss and router entropy on their own panels.

**Noise.** The joint score has no measured seed spread on this stack. Known: the README's
0.002-0.010 eval-noise band (measured on the ASR score at 200 clips); the crossing's seed spread
0.016-0.017 and T0/T0' 0.009 at step 2,872 are on the *old ASR-only* score; the 0.32 spread of
`04` §7 is a different joint (equal thirds including en->X). None transfers. `x1` and `x1-s43`
measure it where it matters, at the reference point, at the same three steps.

Pre-registered, with g = |score(x1) - score(x1-s43)| at the same step:

- **A difference is called** if it exceeds max(0.010, 2g) at **both** 2,688 and 3,360, with the same sign, and the
  winner is not worse than x1 on either half by more than 0.010.
- **It is actionable** (outcomes C and D) only if it also exceeds 0.020 at both steps, the sign agrees at 1,344, and the loss slope
  does not flag it. For any arm other than x1 this needs a second seed first (a contingent wave-2 replicate, +1 day).
- Otherwise: tie, outcome A or B.
- `x1` against Fondue: compare `train_hours/*` and loss at matched steps once Fondue's log exists (same seed, same data, if the topology matched; it will not at 8 nodes, so only the language mix and loss level are comparable).

0.010 and 0.020 are my proposal, not measured; question 5.

**Same batches.** All arms use `MELT_SEED` 42 and `trainer.seed` 1337 at the same world size, so
they read the same stream and start from the same init; I compare `train_hours/*` at
matched steps, as the MA study did. The launcher's `MELT_SEED` sets the shard seed (it overrides
the YAML's 1337, which is why Fondue's constant phase ran with 42 and its anneal with 43).

**Retention and disk.** `epor48` scratch: 17.82 of 19.53 TB soft, 20.51 hard (`bsc_quota`,
measured 11-09), so 1.71 TB of headroom. A full checkpoint is 28 GB (`du`, Fondue-v1
`checkpoint-2000`: 9.7 GB weights, 19.3 GB optimizer); Fondue keeps 2 (56 GB), the MoE study up to 2 each for
14 runs (~0.8 TB). `save_only_model: true` at `save_steps` 1,344 keeps 3 x 9.7 GB = 29 GB per
arm, **145 GB for five**. Cost: an arm that dies cannot resume (rerun it, <= 560 GPU-h). The
HF option and this trainer's sampler-state save together are untested; the smoke checks the
saved directory. If it fails I will ask rather than fall back to full saves.
Mid-run checkpoints carry no processor or tokenizer (the Fondue-v1 `checkpoint-2000` has none);
the scorer links them from the arm's run root after the final save (as for the crossing).

**Scoring.** On MN5, 1 GPU per job, melt-eval frozen sets (not artemis: scratch there is 96% full),
`--log-format json`, `melt` from the frozen clone on `PYTHONPATH` because the eval image bakes
a stale `melt` without the MoE adapter (`eval-container-code-via-pythonpath.md`; the crossing
script resolves cells from `campaign.yaml`, so I write a small sibling for these run roots). ~1.5 GPU-h per checkpoint
(estimate from 0.8-0.9 per Whisper checkpoint for ASR plus ST sets); 15 checkpoints = ~23 GPU-h. **Scoring jobs also queue**:
1-node jobs waited a median 0.2 h but up to 150 h; I budget a day. Numbers go to the PI;
I do not write `results_master.csv` (question 8).

**Turn-end check.** The in-trainer eval of from-scratch runs ran to the length cap before #165.
`Fondue-v1` `checkpoint-2000` has `eos_token_id` 4 in `generation_config.json`; I will confirm in
the smoke that 4 is `<|im_end|>` and that logged hypotheses end rather than run to 256 tokens.

## 6. Topology

"Same batch" is required and *met by construction*: nodes x 4 x ga = 1,024 micro-batches per
step. Same node count is not.

**Numerics at fixed 1,024 (read from the code).**
- CE: HF normalises by the token count of the whole step across ranks (`average_tokens_across_devices: true`).
- MoE aux term: `_aux_loss_scale` = 1 / (world x ga) inside the model loss, which after HF's x world
  and DDP's mean gives `coef x mean(aux)` over all micro-batches. The product world x ga is 1,024
  in all three shapes, so the weight is identical (#171). A unit test calls it with (64,16), (32,32), (16,64).
- Grad clipping acts on the same global gradient; DDP reduces fp32 (fp32 masters).
- Different: each rank has its own bucketing sampler, so the stream at 32 ranks is not the stream at
  64 (same law, different draws); `x1` is therefore not bit-identical to Fondue's first 3,360 steps
  except at 16 nodes. Arms are paired with each other in every shape.
- `MELT_GPUS_PER_NODE=4` pinned in the script and checked against `world_size` in the first log line.

| | 16 x ga16 | **8 x ga32** | 4 x ga64 |
|---|---|---|---|
| s/step (derived: ~0.8 s fixed + 0.49 s per micro-batch, `board 2026-10-31`; 2-node IB measured 8.55 at ga16) | ~8.7 | ~17 | ~33 |
| wall per arm, 3,360 steps + 24 min start-up | ~8.5 h | ~16.5 h | ~31.5 h |
| GPU-h per arm | ~510 | ~500 | ~490 |
| nodes held, six arms at once | 96 | 48 | 24 |
| queue wait, this user since 10-01 | median 15.5 h (n=4), max 61 h | same bucket | median 8.0 h (n=40, 2-4 nodes), max 39 h |
| `--test-only` today | 11-17 18:55 | 11-17 18:55 | 11-17 18:55 |

GPU-h is flat across shapes (19.6-20.7 per 1,000 h, derived; brief's ceiling 22.2 -> ~555 per arm).
**PI's choice (11-09): 4 nodes x ga64**, 5 arms x 4 = 20 nodes at once, batch unchanged (1,024 micro-batches; the aux scale and CE normalisation are the same, the unit test covers it). Earlier recommendation was 8 x ga32. Median queue + wall comes to ~30 h against ~34 h at 4 nodes and
~24 h at 16, p90 ~43 h / ~54 h / unknown; 16 nodes would put 96 nodes in the queue
beside Fondue's 16, and the history for wide jobs is four samples. Stage 2 re-runs
`--test-only` for the three shapes and I switch only if it shows a clear winner (the PI's call).
Wall request: 36 h. Caveat: 4 nodes should be *scheduled* earlier (median wait 8 h vs 15.5 h) but finishes about as late or later (median ~40 h vs ~32 h to results, p90 ~54 h vs ~43 h); the 8-node figures rest on 4 samples, and `--test-only` shows no difference today.

## 7. Everything else on the allocation

- **Raclette must never delay Fondue.** Fondue's job has priority 68,759; the 14 MoE jobs (2 nodes each, already submitted) have 66,790.
  `--nice` only reorders Raclette against other jobs; it does not touch fairshare, which is computed from *running* usage.
  What Raclette would consume (~2,600 GPU-h) is ~75% of Fondue's own job 1 (64 GPUs x 72 h = 4,600), and
  Fondue's jobs 2-4 queue after it. I do not have the fairshare weights (`sshare` shows one user, FairShare 0.61, EffectvUsage 1.0).
  Proposal: submit with `--dependency=after:48754959` (Slurm starts them eligible only once Fondue's job 1 has *started*; it never
  becomes satisfied if that job is cancelled before starting, which is safe) and a moderate `--nice`; read `sprio` for Fondue's jobs before and after.
  The PI decides (question 6).
- **Budget** (`bsc_acct`, lags ~22 h): epor48 used 1,021.55 of 1,752 khours = 730 khours = ~36,500 GPU-h left. Fondue 6.7-9.8K, MoE 3.35K, Synth-ST arm B 5K,
  Raclette ~2.6K (+13 for the smokes): ~18-21K, leaving ~15K; a Fondue restart (+9.8K) would leave ~4.5K. GPU-h is not the constraint, the queue is.
- **`acc_debug`**: 1 job at a time, 2 h wall. The MoE agent used it for its smokes on 11-08. My smoke queues behind it.
- **MoE E8-K2** (Fondue's LRs at 1/8 of the batch, 10,000 h) is a cross-check of "x1 is healthy at a small batch", not an arm: different batch, hours and schedule.

## 8. Divergence watch

The trainer has no non-finite abort; the pre-#171 bf16 bug appeared as `loss 0, grad_norm nan` from step 2. Logs every 5 steps; the
logged `learning_rate` is the **adapter group only** (group 0), so the encoder/decoder LRs are checked from the `Optimizer group:` lines.

Watcher (read-only polling of `squeue` and each arm's log, over `itpt955676@alogin2.bsc.es`):
1. **Alert** (no action): `grad_norm` > 30 in one line, or > 3x the arm's trailing median in 3 of 20 lines.
2. **Cancel candidate**: `loss` non-finite or exactly 0 in 3 lines, or `grad_norm` > 100 in 3 of the last 200 lines **and** loss more than 1.0 above its running minimum for 20 lines.
3. Right before any `scancel`, re-read the log and compare with what I told the PI (`feedback-recheck-state-before-cancelling.md`); only ids in the arm list; never Fondue or MoE.

A watcher does not wake an idle session (`feedback-alerts-reach-idle-session-late.md`).
The PI has not granted a standing `scancel` for Raclette and I do not assume the crossing's; question 7.
A diverged x4 is a result, not an incident: I record it and cancel to save GPU-h.

## 9. Schedule (today Mon 11-09; the grant ends Mon 11-30)

| when | what |
|---|---|
| Mon 11-09 | This plan, board entry, questions to the PI. |
| Tue 11-10 (after answers) | Renderer, `plan | smoke | submit` script, tests; diff shown to the PI. New frozen clone `~/training-raclette`, never reused (code over `git push` to `alogin2` or the shared home, as for the other clones). |
| Tue-Wed 11-10/11 | Two `acc_debug` smokes (x1 and x4), 2 nodes, ~40 steps, ~13 GPU-h: peak memory, s/step, per-group LRs, `train_hours` identical, `save_only_model` directory, turn-end. Report; wait for the **go**. |
| go (D) | `submit` with the dependency on Fondue's job 1. |
| D + 1 to D + 3 | queue (estimate). |
| + 11-17 h | the five arms run side by side (4 nodes each). |
| + ~1 day | scoring (15 checkpoints), then the readout. |

Earliest read ~11-15; with the §0 queue estimate for D = 11-11, a read in 11-17 to 11-20 is plausible.

## 9a. Smoke results (2026-11-09, `acc_debug`, 2 nodes x ga16, 40 steps, frozen clone `~/training-raclette` @ 608fb9d)

x1 (job 48800465) and x4 (48801761), same seed and data. All measured.

| check | result |
|---|---|
| per-group LRs | x1 `Optimizer group:` lines 2e-4 / 6e-6 / 2e-5 (adapter / encoder / decoder); x4 8e-4 / 2.4e-5 / 8e-5. The logged `learning_rate` is the adapter's: 2e-4 and 8e-4 at peak, reached at step 5 (warmup 4), 1.74e-4 / 6.9e-4 and 2.7e-5 / 1.07e-4 in the 8-step decay |
| same batches | `train_hours/*` identical in the 8 logged steps of x1 and x4; audio per step identical |
| step time | 4.4-6.0 s/step in 5-step windows, mean ~5.1 (x1 and x4 alike), at 2 nodes, ga16; the closing tqdm averages (6.4 and 37.5 s/it) include the first step and are not rates |
| start-up | ~36 min from job start to the first step (dataloader build with the cut filter, then the preallocation passes) |
| memory | the preallocation pass peaks at 31.1 GB (rank 0); the training-step peak is not in `perf.jsonl` (no `gpu_peak` key), so the earlier figure for Fondue (54 of 63 GB) stands unchecked |
| `save_only_model` | `checkpoint-20` holds `model.safetensors` (9.66 GB), config, generation config, sampler state, trainer state: no optimizer, scheduler or RNG files. The run root has the processor and tokenizer files, as expected |
| turn end | token id 4 is `<|im_end|>` (EuroLLM tokenizer) and is the `eos_token_id` of the saved `generation_config.json`. Hypotheses after 40 steps are not informative about stopping (one repeats to the cap, two end); the first scored arm checkpoint settles it |
| final step | a run's last checkpoint is written only after the final in-trainer eval (x1's 40 min wall died in that eval, x4 with 1.5 h finished); real arms have 36 h |

**Consequence for the estimates.** 5.1 s/step at ga16 is faster than the 8.55 s of the IB profile
that §6 used (a ~0.8 s fixed part plus ~0.27 s per micro-batch fits both smokes). Derived, not measured at 4 nodes:
ga64 ~19 s/step, so ~18 h per arm plus ~36 min start-up, ~19 h in all against the 36 h wall, and
~330 GPU-h per arm (16 GPUs). The §6 table's s/step and wall rows are the pessimistic side.
The two smokes cost about 11 GPU-h (estimate).

## 10. Questions for the PI

Recommendation first in each.

1. **Is it worth running?** Yes, as a paper result and a possible anneal input (§1), not as a rescue. If cost matters, 4 arms (x1, x1-s43, x2, x4; ~2,000 GPU-h).
2. **Arms.** The five of §3 (adapter-only dropped by the PI). Drop `x0.5` first if cost matters.
3. **Length, warmup, topology.** 25,136 h = 3,360 steps, warmup 336, one decaying phase, 4 nodes x ga64, 36 h wall (PI).
4. **Longer-horizon check** (x1 against the winner at 4x length, ~4,200 GPU-h)? No, unless the paper needs the claim.
5. **Thresholds.** Called: max(0.010, 2g) at 2,688 and 3,360; actionable: 0.020 plus a second seed. Confirm or change.
6. **Timing against Fondue.** Submit at your go with `--dependency=after:48754959` and a moderate `--nice`; I check `sprio`/`sshare` first. Alternative: hold until Fondue has started.
7. **Watcher.** A standing `scancel` limited to the five Raclette job ids under §8's rule 2, plus scheduled wake-ups so an idle session is not the only net? Not assumed.
8. **Who ingests the scored numbers** into `results_master.csv`? I hand them to the PI and do not write it.
9. **Mid-run LR patch script** (§1, offline, outside `melt/`): write it now, or only if outcome C or D happens? Recommend only then.
10. **`save_only_model`** for the arms (§5), accepting that a dead arm is rerun, not resumed.
