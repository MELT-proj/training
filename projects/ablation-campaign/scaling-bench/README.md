# Scaling benchmark: 1 to 32 MN5 nodes

What this measures, for the Fondue recipe, at every node count we might use:

1. **Throughput** — TFLOP/s per GPU
2. **MFU** — the same number as a fraction of the 989 TFLOP/s dense bf16 peak
3. **Wall clock** — hours to consume 100,000 hours of audio

It exists because the repo can predict Fondue's *cost* (GPU-hours) but not its
*efficiency*, and the two open questions the cost-axis study left behind are both
answerable with one sweep:

- Measured points exist only for 2 / 8 / 16 nodes, on the Qwen IFT arm
  (`plan/06-fondue.md` §2), and the 2 -> 16 curve is described as "flat". 4
  nodes was never measured and 32 never attempted.
- `cost-axis/README.md` asks for *"a compute-only measure (FLOPs per audio second
  from the configs, or synthetic-batch timing)"* and *"a dataloader-only (CPU)
  benchmark vs world_size"*. Both are arms of this campaign.

**Status: built, not yet run.** Nothing here has produced a measurement. The
tables in `tables/` are the artifact of `calibrate_flops.py static` (parameter
counts, which are real) plus a synthetic round-trip through `collect.py` that
reproduces the hand-computed anchor. Read the numbers below as the *prediction*
the campaign is built to test.

## The prediction

From the documented 32.3 s/step at 8 nodes (`batch_duration 30`,
`grad_accum 20`, 32 GPUs, so 19,200 audio-seconds per step):

| | |
|---|---|
| audio per wall second | 594 (594 audio-hours per wall-hour) |
| TFLOP/s per GPU | 19.5 |
| **MFU** | **1.97%** of 989 TFLOP/s |
| encoder share of the step's FLOPs | 25% (unfrozen; ~2% frozen) |
| GPU-hours per 1,000 audio-hours | 53.8 |
| **100,000 audio hours at 8 nodes** | **168 wall hours = 7.0 days**, 5,383 GPU-h |

The cost side of that table agrees with the repo's own published figures, which
is what makes it credible. The **2% MFU is the surprising part**: a 2 B decoder
on H100s at 40% MFU would step this batch in well under a second. Either the
rate is set by the data path, or the FLOPs are not where the config says they
are. Distinguishing those two is the point of the extra arms below, and it is
the answer the next Fondue decision actually needs.

## Method

### What is held fixed

Everything except `nodes`. `bench.py render --all` asserts this mechanically.
`batch_duration`, `grad_accum`, dtype, attention backend, gradient checkpointing,
the three freeze flags, `max_duration`/`max_tokens`, `num_workers`, the bucket
bins and the seed are identical at every point, so the only variable is world
size. Per-rank batch stays fixed, so the effective batch grows with node count:
this is the campaign's convention ("keep `grad_accum` and add nodes"), not a
choice made here.

The recipe is derived from `ABL-IFT-700.yaml` by override list, with the encoder,
adapter and decoder **all unfrozen** — a full fine-tune, which is not what the
drafted Fondue IFT does (it freezes encoder and adapter). Three lines in
`bench.py:bench_overrides` switch it to the frozen recipe, and the sweep can be
re-run after the config freeze on 2026-10-25 without any other change.

### Timing hygiene

Everything that is not training is off: `eval_on_start`, eval, saving, memory
preallocation, checkpoint resume. `data.validation_ds.input_cfg` is nulled so
`MELTTrainer.__init__` skips building the ~100k-cut eval dataset — several
minutes of startup, at every node count, that would otherwise be in the way.

`report_to` deliberately stays `wandb` in offline mode: `resolved_config.json` is
only written when the tracker is on, and the reporter needs it to prove the
recipe did not drift between points.

### Steady state

The first 10 steps are discarded. In this repo the first step alone has been
measured at 13-21 minutes (shuffle-buffer fill, CUDA allocator growth,
cuDNN/triton autotune), which would dominate a 110-step mean. `s_per_step` is
the mean of the retained intervals; p10/p90 are reported beside it, because a
mean hides a straggler tail and a tail is what a wall-clock estimate has to
survive.

### Definitions

With `B` = `batch_duration`, `G` = `grad_accum`, `R` = world size (4 per node):

```
audio_h_per_step = B * G * R / 3600
s_per_step       = mean wall time per optimizer step
rate             = audio_h_per_step / s_per_step        [audio-h per wall-h]
wall_h(H)        = H / rate
gpu_h(H)         = wall_h * R
```

`quadratic_duration` is null in these configs, so `B` is real audio seconds and
the product is exact. What is *not* exact is the fill: a duration-bucketed batch
rarely holds the nominal `B`, and the board measured 78-86% on the 2,100 h mix.
`collect.py` reports the measured fill next to the nominal figure, so the
measured wall time is the quoted one and the old nominal extrapolations remain
comparable.

### MFU

```
flops = R * G * (encoder + adapter + decoder)
  body         k * N * tokens/frames,  k = 2 (frozen) | 6 (fwd+bwd) | 8 (with recompute)
  attention    4/12/16 * T^2 * d per layer, 12/16 for the decoder's 6 full-attention layers only
encoder attention is quadratic within its 1500-frame chunk, not over the cut
mfu = flops / s_per_step / (R * 989e12)
```

Parameter counts are not assumed: `calibrate_flops.py static` builds the modules
on the `meta` device and counts them. Current values, transformers 5.16.1:

| | params | notes |
|---|---|---|
| `facebook/w2v-bert-2.0` | 580.5 M | 24 layers, dim 1024, 50 Hz frames |
| `Qwen/Qwen3.5-2B` | 1.8818 B | body 1.3733 B + head 0.5086 B, tied embeddings |
| MELT MLP adapter | 6.30 M | 1024 -> 2048 -> 2048 |
| decoder attention | 6 of 24 layers | 18 are Gated-DeltaNet (linear, no quadratic term) |

The head is counted even though it is tied to the embedding: the forward still
does `hidden @ vocab` on every position. The input *lookup* is not counted, so
body + head is exactly the parameter count — counting the whole embedding matrix
as 6N per token would overstate the decoder by a quarter.

Two MFU columns are reported. `mfu_total` counts everything the model runs;
`mfu_trainable` counts only what produces a gradient. With everything unfrozen
they agree. With the encoder frozen they do not, and the gap is precisely the
compute that freezing is supposed to save — which is the column to read before
deciding to freeze the encoder for Fondue.

## The extra arms (no GPU-hours)

1. **`dataloader_bench.py`** — builds the same Lhotse dataloader and times
   iteration with no model. Runs on `gp_debug` (CPU-only, 32 nodes, priority
   10000) and costs nothing. Compare a batch's median time against
   `s_per_step / grad_accum`: if the data path eats a large share of the
   per-rank step budget, more nodes cannot help, and that is the answer to the
   cost-axis question.
2. **`calibrate_flops.py count`** — one GPU, runs the real decoder under
   `torch.utils.flop_counter.FlopCounterMode` and prints measured vs analytic
   FLOPs per duration. The counter sees matmuls and SDPA forward/backward but
   not fused pointwise/norm kernels or the Gated-DeltaNet recurrence, so it is a
   lower bound; the residual is recorded rather than silently scaled by.
3. **`gpu_peak.py`** — the achievable bf16 TFLOP/s on the actual card. MN5's
   H100 is the 64 GB variant and the SXM/PCIe/NVL parts differ by up to a third,
   so the report will name which denominator its MFU uses.

### Which of these need to run on MN5

Only one of them, and it is worth being precise about why, because "run the
calibration on artemis first" is the obvious way to save queue time and it is
half wrong.

**FLOPs are a property of the computation graph, not of the machine.** A 1.4 B ×
4096 matmul is the same number of FLOPs on an A6000, an H100 or a CPU; clocks,
bandwidth, power limits and cooling do not enter `FlopCounterMode` at all. So
`calibrate_flops.py static` (parameter counts, already run) and `count` (the
analytic constants) can run anywhere — including a laptop with no GPU for the
static half. Three things could still move the answer, and none of them is the
GPU:

- *which kernel dispatches* — flash vs efficient-attention vs math SDPA are
  different code paths. Artemis `dionysus` is 4 × H100, the same sm90 silicon as
  MN5's `acc` nodes, which is the reason to pin `--partition=h100` rather than
  land on an A6000;
- *library version* — the same `.sif` runs both sides;
- *shapes* — the real risk, and it is shape bias rather than hardware bias: a
  synthetic 5/15/30 s sequence that does not resemble the padded distribution
  gives a wrong ratio. This is why the analytic model consumes the trainer's
  *measured* padded frame counts, and the counter is a cross-check rather than
  the source of truth.

**`gpu_peak.py` is the opposite case and must run on MN5.** Achievable TFLOP/s
depends on clocks, power capping, cooling and the exact board variant, so an
artemis number would be a genuine bias in the MFU denominator.

The practical consequence: `calibrate_flops.py count` takes seconds, the model is
already built, so fold it into the 1-node MN5 smoke instead of spending a
separate artemis allocation on it. Artemis stays useful only for iterating on the
script itself if it breaks.

## Running it

From `alogin1.bsc.es`, in the repo, after `infra/sync_repo.sh mn5`:

```bash
cd ~/training

# 1. Shape and constants (CPU only, no cluster time)
python3 projects/ablation-campaign/scaling-bench/calibrate_flops.py static

# 2. Validate every point without consuming the acc_debug submission slot
for p in ift-qwen-n1 ift-qwen-n2 ift-qwen-n4 ift-qwen-n8 ift-qwen-n16 ift-qwen-n32; do
  python3 projects/ablation-campaign/scaling-bench/bench.py check "$p"
done

# 3. Smoke first: 1 node proves the model fits, the [perf] lines appear, and the
#    rate is plausible before any of the expensive points are queued.
python3 projects/ablation-campaign/scaling-bench/bench.py run ift-qwen-n1

# 4. The sweep. acc_debug allows ONE submitted job at a time, so the four small
#    points are chained by a polling driver -- --dependency is not an option,
#    because the dependent job is itself a second submission and is rejected.
for p in ift-qwen-n2 ift-qwen-n4 ift-qwen-n8; do
  while [[ -n "$(squeue -u $USER -h -o %i)" ]]; do sleep 60; done
  python3 projects/ablation-campaign/scaling-bench/bench.py run "$p"
done
#    ...and 16/32 nodes on acc_ehpc, one at a time so repeats do not share the
#    same GPFS contention.
for p in ift-qwen-n16 ift-qwen-n32; do
  python3 projects/ablation-campaign/scaling-bench/bench.py run "$p"
  while [[ -n "$(squeue -u $USER -h -o %i)" ]]; do sleep 60; done
done

# 5. Collect
python3 projects/ablation-campaign/scaling-bench/collect.py --raw raw/

# 6. Data path, on the CPU partition, at no GPU cost
sbatch --nodes=32 projects/ablation-campaign/scaling-bench/dataloader_bench.sbatch
```

`bench.py fetch <point>` pulls `perf.jsonl`, `resolved_config.json` and the
SLURM `.out` into `raw/<exp_name>/`; `collect.py` never touches the cluster.

## Files

| | |
|---|---|
| `bench.yaml` | recipes and points (what to run) |
| `bench.py` | `list` / `render` / `check` / `run` / `status` |
| `flops.py` | the FLOPs, MFU and walltime arithmetic (pure Python) |
| `calibrate_flops.py` | `static` (real parameter counts) and `count` (FlopCounterMode) |
| `collect.py` | perf records -> `tables/points.tsv`, `tables/walltime.tsv` |
| `dataloader_bench.py` | the CPU-only data-path benchmark |
| `gpu_peak.py` | achievable bf16 TFLOP/s |
| `tables/` | committed results; **no job ids** |
| `raw/` | gitignored: `perf.jsonl`, `resolved_config.json`, SLURM logs |

Job ids go in `../arms.tsv`, the campaign's append-only ledger, and nowhere in
`tables/`. (A directory named `results/` anywhere in this repo is silently
gitignored, which is why the committed output is `tables/`.)

## Budget

One pass, sized on elapsed time (MN5 bills elapsed, not requested — a 150-step
point runs ~80 min of stepping plus a ~20 min first step):

| point | GPUs | request | expected GPU-h |
|---|---|---|---|
| 1 node | 4 | 2 h | 6 |
| 2 nodes | 8 | 2 h | 13 |
| 4 nodes | 16 | 2 h | 26 |
| 8 nodes | 32 | 2 h | 51 |
| 16 nodes | 64 | 2 h | 102 |
| 32 nodes | 128 | 3 h | ~224 |
| | | **one pass** | **~420** |

~500 GPU-h with the repeat on the worst-fitting points. The 32-node point alone
is over half of it — that is the price of the one number no other point can give.

## Confounds, and what is done about them

- **Rack variance.** Identical configs have differed by 1.4-1.9x in wall time in
  this repo, which is larger than the effects this sweep is looking for. Every
  job records `SLURM_JOB_NODELIST`; the worst-fitting points get a repeat on a
  different rack, interleaved in time; and no trend smaller than the observed
  spread is claimed.
- **Page-cache warmth.** A 110-step run re-reads a small slice of the corpus, so
  its I/O is warmer than a 100,000-hour run. The measured wall time is therefore
  a *floor*, and `dataloader_bench.py`'s cold first batch is quoted beside it.
- **`num_workers: 1` on the training path.** Untested suspicion in this repo.
  The loader-only arm prices it directly, with `--num-workers 2` as a variant.
- **128 ranks at 32 nodes.** The indexed-Shar sampler is designed for it, but
  this would be the first run at that width. 1 -> 8 -> 16 precede it, and the
  loader-only arm runs at 32 on the CPU partition first.
- **The 989 TFLOP/s denominator may not describe these cards.** `gpu_peak.py`
  measures what they actually reach.
- **QOS differs between small and large points.** `acc_debug` for 1-8 nodes,
  `acc_ehpc` for 16-32: it changes queue time, not compute, and the per-job
  nodelist reveals any scheduling effect in the step-time spread.

## Verified so far

- `tests/test_scaling_bench.py` (27 tests): the FLOPs arithmetic against
  hand-computed cases, the 6/8/2 multipliers, encoder chunking, the trainable
  split, the model-free rate invariance to window length, and the trainer's
  perf-window accounting including the log-time subtraction and the jsonl
  append. `pytest tests/test_scaling_bench.py` passes on nyx.
- The full override set parses: `--run.dry_run` renders `perf_logging: true`,
  three `freeze: false`, `eval_strategy: 'no'` as a *string* (the OmegaConf
  dotlist trap), `batch_duration: 30`, `grad_accum: 20`, `max_steps: 110`.
- `bench.py render --all` reports identical argv at every node count.
- The site file sources on nyx and builds
  `--nodes=8 --gpus-per-node=4 --account=epor48 --qos=acc_debug --cpus-per-task=80`.
- `calibrate_flops.py static` produced the parameter table above from the real
  configs.
- `collect.py` reproduces the hand-computed anchor exactly from a synthetic
  perf log: 594 audio-s/wall-s, 19.5 TFLOP/s per GPU, 1.97% MFU, 53.8 GPU-h per
  1K h, 7.0 days per 100K h.

Not verified: anything that needs an allocation. The first real check is the
1-node smoke, and the first trustworthy number is the 8-node point reproducing
31-33 s/step.
