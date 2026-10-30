# Where the Fondue step goes

Results are in the 2026-10-31 entry of `plan/board.md`. This directory holds the tools and the raw data.

| file | what |
|---|---|
| `run_profile.py` | submits one profile job (`run`), the isolated lever benchmark (`levers`), or prints/validates a command (`render`, `check`). `acc_debug` only; writes `BENCH-profile-*` and no weights |
| `analyze_profile.py` | `phases` (decomposition from the marks), `trace` (NCCL overlap, per-phase kernels and host syncs from a profiler trace), `fit` (s/step = a + b x grad_accum) |
| `lever_bench.py` | all-reduce by bucket size and dtype, MoE dispatch variants, AdamW foreach vs fused; also prints what the container sees of `/dev/infiniband` |
| `raw/<run>/` | `step_phases_rank0.jsonl` (raw marks), `perf.jsonl`, `unused_grads_rank0.json`; traces only for the ga40 run (the rest stay on MN5 under `/gpfs/scratch/epor48/outputs/BENCH-profile-*`) |

Run names: `n<nodes>-ga<grad_accum>[-bd<batch_duration>]ckptsk5[-nccl]`. All runs: fp32 master weights, gradient checkpointing, `stack_factor 5`, bd 15 unless the name says otherwise. `-nccl` runs set `NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8 NCCL_BUFFSIZE=16777216` and `trainer.ddp_bucket_cap_mb=256`.

```bash
# on MN5, from your own clone, after infra/sync_repo.sh
python3 projects/ablation-campaign/scaling-bench/profile/run_profile.py run --nodes 2 --grad-accum 20 \
    --steps 40 --tag ckptsk5-nccl --env NCCL_SOCKET_NTHREADS=8 --env NCCL_NSOCKS_PERTHREAD=8 \
    --env NCCL_BUFFSIZE=16777216 --extra=--trainer.gradient_checkpointing=true \
    --extra=--model.adapter.stack_factor=5 --extra=--trainer.ddp_bucket_cap_mb=256

# anywhere, on the fetched files
python3 analyze_profile.py phases raw/n2-ga20-ckptsk5-nccl
python3 analyze_profile.py trace  raw/n2-ga40-ckptsk5/trace_rank0_step8_final.json.gz
python3 analyze_profile.py fit    raw/n2-ga10-ckptsk5-nccl raw/n2-ga20-ckptsk5-nccl
```

Two things to know before reading a number:

- The trained step is not the scaling sweep of 2026-10-24, which ran on a bf16-loaded model that went NaN at step 2.
- `fit` adds a fixed 0.8 s to go from 2 to 16 nodes. That figure comes from the old bf16 sweep; the true fp32 value is unmeasured.
