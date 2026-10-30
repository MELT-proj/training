"""Isolated benchmarks for the levers the step profile points at.

Launched through the training launcher so it gets the same container, ranks and
NCCL setup as a real run (``run_profile.py levers``, which sets
``MELT_TRAIN_MODULE=lever_bench``). No data and no checkpoint: random tensors of
the Fondue recipe's real shapes, so it costs minutes, not an allocation.

    all-reduce   the full gradient (2.33 B fp32 parameters = 9.3 GB) in DDP-sized
                 buckets, fp32 and bf16, across the whole job and inside one node
    moe          the adapter's expert dispatch: the repo's loop, the same loop on
                 valid frames only, a dense dispatch, and a grouped GEMM
    optimizer    AdamW foreach vs fused, and clip_grad_norm_, at the full size

Results go to ``<out>/levers.json`` (rank 0). Every variant that changes a
number is compared with the baseline's output, so the lever table can say
"identical", "close" or "changes the model" from a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


GRAD_ELEMS = 2_330_000_000  # trainable parameters of the Fondue recipe


def _ms(fn, reps: int = 5, warmup: int = 2) -> dict:
    """Median and spread of fn() in milliseconds, device time between two events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1]}


# ---------------------------------------------------------------------------
# All-reduce
# ---------------------------------------------------------------------------


def allreduce_bench(rank: int, world: int, local_world: int) -> list[dict]:
    """The whole gradient reduced bucket by bucket, as DDP's reducer does."""
    node = rank // local_world
    groups = [dist.new_group(list(range(n * local_world, (n + 1) * local_world))) for n in range(world // local_world)]
    scopes = {"job": dist.group.WORLD, "node": groups[node]}
    out = []
    for dtype in (torch.float32, torch.bfloat16):
        flat = torch.randn(GRAD_ELEMS, device="cuda").to(dtype)
        nbytes = flat.numel() * flat.element_size()
        for bucket_mb in (25, 100, 256, 1024):
            elems = bucket_mb * 1024 * 1024 // flat.element_size()
            chunks = list(flat.split(elems))
            for scope_name, group in scopes.items():
                size = dist.get_world_size(group)
                if size == 1:
                    continue

                def once(chunks=chunks, group=group):
                    works = [dist.all_reduce(c, group=group, async_op=True) for c in chunks]
                    for w in works:
                        w.wait()

                dist.barrier()
                stats = _ms(once, reps=3, warmup=1)
                seconds = stats["median_ms"] / 1000.0
                out.append(
                    {
                        "dtype": str(dtype).split(".")[-1],
                        "bucket_mb": bucket_mb,
                        "scope": scope_name,
                        "ranks": size,
                        "gigabytes": nbytes / 1e9,
                        "median_s": seconds,
                        "algbw_GBps": nbytes / 1e9 / seconds,
                        "busbw_GBps": nbytes / 1e9 / seconds * 2 * (size - 1) / size,
                    }
                )
        del flat, chunks
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# MoE adapter dispatch
# ---------------------------------------------------------------------------


def _moe_config(stack_factor: int) -> SimpleNamespace:
    return SimpleNamespace(
        audio_encoder_config=SimpleNamespace(hidden_size=1280, d_model=1280),
        text_decoder_config=SimpleNamespace(hidden_size=2048),
        adapter_config=SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=1024,
            use_shared_expert=False,
            shared_expert_intermediate_size=1024,
            stack_factor=stack_factor,
        ),
    )


def _stacked(adapter):
    """Expert weights as (E, in, out) stacks, for the batched variants."""
    gate = torch.stack([e.gate_proj.weight.t() for e in adapter.experts])
    up = torch.stack([e.up_proj.weight.t() for e in adapter.experts])
    down = torch.stack([e.down_proj.weight.t() for e in adapter.experts])
    return gate, up, down


def _route(adapter, flat):
    logits = adapter.router(flat)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    weights, indices = probs.topk(adapter.num_experts_per_tok, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(flat.dtype)
    return weights, indices


def _dense_dispatch(adapter, flat, stacks):
    """Every expert over every token, combined with the (zero outside top-k) routing weights."""
    gate, up, down = stacks
    weights, indices = _route(adapter, flat)
    combine = torch.zeros(flat.shape[0], adapter.num_experts, dtype=flat.dtype, device=flat.device)
    combine.scatter_(1, indices, weights)
    h = F.silu(torch.einsum("nd,edh->enh", flat, gate)) * torch.einsum("nd,edh->enh", flat, up)
    y = torch.einsum("enh,eho->eno", h, down)
    return torch.einsum("eno,ne->no", y, combine)


def _grouped_dispatch(adapter, flat, stacks):
    """Tokens sorted by expert, one grouped GEMM per projection; no host sync."""
    gate, up, down = stacks
    weights, indices = _route(adapter, flat)
    n, k = indices.shape
    order = torch.argsort(indices.reshape(-1), stable=True)
    token = order // k
    expert_of = indices.reshape(-1)[order]
    counts = torch.bincount(expert_of, minlength=adapter.num_experts)
    offs = torch.cumsum(counts, 0).to(torch.int32)
    x = flat[token]
    gate, up, down = (w.to(x.dtype).contiguous() for w in (gate, up, down))
    h = F.silu(torch._grouped_mm(x, gate, offs=offs)) * torch._grouped_mm(x, up, offs=offs)
    y = torch._grouped_mm(h, down, offs=offs)
    y = y * weights.reshape(-1)[order].unsqueeze(-1)
    out = torch.zeros(n, y.shape[-1], dtype=y.dtype, device=y.device)
    return out.index_add_(0, token, y)


def moe_bench(cuts: int, valid_fraction: float, stack_factor: int) -> list[dict]:
    from melt.modeling.modeling_melt import MELTMoEAdapter

    torch.manual_seed(0)
    adapter = MELTMoEAdapter(_moe_config(stack_factor)).cuda()
    frames = 1500 // stack_factor
    x = torch.randn(cuts, frames, 1280 * stack_factor, device="cuda", dtype=torch.bfloat16)
    valid_len = max(int(frames * valid_fraction), 1)
    mask = (torch.arange(frames, device="cuda")[None, :] < valid_len).expand(cuts, -1).contiguous()
    valid_x = x[mask]  # (valid tokens, D): what the decoder actually receives

    def with_grad(fn):
        def run():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fn()
            out.float().pow(2).mean().backward()

        return run

    stacks_params = [p.detach().clone().requires_grad_(True) for p in _stacked(adapter)]
    results: list[dict] = []

    def baseline():
        return adapter(x, attention_mask=mask.to(torch.long))

    def valid_only():
        return adapter(valid_x.unsqueeze(0))

    def dense():
        return _dense_dispatch(adapter, x.reshape(-1, x.shape[-1]), stacks_params).reshape(cuts, frames, -1)

    def grouped():
        return _grouped_dispatch(adapter, x.reshape(-1, x.shape[-1]), stacks_params).reshape(cuts, frames, -1)

    variants = {
        "loop (repo), all frames": baseline,
        "loop (repo), valid frames only": valid_only,
        "dense dispatch, all frames": dense,
    }
    if hasattr(torch, "_grouped_mm"):
        variants["grouped GEMM, all frames"] = grouped

    for name, fn in variants.items():
        try:
            stats = _ms(with_grad(fn), reps=10, warmup=3)
            results.append({"variant": name, "tokens": int(x.shape[0] * x.shape[1]), **stats})
        except Exception as exc:  # a missing kernel must not lose the other variants
            results.append({"variant": name, "error": repr(exc)})
    return results


def moe_numerics(cuts: int, valid_fraction: float) -> dict:
    """Max difference of each variant's expert mixture from the repo loop, before post_norm."""
    from melt.modeling.modeling_melt import MELTMoEAdapter

    torch.manual_seed(0)
    adapter = MELTMoEAdapter(_moe_config(1)).cuda()
    x = torch.randn(cuts * 1500, 1280, device="cuda", dtype=torch.bfloat16)
    stacks = [p.detach() for p in _stacked(adapter)]
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        weights, indices = _route(adapter, x)
        loop = torch.zeros(x.shape[0], 2048, dtype=x.dtype, device=x.device)
        mask = F.one_hot(indices, num_classes=8).permute(2, 1, 0)
        for e in range(8):
            slot, tok = torch.where(mask[e])
            if tok.numel():
                loop.index_add_(0, tok, adapter.experts[e](x[tok]) * weights[tok, slot, None])
        loop = loop.float()
        out["dense_vs_loop_max_abs"] = (_dense_dispatch(adapter, x, stacks).float() - loop).abs().max().item()
        if hasattr(torch, "_grouped_mm"):
            out["grouped_vs_loop_max_abs"] = (_grouped_dispatch(adapter, x, stacks).float() - loop).abs().max().item()
        out["loop_abs_mean"] = loop.abs().mean().item()
    return out


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def optimizer_bench() -> dict:
    sizes = torch.tensor([1.0] * 700 + [8.0] * 60 + [40.0] * 13)
    sizes = (sizes / sizes.sum() * GRAD_ELEMS).long().tolist()
    out: dict = {"tensors": len(sizes)}
    for name, kwargs in (("foreach", {"foreach": True}), ("fused", {"fused": True})):
        params = [torch.nn.Parameter(torch.randn(n, device="cuda")) for n in sizes]
        for p in params:
            p.grad = torch.randn_like(p) * 1e-3
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), **kwargs)
        out[name] = _ms(opt.step, reps=5, warmup=2)
        if name == "foreach":
            out["clip_grad_norm"] = _ms(lambda: torch.nn.utils.clip_grad_norm_(params, 1.0), reps=5, warmup=2)
        # Numerics: one identical first step from identical state.
        torch.manual_seed(1)
        a = [torch.nn.Parameter(torch.randn(1_000_000, device="cuda")) for _ in range(4)]
        b = [torch.nn.Parameter(t.detach().clone()) for t in a]
        for t, q in zip(a, b):
            t.grad = torch.randn_like(t)
            q.grad = t.grad.clone()
        torch.optim.AdamW(a, lr=1e-3, foreach=True).step()
        torch.optim.AdamW(b, lr=1e-3, fused=True).step()
        out["fused_vs_foreach_max_abs_after_one_step"] = max((t - q).abs().max().item() for t, q in zip(a, b))
        params = opt = None
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_world = torch.cuda.device_count()

    results: dict = {"world": world, "local_world": local_world, "gpu": torch.cuda.get_device_name(0)}
    t0 = time.time()
    results["allreduce"] = allreduce_bench(rank, world, local_world)
    dist.barrier()
    if rank == 0:
        results["moe"] = {}
        for label, cuts, valid, stack in (
            ("4 cuts, 36% valid, 50 Hz", 4, 0.36, 1),
            ("4 cuts, 36% valid, 10 Hz (stack 5)", 4, 0.36, 5),
            ("12 cuts, 36% valid, 50 Hz", 12, 0.36, 1),
        ):
            results["moe"][label] = moe_bench(cuts, valid, stack)
        results["moe_numerics"] = moe_numerics(4, 0.36)
        results["optimizer"] = optimizer_bench()
        results["seconds"] = time.time() - t0
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "levers.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print("[lever_bench] wrote", os.path.join(args.out, "levers.json"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
