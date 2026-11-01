#!/usr/bin/env python3
"""Turn step profiles into the decomposition, the grad_accum fit and trace statistics.

    analyze_profile.py phases RUN_DIR [--skip 3] [--json out.json]
    analyze_profile.py trace  TRACE.json.gz [...] [--json out.json]
    analyze_profile.py fit    RUN_DIR [RUN_DIR ...] [--json out.json]

Standard library only, so it runs on a login node, on nyx, or anywhere the
files land. ``RUN_DIR`` holds ``step_phases_rank0.jsonl`` as written by
``melt/training/step_profile.py``.

``phases``
    Cuts every optimizer step into the intervals between consecutive marks and
    names each interval from the marks at its two ends (``PAIRS``). The names
    partition the step, so they sum to the step time exactly; an interval whose
    two marks are not in ``PAIRS`` is reported as ``other:<a>-><b>`` rather
    than dropped, which is how the accounting stays honest.

``trace``
    From a torch.profiler Chrome trace: GPU busy and idle time, NCCL time split
    into the part that overlaps compute and the exposed part, and kernels,
    kernel time and host syncs per ``melt/*`` range.

``fit``
    s/step = a + b * grad_accum over several runs of one recipe, and the
    16-node prediction.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


#: (previous mark, this mark) -> what the interval between them is.
PAIRS = {
    ("dl0", "dl1"): "dataloader",
    ("dl1", "mb0"): "after_dataloader",
    ("mb0", "h2d0"): "micro_prep",
    ("h2d0", "h2d1"): "h2d",
    ("h2d1", "fwd0"): "fwd_setup",
    ("fwd0", "enc0"): "embed_text",
    ("enc0", "enc1"): "enc_fwd",
    ("enc1", "ad0"): "enc_to_adapter",
    ("ad0", "ad1"): "adapter_fwd",
    ("ad1", "dec0"): "merge",
    ("dec0", "dec1"): "dec_fwd",
    ("dec1", "fwd1"): "loss_fwd",
    ("fwd1", "ddp1"): "ddp_after_fwd",
    ("ddp1", "bwd0"): "loss_prep",
    ("fwd1", "bwd0"): "loss_prep",
    ("bwd0", "g_logits"): "bwd_loss",
    ("g_logits", "g_adapter"): "bwd_dec",
    ("g_adapter", "g_enc"): "bwd_adapter",
    ("g_enc", "g_encend"): "bwd_enc",
    ("g_encend", "mb1"): "bwd_tail",
    ("mb1", "mb0"): "between_micro",
    ("mb1", "clip0"): "after_last_micro",
    ("clip0", "clip1"): "clip",
    ("clip1", "opt0"): "before_optimizer",
    ("opt0", "opt1"): "optimizer",
    ("opt1", "step_end"): "zero_grad_sched",
    ("step_end", "close"): "logging_and_sync",
}

#: Report groups, in the order of the board table.
GROUPS = {
    "dataloader wait": ["dataloader", "after_dataloader"],
    "host-to-device copy": ["h2d"],
    "encoder forward": ["embed_text", "enc_fwd"],
    "encoder backward": ["bwd_enc"],
    "adapter forward": ["enc_to_adapter", "adapter_fwd"],
    "adapter backward": ["bwd_adapter"],
    "decoder forward": ["merge", "dec_fwd"],
    "decoder backward": ["bwd_dec"],
    "loss (forward + backward)": ["loss_fwd", "loss_prep", "bwd_loss"],
    "DDP forward hook (find_unused traversal)": ["ddp_after_fwd"],
    "backward tail (all-reduce not hidden by backward)": ["bwd_tail"],
    "clip": ["clip"],
    "optimizer": ["optimizer", "before_optimizer"],
    "H2D, bookkeeping between micro-batches": ["micro_prep", "fwd_setup", "between_micro", "after_last_micro"],
    "logging, scheduler, sync at step end": ["zero_grad_sched", "logging_and_sync"],
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def step_intervals(record: dict) -> tuple[dict[str, float], dict[str, float], list[dict]]:
    """Per-label seconds on the GPU timeline and on the host, plus one dict per micro-batch.

    The GPU timeline is the truth for "how long did the step take"; the host
    timeline says whether the host was ahead of the device (it reaches a mark
    sooner than the device does) or waiting for it.
    """
    gpu: dict[str, float] = defaultdict(float)
    cpu: dict[str, float] = defaultdict(float)
    micro: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    marks = record["marks"]
    for (a_name, a_micro, a_gpu, a_cpu), (b_name, b_micro, b_gpu, b_cpu) in zip(marks, marks[1:]):
        label = PAIRS.get((a_name, b_name), f"other:{a_name}->{b_name}")
        gpu[label] += (b_gpu - a_gpu) / 1000.0
        cpu[label] += (b_cpu - a_cpu) / 1000.0
        if b_micro >= 0 and label not in (
            "clip",
            "optimizer",
            "before_optimizer",
            "zero_grad_sched",
            "logging_and_sync",
        ):
            micro[b_micro][label] += (b_gpu - a_gpu) / 1000.0
    per_micro = []
    for meta in record.get("micro", []):
        entry = dict(micro.get(meta["micro"], {}))
        entry.update({"micro": meta["micro"], "sync": meta.get("sync"), "audio_s": meta.get("audio_s")})
        per_micro.append(entry)
    return dict(gpu), dict(cpu), per_micro


def usable(records: list[dict], skip: int) -> list[dict]:
    """Steady-state steps: past the warm-up, not a traced step, with a closing mark."""
    kept = [r for i, r in enumerate(records) if i >= skip and not r.get("traced") and r["marks"][-1][0] == "close"]
    return kept


def phases(run_dir: Path, skip: int = 3, rank: int = 0) -> dict:
    records = read_jsonl(run_dir / f"step_phases_rank{rank}.jsonl")
    kept = usable(records, skip)
    if not kept:
        raise SystemExit(f"{run_dir}: no usable steps (have {len(records)}, skip {skip})")
    totals: dict[str, list[float]] = defaultdict(list)
    cpu_totals: dict[str, list[float]] = defaultdict(list)
    step_s = []
    micro_count = []
    nonsync_total, sync_total = [], []
    audio_per_micro = []
    for record in kept:
        gpu, cpu, per_micro = step_intervals(record)
        step_s.append(sum(gpu.values()))
        micro_count.append(len(per_micro))
        for label in set(gpu) | set(cpu):
            totals[label].append(gpu.get(label, 0.0))
            cpu_totals[label].append(cpu.get(label, 0.0))
        for entry in per_micro:
            seconds = sum(v for k, v in entry.items() if k not in ("micro", "sync", "audio_s") and v is not None)
            (sync_total if entry.get("sync") else nonsync_total).append(seconds)
            if entry.get("audio_s") is not None:
                audio_per_micro.append(entry["audio_s"])
    mean_step = statistics.fmean(step_s)
    n_micro = statistics.fmean(micro_count)
    labels = {k: statistics.fmean(v) for k, v in totals.items()}
    cpu_labels = {k: statistics.fmean(v) for k, v in cpu_totals.items()}

    groups = {}
    used = set()
    for name, members in GROUPS.items():
        groups[name] = sum(labels.get(m, 0.0) for m in members)
        used.update(members)
    other = {k: v for k, v in labels.items() if k not in used}
    result = {
        "run": str(run_dir),
        "steps_used": len(kept),
        "steps_total": len(records),
        "micro_per_step": n_micro,
        "s_per_step": mean_step,
        "s_per_step_p10_p90": [
            sorted(step_s)[len(step_s) // 10],
            sorted(step_s)[min(len(step_s) - 1, (9 * len(step_s)) // 10)],
        ],
        "audio_s_per_micro": statistics.fmean(audio_per_micro) if audio_per_micro else None,
        "groups_s": groups,
        "unlabelled_s": other,
        "labels_s": labels,
        "labels_host_s": cpu_labels,
        "accounted_fraction": (sum(groups.values()) / mean_step) if mean_step else None,
        "micro_s_nonsync_mean": statistics.fmean(nonsync_total) if nonsync_total else None,
        "micro_s_sync": statistics.fmean(sync_total) if sync_total else None,
    }
    return result


def print_phases(result: dict) -> None:
    step = result["s_per_step"]
    n = result["micro_per_step"]
    print(
        f"{result['run']}: {result['steps_used']} of {result['steps_total']} steps, "
        f"{n:.0f} micro-batches/step, {step:.3f} s/step (p10-p90 {result['s_per_step_p10_p90'][0]:.2f}-"
        f"{result['s_per_step_p10_p90'][1]:.2f})"
    )
    print(f"{'phase':<54}{'s/step':>8}{'%':>7}{'ms/micro':>10}")
    for name, seconds in result["groups_s"].items():
        print(f"{name:<54}{seconds:>8.3f}{100 * seconds / step:>6.1f}%{1000 * seconds / n:>10.1f}")
    for name, seconds in result["unlabelled_s"].items():
        print(f"{name:<54}{seconds:>8.3f}{100 * seconds / step:>6.1f}%{1000 * seconds / n:>10.1f}")
    print(
        f"accounted: {100 * result['accounted_fraction']:.1f}% of the step; "
        f"micro-batch (non-sync) {1000 * (result['micro_s_nonsync_mean'] or 0):.0f} ms, "
        f"sync micro-batch {1000 * (result['micro_s_sync'] or 0):.0f} ms"
    )


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def total(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Both inputs are unions (sorted, disjoint)."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["traceEvents"]


SYNC_CALLS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpyAsync",
    "cudaMemcpy",
)


def trace_stats(events: list[dict]) -> dict:
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kernels:
        raise SystemExit("trace has no GPU activity")
    runtime = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    ranges = sorted(
        (
            (e["ts"], e["ts"] + e["dur"], e["name"])
            for e in events
            if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e["name"].startswith("melt/")
        ),
    )

    def phase_of(ts: float) -> str:
        for start, end, name in ranges:
            if start <= ts <= end:
                return name
        return "(outside melt/* ranges)"

    nccl = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if "nccl" in k["name"].lower()]
    compute = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if "nccl" not in k["name"].lower()]
    compute_u, nccl_u = union(compute), union(nccl)
    overlap = total(intersect(compute_u, nccl_u))
    start = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    busy_u = union(compute + nccl)

    by_phase: dict[str, dict] = defaultdict(lambda: {"kernels": 0, "kernel_us": 0.0, "spans": []})
    top: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for k in kernels:
        launch = runtime.get(k.get("args", {}).get("correlation"))
        phase = phase_of(launch["ts"]) if launch else "(no launch record)"
        entry = by_phase[phase]
        entry["kernels"] += 1
        entry["kernel_us"] += k["dur"]
        entry["spans"].append((k["ts"], k["ts"] + k["dur"]))
        top[k["name"][:90]][0] += 1
        top[k["name"][:90]][1] += k["dur"]
    phase_out = {}
    for name, entry in by_phase.items():
        spans = union(entry["spans"])
        phase_out[name] = {
            "kernels": entry["kernels"],
            "kernel_ms": entry["kernel_us"] / 1000.0,
            "busy_ms": total(spans) / 1000.0,
            "span_ms": (spans[-1][1] - spans[0][0]) / 1000.0,
        }

    syncs: dict[str, dict] = defaultdict(lambda: {"calls": 0, "ms": 0.0})
    for e in runtime.values():
        if e["name"] in SYNC_CALLS:
            phase = phase_of(e["ts"])
            syncs[phase]["calls"] += 1
            syncs[phase]["ms"] += e["dur"] / 1000.0
    ddp = [
        (e["name"], e["dur"] / 1000.0)
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "user_annotation" and "DistributedDataParallel" in e["name"]
    ]

    return {
        "window_ms": (end - start) / 1000.0,
        "gpu_busy_ms": total(busy_u) / 1000.0,
        "gpu_idle_ms": ((end - start) - total(busy_u)) / 1000.0,
        "compute_busy_ms": total(compute_u) / 1000.0,
        "nccl_ms": total(nccl_u) / 1000.0,
        "nccl_overlapped_ms": overlap / 1000.0,
        "nccl_exposed_ms": (total(nccl_u) - overlap) / 1000.0,
        "nccl_kernels": len(nccl),
        "phases": phase_out,
        "host_syncs": dict(syncs),
        "ddp_ranges": ddp,
        "top_kernels": sorted(((n, c, us / 1000.0) for n, (c, us) in top.items()), key=lambda t: -t[2])[:15],
    }


def print_trace(path: str, stats: dict) -> None:
    print(f"\n{path}")
    print(
        f"  window {stats['window_ms']:.1f} ms | GPU busy {stats['gpu_busy_ms']:.1f} ms, idle {stats['gpu_idle_ms']:.1f} ms"
    )
    print(
        f"  NCCL {stats['nccl_ms']:.1f} ms in {stats['nccl_kernels']} kernels: "
        f"{stats['nccl_overlapped_ms']:.1f} ms overlapped with compute, {stats['nccl_exposed_ms']:.1f} ms exposed"
    )
    print(f"  {'range':<28}{'kernels':>9}{'kernel ms':>11}{'busy ms':>9}{'span ms':>9}{'syncs':>7}{'sync ms':>9}")
    for name, p in sorted(stats["phases"].items(), key=lambda kv: -kv[1]["span_ms"]):
        s = stats["host_syncs"].get(name, {"calls": 0, "ms": 0.0})
        print(
            f"  {name:<28}{p['kernels']:>9}{p['kernel_ms']:>11.1f}{p['busy_ms']:>9.1f}{p['span_ms']:>9.1f}"
            f"{s['calls']:>7}{s['ms']:>9.1f}"
        )
    for name, ms in stats["ddp_ranges"][:4]:
        print(f"  host range {name}: {ms:.1f} ms")
    for name, count, ms in stats["top_kernels"][:8]:
        print(f"  {ms:>9.1f} ms {count:>6}x  {name}")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """y = a + b x. Returns (a, b, rms residual)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    rms = (sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return a, b, rms


def fit(run_dirs: list[Path], skip: int = 3, n16_delta_s: float = 0.8, ranks_2: int = 8, ranks_16: int = 64) -> dict:
    """s/step against grad_accum, and the prediction at 16 nodes.

    The 16-node fixed cost is the 2-node one plus ``n16_delta_s``: the measured
    rise of s/step from 2 to 16 nodes at grad_accum 40 (15.7 -> 16.5 s in the
    scaling sweep). It is attributed to the fixed part because the per-micro-
    batch work does not depend on the world size, and the all-reduce is the
    only thing that does.
    """
    points = []
    for run in run_dirs:
        result = phases(run, skip)
        points.append(
            {
                "run": str(run),
                "grad_accum": round(result["micro_per_step"]),
                "s_per_step": result["s_per_step"],
                "audio_s_per_micro": result["audio_s_per_micro"],
                "steps_used": result["steps_used"],
            }
        )
    points.sort(key=lambda p: p["grad_accum"])
    xs = [p["grad_accum"] for p in points]
    ys = [p["s_per_step"] for p in points]
    a, b, rms = least_squares(xs, ys)
    audio_micro = statistics.fmean(p["audio_s_per_micro"] for p in points if p["audio_s_per_micro"])
    predictions = []
    for g in (40, 20, 10, 5):
        s2 = a + b * g
        s16 = s2 + n16_delta_s
        audio_rank = audio_micro * g
        predictions.append(
            {
                "grad_accum": g,
                "s_per_step_2_nodes": s2,
                "s_per_step_16_nodes": s16,
                "audio_s_per_rank_per_step": audio_rank,
                "global_batch_audio_h_16_nodes": audio_rank * ranks_16 / 3600.0,
                "gpu_h_per_1000_audio_h_16_nodes": 1000.0 * s16 / audio_rank,
                "fixed_share_of_step": (a + n16_delta_s) / s16,
            }
        )
    return {
        "points": points,
        "a_s": a,
        "b_s_per_micro": b,
        "rms_s": rms,
        "predictions": predictions,
        "n16_delta_s": n16_delta_s,
    }


def print_fit(result: dict) -> None:
    print("grad_accum  s/step   (steps)")
    for p in result["points"]:
        print(f"{p['grad_accum']:>10}  {p['s_per_step']:>6.2f}   ({p['steps_used']})")
    print(
        f"\ns/step = a + b * grad_accum: a = {result['a_s']:.2f} s, b = {1000 * result['b_s_per_micro']:.1f} ms per micro-batch, "
        f"rms residual {result['rms_s']:.2f} s"
    )
    print(f"16 nodes = 2-node fit + {result['n16_delta_s']:.1f} s (measured 2 -> 16 node rise at grad_accum 40)\n")
    print(f"{'grad_accum':>10}{'s/step':>9}{'batch audio-h':>15}{'GPU-h/1K h':>12}{'fixed share':>13}")
    for p in result["predictions"]:
        print(
            f"{p['grad_accum']:>10}{p['s_per_step_16_nodes']:>9.2f}{p['global_batch_audio_h_16_nodes']:>15.1f}"
            f"{p['gpu_h_per_1000_audio_h_16_nodes']:>12.1f}{100 * p['fixed_share_of_step']:>12.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_phases = sub.add_parser("phases")
    p_phases.add_argument("run_dir", type=Path)
    p_phases.add_argument("--skip", type=int, default=3)
    p_phases.add_argument("--rank", type=int, default=0)
    p_trace = sub.add_parser("trace")
    p_trace.add_argument("traces", type=Path, nargs="+")
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("run_dirs", type=Path, nargs="+")
    p_fit.add_argument("--skip", type=int, default=3)
    p_fit.add_argument("--n16-delta", type=float, default=0.8)
    for p in (p_phases, p_trace, p_fit):
        p.add_argument("--json", type=Path, help="also write the numbers here")
    args = parser.parse_args()

    if args.command == "phases":
        result = phases(args.run_dir, args.skip, args.rank)
        print_phases(result)
    elif args.command == "trace":
        result = {}
        for path in args.traces:
            result[str(path)] = trace_stats(load_trace(path))
            print_trace(str(path), result[str(path)])
    else:
        result = fit(args.run_dirs, args.skip, args.n16_delta)
        print_fit(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
