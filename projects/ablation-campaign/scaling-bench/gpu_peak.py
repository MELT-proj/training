#!/usr/bin/env python3
"""What bf16 TFLOP/s these GPUs actually reach, for the MFU denominator.

MFU is normally quoted against the nominal dense bf16 peak -- 989 TFLOP/s for
an H100 SXM5. MN5's `acc` nodes ship the 64 GB H100, and the SXM/PCIe/NVL
variants differ by up to a third in dense bf16, so anchoring on the nominal
number without checking is how an MFU ends up either flattering or punitive for
reasons that have nothing to do with the training code.

Runs a large, purely matmul-bound bf16 GEMM with warmup and CUDA-event timing,
and prints the achieved rate next to the nominal one. `collect.py` does not read
this; the report quotes it beside the MFU column.

    gpu_peak.py                       # 8192^3, 5 warmup + 20 timed
    gpu_peak.py --size 16384 --iters 10
"""

from __future__ import annotations

import argparse

import torch

NOMINAL_BF16_TFLOPS = {"H100": 989.0}


def measure(size: int = 8192, warmup: int = 5, iters: int = 20) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this must run inside a GPU allocation")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    out = torch.empty(size, size, device=device, dtype=dtype)

    for _ in range(warmup):
        torch.matmul(a, b, out=out)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.matmul(a, b, out=out)
    end.record()
    torch.cuda.synchronize()

    seconds = start.elapsed_time(end) / 1000.0
    flops = 2 * size**3 * iters  # two flops per multiply-accumulate
    tflops = flops / seconds / 1e12
    return {
        "device": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "size": size,
        "iters": iters,
        "seconds": seconds,
        "tflops": tflops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    result = measure(args.size, args.warmup, args.iters)
    nominal = next(
        (v for k, v in NOMINAL_BF16_TFLOPS.items() if k in result["device"]), None
    )
    print(f"device     {result['device']} ({result['memory_gb']:.1f} GiB)")
    print(f"gemm       {result['size']}^3 x {result['iters']} in {result['seconds']:.3f} s")
    print(f"achieved   {result['tflops']:.1f} TFLOP/s bf16")
    if nominal:
        print(f"nominal    {nominal:.0f} TFLOP/s  ->  {result['tflops'] / nominal:.1%} of it")
        print("Quote MFU against whichever of the two the report names explicitly.")
    else:
        print(f"nominal    unknown for this device; use the achieved figure as the denominator")


if __name__ == "__main__":
    main()
