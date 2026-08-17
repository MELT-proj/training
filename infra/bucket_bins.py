#!/usr/bin/env python3
"""Duration-bucket bin estimation, shared by the tools that need it.

This lives apart from ``estimate_bucket_bins.py`` so that it can be imported
without that module's heavier dependencies (joblib, omegaconf, lhotse, tqdm).
Only numpy is required here, which matters because the lhotse 2 venv on nyx has
numpy but neither joblib nor omegaconf, so ``import estimate_bucket_bins`` fails
there outright.

Two entry points, both implementing the same algorithm as Lhotse's
``estimate_duration_buckets``: sort the durations, divide total duration equally
among the buckets, and cut a boundary wherever the running sum crosses a
threshold.

- :func:`estimate_bins_from_durations` takes every duration. Exact, but the
  caller has to hold one float per cut (120M+ for the full SFT mixture).
- :func:`estimate_bins_from_histogram` takes a quantised histogram instead, so a
  measurement pass can be cached compactly and replayed. At 0.01 s resolution
  the bins land within one resolution step of the exact answer, which is below
  the 2 decimal places the configs are written at.
"""

from __future__ import annotations

import numpy as np


def estimate_bins_from_durations(
    durations: list[float],
    num_buckets: int,
) -> list[float]:
    """
    Estimate bucket duration bins from a list of durations.

    This implements the same algorithm as Lhotse's estimate_duration_buckets:
    - Sort durations
    - Divide total duration equally among buckets
    - Place bin boundaries where cumulative duration crosses thresholds

    Args:
        durations: List of cut durations.
        num_buckets: Desired number of buckets.

    Returns:
        List of (num_buckets - 1) boundary duration values.
    """
    if len(durations) == 0:
        raise ValueError("No durations provided")

    if num_buckets > len(durations):
        raise ValueError(f"Number of buckets ({num_buckets}) must be <= number of cuts ({len(durations)})")

    # Sort durations
    sizes = np.array(durations)
    sizes.sort()

    # Target duration per bucket
    size_per_bucket = sizes.sum() / num_buckets

    bins = []
    tot = 0.0
    for size in sizes:
        if tot > size_per_bucket:
            bins.append(float(size))
            tot = 0.0
        tot += size

    return bins


def estimate_bins_from_histogram(
    counts: dict[int, int] | dict[str, int],
    resolution: float,
    num_buckets: int,
) -> list[float]:
    """Same estimate, from a quantised duration histogram.

    ``counts`` maps ``round(duration / resolution)`` to the number of cuts at
    that quantised duration; JSON round-trips turn those keys into strings, so
    both are accepted. Histograms from several sources may simply be summed
    key-wise before calling this, which is what makes one measurement pass
    answer bin questions for any subset of the mixture and any bucket count.

    The loop walks each duration individually rather than in bulk per bin
    because the threshold crossing is sequential: a bin holding many cuts can
    span several boundaries, and collapsing it would move them.
    """
    hist = {int(k): int(v) for k, v in counts.items() if int(v) > 0}
    if not hist:
        raise ValueError("No durations provided")

    n_cuts = sum(hist.values())
    if num_buckets > n_cuts:
        raise ValueError(f"Number of buckets ({num_buckets}) must be <= number of cuts ({n_cuts})")

    keys = sorted(hist)
    total = sum(k * resolution * hist[k] for k in keys)
    size_per_bucket = total / num_buckets

    bins: list[float] = []
    tot = 0.0
    for key in keys:
        size = key * resolution
        for _ in range(hist[key]):
            if tot > size_per_bucket:
                bins.append(float(size))
                tot = 0.0
            tot += size

    return bins


def histogram_of(durations: list[float], resolution: float = 0.01) -> dict[int, int]:
    """Quantise durations into the histogram :func:`estimate_bins_from_histogram` reads."""
    hist: dict[int, int] = {}
    for duration in durations:
        key = int(round(duration / resolution))
        hist[key] = hist.get(key, 0) + 1
    return hist


def format_bins_for_yaml(bins: list[float], precision: int = 2) -> str:
    """Format bins as a YAML-compatible list string."""
    formatted = [round(b, precision) for b in bins]
    return f"[{', '.join(map(str, formatted))}]"
