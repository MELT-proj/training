#!/usr/bin/env python3
"""
measure_frames.py

Small utility to measure how many time-frames a Hugging Face feature extractor
produces for a synthetic audio input (silence) of configurable duration.

Usage examples:

# Measure frames for Facebook w2v-bert-2.0 using 10s of silence
python utils/measure_frames.py --model facebook/w2v-bert-2.0 --duration 10

# Use a specific sampling rate and request PyTorch tensors (not used for frame measurement)
python utils/measure_frames.py --model facebook/w2v-bert-2.0 --duration 5 --sampling-rate 16000 --return-tensors np

"""

import argparse
import sys
from typing import Any

import numpy as np

from transformers import AutoFeatureExtractor


def detect_time_dim(output: dict[str, Any]) -> int | None:
    """Return length of the time dimension found in the extractor output.

    Heuristics used:
    - If `input_values` exists, use its last dimension
    - If `input_features` exists, use its second-to-last dimension
    - Otherwise inspect all arrays: pick the first ndarray with ndim >= 2 and choose the
      largest plausible time dimension (preferring last then second last).

    Returns the number of frames (int) or None if no suitable array is found.
    """
    if "input_values" in output:
        arr = output["input_values"]
        if hasattr(arr, "shape"):
            return int(arr.shape[-1])

    if "input_features" in output:
        arr = output["input_features"]
        if hasattr(arr, "shape") and arr.ndim >= 2:
            return int(arr.shape[-2])
    return None


def measure_frames(
    model: str | None = None,
    duration_seconds: float = 10.0,
    return_tensors: str | None = "np",
) -> int | None:
    """Load AutoFeatureExtractor and measure frames produced for a zero audio signal.

    Arguments
    - model: model id passed to AutoFeatureExtractor.from_pretrained(). If None, uses
      default AutoFeatureExtractor().
    - duration_seconds: length of synthetic audio in seconds
    - sampling_rate: optional override; if None the extractor's sampling_rate will be read
    - return_tensors: pass-through to extractor call (commonly 'np' or 'pt' or 'tf')

    Returns the number of frames or None if not found.
    """

    fe = AutoFeatureExtractor.from_pretrained(model)
    sr = 16000
    audio = np.zeros(int(sr * duration_seconds), dtype=np.float32)

    # many extractors expect a single sample (mono) or a list/batch of samples
    # feed a single sample; pass return_tensors only if provided
    kwargs: dict[str, Any] = {"sampling_rate": sr}
    if return_tensors:
        kwargs["return_tensors"] = return_tensors

    out = fe(audio, **kwargs)

    frames = detect_time_dim(out)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure time-frames produced by a Hugging Face feature extractor")
    parser.add_argument("--model", type=str, help="model id for AutoFeatureExtractor.from_pretrained")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="duration of synthetic audio in seconds (default 10)"
    )
    parser.add_argument(
        "--return-tensors", type=str, default="np", help="return_tensors arg passed to extractor (np/pt/tf)"
    )

    args = parser.parse_args(argv)

    try:
        frames = measure_frames(args.model, args.duration, args.return_tensors)
    except Exception as e:  # pragma: no cover - runtime helper
        print("Error running extractor:", e, file=sys.stderr)
        return 2

    if frames is None:
        print("frames: <could not detect time dimension from extractor output>")
        return 1

    print("frames:", frames)
    print(f"frame resolution: {1000 * args.duration / frames:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
