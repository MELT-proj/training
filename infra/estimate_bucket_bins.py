#!/usr/bin/env python3
"""
Estimate bucket duration bins for dynamic batching using Lhotse.

This script reads a YAML training config, loads all datasets specified in train_ds
and validation_ds, and estimates optimal bucket duration bins for dynamic bucketing.

Usage:
    python utils/estimate_bucket_bins.py --config config/train/asr.yaml
    python utils/estimate_bucket_bins.py --config config/train/asr.yaml --train-buckets 30 --val-buckets 20

The script outputs the estimated bucket_duration_bins that can be used in the config.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from lhotse import CutSet
from omegaconf import DictConfig, OmegaConf


def load_yaml_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file using OmegaConf.

    This properly handles OmegaConf syntax like ${oc.env:VAR} for environment variables.
    """
    config = OmegaConf.load(config_path)
    # Resolve all interpolations (environment variables, references, etc.)
    OmegaConf.resolve(config)
    return config


def get_durations_from_shar(
    shar_path: str | Path,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> list[float]:
    """
    Read durations from SHAR manifest files without loading audio.

    Args:
        shar_path: Path to the SHAR directory.
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.

    Returns:
        List of durations for cuts that pass the filter.
    """
    from glob import glob

    shar_path = Path(shar_path)
    durations = []

    # Find all cuts manifest files (cuts.*.jsonl.gz pattern)
    manifest_files = sorted(glob(str(shar_path / "cuts.*.jsonl.gz")))

    if not manifest_files:
        print(f"  Warning: No manifest files found in {shar_path}")
        return durations

    for manifest_file in manifest_files:
        try:
            with gzip.open(manifest_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cut_data = json.loads(line)
                        duration = cut_data.get("duration", 0.0)
                        # Apply duration filter
                        if min_duration <= duration <= max_duration:
                            durations.append(duration)
        except Exception as e:
            print(f"  Warning: Error reading manifest {manifest_file}: {e}")
            continue

    return durations


def load_cutset_from_config(
    input_cfg: list[dict] | DictConfig,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> list[float]:
    """
    Load durations from all data sources specified in input_cfg.

    Args:
        input_cfg: List of data source configurations (can be DictConfig from OmegaConf).
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.

    Returns:
        List of all durations across all data sources.
    """
    all_durations = []

    for source_cfg in input_cfg:
        # Handle both dict and DictConfig
        if isinstance(source_cfg, DictConfig):
            source_type = source_cfg.get("type", "lhotse_shar")
        else:
            source_type = source_cfg.get("type", "lhotse_shar")

        if source_type == "lhotse_shar":
            shar_path = source_cfg.get("shar_path")
            if shar_path is None:
                continue

            # Convert to string (OmegaConf already resolved interpolations)
            shar_path = str(shar_path)

            if not Path(shar_path).exists():
                print(f"  Warning: Shar path not found: {shar_path}")
                continue

            print(f"  Loading from: {shar_path}")
            durations = get_durations_from_shar(shar_path, min_duration, max_duration)
            print(f"    Found {len(durations)} cuts")
            all_durations.extend(durations)

        elif source_type == "lhotse_cuts":
            cuts_path = source_cfg.get("cuts_path")
            if cuts_path is None:
                continue

            # Convert to string (OmegaConf already resolved interpolations)
            cuts_path = str(cuts_path)

            if not Path(cuts_path).exists():
                print(f"  Warning: Cuts path not found: {cuts_path}")
                continue

            print(f"  Loading from: {cuts_path}")
            try:
                cuts = CutSet.from_file(cuts_path)
                for cut in cuts:
                    if min_duration <= cut.duration <= max_duration:
                        all_durations.append(cut.duration)
                print(f"    Found {len(all_durations)} cuts")
            except Exception as e:
                print(f"  Warning: Error reading cuts {cuts_path}: {e}")
                continue

    return all_durations


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


def format_bins_for_yaml(bins: list[float], precision: int = 2) -> str:
    """Format bins as a YAML-compatible list string."""
    formatted = [round(b, precision) for b in bins]
    return f"[{', '.join(map(str, formatted))}]"


def main():
    parser = argparse.ArgumentParser(
        description="Estimate bucket duration bins for dynamic batching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default bucket counts (30 for train, 20 for validation)
    python utils/estimate_bucket_bins.py --config config/train/asr.yaml

    # Custom bucket counts
    python utils/estimate_bucket_bins.py --config config/train/asr.yaml \\
        --train-buckets 40 --val-buckets 25

    # Only estimate for training data
    python utils/estimate_bucket_bins.py --config config/train/asr.yaml --train-only
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML training configuration file.",
    )
    parser.add_argument(
        "--train-buckets",
        type=int,
        default=30,
        help="Number of buckets for train_ds (default: 30).",
    )
    parser.add_argument(
        "--val-buckets",
        type=int,
        default=20,
        help="Number of buckets for validation_ds (default: 20).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only estimate bins for training data.",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only estimate bins for validation data.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for bin values (default: 2).",
    )

    args = parser.parse_args()

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_yaml_config(args.config)

    data_config = config.get("data", {})

    results = {}

    # Process train_ds
    if not args.val_only:
        train_ds = data_config.get("train_ds", {})
        train_input_cfg = train_ds.get("input_cfg", [])

        if train_input_cfg:
            print("\n" + "=" * 60)
            print("Processing train_ds...")
            print("=" * 60)

            min_duration = train_ds.get("min_duration", 0.0)
            max_duration = train_ds.get("max_duration", float("inf"))

            print(f"Duration filter: [{min_duration}, {max_duration}] seconds")

            train_durations = load_cutset_from_config(
                train_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if train_durations:
                print(f"\nTotal training cuts: {len(train_durations)}")
                print(f"Total duration: {sum(train_durations) / 3600:.2f} hours")
                print(f"Duration range: [{min(train_durations):.2f}, {max(train_durations):.2f}] seconds")
                print(f"Mean duration: {np.mean(train_durations):.2f} seconds")

                train_bins = estimate_bins_from_durations(train_durations, args.train_buckets)
                results["train_ds"] = train_bins

                print(f"\nEstimated {args.train_buckets} bucket bins for train_ds:")
                print(f"  bucket_duration_bins: {format_bins_for_yaml(train_bins, args.precision)}")
            else:
                print("\nNo training data found!")
        else:
            print("\nNo train_ds.input_cfg found in config.")

    # Process validation_ds
    if not args.train_only:
        val_ds = data_config.get("validation_ds", {})
        val_input_cfg = val_ds.get("input_cfg", [])

        if val_input_cfg:
            print("\n" + "=" * 60)
            print("Processing validation_ds...")
            print("=" * 60)

            min_duration = val_ds.get("min_duration", 0.0)
            max_duration = val_ds.get("max_duration", float("inf"))

            print(f"Duration filter: [{min_duration}, {max_duration}] seconds")

            val_durations = load_cutset_from_config(
                val_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if val_durations:
                print(f"\nTotal validation cuts: {len(val_durations)}")
                print(f"Total duration: {sum(val_durations) / 3600:.2f} hours")
                print(f"Duration range: [{min(val_durations):.2f}, {max(val_durations):.2f}] seconds")
                print(f"Mean duration: {np.mean(val_durations):.2f} seconds")

                val_bins = estimate_bins_from_durations(val_durations, args.val_buckets)
                results["validation_ds"] = val_bins

                print(f"\nEstimated {args.val_buckets} bucket bins for validation_ds:")
                print(f"  bucket_duration_bins: {format_bins_for_yaml(val_bins, args.precision)}")
            else:
                print("\nNo validation data found!")
        else:
            print("\nNo validation_ds.input_cfg found in config.")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY - Copy these to your config file:")
        print("=" * 60)

        if "train_ds" in results:
            print("\ntrain_ds:")
            print(f"  bucket_duration_bins: {format_bins_for_yaml(results['train_ds'], args.precision)}")

        if "validation_ds" in results:
            print("\nvalidation_ds:")
            print(f"  bucket_duration_bins: {format_bins_for_yaml(results['validation_ds'], args.precision)}")

        print()


if __name__ == "__main__":
    main()
