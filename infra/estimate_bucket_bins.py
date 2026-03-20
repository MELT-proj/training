#!/usr/bin/env python3
"""
Estimate bucket duration bins for dynamic batching using Lhotse.

This script reads a YAML training config, loads all datasets specified in train_ds
and validation_ds, and estimates optimal bucket duration bins for dynamic bucketing.

Usage:
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml --train-buckets 30 --val-buckets 20

The script outputs the estimated bucket_duration_bins that can be used in the config,
and writes all results to a JSON file for caching and inspection.
"""

from __future__ import annotations

import argparse
import gzip
import json
from glob import glob
from pathlib import Path

from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from lhotse import CutSet
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm


def load_yaml_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file using OmegaConf.

    This properly handles OmegaConf syntax like ${oc.env:VAR} for environment variables.
    """
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected DictConfig at root of {config_path}, got {type(config).__name__}")
    # Resolve all interpolations (environment variables, references, etc.)
    OmegaConf.resolve(config)
    return config


def _extract_language(cut_data: dict) -> str:
    """Extract language from a Lhotse cut record.

    Looks in supervisions[0].language first, then falls back to cut-level 'language'.
    """
    supervisions = cut_data.get("supervisions", [])
    if supervisions and isinstance(supervisions, list):
        lang = supervisions[0].get("language")
        if lang:
            return str(lang)
    lang = cut_data.get("language")
    if lang:
        return str(lang)
    return "unknown"


def _base_language(language: str) -> str:
    """Collapse a language variant to its macro language code."""
    for separator in ("-", "_"):
        if separator in language:
            return language.split(separator, 1)[0]
    return language


def _build_duration_stats(durations: list[float], **extra: str) -> dict[str, int | float | str]:
    """Build a duration statistics record with optional metadata fields."""
    stats: dict[str, int | float | str] = {
        "num_cuts": len(durations),
        "total_duration_hours": round(sum(durations) / 3600, 4),
    }
    stats.update(extra)
    return stats


def _merge_stats_entry(
    stats_map: dict[str, dict[str, int | float | str]],
    key: str,
    durations: list[float],
    **extra: str,
) -> None:
    """Insert a stats entry into a keyed stats map."""
    stats_map[key] = _build_duration_stats(durations, **extra)


def _build_source_stats(
    source_cfg: dict[str, object],
    durations: list[float],
    sample_record: dict | None,
) -> dict[str, dict[str, dict[str, int | float | str]]]:
    """Build per-source stats keyed by task, language variant, and macro language."""
    raw_tags = source_cfg.get("tags", {})
    tags = raw_tags if isinstance(raw_tags, dict) else {}
    task = str(tags.get("task", "unknown"))

    source_stats: dict[str, dict[str, dict[str, int | float | str]]] = {
        "task_stats": {
            task: _build_duration_stats(durations, task=task),
        }
    }

    if task == "st":
        src_lang = str(tags.get("src_lang") or "en")
        tgt_lang = str(tags.get("tgt_lang") or tags.get("lang") or "unknown")
        pair_key = f"{src_lang}->{tgt_lang}"
        macro_src_lang = _base_language(src_lang)
        macro_tgt_lang = _base_language(tgt_lang)
        macro_pair_key = f"{macro_src_lang}->{macro_tgt_lang}"
        source_stats["language_pair_stats"] = {
            pair_key: _build_duration_stats(
                durations,
                task=task,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
        }
        source_stats["base_language_pair_stats"] = {
            macro_pair_key: _build_duration_stats(
                durations,
                task=task,
                src_lang=macro_src_lang,
                tgt_lang=macro_tgt_lang,
            )
        }
        return source_stats

    lang = str(tags.get("lang") or _extract_language(sample_record or {}))
    macro_lang = _base_language(lang)
    source_stats["language_variant_stats"] = {
        lang: _build_duration_stats(durations, task=task, lang=lang),
    }
    source_stats["language_stats"] = {
        macro_lang: _build_duration_stats(durations, task=task, lang=macro_lang),
    }
    return source_stats


def _ensure_source_stats(
    source_cfg: dict[str, object],
    source_info: dict,
) -> dict:
    """Backfill task-aware stats for newly computed or cached source info."""
    normalized = dict(source_info)
    if {
        "task_stats",
        "language_stats",
        "language_variant_stats",
        "language_pair_stats",
        "base_language_pair_stats",
    }.isdisjoint(normalized):
        normalized.update(
            _build_source_stats(
                source_cfg,
                normalized.get("durations", []),
                normalized.get("sample_record"),
            )
        )
    return normalized


def _normalize_source_cfg(source_cfg: object) -> dict[str, object]:
    """Convert a source config from OmegaConf or dict into a plain dict."""
    if isinstance(source_cfg, DictConfig):
        normalized = OmegaConf.to_container(source_cfg, resolve=True)
        if isinstance(normalized, dict):
            return {str(key): value for key, value in normalized.items()}
        raise TypeError(f"Expected source config mapping, got {type(normalized).__name__}")
    if isinstance(source_cfg, dict):
        return {str(key): value for key, value in source_cfg.items()}
    raise TypeError(f"Expected source config mapping, got {type(source_cfg).__name__}")


def _process_manifest_file(
    manifest_file: str,
    min_duration: float,
    max_duration: float,
    is_first: bool,
) -> tuple[list[float], dict | None]:
    """Process a single manifest file and return durations and optionally a sample record.

    Args:
        manifest_file: Path to the gzipped JSONL manifest.
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        is_first: If True, capture and return the first JSON record as a sample.

    Returns:
        Tuple of (filtered durations, sample_record or None).
    """
    durations = []
    sample_record = None
    try:
        with gzip.open(manifest_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cut_data = json.loads(line)
                    if is_first and sample_record is None:
                        sample_record = cut_data
                    duration = cut_data.get("duration", 0.0)
                    if min_duration <= duration <= max_duration:
                        durations.append(duration)
    except Exception as e:
        print(f"  Warning: Error reading manifest {manifest_file}: {e}")
    return durations, sample_record


def get_durations_from_shar(
    shar_path: str | Path,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
    n_jobs: int = 8,
) -> tuple[list[float], dict | None]:
    """
    Read durations from SHAR manifest files without loading audio.

    Uses joblib to process manifest files in parallel.

    Args:
        shar_path: Path to the SHAR directory.
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        n_jobs: Number of parallel workers.

    Returns:
        Tuple of (list of durations, sample_record).
    """
    shar_path = Path(shar_path)

    manifest_files = sorted(glob(str(shar_path / "cuts.*.jsonl.gz")))

    if not manifest_files:
        print(f"  Warning: No manifest files found in {shar_path}")
        return [], None

    # Only the first manifest file should capture a sample record
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_manifest_file)(mf, min_duration, max_duration, i == 0)
        for i, mf in enumerate(tqdm(manifest_files, desc=f"  Reading {shar_path.name}", unit="file"))
    )

    durations = []
    sample_record = None
    for result in results:
        if result is None:
            continue
        file_durations, file_sample = result
        durations.extend(file_durations)
        if sample_record is None and file_sample is not None:
            sample_record = file_sample

    return durations, sample_record


def _source_cache_key(source_cfg: dict[str, object]) -> str:
    """Build a unique cache key for a source configuration."""
    source_type = source_cfg.get("type", "lhotse_shar")
    if source_type == "lhotse_shar":
        return f"shar::{source_cfg.get('shar_path', '')}"
    elif source_type == "lhotse_cuts":
        return f"cuts::{source_cfg.get('cuts_path', '')}"
    return f"{source_type}::{id(source_cfg)}"


def load_cutset_from_config(
    input_cfg: list[dict[str, object]] | ListConfig,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
    cached_sources: dict[str, dict] | None = None,
    force_recompute: bool = False,
    n_jobs: int = 8,
) -> tuple[list[float], dict[str, dict]]:
    """
    Load durations from all data sources specified in input_cfg.

    Args:
        input_cfg: List of data source configurations (can be DictConfig from OmegaConf).
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        cached_sources: Dict mapping source cache keys to previously computed results.
        force_recompute: If True, ignore cached results and recompute everything.
        n_jobs: Number of parallel workers for manifest reading.

    Returns:
        Tuple of (all_durations, per_source_results dict keyed by cache key).
    """
    all_durations = []
    per_source_results: dict[str, dict] = {}
    if cached_sources is None:
        cached_sources = {}

    for raw_source_cfg in input_cfg:
        source_cfg = _normalize_source_cfg(raw_source_cfg)
        source_type = source_cfg.get("type", "lhotse_shar")
        cache_key = _source_cache_key(source_cfg)

        # Check cache
        if not force_recompute and cache_key in cached_sources:
            cached = _ensure_source_stats(source_cfg, cached_sources[cache_key])
            cached_durations = cached.get("durations")
            if isinstance(cached_durations, list):
                print(f"  [cached] {cache_key} — {cached['num_cuts']} cuts")
                all_durations.extend(cached_durations)
                per_source_results[cache_key] = cached
                continue
            print(f"  [cached-metadata-only] {cache_key} — recomputing durations")

        if source_type == "lhotse_shar":
            shar_path = source_cfg.get("shar_path")
            if shar_path is None:
                raise ValueError(f"source_cfg of type 'lhotse_shar' is missing required key 'shar_path': {source_cfg}")

            shar_path = str(shar_path)
            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")

            print(f"  Loading from: {shar_path}")
            durations, sample_record = get_durations_from_shar(
                shar_path, min_duration, max_duration, n_jobs=n_jobs
            )
            print(f"    Found {len(durations)} cuts")
            all_durations.extend(durations)

            per_source_results[cache_key] = _ensure_source_stats(source_cfg, {
                "source_type": source_type,
                "path": shar_path,
                "num_cuts": len(durations),
                "total_duration_hours": sum(durations) / 3600 if durations else 0.0,
                "durations": durations,
                "sample_record": sample_record,
            })

        elif source_type == "lhotse_cuts":
            cuts_path = source_cfg.get("cuts_path")
            if cuts_path is None:
                raise ValueError(f"source_cfg of type 'lhotse_cuts' is missing required key 'cuts_path': {source_cfg}")

            cuts_path = str(cuts_path)
            if not Path(cuts_path).exists():
                raise FileNotFoundError(f"Cuts path not found: {cuts_path}")

            print(f"  Loading from: {cuts_path}")
            try:
                cuts = CutSet.from_file(cuts_path)
                durations = []
                sample_record = None
                for i, cut in enumerate(cuts):
                    if i == 0:
                        # Capture sample record from first cut
                        sample_record = cut.to_dict() if hasattr(cut, "to_dict") else {"id": cut.id, "duration": cut.duration}
                    if min_duration <= cut.duration <= max_duration:
                        durations.append(cut.duration)
                print(f"    Found {len(durations)} cuts")
                all_durations.extend(durations)

                per_source_results[cache_key] = _ensure_source_stats(source_cfg, {
                    "source_type": source_type,
                    "path": cuts_path,
                    "num_cuts": len(durations),
                    "total_duration_hours": sum(durations) / 3600 if durations else 0.0,
                    "durations": durations,
                    "sample_record": sample_record,
                })
            except Exception as e:
                raise RuntimeError(f"Failed to read cuts from {cuts_path}: {e}") from e

    return all_durations, per_source_results


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


def load_cached_results(output_path: Path) -> dict:
    """Load previously cached results from a JSON file."""
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def save_results(output_path: Path, results: dict) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {output_path}")


def _compact_sources_for_output(per_source: dict[str, dict]) -> dict[str, dict]:
    """Drop per-cut arrays from per-source stats before persisting to JSON.

    ``sample_record`` (a single cut dict) is retained; only the ``durations``
    list (one float per cut) is stripped to keep the output compact.
    """
    compact_sources: dict[str, dict] = {}
    for cache_key, source_info in per_source.items():
        compact_info = {
            key: value
            for key, value in source_info.items()
            if key != "durations"
        }
        compact_sources[cache_key] = compact_info
    return compact_sources


def _compact_cached_output(cached_data: dict) -> dict:
    """Strip per-cut duration arrays from previously cached output structure."""
    compact_data = dict(cached_data)
    for split_key in ("train_ds", "validation_ds"):
        split_data = compact_data.get(split_key)
        if not isinstance(split_data, dict):
            continue
        split_copy = dict(split_data)
        sources = split_copy.get("sources", {})
        if isinstance(sources, dict):
            split_copy["sources"] = _compact_sources_for_output(sources)
        compact_data[split_key] = split_copy
    return compact_data


def _aggregate_stats(per_source: dict[str, dict], stats_key: str) -> dict[str, dict]:
    """Aggregate keyed statistics across all sources in a split."""
    totals: dict[str, dict[str, int | float | str]] = defaultdict(
        lambda: {"num_cuts": 0, "total_duration_hours": 0.0}
    )
    for source_info in per_source.values():
        for key, stats in source_info.get(stats_key, {}).items():
            totals[key]["num_cuts"] += stats["num_cuts"]
            totals[key]["total_duration_hours"] += stats["total_duration_hours"]
            for field in ("task", "lang", "src_lang", "tgt_lang"):
                if field in stats:
                    totals[key][field] = stats[field]

    return {
        key: {
            **{field: stats[field] for field in ("task", "lang", "src_lang", "tgt_lang") if field in stats},
            "num_cuts": int(stats["num_cuts"]),
            "total_duration_hours": round(float(stats["total_duration_hours"]), 4),
        }
        for key, stats in sorted(totals.items())
    }


def _aggregate_language_stats(per_source: dict[str, dict]) -> dict[str, dict]:
    """Aggregate macro-language statistics across all sources in a split."""
    return _aggregate_stats(per_source, "language_stats")


def _aggregate_language_variant_stats(per_source: dict[str, dict]) -> dict[str, dict]:
    """Aggregate exact language-variant statistics across all sources in a split."""
    return _aggregate_stats(per_source, "language_variant_stats")


def _aggregate_language_pair_stats(per_source: dict[str, dict]) -> dict[str, dict]:
    """Aggregate per-language-pair statistics across ST sources in a split."""
    return _aggregate_stats(per_source, "language_pair_stats")


def _aggregate_base_language_pair_stats(per_source: dict[str, dict]) -> dict[str, dict]:
    """Aggregate macro language-pair statistics across ST sources in a split."""
    return _aggregate_stats(per_source, "base_language_pair_stats")


def _aggregate_task_stats(per_source: dict[str, dict]) -> dict[str, dict]:
    """Aggregate per-task statistics across all sources in a split."""
    return _aggregate_stats(per_source, "task_stats")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate bucket duration bins for dynamic batching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default bucket counts (30 for train, 20 for validation)
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml

    # Custom bucket counts
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml \\
        --train-buckets 40 --val-buckets 25

    # Only estimate for training data
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml --train-only

    # Force recompute even if cached results exist
    python infra/estimate_bucket_bins.py --config config/train/asr.yaml --force-recompute
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML training configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output JSON file. Defaults to <config_stem>_bucket_info.json next to the config.",
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
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all sources, ignoring any cached results in the output JSON.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel workers for reading manifest files (default: 8).",
    )

    args = parser.parse_args()

    # Determine output path
    config_path = Path(args.config)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config_path.parent / f"{config_path.stem}_bucket_info.json"

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_yaml_config(args.config)

    data_config = config.get("data", {})

    # Load cached results if they exist (and we're not forcing recompute)
    cached_data = {}
    if not args.force_recompute and output_path.exists():
        print(f"Loading cached results from: {output_path}")
        cached_data = load_cached_results(output_path)
        cached_data = _compact_cached_output(cached_data)

    # The full output structure
    output: dict = {
        "config_path": str(config_path.resolve()),
        "train_ds": cached_data.get("train_ds", {}),
        "validation_ds": cached_data.get("validation_ds", {}),
    }

    results = {}

    # Helper to build cached_sources lookup from previously saved output
    def _get_cached_sources(split_key: str) -> dict[str, dict]:
        split_data = cached_data.get(split_key, {})
        return split_data.get("sources", {})

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

            cached_sources = _get_cached_sources("train_ds")
            train_durations, per_source = load_cutset_from_config(
                train_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
                cached_sources=cached_sources,
                force_recompute=args.force_recompute,
                n_jobs=args.n_jobs,
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

                output["train_ds"] = {
                    "num_buckets": args.train_buckets,
                    "bucket_duration_bins": [round(b, args.precision) for b in train_bins],
                    "total_cuts": len(train_durations),
                    "total_duration_hours": round(sum(train_durations) / 3600, 4),
                    "min_duration": round(min(train_durations), 4),
                    "max_duration": round(max(train_durations), 4),
                    "mean_duration": round(float(np.mean(train_durations)), 4),
                    "task_stats": _aggregate_task_stats(per_source),
                    "language_stats": _aggregate_language_stats(per_source),
                    "language_variant_stats": _aggregate_language_variant_stats(per_source),
                    "language_pair_stats": _aggregate_language_pair_stats(per_source),
                    "base_language_pair_stats": _aggregate_base_language_pair_stats(per_source),
                    "sources": _compact_sources_for_output(per_source),
                }
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

            cached_sources = _get_cached_sources("validation_ds")
            val_durations, per_source = load_cutset_from_config(
                val_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
                cached_sources=cached_sources,
                force_recompute=args.force_recompute,
                n_jobs=args.n_jobs,
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

                output["validation_ds"] = {
                    "num_buckets": args.val_buckets,
                    "bucket_duration_bins": [round(b, args.precision) for b in val_bins],
                    "total_cuts": len(val_durations),
                    "total_duration_hours": round(sum(val_durations) / 3600, 4),
                    "min_duration": round(min(val_durations), 4),
                    "max_duration": round(max(val_durations), 4),
                    "mean_duration": round(float(np.mean(val_durations)), 4),
                    "task_stats": _aggregate_task_stats(per_source),
                    "language_stats": _aggregate_language_stats(per_source),
                    "language_variant_stats": _aggregate_language_variant_stats(per_source),
                    "language_pair_stats": _aggregate_language_pair_stats(per_source),
                    "base_language_pair_stats": _aggregate_base_language_pair_stats(per_source),
                    "sources": _compact_sources_for_output(per_source),
                }
            else:
                print("\nNo validation data found!")
        else:
            print("\nNo validation_ds.input_cfg found in config.")

    # Write output JSON
    save_results(output_path, output)

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
