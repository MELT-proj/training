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
import csv
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


def _has_pnc_text(sample_record: dict | None) -> bool | None:
    """Return whether the cutset has a custom['pnc_text'] field.

    Checks the metadata of a single representative cut. Returns None if no
    sample record is available.
    """
    if sample_record is None:
        return None
    custom = sample_record.get("custom")
    if not isinstance(custom, dict):
        return False
    return "pnc_text" in custom


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
    if "has_pnc_text" not in normalized:
        normalized["has_pnc_text"] = _has_pnc_text(normalized.get("sample_record"))
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
) -> tuple[list[float], dict | None, int, int, list[int]]:
    """Process a single manifest file and return durations, a sample record, and word counts.

    Args:
        manifest_file: Path to the gzipped JSONL manifest.
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        is_first: If True, capture and return the first JSON record as a sample.

    Returns:
        Tuple of (filtered durations, sample_record or None, total_words, cuts_with_text,
        word_counts).  All word fields are accumulated only for cuts that pass the duration
        filter.
    """
    durations: list[float] = []
    sample_record = None
    total_words = 0
    cuts_with_text = 0
    word_counts: list[int] = []
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
                        supervisions = cut_data.get("supervisions", [])
                        if supervisions and isinstance(supervisions, list):
                            text = supervisions[0].get("text", "") or ""
                            if text:
                                wc = len(text.split())
                                total_words += wc
                                cuts_with_text += 1
                                word_counts.append(wc)
    except Exception as e:
        print(f"  Warning: Error reading manifest {manifest_file}: {e}")
    return durations, sample_record, total_words, cuts_with_text, word_counts


def get_durations_from_shar(
    shar_path: str | Path,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
    n_jobs: int = 8,
) -> tuple[list[float], dict | None, int, int, list[int]]:
    """
    Read durations from SHAR manifest files without loading audio.

    Uses joblib to process manifest files in parallel.

    Args:
        shar_path: Path to the SHAR directory.
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        n_jobs: Number of parallel workers.

    Returns:
        Tuple of (list of durations, sample_record, total_words, cuts_with_text).
        total_words and cuts_with_text only cover cuts that pass the duration filter.
    """
    shar_path = Path(shar_path)

    manifest_files = sorted(glob(str(shar_path / "cuts.*.jsonl.gz")))

    if not manifest_files:
        print(f"  Warning: No manifest files found in {shar_path}")
        return [], None, 0, 0, []

    # Only the first manifest file should capture a sample record
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_manifest_file)(mf, min_duration, max_duration, i == 0)
        for i, mf in enumerate(tqdm(manifest_files, desc=f"  Reading {shar_path.name}", unit="file"))
    )

    durations: list[float] = []
    sample_record = None
    total_words = 0
    cuts_with_text = 0
    word_counts: list[int] = []
    for result in results:
        if result is None:
            continue
        file_durations, file_sample, file_words, file_cwt, file_wc = result
        durations.extend(file_durations)
        if sample_record is None and file_sample is not None:
            sample_record = file_sample
        total_words += file_words
        cuts_with_text += file_cwt
        word_counts.extend(file_wc)

    return durations, sample_record, total_words, cuts_with_text, word_counts


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
) -> tuple[list[float], dict[str, dict], bool]:
    """
    Load durations from all data sources specified in input_cfg.

    Fast path: if every source in input_cfg has complete metadata in
    cached_sources, returns immediately without reading any manifest files.
    The returned durations list will be empty in that case and all_from_cache
    will be True.

    Slow path: when any source is missing from cache (or --force-recompute),
    ALL sources are re-read from disk so that bin estimates are always computed
    from the full dataset.

    Args:
        input_cfg: List of data source configurations (can be DictConfig from OmegaConf).
        min_duration: Minimum cut duration to include.
        max_duration: Maximum cut duration to include.
        cached_sources: Dict mapping source cache keys to previously computed results.
        force_recompute: If True, ignore cached results and recompute everything.
        n_jobs: Number of parallel workers for manifest reading.

    Returns:
        Tuple of (all_durations, per_source_results dict keyed by cache key,
        all_from_cache bool).
    """
    if cached_sources is None:
        cached_sources = {}

    # ── Fast path: all sources cached with complete metadata ──────────────
    if not force_recompute:
        cached_results: dict[str, dict] = {}
        all_in_cache = True
        for raw_source_cfg in input_cfg:
            source_cfg = _normalize_source_cfg(raw_source_cfg)
            cache_key = _source_cache_key(source_cfg)
            entry = cached_sources.get(cache_key)
            if entry is None or entry.get("num_cuts") is None:
                all_in_cache = False
                break
            cached_results[cache_key] = _ensure_source_stats(source_cfg, entry)

        if all_in_cache and cached_results:
            for cache_key, info in cached_results.items():
                print(f"  [cached] {cache_key} — {info['num_cuts']} cuts")
            return [], cached_results, True

    # ── Slow path: read all sources from disk ─────────────────────────────
    all_durations: list[float] = []
    per_source_results: dict[str, dict] = {}

    for raw_source_cfg in input_cfg:
        source_cfg = _normalize_source_cfg(raw_source_cfg)
        source_type = source_cfg.get("type", "lhotse_shar")
        cache_key = _source_cache_key(source_cfg)

        if source_type == "lhotse_shar":
            shar_path = source_cfg.get("shar_path")
            if shar_path is None:
                raise ValueError(f"source_cfg of type 'lhotse_shar' is missing required key 'shar_path': {source_cfg}")

            shar_path = str(shar_path)
            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")

            print(f"  Loading from: {shar_path}")
            durations, sample_record, total_words, cuts_with_text, word_counts = get_durations_from_shar(
                shar_path, min_duration, max_duration, n_jobs=n_jobs
            )
            print(f"    Found {len(durations)} cuts")
            all_durations.extend(durations)
            median_duration = float(np.median(durations)) if durations else None
            median_words = float(np.median(word_counts)) if word_counts else None

            per_source_results[cache_key] = _ensure_source_stats(source_cfg, {
                "source_type": source_type,
                "path": shar_path,
                "num_cuts": len(durations),
                "total_duration_hours": sum(durations) / 3600 if durations else 0.0,
                "median_duration": median_duration,
                "median_words_per_cut": median_words,
                "durations": durations,
                "word_counts": word_counts,
                "sample_record": sample_record,
                "has_pnc_text": _has_pnc_text(sample_record),
                "total_words": total_words,
                "cuts_with_text": cuts_with_text,
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
                word_counts = []
                sample_record = None
                total_words = 0
                cuts_with_text = 0
                for i, cut in enumerate(cuts):
                    if i == 0:
                        sample_record = cut.to_dict() if hasattr(cut, "to_dict") else {"id": cut.id, "duration": cut.duration}
                    if min_duration <= cut.duration <= max_duration:
                        durations.append(cut.duration)
                        if hasattr(cut, "supervisions") and cut.supervisions:
                            text = getattr(cut.supervisions[0], "text", None) or ""
                            if text:
                                wc = len(text.split())
                                total_words += wc
                                cuts_with_text += 1
                                word_counts.append(wc)
                print(f"    Found {len(durations)} cuts")
                all_durations.extend(durations)
                median_duration = float(np.median(durations)) if durations else None
                median_words = float(np.median(word_counts)) if word_counts else None

                per_source_results[cache_key] = _ensure_source_stats(source_cfg, {
                    "source_type": source_type,
                    "path": cuts_path,
                    "num_cuts": len(durations),
                    "total_duration_hours": sum(durations) / 3600 if durations else 0.0,
                    "median_duration": median_duration,
                    "median_words_per_cut": median_words,
                    "durations": durations,
                    "word_counts": word_counts,
                    "sample_record": sample_record,
                    "has_pnc_text": _has_pnc_text(sample_record),
                    "total_words": total_words,
                    "cuts_with_text": cuts_with_text,
                })
            except Exception as e:
                raise RuntimeError(f"Failed to read cuts from {cuts_path}: {e}") from e

    return all_durations, per_source_results, False


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

    ``sample_record`` (a single cut dict) is retained; ``durations`` and
    ``word_counts`` (one value per cut) are stripped to keep the output compact.
    Scalar summary stats (median_duration, median_words_per_cut, etc.) are kept.
    """
    _STRIP_KEYS = {"durations", "word_counts"}
    compact_sources: dict[str, dict] = {}
    for cache_key, source_info in per_source.items():
        compact_info = {
            key: value
            for key, value in source_info.items()
            if key not in _STRIP_KEYS
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


def _fmt_optional(value: float | None, precision: int = 2) -> str:
    """Format an optional float, returning empty string for None."""
    return "" if value is None else str(round(value, precision))


def _fmt_mean_dur(total_hours: float, num_cuts: int) -> str:
    """Format mean duration in seconds, or empty string if unavailable."""
    if num_cuts == 0:
        return ""
    return str(round(total_hours * 3600 / num_cuts, 2))


def _fmt_avg_words(total_words: int, cuts_with_text: int) -> str:
    """Format average words per cut, or empty string if no text was available."""
    if cuts_with_text == 0:
        return ""
    return str(round(total_words / cuts_with_text, 1))


def generate_csv(data: dict, csv_path: Path) -> None:
    """Generate a CSV summary table from the computed bucket-info JSON.

    Produces one row per source dataset plus aggregate rows at four granularities:
    task, language (ASR), language-pair (ST), split total, and grand total.
    Word-count columns are left blank when the underlying JSON was produced by an
    older version of the script that did not track word counts.

    Args:
        data: Parsed JSON dict (as written by save_results).
        csv_path: Destination path for the CSV file.
    """
    fieldnames = [
        "split", "dataset", "name",
        "task", "src_lang", "tgt_lang", "has_pnc_text",
        "num_cuts", "total_hours",
        "mean_duration_sec", "median_duration_sec",
        "avg_words_per_cut", "median_words_per_cut",
    ]

    rows: list[dict] = []
    grand_cuts = 0
    grand_hours = 0.0
    grand_words = 0
    grand_cwt = 0

    for split_key in ("train_ds", "validation_ds"):
        split_data = data.get(split_key)
        if not isinstance(split_data, dict) or not split_data:
            continue

        split_label = "train" if split_key == "train_ds" else "validation"
        sources = split_data.get("sources", {})

        # Accumulate per-group word counts from source-level data.
        # Keys: "task::<task>", "lang::<lang>", "base_pair::<src>-><tgt>"
        group_words: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        split_words = 0
        split_cwt = 0

        for source_info in sources.values():
            tw = source_info.get("total_words", 0) or 0
            cwt = source_info.get("cuts_with_text", 0) or 0
            split_words += tw
            split_cwt += cwt
            for task in source_info.get("task_stats", {}):
                group_words[f"task::{task}"][0] += tw
                group_words[f"task::{task}"][1] += cwt
            for lang in source_info.get("language_stats", {}):
                group_words[f"lang::{lang}"][0] += tw
                group_words[f"lang::{lang}"][1] += cwt
            for pair in source_info.get("language_pair_stats", {}):
                if "->" in pair:
                    src_p, tgt_p = pair.split("->", 1)
                    macro_pair = f"{_base_language(src_p)}->{_base_language(tgt_p)}"
                else:
                    macro_pair = pair
                group_words[f"base_pair::{macro_pair}"][0] += tw
                group_words[f"base_pair::{macro_pair}"][1] += cwt

        # ── Source-level rows ──────────────────────────────────────────────
        for source_info in sorted(sources.values(), key=lambda s: s.get("path", "")):
            task_stats = source_info.get("task_stats", {})
            task = max(task_stats, key=lambda k: task_stats[k]["num_cuts"]) if task_stats else "unknown"

            src_lang = ""
            tgt_lang = ""
            if task == "st":
                lp_stats = source_info.get("language_pair_stats", {})
                if lp_stats:
                    pair = max(lp_stats, key=lambda k: lp_stats[k]["num_cuts"])
                    parts = pair.split("->", 1)
                    src_lang = parts[0]
                    tgt_lang = parts[1] if len(parts) > 1 else ""
            else:
                lang_stats = source_info.get("language_stats", {})
                if lang_stats:
                    src_lang = max(lang_stats, key=lambda k: lang_stats[k]["num_cuts"])

            has_pnc = source_info.get("has_pnc_text")
            has_pnc_str = "" if has_pnc is None else ("yes" if has_pnc else "no")
            num_cuts = source_info.get("num_cuts", 0)
            total_hours = source_info.get("total_duration_hours", 0.0)
            tw = source_info.get("total_words", 0) or 0
            cwt = source_info.get("cuts_with_text", 0) or 0
            path_str = source_info.get("path", "")
            path_parts = Path(path_str).parts
            dataset_name = path_parts[-3] if len(path_parts) >= 3 else (path_parts[-1] if path_parts else path_str)
            name = path_parts[-1] if path_parts else path_str

            rows.append({
                "split": split_label, "dataset": dataset_name, "name": name,
                "task": task, "src_lang": src_lang, "tgt_lang": tgt_lang,
                "has_pnc_text": has_pnc_str,
                "num_cuts": num_cuts,
                "total_hours": round(total_hours, 4),
                "mean_duration_sec": _fmt_mean_dur(total_hours, num_cuts),
                "median_duration_sec": _fmt_optional(source_info.get("median_duration")),
                "avg_words_per_cut": _fmt_avg_words(tw, cwt),
                "median_words_per_cut": _fmt_optional(source_info.get("median_words_per_cut")),
            })

        # ── Per-task rows ──────────────────────────────────────────────────
        for task, stats in sorted(split_data.get("task_stats", {}).items()):
            gw = group_words.get(f"task::{task}", [0, 0])
            num_cuts = stats["num_cuts"]
            total_hours = stats["total_duration_hours"]
            rows.append({
                "split": split_label, "dataset": f"[task:{task}]", "name": task,
                "task": task, "src_lang": "", "tgt_lang": "", "has_pnc_text": "",
                "num_cuts": num_cuts,
                "total_hours": round(total_hours, 4),
                "mean_duration_sec": _fmt_mean_dur(total_hours, num_cuts),
                "median_duration_sec": "",
                "avg_words_per_cut": _fmt_avg_words(gw[0], gw[1]),
                "median_words_per_cut": "",
            })

        # ── Per-language rows (ASR) ────────────────────────────────────────
        for lang, stats in sorted(split_data.get("language_stats", {}).items()):
            gw = group_words.get(f"lang::{lang}", [0, 0])
            num_cuts = stats["num_cuts"]
            total_hours = stats["total_duration_hours"]
            rows.append({
                "split": split_label, "dataset": f"[lang:{lang}]", "name": lang,
                "task": stats.get("task", ""), "src_lang": lang, "tgt_lang": "",
                "has_pnc_text": "",
                "num_cuts": num_cuts,
                "total_hours": round(total_hours, 4),
                "mean_duration_sec": _fmt_mean_dur(total_hours, num_cuts),
                "median_duration_sec": "",
                "avg_words_per_cut": _fmt_avg_words(gw[0], gw[1]),
                "median_words_per_cut": "",
            })

        # ── Per-language-pair rows (ST, macro level) ───────────────────────
        for pair, stats in sorted(split_data.get("base_language_pair_stats", {}).items()):
            gw = group_words.get(f"base_pair::{pair}", [0, 0])
            num_cuts = stats["num_cuts"]
            total_hours = stats["total_duration_hours"]
            pair_parts = pair.split("->", 1)
            src_lang = pair_parts[0]
            tgt_lang = pair_parts[1] if len(pair_parts) > 1 else ""
            rows.append({
                "split": split_label, "dataset": f"[pair:{pair}]", "name": pair,
                "task": "st", "src_lang": src_lang, "tgt_lang": tgt_lang,
                "has_pnc_text": "",
                "num_cuts": num_cuts,
                "total_hours": round(total_hours, 4),
                "mean_duration_sec": _fmt_mean_dur(total_hours, num_cuts),
                "median_duration_sec": "",
                "avg_words_per_cut": _fmt_avg_words(gw[0], gw[1]),
                "median_words_per_cut": "",
            })

        # ── Split-total row ────────────────────────────────────────────────
        num_cuts = split_data.get("total_cuts", 0)
        total_hours = split_data.get("total_duration_hours", 0.0)
        rows.append({
            "split": split_label, "dataset": "[total]", "name": f"{split_label}_total",
            "task": "", "src_lang": "", "tgt_lang": "", "has_pnc_text": "",
            "num_cuts": num_cuts,
            "total_hours": round(total_hours, 4),
            "mean_duration_sec": _fmt_mean_dur(total_hours, num_cuts),
            "median_duration_sec": "",
            "avg_words_per_cut": _fmt_avg_words(split_words, split_cwt),
            "median_words_per_cut": "",
        })

        grand_cuts += num_cuts
        grand_hours += total_hours
        grand_words += split_words
        grand_cwt += split_cwt

    # ── Grand-total row ────────────────────────────────────────────────────
    rows.append({
        "split": "all", "dataset": "[total]", "name": "grand_total",
        "task": "", "src_lang": "", "tgt_lang": "", "has_pnc_text": "",
        "num_cuts": grand_cuts,
        "total_hours": round(grand_hours, 4),
        "mean_duration_sec": _fmt_mean_dur(grand_hours, grand_cuts),
        "median_duration_sec": "",
        "avg_words_per_cut": _fmt_avg_words(grand_words, grand_cwt),
        "median_words_per_cut": "",
    })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV table written to:  {csv_path}")


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
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path for the output CSV file. Defaults to <config_stem>_bucket_info.csv next to the config.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV table generation.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only regenerate the CSV from an existing JSON output; skip all data processing.",
    )

    args = parser.parse_args()

    # Determine output paths
    config_path = Path(args.config)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config_path.parent / f"{config_path.stem}_bucket_info.json"

    if args.csv_output:
        csv_path = Path(args.csv_output)
    else:
        csv_path = output_path.with_suffix(".csv")

    # --csv-only: regenerate CSV from an existing JSON, then exit.
    if args.csv_only:
        if not output_path.exists():
            print(f"Error: JSON output not found at {output_path}. Run without --csv-only first.")
            raise SystemExit(1)
        print(f"Loading existing results from: {output_path}")
        generate_csv(load_cached_results(output_path), csv_path)
        return

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
            train_durations, per_source, all_cached = load_cutset_from_config(
                train_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
                cached_sources=cached_sources,
                force_recompute=args.force_recompute,
                n_jobs=args.n_jobs,
            )

            if all_cached and per_source:
                total_cuts = sum(s.get("num_cuts", 0) for s in per_source.values())
                total_hours = sum(s.get("total_duration_hours", 0.0) for s in per_source.values())
                print(f"\nAll sources loaded from cache — skipping manifest reads.")
                print(f"Total training cuts (cached): {total_cuts}")
                print(f"Total duration (cached): {total_hours:.2f} hours")
                # Preserve existing split-level output; refresh per-source entries
                # (picks up any newly backfilled fields like has_pnc_text, median_*, etc.)
                if output["train_ds"]:
                    output["train_ds"]["sources"] = _compact_sources_for_output(per_source)
                    cached_bins = output["train_ds"].get("bucket_duration_bins", [])
                    if cached_bins:
                        results["train_ds"] = cached_bins
            elif train_durations:
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
            val_durations, per_source, all_cached = load_cutset_from_config(
                val_input_cfg,
                min_duration=min_duration,
                max_duration=max_duration,
                cached_sources=cached_sources,
                force_recompute=args.force_recompute,
                n_jobs=args.n_jobs,
            )

            if all_cached and per_source:
                total_cuts = sum(s.get("num_cuts", 0) for s in per_source.values())
                total_hours = sum(s.get("total_duration_hours", 0.0) for s in per_source.values())
                print(f"\nAll sources loaded from cache — skipping manifest reads.")
                print(f"Total validation cuts (cached): {total_cuts}")
                print(f"Total duration (cached): {total_hours:.2f} hours")
                if output["validation_ds"]:
                    output["validation_ds"]["sources"] = _compact_sources_for_output(per_source)
                    cached_bins = output["validation_ds"].get("bucket_duration_bins", [])
                    if cached_bins:
                        results["validation_ds"] = cached_bins
            elif val_durations:
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

    # Write CSV summary table
    if not args.no_csv:
        generate_csv(output, csv_path)

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
