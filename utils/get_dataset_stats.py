#!/usr/bin/env python3
"""Compute statistics from Lhotse SHAR manifests.

This script scans a parent directory containing SHAR-format datasets
and computes duration and word count statistics per dataset, language, and split.

Usage:
    python infra/get_dataset_stats.py /path/to/shar/parent --output stats.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from statistics import mean, median, stdev


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Dataset Configuration
# =============================================================================

# Datasets where the second level is language (multilingual)
MULTILINGUAL_DATASETS = {"cv22_sidon", "fleurs", "voxpopuli", "mls_sidon"}

# Datasets where the second level is config (monolingual, assume English)
MONOLINGUAL_DATASETS = {"librispeech", "peoples_speech"}

# Language folder name → standardized ISO 639-1 code
LANGUAGE_MAP = {
    # Already standard
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "cs": "cs",
    "ja": "ja",
    # Common Voice variants
    "zh-CN": "zh",
    "zh-HK": "zh",
    "zh-TW": "zh",
    # Fleurs-style with region
    "de_de": "de",
    "en_us": "en",
    "es_419": "es",
    "es_es": "es",
    "fr_fr": "fr",
    "it_it": "it",
    "pt_br": "pt",
    "pt_pt": "pt",
    # Full language names (MLS)
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "polish": "pl",
    "portuguese": "pt",
    "spanish": "es",
}


def standardize_language(folder_name: str) -> str:
    """Convert folder name to standardized language code."""
    folder_lower = folder_name.lower()
    return LANGUAGE_MAP.get(folder_lower, folder_lower)


# =============================================================================
# Statistics Computation
# =============================================================================


@dataclass
class Stats:
    """Accumulator for duration and word count statistics."""

    durations: list[float] = field(default_factory=list)
    word_counts: list[int] = field(default_factory=list)

    def add(self, duration: float, word_count: int) -> None:
        self.durations.append(duration)
        self.word_counts.append(word_count)

    def merge(self, other: "Stats") -> None:
        self.durations.extend(other.durations)
        self.word_counts.extend(other.word_counts)

    @property
    def count(self) -> int:
        return len(self.durations)

    def compute(self) -> dict[str, float]:
        """Compute summary statistics."""
        if not self.durations:
            return {
                "num_cuts": 0,
                "duration_total_hrs": 0.0,
                "duration_mean_sec": 0.0,
                "duration_median_sec": 0.0,
                "duration_std_sec": 0.0,
                "words_total": 0,
                "words_mean": 0.0,
                "words_median": 0.0,
                "words_std": 0.0,
            }

        dur_total = sum(self.durations)
        wc_total = sum(self.word_counts)

        return {
            "num_cuts": len(self.durations),
            "duration_total_hrs": dur_total / 3600.0,
            "duration_mean_sec": mean(self.durations),
            "duration_median_sec": median(self.durations),
            "duration_std_sec": stdev(self.durations) if len(self.durations) > 1 else 0.0,
            "words_total": wc_total,
            "words_mean": mean(self.word_counts),
            "words_median": median(self.word_counts),
            "words_std": stdev(self.word_counts) if len(self.word_counts) > 1 else 0.0,
        }


def read_shar_manifests(shar_dir: Path) -> Stats:
    """Read all cuts manifests from a SHAR directory and compute stats."""
    stats = Stats()
    manifest_files = sorted(glob(str(shar_dir / "cuts.*.jsonl.gz")))

    if not manifest_files:
        logger.warning(f"No manifest files found in {shar_dir}")
        return stats

    for manifest_file in manifest_files:
        try:
            with gzip.open(manifest_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cut_data = json.loads(line)
                    duration = cut_data.get("duration", 0.0)

                    # Extract text from supervisions
                    text = ""
                    supervisions = cut_data.get("supervisions", [])
                    if supervisions:
                        text = supervisions[0].get("text", "")

                    word_count = len(text.split()) if text else 0
                    stats.add(duration, word_count)
        except Exception as e:
            logger.warning(f"Error reading manifest {manifest_file}: {e}")
            continue

    return stats


# =============================================================================
# Directory Scanning
# =============================================================================


@dataclass
class DatasetEntry:
    """A single dataset/language/split entry."""

    dataset: str
    language: str
    split: str
    shar_path: Path
    stats: Stats = field(default_factory=Stats)


def discover_datasets(parent_dir: Path) -> list[DatasetEntry]:
    """Discover all dataset/language/split combinations in the parent directory."""
    entries = []

    for dataset_dir in sorted(parent_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Skip hidden/marker directories
        if dataset_name.startswith("."):
            continue

        if dataset_name in MULTILINGUAL_DATASETS:
            # Structure: dataset/language/split
            for lang_dir in sorted(dataset_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue

                language = standardize_language(lang_dir.name)

                for split_dir in sorted(lang_dir.iterdir()):
                    if not split_dir.is_dir():
                        continue

                    entries.append(
                        DatasetEntry(
                            dataset=dataset_name,
                            language=language,
                            split=split_dir.name,
                            shar_path=split_dir,
                        )
                    )

        elif dataset_name in MONOLINGUAL_DATASETS:
            # Structure: dataset/config/split (all English)
            for config_dir in sorted(dataset_dir.iterdir()):
                if not config_dir.is_dir():
                    continue

                for split_dir in sorted(config_dir.iterdir()):
                    if not split_dir.is_dir():
                        continue

                    # Use config as part of the identifier, language is English
                    entries.append(
                        DatasetEntry(
                            dataset=f"{dataset_name}/{config_dir.name}",
                            language="en",
                            split=split_dir.name,
                            shar_path=split_dir,
                        )
                    )

        else:
            logger.warning(f"Unknown dataset type: {dataset_name}, skipping")

    return entries


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute statistics from Lhotse SHAR manifests")
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing SHAR dataset folders",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    parent_dir = args.parent_dir.resolve()
    if not parent_dir.exists():
        logger.error(f"Parent directory not found: {parent_dir}")
        return

    logger.info(f"Scanning {parent_dir} for SHAR datasets...")

    # Discover all datasets
    entries = discover_datasets(parent_dir)
    logger.info(f"Found {len(entries)} dataset/language/split combinations")

    # Compute statistics for each entry
    for entry in entries:
        logger.info(f"Processing {entry.dataset}/{entry.language}/{entry.split}...")
        entry.stats = read_shar_manifests(entry.shar_path)

    # Prepare results
    results = []
    for entry in entries:
        computed = entry.stats.compute()
        results.append(
            {
                "dataset": entry.dataset,
                "language": entry.language,
                "split": entry.split,
                **computed,
            }
        )

    # Compute aggregates per language
    lang_stats: dict[str, Stats] = defaultdict(Stats)
    for entry in entries:
        lang_stats[entry.language].merge(entry.stats)

    for lang, stats in sorted(lang_stats.items()):
        computed = stats.compute()
        results.append(
            {
                "dataset": "[TOTAL]",
                "language": lang,
                "split": "[all]",
                **computed,
            }
        )

    # Compute aggregates per dataset
    dataset_stats: dict[str, Stats] = defaultdict(Stats)
    for entry in entries:
        dataset_stats[entry.dataset].merge(entry.stats)

    for dataset, stats in sorted(dataset_stats.items()):
        computed = stats.compute()
        results.append(
            {
                "dataset": dataset,
                "language": "[all]",
                "split": "[all]",
                **computed,
            }
        )

    # Compute global total
    global_stats = Stats()
    for entry in entries:
        global_stats.merge(entry.stats)

    computed = global_stats.compute()
    results.append(
        {
            "dataset": "[GLOBAL]",
            "language": "[all]",
            "split": "[all]",
            **computed,
        }
    )

    # Output
    fieldnames = [
        "dataset",
        "language",
        "split",
        "num_cuts",
        "duration_total_hrs",
        "duration_mean_sec",
        "duration_median_sec",
        "duration_std_sec",
        "words_total",
        "words_mean",
        "words_median",
        "words_std",
    ]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results written to {args.output}")
    else:
        # Print to stdout as table
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    global_computed = global_stats.compute()
    logger.info(
        f"\nGlobal summary: {global_computed['num_cuts']:,} cuts, "
        f"{global_computed['duration_total_hrs']:.2f} hours, "
        f"{global_computed['words_total']:,} words"
    )


if __name__ == "__main__":
    main()
