"""Convert LibriSpeech dataset from HuggingFace to Lhotse Shar format.

This script downloads the LibriSpeech dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Structure:
------------------
LibriSpeech has the following structure:
- config: "all" (single HuggingFace config)
- splits: train.clean.100, train.clean.360, train.other.500, validation.clean,
          validation.other, test.clean, test.other

We organize output as:
    {BASE_OUTPUT_DIR}/librispeech/{our_config}/{our_split}/
    Example: shar/librispeech/clean/train/

Reference: https://huggingface.co/datasets/openslr/librispeech_asr
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_subset_to_shar_batched


# --- Configuration ---
DATASET_NAME = "openslr/librispeech_asr"
DATASET_NICKNAME = "librispeech"
HF_CONFIG = "all"  # LibriSpeech uses a single config in HuggingFace
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# Dataset properties
DEFAULT_LANGUAGE = "en"

# Mapping from our config/split names to HuggingFace split names
# Format: our_config -> {our_split: hf_split}
CONFIGS_AND_SPLITS = {
    "clean": {
        "train": "train.clean.100+train.clean.360",  # Combined clean training
        "validation": "validation.clean",
        "test": "test.clean",
    },
    "other": {
        "train": "train.other.500",
        "validation": "validation.other",
        "test": "test.other",
    },
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_dir(config: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / config / split


def _marker_path_for_output(output_dir: Path) -> Path:
    try:
        rel = output_dir.relative_to(BASE_OUTPUT_DIR)
    except Exception:
        rel = Path(output_dir.name)
    marker = MARKER_ROOT / rel
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker.with_suffix(".done")


def is_conversion_complete(output_dir: Path) -> bool:
    """Check if conversion is already complete."""
    marker = _marker_path_for_output(output_dir)
    if marker.exists():
        return True
    return output_dir.exists() and any(output_dir.iterdir())


def mark_conversion_complete(output_dir: Path, count: int, errors: int) -> None:
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(f"Conversion completed successfully.\nCuts processed: {count}\nErrors: {errors}\n")
    logger.info(f"Created completion marker: {marker_path}")


def convert_subset_to_shar(
    config: str,
    split: str,
    hf_split: str,
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single configuration/split combination to Shar format.

    Args:
        config: Our configuration name (e.g., "clean", "other").
        split: Our split name (e.g., "train", "validation", "test").
        hf_split: HuggingFace split name (e.g., "train.clean.100").
        force: If True, re-run conversion even if already complete.
        num_workers: Number of parallel workers (default: cpu_count).
        hf_num_proc: Number of HuggingFace loading processes.

    Returns:
        Tuple of (count, errors) if processed, or (None, None) if skipped.
    """
    output_dir = get_output_dir(config, split)

    # Check if conversion is already complete
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(f"SKIPPING {config}/{split} - already complete (marker: {marker})")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {config}/{split} (HF split: {hf_split}) to: {output_dir}")

    # Use the batched converter with multiprocessing
    count, errors = convert_subset_to_shar_batched(
        dataset_name=DATASET_NAME,
        hf_config=HF_CONFIG,
        hf_split=hf_split,
        output_dir=output_dir,
        audio_format=AUDIO_FORMAT,
        shard_size=SHARD_SIZE,
        language=DEFAULT_LANGUAGE,
        num_workers=num_workers,
        hf_num_proc=hf_num_proc,
    )

    logger.info(f"Finished {config}/{split}! Processed {count} cuts with {errors} errors.")

    # Mark conversion as complete only if no errors
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking {config}/{split} as complete due to {errors} errors")

    return count, errors


def convert_all_to_shar(
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
):
    """Convert all configurations and splits to Shar format.

    Args:
        force: If True, re-run conversion even for already completed subsets.
        num_workers: Number of parallel workers (default: cpu_count).
        hf_num_proc: Number of HuggingFace loading processes.
    """
    (BASE_OUTPUT_DIR / DATASET_NICKNAME).mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")

    total_count = 0
    total_errors = 0
    skipped = 0

    for config, splits_mapping in CONFIGS_AND_SPLITS.items():
        for our_split, hf_split in splits_mapping.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {config}/{our_split} (HF: {hf_split})")
            logger.info(f"{'=' * 60}")

            count, errors = convert_subset_to_shar(
                config,
                our_split,
                hf_split,
                force=force,
                num_workers=num_workers,
                hf_num_proc=hf_num_proc,
            )

            if count is None or errors is None:
                skipped += 1
            else:
                total_count += count
                total_errors += errors

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! Processed: {total_count} cuts, Errors: {total_errors}, Skipped: {skipped} subsets")
    logger.info(f"{'=' * 60}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LibriSpeech dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default conversion with multiprocessing:
  python librispeech.py

  # Custom number of workers:
  python librispeech.py --num-workers 8

  # Force re-conversion of all subsets:
  python librispeech.py --force
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even for already completed subsets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for conversion (default: cpu_count).",
    )
    parser.add_argument(
        "--hf-num-proc",
        type=int,
        default=4,
        help="Number of HuggingFace data loading processes (default: 4).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_all_to_shar(
        force=args.force,
        num_workers=args.num_workers,
        hf_num_proc=args.hf_num_proc,
    )
