"""Convert People's Speech dataset from HuggingFace to Lhotse Shar format.

This script downloads the People's Speech dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Dimensions:
-------------------
1. CONFIG (subset/configuration):
   - For People's Speech: "clean", "dirty", "clean_sa", "dirty_sa", "microset"
   - These correspond to HuggingFace dataset configs

2. SPLIT:
   - Standard ML data splits: "train", "validation", "test"

Output Directory Structure:
--------------------------
    {BASE_OUTPUT_DIR}/peoples_speech/{config}/{split}/
    Example: shar/peoples_speech/clean/train/

Reference: https://huggingface.co/datasets/MLCommons/peoples_speech
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_subset_to_shar_batched


# --- Configuration ---
DATASET_NAME = "MLCommons/peoples_speech"
DATASET_NICKNAME = "peoples_speech"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# Dataset properties
DEFAULT_LANGUAGE = "en"

# Define all configurations and their available splits
CONFIGS_AND_SPLITS = {
    "clean": ["train", "validation", "test"],
    # "clean_sa": ["train", "validation", "test"],
    # "dirty": ["train", "validation", "test"],
    # "dirty_sa": ["train", "validation", "test"],
    # "microset": ["train"],
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
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single configuration/split combination to Shar format.

    Args:
        config: HuggingFace dataset configuration (e.g., "clean", "dirty").
        split: Data split (e.g., "train", "validation", "test").
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
    logger.info(f"Converting {config}/{split} to: {output_dir}")

    # Use the batched converter with multiprocessing
    count, errors = convert_subset_to_shar_batched(
        dataset_name=DATASET_NAME,
        hf_config=config,
        hf_split=split,
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

    for config, splits in CONFIGS_AND_SPLITS.items():
        for split in splits:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing configuration: {config}, split: {split}")
            logger.info(f"{'=' * 60}")

            count, errors = convert_subset_to_shar(
                config,
                split,
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
        description="Convert People's Speech dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default conversion with multiprocessing:
  python peoples_speech.py

  # Custom number of workers:
  python peoples_speech.py --num-workers 8

  # Force re-conversion of all subsets:
  python peoples_speech.py --force
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
