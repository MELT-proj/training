"""Convert VoxPopuli dataset from HuggingFace to Lhotse Shar format.

This script downloads the VoxPopuli dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Structure:
------------------
VoxPopuli is organized by language. Each language has train/validation/test splits.
The dataset uses "raw_text" as the transcription field.

Output Directory Structure:
--------------------------
    {BASE_OUTPUT_DIR}/voxpopuli/{lang}/{split}/
    Example: shar/voxpopuli/en/train/

Reference: https://huggingface.co/datasets/facebook/voxpopuli
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_subset_to_shar_batched


# --- Configuration ---
DATASET_NAME = "facebook/voxpopuli"
DATASET_NICKNAME = "voxpopuli"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# Available ASR languages (VoxPopuli ASR subset)
_ASR_LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
]

# Default languages to convert
DEFAULT_LANGUAGES = ["en", "de", "fr", "es", "it"]

# Default splits
DEFAULT_SPLITS = ["train", "validation", "test"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_dir(lang: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / lang / split


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


def convert_one(
    lang: str,
    split: str,
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single language/split combination to Shar format.

    Args:
        lang: Language code (e.g., "en", "de", "fr").
        split: Data split (e.g., "train", "validation", "test").
        force: If True, re-run conversion even if already complete.
        num_workers: Number of parallel workers (default: cpu_count).
        hf_num_proc: Number of HuggingFace loading processes.

    Returns:
        Tuple of (count, errors) if processed, or (None, None) if skipped.
    """
    output_dir = get_output_dir(lang, split)

    # Check if conversion is already complete
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(f"SKIPPING {lang}/{split} - already complete (marker: {marker})")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {lang}/{split} to: {output_dir}")

    # Use the batched converter with multiprocessing
    count, errors = convert_subset_to_shar_batched(
        dataset_name=DATASET_NAME,
        hf_config=lang,  # VoxPopuli uses language as config
        hf_split=split,
        output_dir=output_dir,
        audio_format=AUDIO_FORMAT,
        shard_size=SHARD_SIZE,
        language=lang,
        num_workers=num_workers,
        hf_num_proc=hf_num_proc,
        text_field="raw_text",  # VoxPopuli uses "raw_text" field
    )

    logger.info(f"Finished {lang}/{split}! Processed {count} cuts with {errors} errors.")

    # Mark conversion as complete only if no errors
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking {lang}/{split} as complete due to {errors} errors")

    return count, errors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert VoxPopuli dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Convert default languages:
  python voxpopuli.py

  # Convert all ASR languages:
  python voxpopuli.py --configs {' '.join(_ASR_LANGUAGES)}

  # Convert specific languages:
  python voxpopuli.py --configs en de fr

  # Custom number of workers:
  python voxpopuli.py --num-workers 8

  # Force re-conversion:
  python voxpopuli.py --force

Available ASR languages: {', '.join(_ASR_LANGUAGES)}
        """,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=f"Language codes to convert (default: {DEFAULT_LANGUAGES}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to convert (default: {DEFAULT_SPLITS}).",
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, ...).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    configs = args.configs if args.configs else DEFAULT_LANGUAGES

    # Validate languages
    for lang in configs:
        if lang not in _ASR_LANGUAGES:
            logger.warning(f"Language '{lang}' not in known ASR languages: {_ASR_LANGUAGES}")

    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")
    logger.info(f"Languages: {configs}")
    logger.info(f"Splits: {args.splits}")

    total_count = 0
    total_errors = 0
    skipped = 0

    for lang in configs:
        for split in args.splits:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {lang}/{split}")
            logger.info(f"{'=' * 60}")

            count, errors = convert_one(
                lang,
                split,
                force=args.force,
                num_workers=args.num_workers,
                hf_num_proc=args.hf_num_proc,
            )

            if count is None:
                skipped += 1
            else:
                total_count += count
                total_errors += errors

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! Processed: {total_count} cuts, Errors: {total_errors}, Skipped: {skipped} subsets")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
