"""Convert espnet/yodas-granary dataset from HuggingFace to Lhotse Shar format.

This script downloads the YODAS-Granary dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Structure:
------------------
YODAS-Granary provides high-quality pseudo-labeled speech data across 23
European languages for two tasks:
  - ASR (Automatic Speech Recognition): transcriptions in the source language.
  - AST (Automatic Speech Translation): translations into English.

The HuggingFace dataset is loaded via:
    load_dataset("espnet/yodas-granary", "<Language>", split="<ast|asr_only>")

All non-English languages have both "ast" and "asr_only" splits.
English has only "asr_only".

Fields per sample:
  utt_id, audio, duration, lang, task, text, translation_en,
  original_audio_id, original_audio_offset

Stored in Shar cuts:
  - supervision.text: Source-language transcription (from 'text' field).
  - cut.custom: Contains extra metadata fields:
      duration, translation_en, original_audio_id, original_audio_offset.
    Note: 'translation_en' is null for asr_only samples.

Output Directory Structure:
--------------------------
    {BASE_OUTPUT_DIR}/yodas-granary/{Language}/{split}/
    Example: shar/yodas-granary/Italian/asr_only/
             shar/yodas-granary/Italian/ast/

Reference: https://huggingface.co/datasets/espnet/yodas-granary
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_subset_to_shar_batched


# --- Configuration ---
DATASET_NAME = "espnet/yodas-granary"
DATASET_NICKNAME = "yodas-granary"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# All available languages
ALL_LANGUAGES = [
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Slovak",
    "Spanish",
    "Swedish",
    "Ukrainian",
]

# Language name → ISO 639-1 code (for supervision segments)
LANGUAGE_CODES: dict[str, str] = {
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hungarian": "hu",
    "Italian": "it",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Spanish": "es",
    "Swedish": "sv",
    "Ukrainian": "uk",
}

# Splits available per language. All have both; English only has asr_only.
AST_AND_ASR = ["ast", "asr_only"]
ASR_ONLY = ["asr_only"]


# Default languages to convert (smaller ones for quick testing)
DEFAULT_LANGUAGES = ALL_LANGUAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _splits_for_language(lang: str) -> list[str]:
    """Return the available splits for a given language."""
    return ASR_ONLY if lang == "English" else AST_AND_ASR


def get_output_dir(lang: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / lang / split


def _marker_path_for_output(output_dir: Path) -> Path:
    """Derive a completion-marker file path from the output directory."""
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
    """Write a completion marker after a successful conversion."""
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(
        f"Conversion completed successfully.\n"
        f"Cuts processed: {count}\n"
        f"Errors: {errors}\n"
    )
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
        lang: Language name (e.g., "Italian", "English").
        split: Data split ("ast" or "asr_only").
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

    iso_code = LANGUAGE_CODES.get(lang, "und")

    # Use the batched converter with multiprocessing
    count, errors = convert_subset_to_shar_batched(
        dataset_name=DATASET_NAME,
        hf_config=lang,
        hf_split=split,
        output_dir=output_dir,
        audio_format=AUDIO_FORMAT,
        shard_size=SHARD_SIZE,
        language=iso_code,
        num_workers=num_workers,
        hf_num_proc=hf_num_proc,
        id_field="utt_id",
        text_field="text",
        audio_field="audio",
        custom_fields=["duration", "translation_en", "original_audio_id", "original_audio_offset"],
    )

    logger.info(f"Finished {lang}/{split}! Processed {count} cuts with {errors} errors.")

    # Mark conversion as complete only if no errors
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking {lang}/{split} as complete due to {errors} errors")

    return count, errors


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert espnet/yodas-granary dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Convert default (smaller) languages, both splits:
  python convert_yodas-granary.py

  # Convert all languages:
  python convert_yodas-granary.py --configs {' '.join(ALL_LANGUAGES[:3])} ...

  # Convert specific languages:
  python convert_yodas-granary.py --configs Italian German French

  # Convert only the ASR split:
  python convert_yodas-granary.py --splits asr_only

  # Custom number of workers:
  python convert_yodas-granary.py --num-workers 8

  # Force re-conversion:
  python convert_yodas-granary.py --force

Available languages: {', '.join(ALL_LANGUAGES)}
        """,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=f"Language names to convert (default: {DEFAULT_LANGUAGES}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Splits to convert. If not specified, converts all available splits "
            "per language (ast + asr_only for non-English, asr_only for English)."
        ),
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
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Cuts per SHAR shard (default: {SHARD_SIZE}).",
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

    global SHARD_SIZE
    SHARD_SIZE = args.shard_size

    configs = args.configs if args.configs else DEFAULT_LANGUAGES

    # Validate language names
    for lang in configs:
        if lang not in ALL_LANGUAGES:
            logger.warning(
                f"Language '{lang}' not in known languages: {ALL_LANGUAGES}. "
                "Make sure the name matches the HuggingFace config exactly."
            )

    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")
    logger.info(f"Languages: {configs}")

    total_count = 0
    total_errors = 0
    skipped = 0

    for lang in configs:
        # Determine splits: CLI override or per-language defaults
        if args.splits is not None:
            splits = args.splits
        else:
            splits = _splits_for_language(lang)

        # Warn if requesting 'ast' for English
        if lang == "English" and "ast" in splits:
            logger.warning("English does not have an 'ast' split — skipping it.")
            splits = [s for s in splits if s != "ast"]

        for split in splits:
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
