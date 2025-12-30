"""Convert People's Speech dataset from HuggingFace to Lhotse Shar format.

This script downloads the People's Speech dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training.

Dataset Dimensions:
-------------------
Different datasets have different organizational dimensions:

1. CONFIG (subset/configuration):
   - Refers to different versions or quality levels of the same dataset
   - For People's Speech: "clean", "dirty", "clean_sa", "dirty_sa", "microset"
   - These correspond to HuggingFace dataset configs (the second argument to load_dataset)
   - Controls data quality/licensing: clean=higher quality, dirty=noisier, _sa=CC-BY-SA license

2. SPLIT:
   - Standard ML data splits: "train", "validation", "test"
   - Used for training, hyperparameter tuning, and final evaluation
   - Passed as the `split` argument to load_dataset

3. LANGUAGE (not applicable for People's Speech):
   - For multilingual datasets, language would be another dimension
   - People's Speech is English-only, so we don't iterate over languages
   - For multilingual datasets, structure would be: {base}/{config}/{lang}/{split}/ or {base}/{lang}/{split}/

Output Directory Structure:
--------------------------
For People's Speech (monolingual with configs):
    {BASE_OUTPUT_DIR}/peoples_speech/{config}/{split}/
    Example: shar/peoples_speech/clean/train/

For a hypothetical multilingual dataset without configs:
    {BASE_OUTPUT_DIR}/{dataset_name}/{lang}/{split}/
    Example: shar/common_voice/en/train/

For a hypothetical multilingual dataset with configs:
    {BASE_OUTPUT_DIR}/{dataset_name}/{config}/{lang}/{split}/
    Example: shar/mls/default/en/train/

Processing Modes:
-----------------
This script supports two processing modes:

1. STREAMING (default): Memory-efficient, processes one item at a time.
   - Lower memory usage
   - No parallelization
   - Good for large datasets when memory is limited

2. BATCHED (optional): Faster processing with parallelization.
   - Loads data in batches (configurable size)
   - Parallel cut creation using multiple workers
   - Higher throughput but requires more RAM
   - Enable with --batched flag

Reference: https://huggingface.co/datasets/MLCommons/peoples_speech
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from datasets import Audio, load_dataset
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm


# --- Configuration ---
DATASET_NAME = "MLCommons/peoples_speech"
DATASET_NICKNAME = "peoples_speech"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"
DONE_MARKER = ".done"  # legacy name (kept for human-readable filenames)

# Dataset properties
IS_MULTILINGUAL = False
DEFAULT_LANGUAGE = "en"  # Used in supervision segments

# Define all configurations and their available splits
# For People's Speech: config -> list of splits
# Config corresponds to HuggingFace's dataset config (2nd arg to load_dataset)
# Split corresponds to HuggingFace's split argument
CONFIGS_AND_SPLITS = {
    "clean": ["train", "validation", "test"],
    "clean_sa": ["train", "validation", "test"],
    # "dirty": ["train", "validation", "test"],
    # "dirty_sa": ["train", "validation", "test"],
    # "microset": ["train"],
}

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_output_dir(
    config: str,
    split: str,
    lang: Optional[str] = None,
) -> Path:
    """Construct output directory path based on dataset structure.

    For monolingual datasets: {base}/{dataset}/{config}/{split}/
    For multilingual datasets: {base}/{dataset}/{config}/{lang}/{split}/
                           or: {base}/{dataset}/{lang}/{split}/ (if no configs)
    """
    base = BASE_OUTPUT_DIR / DATASET_NICKNAME
    if IS_MULTILINGUAL and lang:
        return base / config / lang / split
    return base / config / split


def _marker_path_for_output(output_dir: Path) -> Path:
    try:
        rel = output_dir.relative_to(BASE_OUTPUT_DIR)
    except Exception:
        rel = Path(output_dir.name)
    marker = MARKER_ROOT / rel
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker.with_suffix(".done")


def is_conversion_complete(output_dir: Path) -> bool:
    """Check if conversion is already complete (without touching leaf directories).

    Strategy:
    1. Check for a central marker file under BASE_OUTPUT_DIR/.conversion_markers
    2. Fallback to checking whether the output_dir exists and contains files
    """
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
    lang: Optional[str] = None,
    force: bool = False,
    use_batched: bool = False,
    batch_size: int = 5000,
    num_workers: int = 4,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single configuration/split/language combination to Shar format.

    Args:
        config: HuggingFace dataset configuration (e.g., "clean", "dirty").
        split: Data split (e.g., "train", "validation", "test").
        lang: Language code for multilingual datasets (not used for People's Speech).
        force: If True, re-run conversion even if already complete.
        use_batched: If True, use batched processing instead of streaming.
        batch_size: Number of items per batch (only used if use_batched=True).
        num_workers: Number of parallel workers for cut creation (only if batched).
        hf_num_proc: Number of HuggingFace loading processes (only if batched).

    Returns:
        Tuple of (count, errors) if processed, or (None, None) if skipped.
    """
    output_dir = get_output_dir(config, split, lang)
    lang_str = f"/{lang}" if lang else ""

    # Check if conversion is already complete
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(
            f"SKIPPING {config}{lang_str}/{split} - already complete (marker: {marker} or existing output files)"
        )
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {config}{lang_str}/{split} to: {output_dir}")

    # Language for supervision segments
    # For multilingual datasets, use the provided lang; otherwise use default
    supervision_lang: str = lang if (IS_MULTILINGUAL and lang) else DEFAULT_LANGUAGE

    # Choose processing mode
    if use_batched:
        # Use batched processing with parallelization
        logger.info(
            f"Using BATCHED mode: batch_size={batch_size}, num_workers={num_workers}, hf_num_proc={hf_num_proc}"
        )
        from batch_utils import convert_subset_to_shar_batched

        count, errors = convert_subset_to_shar_batched(
            dataset_name=DATASET_NAME,
            hf_config=config,
            hf_split=split,
            output_dir=output_dir,
            audio_format=AUDIO_FORMAT,
            shard_size=SHARD_SIZE,
            language=supervision_lang,
            batch_size=batch_size,
            num_workers=num_workers,
            hf_num_proc=hf_num_proc,
        )
    else:
        # Use streaming mode (default)
        logger.info("Using STREAMING mode (memory-efficient, sequential)")
        count, errors = _convert_streaming(
            hf_config=config,
            hf_split=split,
            output_dir=output_dir,
            supervision_lang=supervision_lang,
            config=config,
            split=split,
            lang_str=lang_str,
        )

    logger.info(f"Finished {config}{lang_str}/{split}! Processed {count} cuts with {errors} errors.")
    logger.info(f"Data is ready in: {output_dir}")

    # Mark conversion as complete
    mark_conversion_complete(output_dir, count, errors)

    return count, errors


def _convert_streaming(
    hf_config: str,
    hf_split: str,
    output_dir: Path,
    supervision_lang: str,
    config: str,
    split: str,
    lang_str: str,
) -> tuple[int, int]:
    """Convert using streaming mode (original implementation).

    Memory-efficient but sequential processing.
    """
    logger.info(f"Initializing stream for {DATASET_NAME} ({config}/{split})...")
    dataset = load_dataset(DATASET_NAME, hf_config, split=hf_split, streaming=True)

    # Prevent HF from decoding; get raw bytes
    dataset = dataset.cast_column("audio", Audio(decode=False))

    writer = SharWriter(output_dir=output_dir, fields={"recording": AUDIO_FORMAT}, shard_size=SHARD_SIZE)

    count = 0
    errors = 0

    logger.info(f"Starting conversion loop for {config}{lang_str}/{split}...")
    with writer:
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {config}{lang_str}/{split}", unit="cut")):
            try:
                # 1. Get raw info
                hf_id = item.get("id", f"no_id_{i}")
                cut_id = hf_id.replace("/", "_").replace(".", "_")
                audio_bytes = item["audio"]["bytes"]
                text = item["text"]

                # 2. Create Recording from raw bytes
                # This automatically detects duration and sampling rate from the file header
                recording = Recording.from_bytes(data=audio_bytes, recording_id=cut_id)

                # 3. Create Supervision
                supervision = SupervisionSegment(
                    id=cut_id,
                    recording_id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    text=text,
                    language=supervision_lang,
                )

                # 4. Create Cut
                cut = MonoCut(
                    id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    recording=recording,
                    supervisions=[supervision],
                )

                # 5. Write to Shar
                # SharWriter will decode the 'from_bytes' recording and re-encode
                # it to your target format (flac) inside the tar.
                writer.write(cut)
                count += 1

            except Exception as e:
                errors += 1
                logger.error(f"Failed to process item {i}: {e}")
                if errors > 100:
                    logger.critical("Too many errors, stopping script.")
                    break

    return count, errors


def convert_all_to_shar(
    force: bool = False,
    use_batched: bool = False,
    batch_size: int = 5000,
    num_workers: int = 4,
    hf_num_proc: int = 4,
):
    """Convert all configurations and splits to Shar format.

    Iterates over:
    - All configs (clean, dirty, clean_sa, dirty_sa, microset)
    - All splits per config (train, validation, test)
    - All languages (only "en" for People's Speech since it's monolingual)

    Args:
        force: If True, re-run conversion even for already completed subsets.
        use_batched: If True, use batched processing instead of streaming.
        batch_size: Number of items per batch (only used if use_batched=True).
        num_workers: Number of parallel workers for cut creation (only if batched).
        hf_num_proc: Number of HuggingFace loading processes (only if batched).
    """
    (BASE_OUTPUT_DIR / DATASET_NICKNAME).mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")

    if use_batched:
        logger.info(f"BATCHED MODE: batch_size={batch_size}, num_workers={num_workers}, hf_num_proc={hf_num_proc}")
    else:
        logger.info("STREAMING MODE (sequential, memory-efficient)")

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
                use_batched=use_batched,
                batch_size=batch_size,
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
  # Default streaming mode (memory-efficient):
  python peoples_speech.py

  # Batched mode with parallelization (faster but more RAM):
  python peoples_speech.py --batched

  # Batched mode with custom settings:
  python peoples_speech.py --batched --batch-size 10000 --num-workers 8

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
        "--batched",
        action="store_true",
        help="Use batched processing instead of streaming (faster but more RAM).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of items per batch when using batched mode (default: 5000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for cut creation in batched mode (default: 4).",
    )
    parser.add_argument(
        "--hf-num-proc",
        type=int,
        default=4,
        help="Number of HuggingFace data loading processes in batched mode (default: 4).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_all_to_shar(
        force=args.force,
        use_batched=args.batched,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hf_num_proc=args.hf_num_proc,
    )
