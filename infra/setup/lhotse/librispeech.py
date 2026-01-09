"""Convert LibriSpeech dataset from HuggingFace to Lhotse Shar format.

This script downloads the LibriSpeech dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training.

Dataset Dimensions:
-------------------
LibriSpeech has a unique structure where splits encode both the quality (clean/other)
and the data partition (train/validation/test):

1. CONFIG (quality level):
   - "clean": Higher quality recordings with clearer speech and less noise
   - "other": More challenging recordings with varied accents and recording quality
   - These are NOT HuggingFace configs - LibriSpeech uses "all" config
   - Instead, quality is encoded in the split names

2. SPLIT (data partition):
   - For "clean": train.100, train.360, validation, test
   - For "other": train.500, validation, test
   - HuggingFace split names: train.clean.100, train.clean.360, validation.clean,
     test.clean, train.other.500, validation.other, test.other

3. LANGUAGE:
   - English only (monolingual dataset)

HuggingFace to Output Mapping:
------------------------------
HuggingFace split name     -> Output directory
train.clean.100            -> librispeech/clean/train.100/
train.clean.360            -> librispeech/clean/train.360/
validation.clean           -> librispeech/clean/validation/
test.clean                 -> librispeech/clean/test/
train.other.500            -> librispeech/other/train.500/
validation.other           -> librispeech/other/validation/
test.other                 -> librispeech/other/test/

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

Reference: https://huggingface.co/datasets/openslr/librispeech_asr
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
DATASET_NAME = "openslr/librispeech_asr"
DATASET_NICKNAME = "librispeech"
HF_CONFIG = "all"  # LibriSpeech uses a single "all" config, splits encode quality
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 2000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"
DONE_MARKER = ".done"  # legacy name (kept for human-readable filenames)

# Dataset properties
IS_MULTILINGUAL = False
DEFAULT_LANGUAGE = "en"  # Used in supervision segments

# Mapping from our config/split to HuggingFace split names
# Structure: {our_config: {our_split: hf_split_name}}
CONFIGS_AND_SPLITS = {
    "clean": {
        "train.100": "train.clean.100",
        "train.360": "train.clean.360",
        "validation": "validation.clean",
        "test": "test.clean",
    },
    "other": {
        "train.500": "train.other.500",
        "validation": "validation.other",
        "test": "test.other",
    },
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_dir(
    config: str,
    split: str,
    lang: Optional[str] = None,
) -> Path:
    """Construct output directory path based on dataset structure.

    For LibriSpeech: {base}/{dataset}/{config}/{split}/
    Example: shar/librispeech/clean/train.100/
    """
    base = BASE_OUTPUT_DIR / DATASET_NICKNAME
    if IS_MULTILINGUAL and lang:
        return base / config / lang / split
    return base / config / split


def _marker_path_for_output(output_dir: Path) -> Path:
    """Return a central marker path for a given output directory.

    The project should not write marker files into leaf data folders because
    Lhotse and other tools may expect certain files there. Instead we store
    markers under BASE_OUTPUT_DIR/.conversion_markers/<relative-path>.done
    """
    try:
        rel = output_dir.relative_to(BASE_OUTPUT_DIR)
    except Exception:
        # If output_dir isn't inside BASE_OUTPUT_DIR for some reason,
        # fallback to using the output dir name as the marker name
        rel = Path(output_dir.name)
    marker = MARKER_ROOT / rel
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker.with_suffix(".done")


def is_conversion_complete(output_dir: Path) -> bool:
    """Check if conversion is already complete.

    Strategy (in order):
    1. If a central marker file exists under BASE_OUTPUT_DIR/.conversion_markers,
       treat the conversion as complete.
    2. Otherwise, if the output directory exists and contains any files,
       assume conversion was already run (useful for manual/partial runs).
    This avoids writing anything into leaf output directories.
    """
    marker = _marker_path_for_output(output_dir)
    if marker.exists():
        return True
    # Fallback: if the output dir already exists and is non-empty, treat
    # it as complete so we avoid re-downloading/re-converting. This is a
    # simpler mechanism that doesn't require writing markers at all.
    return output_dir.exists() and any(output_dir.iterdir())


def mark_conversion_complete(output_dir: Path, count: int, errors: int) -> None:
    """Record a completion marker in the central marker directory.

    We do NOT write into the leaf output directory to avoid confusion with
    Lhotse's own on-disk artifacts. The marker contains basic metadata and
    is stored under BASE_OUTPUT_DIR/.conversion_markers.
    """
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(f"Conversion completed successfully.\nCuts processed: {count}\nErrors: {errors}\n")
    logger.info(f"Created completion marker: {marker_path}")


def convert_subset_to_shar(
    config: str,
    split: str,
    hf_split: str,
    lang: Optional[str] = None,
    force: bool = False,
    use_batched: bool = False,
    batch_size: int = 5000,
    num_workers: int = 4,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single configuration/split combination to Shar format.

    Args:
        config: Our configuration name (e.g., "clean", "other").
        split: Our split name (e.g., "train.100", "validation", "test").
        hf_split: HuggingFace split name (e.g., "train.clean.100", "validation.clean").
        lang: Language code for multilingual datasets (not used for LibriSpeech).
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
            hf_config=HF_CONFIG,
            hf_split=hf_split,
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
            hf_split=hf_split,
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
    logger.info(f"Initializing stream for {DATASET_NAME} (config={HF_CONFIG}, split={hf_split})...")
    dataset = load_dataset(DATASET_NAME, HF_CONFIG, split=hf_split, streaming=True)

    # Prevent HF from decoding; get raw bytes
    dataset = dataset.cast_column("audio", Audio(decode=False))

    writer = SharWriter(output_dir=output_dir, fields={"recording": AUDIO_FORMAT}, shard_size=SHARD_SIZE)

    count = 0
    errors = 0

    logger.info(f"Starting conversion loop for {config}{lang_str}/{split}...")
    with writer:
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {config}{lang_str}/{split}", unit="cut")):
            try:
                # 1. Get raw info - LibriSpeech uses "id" field
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
    - All configs (clean, other)
    - All splits per config (train.100, train.360, validation, test for clean;
      train.500, validation, test for other)

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

    for config, splits_mapping in CONFIGS_AND_SPLITS.items():
        for our_split, hf_split in splits_mapping.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing configuration: {config}, split: {our_split} (HF: {hf_split})")
            logger.info(f"{'=' * 60}")
            count, errors = convert_subset_to_shar(
                config,
                our_split,
                hf_split,
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
        description="Convert LibriSpeech dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default streaming mode (memory-efficient):
  python librispeech.py

  # Batched mode with parallelization (faster but more RAM):
  python librispeech.py --batched

  # Batched mode with custom settings:
  python librispeech.py --batched --batch-size 10000 --num-workers 8

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
