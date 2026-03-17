"""Convert IWSLT2026 Metrics Shared Task dataset from HuggingFace to Lhotse Shar format.

This script downloads the IWSLT2026 Metrics Shared Task dataset from HuggingFace and
converts it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Information:
--------------------
- HF id: maikezu/iwslt2026-metrics-shared-train-dev
- Task: Speech Translation Quality Estimation
- Source language: English (always)
- Target languages: Varies (de, zh, ja, etc.) - determined by `tgt_lang` field

Dataset Dimensions:
-------------------
1. TARGET LANGUAGE (tgt_lang):
   - Dataset is sliced by target language
   - Each language gets its own subfolder

2. SPLIT:
   - train: Mix of IWSLT 2023, WMT 2024, WMT 2025 human annotations
   - train_synthetic: SpeechQE data with automatic (synthetic) scores
   - dev: IWSLT 2025 ACL Talks human annotations

Output Directory Structure:
--------------------------
    {BASE_OUTPUT_DIR}/iwslt2026_metrics/{tgt_lang}/{split}/
    Example: shar/iwslt2026_metrics/de/train/

Features stored in custom metadata:
-----------------------------------
- audio_path: Path to the audio file
- doc_id: Unique identifier for the segment/document
- src_text: Source text
- src_text_system: Source text system (e.g. human, ASR model)
- src_lang: Source language code (e.g., en)
- tgt_text: Target text translation
- tgt_lang: Target language code (e.g., de)
- domain: Domain or dataset source
- tgt_system: Target system or model used for translation
- score: Human or synthetic evaluation score (0–1)

Reference: https://huggingface.co/datasets/maikezu/iwslt2026-metrics-shared-train-dev
"""

import argparse
import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
import pandas as pd

from datasets import Audio, load_dataset, concatenate_datasets, DatasetDict
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm
from typing import Optional
from sklearn.model_selection import train_test_split
import json

# --- Configuration ---

#ls train, voxpoplui
DATASET_NAME = "maikezu/iwslt2026-metrics-shared-train-dev"  # HuggingFace dataset ID
DATASET_NICKNAME = "iwslt_metrics" 
HF_CONFIG = "default"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/scratch-artemis/sonal/IWSLT26/data/"))

# Average audio length is 8-9 seconds
# ~8.5 sec avg @ FLAC 16kHz → ~135 KB/sample → ~2GB shards = ~15000 samples
SHARD_SIZE = 15000
AUDIO_FORMAT = "flac"
TARGET_SR = 16000
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# Dataset properties
SOURCE_LANGUAGE = "en"  # Source is always English
SPLITS = ["train"]

# Fields to store in custom metadata (all dataset fields except audio)
CUSTOM_FIELDS = [
    "audio_path",
    "doc_id",
    "src_text",
    "src_text_system",
    "src_lang",
    "tgt_text",
    "tgt_lang",
    "domain",
    "tgt_system",
    "score",
]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_dir(tgt_lang: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / tgt_lang / split


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


def _extract_custom_metadata(item: dict) -> dict:
    """Extract all custom fields from a dataset item."""
    custom = {}
    for field in CUSTOM_FIELDS:
        if field == "doc_id":
            value = item.get("audio_id")
        if field == "src_text":
            value = item.get("raw_text")
        if field not in item.keys():
            value = "" # Not available in the dataset, set to empty string
            print(f"Warning. Field {field} not found in dataset item. Setting to empty string.")
        value = item.get(field)
        if value is not None:
            # Convert to appropriate type for JSON serialization
            if isinstance(value, (int, float, str, bool)):
                custom[field] = value
            else:
                custom[field] = str(value)
    return custom


def discover_target_languages(split: str) -> list[str]:
    """Discover all unique target languages in a split.

    Args:
        split: The HuggingFace split name.

    Returns:
        Sorted list of unique target language codes.
    """
    logger.info(f"Discovering target languages in split '{split}'...")

    # Load dataset in streaming mode to avoid downloading everything
    dataset = load_dataset(
        DATASET_NAME,
        HF_CONFIG,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    # Collect unique tgt_lang values
    languages = set()
    for item in tqdm(dataset, desc=f"Scanning {split} for languages", unit="item"):
        tgt_lang = item.get("tgt_lang")
        if tgt_lang:
            languages.add(tgt_lang)

    sorted_langs = sorted(languages)
    logger.info(f"Found {len(sorted_langs)} target languages in '{split}': {sorted_langs}")
    return sorted_langs


def write_out(dataset: str,
    split: str,
    force: bool = False,
    ):
    # Don't decode audio - get raw bytes
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Group items by target language
    logger.info("Grouping items by fixed target languages [de or zh]...")
    items_by_lang: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(tqdm(dataset, desc="Grouping by tgt_lang", unit="item")):
        tgt_lang = item.get("tgt_lang", "unknown")
        #language split to de and zh only!
        # if tgt_lang not in ["de","zh"]:
        #     continue
        # else:
        items_by_lang[tgt_lang].append(idx)

    logger.info(f"Found {len(items_by_lang)} target languages: {sorted(items_by_lang.keys())}")
    for lang, indices in sorted(items_by_lang.items()):
        logger.info(f"  {lang}: {len(indices)} items")

    total_count = 0
    total_errors = 0
    skipped = 0

    # Process each language
    for tgt_lang, indices in sorted(items_by_lang.items()):
        output_dir = get_output_dir(tgt_lang, split)

        # Check if already complete
        if not force and is_conversion_complete(output_dir):
            marker = _marker_path_for_output(output_dir)
            logger.info(f"SKIPPING {tgt_lang}/{split} - already complete (marker: {marker})")
            skipped += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Converting {tgt_lang}/{split} ({len(indices)} items) to: {output_dir}")

        writer = SharWriter(
            output_dir=output_dir,
            fields={"recording": AUDIO_FORMAT},
            shard_size=SHARD_SIZE,
        )

        count = 0
        errors = 0

        with writer:
            for idx in tqdm(indices, desc=f"Processing {tgt_lang}/{split}", unit="cut"):
                try:
                    item = dataset[idx]

                    # Generate cut ID from doc_id or index
                    doc_id = item.get("doc_id", f"item_{idx}")
                    cut_id = f"{tgt_lang}_{split}_{doc_id}".replace("/", "_").replace(".", "_").replace(" ", "_")

                    # Extract audio bytes
                    audio_data = item["src_audio"]
                    audio_bytes = audio_data.get("bytes")

                    if audio_bytes is None:
                        audio_path = audio_data.get("path")
                        if audio_path:
                            with open(audio_path, "rb") as f:
                                audio_bytes = f.read()

                    if audio_bytes is None:
                        logger.warning(f"No audio bytes for item {cut_id}, skipping")
                        errors += 1
                        continue

                    # Create Recording from raw bytes
                    recording = Recording.from_bytes(data=audio_bytes, recording_id=cut_id)

                    # Resample to target SR if needed
                    if recording.sampling_rate != TARGET_SR:
                        recording = recording.resample(TARGET_SR)

                    # Use tgt_text as the main text field (this is a translation dataset)
                    text = item.get("tgt_text", "")
                    if not isinstance(text, str):
                        text = str(text) if text is not None else ""

                    # Create Supervision with target language
                    supervision = SupervisionSegment(
                        id=cut_id,
                        recording_id=cut_id,
                        start=0.0,
                        duration=recording.duration,
                        text=text,
                        language=tgt_lang,
                    )

                    # Extract custom metadata (all dataset fields)
                    custom = _extract_custom_metadata(item)

                    # Create Cut with custom metadata
                    cut = MonoCut(
                        id=cut_id,
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        recording=recording,
                        supervisions=[supervision],
                        custom=custom if custom else None,
                    )

                    writer.write(cut)
                    count += 1

                except Exception as e:
                    errors += 1
                    logger.error(f"Failed to process item {idx}: {e}")
                    if errors > 100:
                        logger.critical("Too many errors, stopping.")
                        break

        logger.info(f"Finished {tgt_lang}/{split}! Processed {count} cuts with {errors} errors.")

        # Mark as complete only if no errors
        if errors == 0:
            mark_conversion_complete(output_dir, count, errors)
        else:
            logger.warning(f"Not marking {tgt_lang}/{split} as complete due to {errors} errors")

        total_count += count
        total_errors += errors

    return total_count, total_errors, skipped

def convert_split_by_language(
    split: str,
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
    split_2: Optional[str] = None,
    split_ratio: Optional[float] = None,
) -> tuple[int, int, int]:
    """Convert a single split, creating subfolders per target language.

    Args:
        split: HuggingFace split name (e.g., "train", "dev").
        force: If True, re-run conversion even if already complete.
        num_workers: Number of parallel workers (not used in streaming mode).
        hf_num_proc: Number of HuggingFace loading processes.

    Returns:
        Tuple of (total_count, total_errors, skipped_languages).
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing split: {split}")
    logger.info(f"{'=' * 60}")


    # Load the full split
    logger.info(f"Loading dataset {DATASET_NAME} (config={HF_CONFIG}, split={split})...")
    dataset = load_dataset(
        DATASET_NAME,
        HF_CONFIG,
        split=split,
        num_proc=hf_num_proc,
        trust_remote_code=True,
    )
    #split 
    if split_2:
        logger.info(f"Adding synthetic data to train mix")
        dataset_synthetic = load_dataset(
        DATASET_NAME,
        HF_CONFIG,
        split=split_2,
        num_proc=hf_num_proc,
        trust_remote_code=True,
        )
        
        train_data = concatenate_datasets([dataset, dataset_synthetic])
        dataset = train_data

    logger.info(f"Loaded {len(dataset)} items")

    splits = None
    if split_ratio > 0:
        logger.info(f"Creating train/valid split with {split_ratio} ratio")

        #to stratify
        indices = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(
            indices, train_size=split_ratio, 
            stratify=dataset['tgt_lang'], 
            random_state=42
        )

        train_dataset = dataset.select(train_idx)
        test_dataset = dataset.select(test_idx)

        train_valid_dataset = DatasetDict({
            'train': train_dataset,
            'valid': test_dataset
        })
        dataset = train_valid_dataset
        splits = dataset.keys()

    if splits is not None:
        total_count, total_errors, skipped = 0,0,0
        for splt in splits:
            logger.info(f"Writing {splt} data from split")
            count, errors, skip = write_out(dataset[splt], splt, force)
            total_count += count
            total_errors += errors
            skipped += skip
    else:
        if split == "dev":
            #change split name to test -> iwslt[dev] is our test set
            split = "test"
        logger.info(f"Writing {split} data")
        total_count, total_errors, skipped = write_out(dataset, split, force)
    
    return total_count, total_errors, skipped   

def convert_all_to_shar(
    force: bool = False,
    splits: list[str] | None = None,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
    split_ratio: Optional[float] = 0.3,
    synthetic: Optional[bool] = True,
):
    """Convert all splits to Shar format, organized by target language.

    Args:
        force: If True, re-run conversion even for already completed subsets.
        splits: List of splits to convert. If None, uses all splits.
        num_workers: Number of parallel workers (not used currently).
        hf_num_proc: Number of HuggingFace loading processes.
    """
    if splits is None:
        splits = SPLITS

    (BASE_OUTPUT_DIR / DATASET_NICKNAME).mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")
    logger.info(f"Splits to process: {splits}")

    grand_total_count = 0
    grand_total_errors = 0
    grand_total_skipped = 0

    for split in splits:
        if split == "train" and synthetic:
            # for the train split and if synthetic, then we set the second split to train_synthetic. Otherwise, we do not use train_synthetic at all.
            split_2 =  "train_synthetic"
        else:
            # for dev split and synthetic = False, we do not use the second split.
            split_2 = None

        #now, after concatenating or not the train set, we need to create a dev split from it ONLY if its train, otherwise no splitting. 
        count, errors, skipped = convert_split_by_language(
            split=split,
            force=force,
            num_workers=num_workers,
            hf_num_proc=hf_num_proc,
            split_2=split_2,
            split_ratio=split_ratio if split == "train" else 0,
        )
        grand_total_count += count
        grand_total_errors += errors
        grand_total_skipped += skipped

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"ALL DONE! Processed: {grand_total_count} cuts, "
        f"Errors: {grand_total_errors}, Skipped: {grand_total_skipped} language/split combos"
    )
    logger.info(f"{'=' * 60}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert IWSLT2026 Metrics Shared Task dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Convert all splits:
            python iwslt2026_metrics.py

            # Convert only train and dev:
            python iwslt2026_metrics.py --splits train dev

            # Force re-conversion of all subsets:
            python iwslt2026_metrics.py --force

            # Custom number of HF loading processes:
            python iwslt2026_metrics.py --hf-num-proc 8
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even for already completed subsets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=SPLITS,
        help=f"Splits to convert. If omitted, converts all: {SPLITS}",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count). Currently not used.",
    )
    parser.add_argument(
        "--hf-num-proc",
        type=int,
        default=4,
        help="Number of HuggingFace data loading processes (default: 4).",
    )
    parser.add_argument(
        "--sratio",
        type=float,
        default=0.1,
        help="Train/dev split ratio from data. Choose a number between 0 and 1, default 0.1. Put 0 to not split the data."
    )

    parser.add_argument(
        "--synthetic",
        type=bool,
        default=False,
        help="Include synthetic train split in training. Default True."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_all_to_shar(
        force=args.force,
        splits=args.splits,
        num_workers=args.num_workers,
        hf_num_proc=args.hf_num_proc,
        split_ratio = float(args.sratio),
        synthetic = bool(args.synthetic),
    )
