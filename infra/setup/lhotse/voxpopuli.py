"""Convert VoxPopuli from HuggingFace to Lhotse Shar format.

Dataset:
  - HF id: facebook/voxpopuli
  - Multilingual: yes

HF configs:
  - Per-language configs (e.g. "hr", "de", "fr", ...)
  - Special configs: "multilang" (all languages), "en_accented" (test-only)

This converter focuses on per-language configs by default.

Output directory structure:
  {LHOTSE_DATA_SHAR_ROOT}/voxpopuli/{lang}/{split}/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from datasets import Audio, load_dataset
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm


DATASET_NAME = "facebook/voxpopuli"
DATASET_NICKNAME = "voxpopuli"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))

SHARD_SIZE = 1000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

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
_ASR_ACCENTED_LANGUAGES = [
    "en_accented",
]

logger = logging.getLogger(__name__)


def get_output_dir(lang: str, split: str) -> Path:
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
    marker = _marker_path_for_output(output_dir)
    if marker.exists():
        return True
    return output_dir.exists() and any(output_dir.iterdir())


def mark_conversion_complete(output_dir: Path, count: int, errors: int) -> None:
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(f"Conversion completed successfully.\nCuts processed: {count}\nErrors: {errors}\n")
    logger.info("Created completion marker: %s", marker_path)


def _pick_text(item: dict) -> str:
    # VoxPopuli provides raw and normalized transcripts.
    for key in ("normalized_text", "raw_text", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _splits_for_config(cfg: str) -> list[str]:
    if cfg == "en_accented":
        return ["test"]
    return ["train", "validation", "test"]


def convert_one(
    lang: str,
    split: str,
    *,
    force: bool,
    use_batched: bool,
    batch_size: int,
    num_workers: int,
    io_num_workers: int,
    prefetch_batches: int,
    hf_num_proc: int,
) -> tuple[int, int] | tuple[None, None]:
    output_dir = get_output_dir(lang, split)

    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info("SKIPPING %s/%s - already complete (marker: %s)", lang, split, marker)
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s/%s to: %s", lang, split, output_dir)

    if use_batched:
        from batch_utils import convert_subset_to_shar_batched

        count, errors = convert_subset_to_shar_batched(
            dataset_name=DATASET_NAME,
            hf_config=lang,
            hf_split=split,
            output_dir=output_dir,
            audio_format=AUDIO_FORMAT,
            shard_size=SHARD_SIZE,
            language=lang,
            batch_size=batch_size,
            num_workers=num_workers,
            io_num_workers=io_num_workers,
            prefetch_batches=prefetch_batches,
            hf_num_proc=hf_num_proc,
            id_field="audio_id",
            text_field="normalized_text",
            audio_field="audio",
        )
    else:
        ds = load_dataset(DATASET_NAME, lang, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        writer = SharWriter(output_dir=output_dir, fields={"recording": AUDIO_FORMAT}, shard_size=SHARD_SIZE)
        count = 0
        errors = 0

        with writer:
            for i, item in enumerate(tqdm(ds, desc=f"Processing {lang}/{split}", unit="cut")):
                try:
                    hf_id = item.get("audio_id", item.get("id", f"no_id_{i}"))
                    cut_id = str(hf_id).replace("/", "_").replace(".", "_")

                    audio_bytes = item["audio"]["bytes"]
                    text = _pick_text(item)

                    recording = Recording.from_bytes(data=audio_bytes, recording_id=cut_id)
                    supervision = SupervisionSegment(
                        id=cut_id,
                        recording_id=cut_id,
                        start=0.0,
                        duration=recording.duration,
                        text=text,
                        language=lang,
                    )
                    cut = MonoCut(
                        id=cut_id,
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        recording=recording,
                        supervisions=[supervision],
                    )

                    writer.write(cut)
                    count += 1
                except Exception as e:
                    errors += 1
                    logger.error("Failed to process item %s: %s", i, e)
                    if errors > 100:
                        logger.critical("Too many errors, stopping.")
                        break

    logger.info("Finished %s/%s! Processed %s cuts with %s errors.", lang, split, count, errors)
    mark_conversion_complete(output_dir, count, errors)
    return count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VoxPopuli to Lhotse SHAR archives")

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="HF config names to convert (e.g. hr de fr). If omitted, uses the built-in ASR language list.",
    )
    parser.add_argument(
        "--include-special-configs",
        action="store_true",
        help="Include special configs like 'en_accented' when using the default config list.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to convert. If omitted, uses train/validation/test (or test-only for en_accented).",
    )

    parser.add_argument("--force", action="store_true", help="Re-run conversion even if already complete.")

    parser.add_argument("--batched", action="store_true", help="Use batched processing (faster, more RAM).")
    parser.add_argument("--batch-size", type=int, default=5000, help="Items per batch in batched mode.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for cut creation.")
    parser.add_argument(
        "--io-num-workers",
        type=int,
        default=8,
        help="Worker threads for IO-bound batch materialization (default: 8).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=1,
        help="Prefetch batches ahead (0 or 1; default: 1).",
    )
    parser.add_argument("--hf-num-proc", type=int, default=4, help="HF loading processes in batched mode.")

    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    if args.configs:
        configs = list(args.configs)
    else:
        configs = list(_ASR_LANGUAGES)
        if args.include_special_configs:
            configs.extend(_ASR_ACCENTED_LANGUAGES)

    logger.info("Base output directory: %s", BASE_OUTPUT_DIR / DATASET_NICKNAME)
    logger.info("Configs: %s", configs)

    total_count = 0
    total_errors = 0
    skipped = 0

    for cfg in configs:
        splits = list(args.splits) if args.splits else _splits_for_config(cfg)
        for split in splits:
            count, errors = convert_one(
                cfg,
                split,
                force=bool(args.force),
                use_batched=bool(args.batched),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                io_num_workers=int(args.io_num_workers),
                prefetch_batches=int(args.prefetch_batches),
                hf_num_proc=int(args.hf_num_proc),
            )
            if count is None:
                skipped += 1
            else:
                total_count += count
                total_errors += errors

    logger.info("ALL DONE! Processed: %s cuts, Errors: %s, Skipped: %s subsets", total_count, total_errors, skipped)


if __name__ == "__main__":
    main()
