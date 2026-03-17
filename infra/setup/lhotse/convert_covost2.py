"""Convert CoVoST2 dataset from local disk to Lhotse Shar format.

This script reads CoVoST2 TSV metadata and MP3 audio files from disk and
converts them to Lhotse Shar archives (FLAC, 16 kHz) for efficient training,
using multiprocessing for speed.

Dataset Structure:
------------------
CoVoST2 provides multiple English-to-X language pairs.

Metadata TSV files live under:
    /mnt/scratch-artemis/shared/datasets/facebook/covost2/
    e.g. covost_v2.en_de.train.tsv

TSV columns:
    path        – audio filename (e.g. "common_voice_en_18540003.mp3")
    sentence    – source-language transcription (English)
    translation – target-language translation
    client_id   – speaker identifier

Audio files live under:
    /mnt/scratch-artemis/shared/datasets/mozilla-foundation/cv4/en/clips/

A faulty_files.tsv in the metadata folder lists clips to skip (column "clip").

Stored in Shar cuts:
  - supervision.text   : translation (target-language text)
  - cut.custom fields  : sentence, path, client_id

Output Directory Structure:
--------------------------
    {LHOTSE_DATA_SHAR_ROOT}/covost2/{config}/{split}/
    Example: shar/covost2/en_de/train/

Reference: https://github.com/facebookresearch/covost
"""

import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm

# --- Configuration ---
DATASET_NICKNAME = "covost2"
BASE_OUTPUT_DIR = Path(
    os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar")
)
METADATA_DIR = Path("/mnt/scratch-artemis/shared/datasets/facebook/covost2")
AUDIO_DIR = Path("/mnt/scratch-artemis/shared/datasets/mozilla-foundation/cv4/en/clips")
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
TARGET_SR = 16000
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

ALL_CONFIGS = [
    "en_ar",
    "en_ca",
    "en_cy",
    "en_de",
    "en_et",
    "en_fa",
    "en_id",
    "en_ja",
    "en_lv",
    "en_mn",
    "en_sl",
    "en_sv-SE",
    "en_ta",
    "en_tr",
    "en_zh-CN",
]
ALL_SPLITS = ["train", "dev", "test"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_output_dir(config: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / config / split


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


def load_faulty_clips(metadata_dir: Path) -> set[str]:
    """Load the set of faulty clip filenames to skip."""
    faulty_path = metadata_dir / "faulty_files.tsv"
    faulty: set[str] = set()
    if not faulty_path.exists():
        logger.warning(f"Faulty files list not found: {faulty_path}")
        return faulty
    with open(faulty_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = row.get("clip", "").strip()
            if clip:
                faulty.add(clip)
    logger.info(f"Loaded {len(faulty)} faulty clips to skip")
    return faulty


def load_tsv_rows(
    config: str,
    split: str,
    metadata_dir: Path,
    faulty_clips: set[str],
) -> list[dict[str, str]]:
    """Read a CoVoST2 TSV file and return rows, filtering out faulty clips.

    Returns:
        List of dicts with keys: path, sentence, translation, client_id.
    """
    tsv_path = metadata_dir / f"covost_v2.{config}.{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    rows: list[dict[str, str]] = []
    skipped = 0
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = row.get("path", "").strip()
            if clip in faulty_clips:
                skipped += 1
                continue
            rows.append(row)

    logger.info(f"Loaded {len(rows)} rows from {tsv_path.name} (skipped {skipped} faulty)")
    return rows


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling)
# ---------------------------------------------------------------------------


def _process_chunk(
    rows: list[dict[str, str]],
    chunk_id: int,
    audio_dir: str,
    output_dir: str,
    audio_format: str,
    shard_size: int,
    language: str,
    target_sr: int,
) -> tuple[int, int, str]:
    """Convert a chunk of TSV rows to Shar format.

    Args:
        rows: List of TSV row dicts for this chunk.
        chunk_id: Worker/chunk identifier.
        audio_dir: Path to the directory containing MP3 audio clips.
        output_dir: Base output directory (chunk subdir will be created).
        audio_format: Audio format for Shar (e.g. "flac").
        shard_size: Number of cuts per shard file.
        language: Language code for supervision segments.
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (count, errors, chunk_output_dir).
    """
    chunk_output_dir = Path(output_dir) / f"chunk_{chunk_id:04d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    audio_root = Path(audio_dir)
    count = 0
    errors = 0

    writer = SharWriter(
        output_dir=chunk_output_dir,
        fields={"recording": audio_format},
        shard_size=shard_size,
    )

    with writer:
        pbar = tqdm(
            rows,
            desc=f"Worker {chunk_id}",
            unit="item",
            position=chunk_id,
            leave=True,
        )
        for row in pbar:
            try:
                clip_filename = row.get("path", "").strip()
                if not clip_filename:
                    errors += 1
                    continue

                audio_path = audio_root / clip_filename
                if not audio_path.exists():
                    print(f"[WARNING] Audio file not found: {audio_path}, skipping")
                    errors += 1
                    continue

                cut_id = clip_filename.rsplit(".", 1)[0]  # strip extension

                # Read raw audio bytes
                audio_bytes = audio_path.read_bytes()

                recording = Recording.from_bytes(
                    data=audio_bytes, recording_id=cut_id
                )

                # Resample to target SR if needed
                # if recording.sampling_rate != target_sr:
                recording = recording.resample(target_sr)

                translation = row.get("translation", "")
                if not isinstance(translation, str):
                    translation = str(translation) if translation is not None else ""

                supervision = SupervisionSegment(
                    id=cut_id,
                    recording_id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    text=translation,
                    language=language,
                )

                custom = {
                    "sentence": row.get("sentence", ""),
                    "path": clip_filename,
                    "client_id": row.get("client_id", ""),
                }

                cut = MonoCut(
                    id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    recording=recording,
                    supervisions=[supervision],
                    custom=custom,
                )

                writer.write(cut)
                count += 1
                pbar.set_postfix({"processed": count, "errors": errors})

            except Exception as e:
                errors += 1
                print(f"[ERROR] Failed to convert {row.get('path', '?')}: {e}")
                pbar.set_postfix({"processed": count, "errors": errors})

    return count, errors, str(chunk_output_dir)


# ---------------------------------------------------------------------------
# Merge helper (same logic as batch_utils)
# ---------------------------------------------------------------------------


def _merge_chunk_outputs(
    chunk_dirs: list[str],
    final_output_dir: Path,
) -> None:
    """Merge all chunk outputs into the final output directory."""
    import shutil

    final_output_dir.mkdir(parents=True, exist_ok=True)

    def _next_shard_index(dir_path: Path) -> int:
        existing = sorted(dir_path.glob("recording.*.tar"))
        max_idx = -1
        for f in existing:
            try:
                idx = int(f.stem.split(".")[-1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    shard_idx = _next_shard_index(final_output_dir)

    for chunk_dir in sorted(chunk_dirs):
        chunk_path = Path(chunk_dir)
        if not chunk_path.exists():
            continue

        tar_files = sorted(chunk_path.glob("recording.*.tar"))
        cuts_files = sorted(chunk_path.glob("cuts.*.jsonl.gz"))

        for tar_file, cuts_file in zip(tar_files, cuts_files):
            new_tar = final_output_dir / f"recording.{shard_idx:06d}.tar"
            new_cuts = final_output_dir / f"cuts.{shard_idx:06d}.jsonl.gz"
            shutil.move(str(tar_file), str(new_tar))
            shutil.move(str(cuts_file), str(new_cuts))
            shard_idx += 1

        try:
            shutil.rmtree(chunk_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Conversion entry-point
# ---------------------------------------------------------------------------

def _supervision_language_from_config(config: str) -> str:
    """Infer supervision language code from a CoVoST2 config string.

    Examples:
        en_de -> de
        en_sv-SE -> sv
        en_zh-CN -> zh
    """
    if "_" not in config:
        return "und"

    target = config.split("_", 1)[1]
    if target.startswith("zh"):
        return "zh"

    # Strip regional variants (e.g., sv-SE -> sv)
    return target.split("-", 1)[0] if target else "und"


def convert_one(
    config: str,
    split: str,
    force: bool = False,
    num_workers: int | None = None,
    metadata_dir: Path = METADATA_DIR,
    audio_dir: Path = AUDIO_DIR,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single config/split combination to Shar format.

    Args:
        config: Language config (e.g., "en_de", "en_sv-SE", "en_zh-CN").
        split: Data split ("train", "dev", or "test").
        force: If True, re-run conversion even if already complete.
        num_workers: Number of parallel workers (default: cpu_count).
        metadata_dir: Directory containing TSV files.
        audio_dir: Directory containing MP3 clips.

    Returns:
        Tuple of (count, errors) if processed, or (None, None) if skipped.
    """
    output_dir = get_output_dir(config, split)

    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(f"SKIPPING {config}/{split} - already complete (marker: {marker})")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {config}/{split} to: {output_dir}")

    # Load faulty clips and TSV rows
    faulty_clips = load_faulty_clips(metadata_dir)
    rows = load_tsv_rows(config, split, metadata_dir, faulty_clips)

    if not rows:
        logger.warning(f"No rows to process for {config}/{split}")
        return 0, 0

    language = _supervision_language_from_config(config)

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    # Avoid more workers than shards worth of data
    max_workers = max(1, len(rows) // SHARD_SIZE) if SHARD_SIZE > 0 else num_workers
    num_workers = max(1, min(num_workers, max_workers))

    # Split rows across workers
    chunk_size = (len(rows) + num_workers - 1) // num_workers
    chunks: list[tuple[int, list[dict[str, str]]]] = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(rows))
        if start >= len(rows):
            break
        chunks.append((i, rows[start:end]))

    logger.info(
        f"Processing {len(rows)} rows with {len(chunks)} workers "
        f"(~{len(rows) // len(chunks)} rows/worker)"
    )

    count = 0
    errors = 0
    chunk_outputs: list[str] = []

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {}
        for chunk_id, chunk_rows in chunks:
            future = executor.submit(
                _process_chunk,
                chunk_rows,
                chunk_id,
                str(audio_dir),
                str(output_dir),
                AUDIO_FORMAT,
                SHARD_SIZE,
                language,
                TARGET_SR,
            )
            futures[future] = chunk_id

        with tqdm(total=len(chunks), desc="Chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    c, e, path = future.result()
                    count += c
                    errors += e
                    chunk_outputs.append(path)
                    pbar.update(1)
                    pbar.set_postfix({"processed": count, "errors": errors})
                except Exception as exc:
                    logger.error(f"Chunk {chunk_id} failed: {exc}")
                    errors += 1

    # Merge chunks
    logger.info(f"Merging {len(chunk_outputs)} chunk outputs...")
    _merge_chunk_outputs(chunk_outputs, output_dir)

    logger.info(f"Finished {config}/{split}! Processed {count} cuts with {errors} errors.")

    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking {config}/{split} as complete due to {errors} errors")

    return count, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CoVoST2 dataset from local disk to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Convert all configs and splits:
  python convert_covost2.py

  # Convert only en_de, train split:
  python convert_covost2.py --configs en_de --splits train

  # Force re-conversion with 8 workers:
  python convert_covost2.py --force --num-workers 8

Available configs : {', '.join(ALL_CONFIGS)}
Available splits  : {', '.join(ALL_SPLITS)}
        """,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=f"Language configs to convert (default: all = {ALL_CONFIGS}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=f"Splits to convert (default: all = {ALL_SPLITS}).",
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
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Cuts per Shar shard (default: {SHARD_SIZE}).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=f"Directory containing CoVoST2 TSV files (default: {METADATA_DIR}).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help=f"Directory containing Common Voice MP3 clips (default: {AUDIO_DIR}).",
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

    configs = args.configs if args.configs else ALL_CONFIGS
    splits = args.splits if args.splits else ALL_SPLITS

    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")
    logger.info(f"Metadata directory  : {args.metadata_dir}")
    logger.info(f"Audio directory     : {args.audio_dir}")
    logger.info(f"Configs: {configs}")
    logger.info(f"Splits : {splits}")

    total_count = 0
    total_errors = 0
    skipped = 0

    for config in configs:
        for split in splits:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {config}/{split}")
            logger.info(f"{'=' * 60}")

            count, errors = convert_one(
                config,
                split,
                force=args.force,
                num_workers=args.num_workers,
                metadata_dir=args.metadata_dir,
                audio_dir=args.audio_dir,
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
