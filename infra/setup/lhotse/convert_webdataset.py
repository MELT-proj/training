"""Convert a WebDataset (tar shards) into Lhotse Shar archives.

This script expects WebDataset samples stored as *paired* files inside tar shards:

    123456.flac           # audio bytes (mono)
    123456.metadata.json  # JSON metadata; transcript is in field "transcript"

Output:
- Writes Lhotse cuts using Shar format to the specified output directory.
- Each cut contains a Recording, a single SupervisionSegment (text from
  metadata["transcript"]), and the full metadata dict stored under
  cut.custom["metadata"].

Optional resampling:
- You can resample audio before writing it into SHAR shards.
- This is useful when your source audio is high sample-rate (e.g. 48kHz) but
  you want a smaller/faster training representation (e.g. 16kHz).

Reference: Uses batch_utils.convert_webdataset_to_shar_batched() for multiprocessing.
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_webdataset_to_shar_batched


# --- Configuration ---
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _marker_path_for_output(output_dir: Path) -> Path:
    """Store completion marker under the CLI-provided output directory."""
    marker = Path(output_dir) / ".conversion.done"
    # Ensure the output directory exists so the marker can be written
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a WebDataset directory to Lhotse SHAR format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion:
  python convert_webdataset.py --shards-dir /path/to/webdataset --output-dir /path/to/output

  # With resampling from 48kHz to 16kHz:
  python convert_webdataset.py --shards-dir /path/to/webdataset --output-dir /path/to/output \\
      --resample --orig-sr 48000 --target-sr 16000

  # Recursive search and custom workers:
  python convert_webdataset.py --shards-dir /path/to/webdataset --output-dir /path/to/output \\
      --recursive --num-workers 8
        """,
    )

    parser.add_argument(
        "--shards-dir",
        type=Path,
        required=True,
        help="Path to a directory containing WebDataset tar shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to write SHAR archives.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tar*",
        help="Glob pattern for shard files inside --shards-dir (default: *.tar*).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for shards recursively under --shards-dir.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Cuts per SHAR shard (default: {SHARD_SIZE}).",
    )
    parser.add_argument(
        "--audio-format",
        default=AUDIO_FORMAT,
        help=f"Audio encoding to use inside SHAR (default: {AUDIO_FORMAT}).",
    )
    parser.add_argument(
        "--language",
        default="und",
        help="Language code stored in supervision segments (default: und).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample audio before writing into SHAR.",
    )
    parser.add_argument(
        "--orig-sr",
        type=int,
        default=48000,
        help="Expected original sampling rate (default: 48000).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sampling rate when --resample is enabled (default: 16000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: cpu_count).",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=20000,
        help="Max pending samples for merging audio+metadata (default: 20000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even if already complete.",
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

    output_dir = args.output_dir

    # Check if conversion is already complete
    if not args.force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(f"SKIPPING - already complete (marker: {marker})")
        return

    logger.info(f"Shards directory: {args.shards_dir}")
    logger.info(f"Output directory: {output_dir}")

    count, errors = convert_webdataset_to_shar_batched(
        shards_parent_dir=args.shards_dir,
        output_dir=output_dir,
        shards_pattern=args.pattern,
        recursive=args.recursive,
        shard_size=args.shard_size,
        audio_format=args.audio_format,
        language=args.language,
        num_workers=args.num_workers,
        max_pending=args.max_pending,
        resample=args.resample,
        orig_sr=args.orig_sr,
        target_sr=args.target_sr,
    )

    logger.info(f"Finished! Processed {count} cuts with {errors} errors.")

    # Mark conversion as complete only if no errors
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking as complete due to {errors} errors")


if __name__ == "__main__":
    main()
