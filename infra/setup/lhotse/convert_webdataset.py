"""Convert a WebDataset (tar shards) into Lhotse Shar archives.

This script expects WebDataset samples stored as *paired* files inside tar shards:

    123456.flac           # audio bytes (mono)
    123456.metadata.json  # JSON metadata; transcript is in field "transcript"

Note: WebDataset groups samples by filename *stem* (everything before the last
"."). With the naming above, audio has key "123456" but metadata has key
"123456.metadata". This script merges those two into a single sample by
stripping the ".metadata" suffix.

Output:
- Writes Lhotse cuts using Shar format (like the other converters in this repo)
  to the specified output directory.
- Each cut contains a Recording, a single SupervisionSegment (text from
  metadata["transcript"]), and the full metadata dict stored under
  cut.custom["metadata"].

Optional resampling:
- You can resample audio before writing it into SHAR shards.
- This is useful when your source audio is high sample-rate (e.g. 48kHz) but
    you want a smaller/faster training representation (e.g. 16kHz).
- Implementation note: we rely on Lhotse's `Recording.resample(...)` hook (when
    available) so we don't manually decode/re-encode audio bytes in this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _marker_path_for_output(output_dir: Path) -> Path:
    # Prefer using LHOTSE_DATA_SHAR_ROOT if set, to match the convention used by
    # the other converters; otherwise fall back to the output_dir parent.
    base_root = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", str(output_dir.parent))).resolve()
    marker_root = base_root / ".conversion_markers"
    try:
        rel = output_dir.resolve().relative_to(base_root)
    except Exception:
        rel = Path(output_dir.name)
    marker = marker_root / rel
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker.with_suffix(".done")


def is_conversion_complete(output_dir: Path) -> bool:
    marker = _marker_path_for_output(output_dir)
    if marker.exists():
        return True
    return output_dir.exists() and any(output_dir.iterdir())


def mark_conversion_complete(output_dir: Path, count: int, errors: int) -> None:
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(
        f"Conversion completed successfully.\nCuts processed: {count}\nErrors: {errors}\n"
    )
    logger.info("Created completion marker: %s", marker_path)


def _find_shards(parent_dir: Path, pattern: str, recursive: bool) -> list[str]:
    if recursive:
        paths = sorted(parent_dir.rglob(pattern))
    else:
        paths = sorted(parent_dir.glob(pattern))
    return [str(p) for p in paths if p.is_file()]


def _iter_merged_webdataset_samples(
    shards: list[str],
    *,
    max_pending: int,
) -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    """Yield merged samples of (cut_id, audio_bytes, metadata_dict)."""

    try:
        import webdataset as wds
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'webdataset'. Install it (e.g. `uv pip install webdataset`)."
        ) from e

    ds = wds.WebDataset(shards, shardshuffle=False)

    pending: dict[str, dict[str, Any]] = {}

    def _touch_entry(key: str) -> dict[str, Any]:
        entry = pending.get(key)
        if entry is None:
            entry = {}
            pending[key] = entry
        return entry

    for sample in ds:
        if not isinstance(sample, dict):
            continue

        key = sample.get("__key__")
        if not isinstance(key, str) or not key:
            continue

        # Audio sample: key like "123456" and field "flac".
        if "flac" in sample and isinstance(sample["flac"], (bytes, bytearray)):
            entry = _touch_entry(key)
            entry["audio_bytes"] = bytes(sample["flac"])

            metadata = entry.get("metadata")
            audio_bytes = entry.get("audio_bytes")
            if metadata is not None and audio_bytes is not None:
                pending.pop(key, None)
                yield key, audio_bytes, metadata

        # Metadata sample: key like "123456.metadata" and field "json".
        json_bytes = None
        if "json" in sample and isinstance(sample["json"], (bytes, bytearray)):
            json_bytes = bytes(sample["json"])
        else:
            # Some pipelines may keep a different key name; be forgiving.
            for k, v in sample.items():
                if isinstance(k, str) and k.endswith("json") and isinstance(v, (bytes, bytearray)):
                    json_bytes = bytes(v)
                    break

        if json_bytes is not None:
            base_key = key
            if base_key.endswith(".metadata"):
                base_key = base_key[: -len(".metadata")]

            try:
                metadata = json.loads(json_bytes.decode("utf-8"))
            except Exception:
                metadata = {}

            entry = _touch_entry(base_key)
            entry["metadata"] = metadata

            audio_bytes = entry.get("audio_bytes")
            if audio_bytes is not None:
                pending.pop(base_key, None)
                yield base_key, audio_bytes, metadata

        if len(pending) > max_pending:
            # Prevent unbounded growth in case shards are malformed.
            dropped_key, _ = pending.popitem()
            logger.warning("Pending cache overflow; dropping incomplete sample: %s", dropped_key)

    if pending:
        logger.warning("Finished with %s incomplete samples (missing audio or metadata)", len(pending))


def _maybe_resample_recording(
    recording: Recording,
    *,
    enable: bool,
    orig_sr_expected: int,
    target_sr: int,
) -> tuple[Recording, bool]:
    """Optionally resample a Lhotse Recording.

    We use Lhotse's resampling hook to avoid explicit decode/re-encode logic
    in this converter.
    """

    if not enable:
        return recording, False

    if target_sr <= 0:
        raise ValueError("target_sr must be > 0")

    sr = getattr(recording, "sampling_rate", None)
    if isinstance(sr, int) and sr > 0 and orig_sr_expected > 0 and sr != orig_sr_expected:
        logger.warning("Recording sampling rate %s != expected %s", sr, orig_sr_expected)

    if isinstance(sr, int) and sr == target_sr:
        return recording, False

    if not hasattr(recording, "resample"):
        raise RuntimeError(
            "This Lhotse version does not expose Recording.resample(...). "
            "Either upgrade lhotse or disable --resample."
        )

    # Be defensive about the signature across Lhotse versions.
    try:
        resampled = recording.resample(target_sr)  # type: ignore[attr-defined]
    except TypeError:
        try:
            resampled = recording.resample(sampling_rate=target_sr)  # type: ignore[attr-defined]
        except TypeError:
            resampled = recording.resample(new_sampling_rate=target_sr)  # type: ignore[attr-defined]

    return resampled, True


def convert_webdataset_to_shar(
    *,
    shards_parent_dir: Path,
    output_dir: Path,
    shards_pattern: str,
    recursive: bool,
    shard_size: int,
    audio_format: str,
    language: str,
    resample: bool,
    orig_sr: int,
    target_sr: int,
    max_pending: int,
    force: bool,
) -> tuple[int, int] | tuple[None, None]:
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info("SKIPPING - already complete (marker: %s)", marker)
        return None, None

    shards = _find_shards(shards_parent_dir, shards_pattern, recursive)
    if not shards:
        raise SystemExit(f"No shards found under {shards_parent_dir} with pattern {shards_pattern!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Found %s shard files", len(shards))
    logger.info("Writing Shar to: %s", output_dir)

    writer = SharWriter(output_dir=output_dir, fields={"recording": audio_format}, shard_size=shard_size)

    count = 0
    errors = 0

    with writer:
        for cut_id, audio_bytes, metadata in tqdm(
            _iter_merged_webdataset_samples(shards, max_pending=max_pending),
            desc="Converting webdataset",
            unit="cut",
        ):
            try:
                transcript = metadata.get("transcript", "")
                if not isinstance(transcript, str):
                    transcript = str(transcript)

                recording = Recording.from_bytes(data=audio_bytes, recording_id=cut_id)

                recording, did_resample = _maybe_resample_recording(
                    recording,
                    enable=bool(resample),
                    orig_sr_expected=int(orig_sr),
                    target_sr=int(target_sr),
                )

                metadata = dict(metadata)
                if did_resample:
                    metadata["resampled_from_hz"] = int(orig_sr)
                    metadata["resampled_to_hz"] = int(target_sr)
                else:
                    sr_now = getattr(recording, "sampling_rate", None)
                    if isinstance(sr_now, int) and sr_now > 0:
                        metadata.setdefault("sample_rate_hz", int(sr_now))
                supervision = SupervisionSegment(
                    id=cut_id,
                    recording_id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    text=transcript,
                    language=language,
                )
                cut = MonoCut(
                    id=cut_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    recording=recording,
                    supervisions=[supervision],
                    custom={"metadata": metadata},
                )

                writer.write(cut)
                count += 1
            except Exception as e:
                errors += 1
                logger.error("Failed to convert cut %s: %s", cut_id, e)
                if errors > 100:
                    logger.critical("Too many errors, stopping.")
                    break

    logger.info("Finished conversion. Processed %s cuts with %s errors.", count, errors)
    mark_conversion_complete(output_dir, count, errors)
    return count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a WebDataset directory to Lhotse SHAR")

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
        default=4000,
        help="Cuts per SHAR shard (default: 4000).",
    )
    parser.add_argument(
        "--audio-format",
        default="flac",
        help="Audio encoding to use inside SHAR (default: flac).",
    )
    parser.add_argument(
        "--language",
        default="und",
        help="Language code stored in supervision segments (default: und).",
    )

    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample audio before writing into SHAR (default off). Uses Lhotse's Recording.resample().",
    )
    parser.add_argument(
        "--orig-sr",
        type=int,
        default=48000,
        help="Expected original sampling rate for sanity checks (default: 48000).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sampling rate when --resample is enabled (default: 16000).",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=20000,
        help="Max number of incomplete samples buffered when merging audio+metadata (default: 20000).",
    )

    parser.add_argument("--force", action="store_true", help="Re-run conversion even if already complete.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    convert_webdataset_to_shar(
        shards_parent_dir=args.shards_dir,
        output_dir=args.output_dir,
        shards_pattern=str(args.pattern),
        recursive=bool(args.recursive),
        shard_size=int(args.shard_size),
        audio_format=str(args.audio_format),
        language=str(args.language),
        resample=bool(args.resample),
        orig_sr=int(args.orig_sr),
        target_sr=int(args.target_sr),
        max_pending=int(args.max_pending),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
