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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
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


def _process_sample(
    cut_id: str,
    audio_bytes: bytes,
    metadata: dict[str, Any],
    language: str,
    resample: bool,
    orig_sr: int,
    target_sr: int,
) -> MonoCut:
    """Process a single sample into a MonoCut. Thread-safe."""
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
    return cut


@dataclass
class ChunkTask:
    """A chunk of samples to be processed by a worker."""
    chunk_id: int
    samples: list[tuple[str, bytes, dict[str, Any]]]


def _process_chunk(
    chunk: ChunkTask,
    output_dir: str,
    audio_format: str,
    shard_size: int,
    language: str,
    resample: bool,
    orig_sr: int,
    target_sr: int,
) -> tuple[int, int, list[str]]:
    """Process a chunk of samples and write to a temporary shard directory.
    
    Each worker writes to its own subdirectory to avoid conflicts.
    Returns (count, errors, output_paths).
    """
    chunk_output_dir = Path(output_dir) / f"chunk_{chunk.chunk_id:04d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    errors = 0
    
    writer = SharWriter(
        output_dir=chunk_output_dir,
        fields={"recording": audio_format},
        shard_size=shard_size,
    )
    
    with writer:
        for cut_id, audio_bytes, metadata in chunk.samples:
            try:
                cut = _process_sample(
                    cut_id,
                    audio_bytes,
                    metadata,
                    language,
                    resample,
                    orig_sr,
                    target_sr,
                )
                writer.write(cut)
                count += 1
            except Exception as e:
                errors += 1
                # Log to stderr since we're in a subprocess
                print(f"[ERROR] Failed to convert cut {cut_id}: {e}")
    
    return count, errors, [str(chunk_output_dir)]


def _merge_chunk_outputs(
    chunk_dirs: list[str],
    final_output_dir: Path,
) -> None:
    """Merge all chunk outputs into the final output directory with proper shard numbering."""
    import shutil
    
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine next shard index based on existing files to ensure continuous numbering
    def _next_shard_index(dir_path: Path) -> int:
        existing = sorted(dir_path.glob("recording.*.tar"))
        max_idx = -1
        for f in existing:
            try:
                # Expect pattern recording.%06d.tar
                idx_str = f.stem.split(".")[-1]
                idx = int(idx_str)
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    shard_idx = _next_shard_index(final_output_dir)
    for chunk_dir in sorted(chunk_dirs):
        chunk_path = Path(chunk_dir)
        if not chunk_path.exists():
            continue
        
        # Find all shard files in this chunk
        tar_files = sorted(chunk_path.glob("recording.*.tar"))
        cuts_files = sorted(chunk_path.glob("cuts.*.jsonl.gz"))
        
        for tar_file, cuts_file in zip(tar_files, cuts_files):
            # Rename with new shard index
            new_tar = final_output_dir / f"recording.{shard_idx:06d}.tar"
            new_cuts = final_output_dir / f"cuts.{shard_idx:06d}.jsonl.gz"
            
            shutil.move(str(tar_file), str(new_tar))
            shutil.move(str(cuts_file), str(new_cuts))
            shard_idx += 1
        
        # Clean up empty chunk directory
        try:
            shutil.rmtree(chunk_path)
        except Exception:
            pass


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
    num_workers: int | None = None,
    max_cuts_in_memory: int | None = None,
) -> tuple[int, int] | tuple[None, None]:
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info("SKIPPING - already complete (marker: %s)", marker)
        return None, None

    shards = _find_shards(shards_parent_dir, shards_pattern, recursive)
    if not shards:
        raise SystemExit(f"No shards found under {shards_parent_dir} with pattern {shards_pattern!r}")

    # Default to cpu_count if not specified
    if num_workers is None:
        num_workers = cpu_count()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Found %s shard files", len(shards))
    logger.info("Writing Shar to: %s (using %s worker processes)", output_dir, num_workers)

    count = 0
    errors = 0

    # Helper: process a batch of samples using worker processes
    def _process_samples_batch(batch_samples: list[tuple[str, bytes, dict[str, Any]]]) -> list[str]:
        nonlocal count, errors, num_workers
        batch_total = len(batch_samples)
        if batch_total == 0:
            return []

        # Adjust number of workers per-batch to avoid undersized shards
        local_num_workers = num_workers
        if shard_size and shard_size > 0:
            max_workers_by_shards = max(1, batch_total // shard_size)
            if local_num_workers > max_workers_by_shards:
                logger.info(
                    "Adjusting workers (batch): %s -> %s (shard_size=%s, batch_total=%s)",
                    local_num_workers,
                    max_workers_by_shards,
                    shard_size,
                    batch_total,
                )
                local_num_workers = max_workers_by_shards

        # Create chunks for this batch
        chunk_size = max(1, (batch_total + local_num_workers - 1) // local_num_workers)
        chunks: list[ChunkTask] = []
        for i in range(0, batch_total, chunk_size):
            chunk_samples = batch_samples[i : i + chunk_size]
            chunks.append(ChunkTask(chunk_id=len(chunks), samples=chunk_samples))

        logger.info("Processing batch: %s chunks (~%s samples/chunk)", len(chunks), chunk_size)

        batch_chunk_outputs: list[str] = []
        with ProcessPoolExecutor(max_workers=local_num_workers) as executor:
            futures = {}
            for chunk in chunks:
                future = executor.submit(
                    _process_chunk,
                    chunk,
                    str(output_dir),
                    audio_format,
                    shard_size,
                    language,
                    resample,
                    orig_sr,
                    target_sr,
                )
                futures[future] = chunk.chunk_id

            with tqdm(total=len(chunks), desc="Processing batch", unit="chunk") as pbar:
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        chunk_count, chunk_errors, chunk_paths = future.result()
                        count += chunk_count
                        errors += chunk_errors
                        batch_chunk_outputs.extend(chunk_paths)
                        pbar.update(1)
                        pbar.set_postfix({"processed": count, "errors": errors})
                    except Exception as e:
                        logger.error("Batch chunk %s failed: %s", chunk_id, e)
                        errors += 1

        return batch_chunk_outputs

    # Iterate samples and process in batches to limit RAM usage
    sample_iter = _iter_merged_webdataset_samples(shards, max_pending=max_pending)
    chunk_outputs_all: list[str] = []
    current_batch: list[tuple[str, bytes, dict[str, Any]]] = []

    # Use a streaming progress bar for reading samples
    pbar_read = tqdm(desc="Reading samples", unit="sample")
    for cut_id, audio_bytes, metadata in sample_iter:
        current_batch.append((cut_id, audio_bytes, metadata))
        pbar_read.update(1)
        if max_cuts_in_memory is not None and len(current_batch) >= max_cuts_in_memory:
            logger.info("Batch size %s reached; offloading to disk via workers", len(current_batch))
            batch_outputs = _process_samples_batch(current_batch)
            chunk_outputs_all.extend(batch_outputs)
            current_batch.clear()

    pbar_read.close()

    # Process any remaining samples
    if current_batch:
        logger.info("Processing final batch of %s samples", len(current_batch))
        batch_outputs = _process_samples_batch(current_batch)
        chunk_outputs_all.extend(batch_outputs)
        current_batch.clear()

    # Merge chunk outputs into final directory structure
    logger.info("Merging %s chunk outputs...", len(chunk_outputs_all))
    _merge_chunk_outputs(chunk_outputs_all, output_dir)

    logger.info("Finished conversion. Processed %s cuts with %s errors.", count, errors)
    # Only mark as complete if no errors occurred
    # This ensures runs with any errors are retried on next invocation
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.error("Conversion had %s errors, not marking as complete", errors)
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: cpu_count).",
    )
    parser.add_argument(
        "--max-cuts-in-memory",
        type=int,
        default=None,
        help="Maximum number of cuts to buffer in RAM before offloading to disk (default: unlimited).",
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
        num_workers=int(args.num_workers) if args.num_workers else cpu_count(),
        max_cuts_in_memory=int(args.max_cuts_in_memory) if args.max_cuts_in_memory else None,
    )


if __name__ == "__main__":
    main()
