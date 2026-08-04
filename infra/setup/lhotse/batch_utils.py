"""Batch processing utilities for faster dataset to Lhotse Shar conversion.

This module provides multiprocessing-based parallel conversion of HuggingFace
datasets and WebDataset archives to Lhotse Shar format. The key feature is
memory-efficient processing: each worker loads its own shard of the dataset
and processes it incrementally, avoiding materializing the entire dataset in memory.

Key Features:
-------------
1. Lazy loading: Dataset is loaded with lazy audio decoding
2. Sharded processing: Each worker processes a shard of the dataset
3. Memory-efficient: Only num_workers batches are in memory at once
4. Temporary directories: Workers write to chunk_XXXX subdirectories
5. Merge phase: After all workers complete, shards are renumbered and moved
6. Shard-size aware: Number of workers adapts to avoid undersized shards
7. Resampling support: Optional audio resampling during conversion

Architecture:
-------------
1. Main process loads the dataset (lazy, doesn't decode audio)
2. Fork num_workers processes, each assigned a shard index
3. Each worker loads the dataset and uses .shard() to get its portion
4. Workers iterate over their shard and convert items on-the-fly
5. Workers write to temporary chunk_XXXX directories
6. After all workers complete, chunks are merged with continuous numbering

Usage:
------
    from batch_utils import convert_subset_to_shar_batched

    count, errors = convert_subset_to_shar_batched(
        dataset_name="openslr/librispeech_asr",
        hf_config="all",
        hf_split="train.clean.100",
        output_dir=Path("/path/to/output"),
        audio_format="flac",
        shard_size=4000,
        language="en",
    )

    # WebDataset conversion with resampling:
    from batch_utils import convert_webdataset_to_shar_batched

    count, errors = convert_webdataset_to_shar_batched(
        shards_parent_dir=Path("/path/to/webdataset"),
        output_dir=Path("/path/to/output"),
        resample=True,
        orig_sr=48000,
        target_sr=16000,
    )
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class CutData:
    """Intermediate data structure holding info needed to create a cut."""

    cut_id: str
    audio_bytes: bytes
    text: str
    language: str
    custom: dict | None = None  # Optional custom metadata


@dataclass
class ResampleConfig:
    """Configuration for audio resampling."""

    enabled: bool = False
    orig_sr: int = 48000  # Expected original sample rate
    target_sr: int = 16000  # Target sample rate


@dataclass
class ChunkTask:
    """A chunk task specification for a worker."""

    chunk_id: int
    dataset_name: str
    hf_config: str
    hf_split: str
    num_shards: int
    shard_index: int
    language: str
    id_field: str
    text_field: str
    audio_field: str
    custom_fields: list[str] | None = None  # Extra HF fields to store in cut.custom


# -----------------------------------------------------------------------------
# Worker functions (must be module-level for pickling)
# -----------------------------------------------------------------------------


def _maybe_resample_recording(
    recording: Recording,
    resample_config: ResampleConfig,
) -> tuple[Recording, bool]:
    """Optionally resample a Lhotse Recording.

    Args:
        recording: The recording to potentially resample.
        resample_config: Resampling configuration.

    Returns:
        Tuple of (recording, did_resample).
    """
    if not resample_config.enabled:
        return recording, False

    target_sr = resample_config.target_sr
    if target_sr <= 0:
        raise ValueError("target_sr must be > 0")

    sr = getattr(recording, "sampling_rate", None)
    if isinstance(sr, int) and sr == target_sr:
        return recording, False

    if not hasattr(recording, "resample"):
        raise RuntimeError(
            "This Lhotse version does not expose Recording.resample(...). "
            "Either upgrade lhotse or disable resampling."
        )

    # Be defensive about the signature across Lhotse versions.
    try:
        resampled = recording.resample(target_sr)
    except TypeError:
        try:
            resampled = recording.resample(sampling_rate=target_sr)
        except TypeError:
            resampled = recording.resample(new_sampling_rate=target_sr)

    return resampled, True


def _create_cut_from_data(
    data: CutData,
    resample_config: ResampleConfig | None = None,
) -> MonoCut:
    """Create a MonoCut from CutData.

    Args:
        data: CutData containing all info needed to create a cut.
        resample_config: Optional resampling configuration.

    Returns:
        MonoCut object.

    Raises:
        Exception if cut creation fails.
    """
    recording = Recording.from_bytes(data=data.audio_bytes, recording_id=data.cut_id)

    # Apply resampling if configured
    did_resample = False
    if resample_config is not None:
        recording, did_resample = _maybe_resample_recording(recording, resample_config)

    supervision = SupervisionSegment(
        id=data.cut_id,
        recording_id=data.cut_id,
        start=0.0,
        duration=recording.duration,
        text=data.text,
        language=data.language,
    )

    # Build custom metadata
    custom = dict(data.custom) if data.custom else {}
    if did_resample:
        custom["resampled_from_hz"] = resample_config.orig_sr
        custom["resampled_to_hz"] = resample_config.target_sr

    cut = MonoCut(
        id=data.cut_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
        custom=custom if custom else None,
    )

    return cut


def _process_chunk(
    chunk: ChunkTask,
    output_dir: str,
    audio_format: str,
    shard_size: int,
    hf_num_proc: int,
) -> tuple[int, int, str]:
    """Process a chunk by loading its dataset shard and converting items incrementally.

    Each worker loads its own shard of the dataset and processes it on-the-fly,
    avoiding materializing the entire dataset in memory.

    Args:
        chunk: ChunkTask containing dataset info and shard assignment.
        output_dir: Base output directory (chunk subdir will be created).
        audio_format: Audio format for Shar (e.g., "flac").
        shard_size: Number of cuts per shard file.
        hf_num_proc: Number of processes for HuggingFace data loading.

    Returns:
        Tuple of (count, errors, chunk_output_dir).
    """
    chunk_output_dir = Path(output_dir) / f"chunk_{chunk.chunk_id:04d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0

    # Load the full dataset in this worker (lazy loading)
    dataset = load_dataset(
        chunk.dataset_name,
        chunk.hf_config,
        split=chunk.hf_split,
        num_proc=hf_num_proc,
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    # Don't decode audio - get raw bytes
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Get this worker's shard. Here, num_shards == num_workers
    worker_shard = dataset.shard(num_shards=chunk.num_shards, index=chunk.shard_index)

    writer = SharWriter(
        output_dir=chunk_output_dir,
        fields={"recording": audio_format},
        shard_size=shard_size,
    )

    with writer:
        pbar = tqdm(
            worker_shard,
            desc=f"Worker {chunk.chunk_id}",
            unit="item",
            position=chunk.chunk_id,
            leave=True,
        )
        for i, item in enumerate(pbar):
            try:
                # Extract ID
                hf_id = item.get(chunk.id_field, f"chunk{chunk.chunk_id}_item{i}")
                cut_id = str(hf_id).replace("/", "_").replace(".", "_")

                # Extract audio bytes
                audio_data = item[chunk.audio_field]
                audio_bytes = audio_data.get("bytes")

                # Some HF datasets leave bytes=None; read from path if needed
                if audio_bytes is None:
                    audio_path = audio_data.get("path")
                    if audio_path:
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()

                if audio_bytes is None:
                    print(f"[WARNING] No audio bytes for item {cut_id}, skipping")
                    errors += 1
                    continue

                # Extract text
                text = item.get(chunk.text_field, "")
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""

                # Extract custom metadata fields
                custom = None
                if chunk.custom_fields:
                    custom = {}
                    for field in chunk.custom_fields:
                        if field in item:
                            custom[field] = item[field]

                # Create CutData and convert to MonoCut
                data = CutData(
                    cut_id=cut_id,
                    audio_bytes=audio_bytes,
                    text=text,
                    language=chunk.language,
                    custom=custom,
                )

                cut = _create_cut_from_data(data)
                writer.write(cut)
                count += 1
                
                # Update progress bar postfix
                pbar.set_postfix({"processed": count, "errors": errors})

            except Exception as e:
                errors += 1
                print(f"[ERROR] Failed to convert item {i} in chunk {chunk.chunk_id}: {e}")
                
                # Update progress bar postfix
                pbar.set_postfix({"processed": count, "errors": errors})

    return count, errors, str(chunk_output_dir)


def _merge_chunk_outputs(
    chunk_dirs: list[str],
    final_output_dir: Path,
) -> None:
    """Merge all chunk outputs into the final output directory with proper shard numbering.

    Args:
        chunk_dirs: List of chunk directory paths to merge.
        final_output_dir: Final output directory for merged shards.
    """
    final_output_dir.mkdir(parents=True, exist_ok=True)

    def _next_shard_index(dir_path: Path) -> int:
        """Determine next shard index based on existing files."""
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


# -----------------------------------------------------------------------------
# Main conversion class
# -----------------------------------------------------------------------------


class BatchedSharConverter:
    """Converter that processes HuggingFace datasets with multiprocessing.

    This converter uses dataset sharding to ensure memory-efficient processing.
    Each worker loads the full dataset (lazy) and uses .shard() to get its portion,
    then iterates over it incrementally. This ensures only num_workers batches are
    in memory at once, making it suitable for very large datasets.

    Workers write to temporary directories, and results are merged at the end
    with proper shard numbering.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        shard_size: int = 4000,
        hf_num_proc: int = 4,
    ):
        """Initialize the batched converter.

        Args:
            num_workers: Number of worker processes. Defaults to cpu_count().
                        Will be adjusted based on shard_size and dataset size.
            shard_size: Number of cuts per shard file (default: 4000).
            hf_num_proc: Number of processes for HuggingFace data loading.
        """
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.shard_size = shard_size
        self.hf_num_proc = hf_num_proc

    def _load_dataset(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
    ) -> Any:
        """Load the full dataset (lazy loading - doesn't materialize in memory).

        Args:
            dataset_name: HuggingFace dataset name.
            hf_config: HuggingFace config name.
            hf_split: HuggingFace split name.

        Returns:
            HuggingFace Dataset object (lazy).
        """
        logger.info(
            "Loading dataset %s (config=%s, split=%s)...",
            dataset_name,
            hf_config,
            hf_split,
        )

        dataset = load_dataset(
            dataset_name,
            hf_config,
            split=hf_split,
            num_proc=self.hf_num_proc,
            trust_remote_code=True,
            verification_mode="no_checks",
        )

        logger.info("Dataset loaded: %s items", len(dataset))
        return dataset

    def _compute_num_workers(self, total_samples: int) -> int:
        """Compute optimal number of workers based on samples and shard size.

        Args:
            total_samples: Total number of samples to process.

        Returns:
            Adjusted number of workers.
        """
        num_workers = self.num_workers

        if self.shard_size and self.shard_size > 0:
            max_workers_by_shards = max(1, total_samples // self.shard_size)
            if num_workers > max_workers_by_shards:
                logger.info(
                    "Adjusting workers: %s -> %s (shard_size=%s, total_samples=%s)",
                    num_workers,
                    max_workers_by_shards,
                    self.shard_size,
                    total_samples,
                )
                num_workers = max_workers_by_shards

        return max(1, num_workers)

    def convert_to_shar(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
        output_dir: Path,
        audio_format: str = "flac",
        shard_size: int | None = None,
        language: str = "en",
        id_field: str = "id",
        text_field: str = "text",
        audio_field: str = "audio",
        custom_fields: list[str] | None = None,
    ) -> tuple[int, int]:
        """Convert a HuggingFace dataset split to Lhotse Shar format.

        Args:
            dataset_name: HuggingFace dataset name.
            hf_config: HuggingFace config name.
            hf_split: HuggingFace split name.
            output_dir: Output directory for Shar archives.
            audio_format: Audio format for Shar (default: "flac").
            shard_size: Number of cuts per shard (uses self.shard_size if None).
            language: Language code for supervision segments.
            id_field: Field name containing item ID.
            text_field: Field name containing transcription.
            audio_field: Field name containing audio data.
            custom_fields: Extra HF dataset fields to store in cut.custom.

        Returns:
            Tuple of (total cuts processed, total errors).
        """
        if shard_size is None:
            shard_size = self.shard_size

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset (lazy - doesn't materialize in memory)
        dataset = self._load_dataset(dataset_name, hf_config, hf_split)

        total_samples = len(dataset)
        if total_samples == 0:
            logger.warning("No samples to process")
            return 0, 0

        # Compute number of workers
        num_workers = self._compute_num_workers(total_samples)

        # Create chunk tasks for workers (each worker will load and process its shard)
        chunks: list[ChunkTask] = []
        for i in range(num_workers):
            chunks.append(
                ChunkTask(
                    chunk_id=i,
                    dataset_name=dataset_name,
                    hf_config=hf_config,
                    hf_split=hf_split,
                    num_shards=num_workers,
                    shard_index=i,
                    language=language,
                    id_field=id_field,
                    text_field=text_field,
                    audio_field=audio_field,
                    custom_fields=custom_fields,
                )
            )

        logger.info(
            "Processing %s samples with %s workers (~%s samples/worker)",
            total_samples,
            num_workers,
            total_samples // num_workers,
        )

        # Process chunks in parallel
        count = 0
        errors = 0
        chunk_outputs: list[str] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for chunk in chunks:
                future = executor.submit(
                    _process_chunk,
                    chunk,
                    str(output_dir),
                    audio_format,
                    shard_size,
                    self.hf_num_proc,
                )
                futures[future] = chunk.chunk_id

            with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        chunk_count, chunk_errors, chunk_path = future.result()
                        count += chunk_count
                        errors += chunk_errors
                        chunk_outputs.append(chunk_path)
                        pbar.update(1)
                        pbar.set_postfix({"processed": count, "errors": errors})
                    except Exception as e:
                        logger.error("Chunk %s failed: %s", chunk_id, e)
                        errors += 1

        # Merge chunk outputs
        logger.info("Merging %s chunk outputs...", len(chunk_outputs))
        _merge_chunk_outputs(chunk_outputs, output_dir)

        logger.info(
            "Conversion complete: %s cuts processed, %s errors",
            count,
            errors,
        )

        return count, errors


# -----------------------------------------------------------------------------
# Convenience function
# -----------------------------------------------------------------------------


def convert_subset_to_shar_batched(
    dataset_name: str,
    hf_config: str,
    hf_split: str,
    output_dir: Path,
    audio_format: str = "flac",
    shard_size: int = 4000,
    language: str = "en",
    num_workers: int | None = None,
    hf_num_proc: int = 4,
    id_field: str = "id",
    text_field: str = "text",
    audio_field: str = "audio",
    custom_fields: list[str] | None = None,
) -> tuple[int, int]:
    """Convenience function for batched conversion with multiprocessing.

    This is a simpler interface to BatchedSharConverter for common use cases.

    Args:
        dataset_name: HuggingFace dataset name.
        hf_config: HuggingFace config name.
        hf_split: HuggingFace split name.
        output_dir: Output directory for Shar archives.
        audio_format: Audio format for Shar (default: "flac").
        shard_size: Number of cuts per shard file (default: 4000).
        language: Language code for supervision segments.
        num_workers: Number of worker processes (default: cpu_count()).
        hf_num_proc: HuggingFace loading processes (default: 4).
        id_field: Field name for item ID.
        text_field: Field name for transcription.
        audio_field: Field name for audio data.
        custom_fields: Extra HF dataset fields to store in cut.custom.

    Returns:
        Tuple of (total cuts processed, total errors).
    """
    converter = BatchedSharConverter(
        num_workers=num_workers,
        shard_size=shard_size,
        hf_num_proc=hf_num_proc,
    )

    return converter.convert_to_shar(
        dataset_name=dataset_name,
        hf_config=hf_config,
        hf_split=hf_split,
        output_dir=output_dir,
        audio_format=audio_format,
        shard_size=shard_size,
        language=language,
        id_field=id_field,
        text_field=text_field,
        audio_field=audio_field,
        custom_fields=custom_fields,
    )


# -----------------------------------------------------------------------------
# WebDataset support
# -----------------------------------------------------------------------------


@dataclass
class WebDatasetChunkTask:
    """A chunk task for WebDataset processing."""

    chunk_id: int
    shards: list[str]  # List of shard file paths for this worker
    language: str
    resample_config: ResampleConfig


def _find_shards(parent_dir: Path, pattern: str, recursive: bool) -> list[str]:
    """Find shard files matching the pattern."""
    if recursive:
        paths = sorted(parent_dir.rglob(pattern))
    else:
        paths = sorted(parent_dir.glob(pattern))
    return [str(p) for p in paths if p.is_file()]


def _iter_merged_webdataset_samples(
    shards: list[str],
    max_pending: int = 20000,
):
    """Yield merged samples of (cut_id, audio_bytes, metadata_dict) from WebDataset.

    WebDataset samples are stored as paired files:
        123456.flac           # audio bytes (mono)
        123456.metadata.json  # JSON metadata; transcript in "transcript" field

    This function merges audio and metadata samples by filename stem.
    """
    try:
        import webdataset as wds
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'webdataset'. Install it (e.g. `uv pip install webdataset`)."
        ) from e

    import json

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

        # Audio sample: key like "123456" and field "flac"
        if "flac" in sample and isinstance(sample["flac"], (bytes, bytearray)):
            entry = _touch_entry(key)
            entry["audio_bytes"] = bytes(sample["flac"])

            metadata = entry.get("metadata")
            audio_bytes = entry.get("audio_bytes")
            if metadata is not None and audio_bytes is not None:
                pending.pop(key, None)
                yield key, audio_bytes, metadata

        # Metadata sample: key like "123456.metadata" and field "json"
        json_bytes = None
        if "json" in sample and isinstance(sample["json"], (bytes, bytearray)):
            json_bytes = bytes(sample["json"])
        else:
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
            dropped_key, _ = pending.popitem()
            print(f"[WARNING] Pending cache overflow; dropping incomplete sample: {dropped_key}")

    if pending:
        print(f"[WARNING] Finished with {len(pending)} incomplete samples (missing audio or metadata)")


def _process_webdataset_chunk(
    chunk: WebDatasetChunkTask,
    output_dir: str,
    audio_format: str,
    shard_size: int,
    max_pending: int,
) -> tuple[int, int, str]:
    """Process a WebDataset chunk by iterating over its shards.

    Args:
        chunk: WebDatasetChunkTask with shard files for this worker.
        output_dir: Base output directory.
        audio_format: Audio format for Shar.
        shard_size: Number of cuts per shard file.
        max_pending: Max pending samples for merging.

    Returns:
        Tuple of (count, errors, chunk_output_dir).
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
        sample_iter = _iter_merged_webdataset_samples(chunk.shards, max_pending=max_pending)
        pbar = tqdm(
            sample_iter,
            desc=f"Worker {chunk.chunk_id}",
            unit="item",
            position=chunk.chunk_id,
            leave=True,
        )
        for cut_id, audio_bytes, metadata in pbar:
            try:
                transcript = metadata.get("transcript", "")
                if not isinstance(transcript, str):
                    transcript = str(transcript)

                data = CutData(
                    cut_id=cut_id,
                    audio_bytes=audio_bytes,
                    text=transcript,
                    language=chunk.language,
                    custom={"metadata": metadata},
                )

                cut = _create_cut_from_data(data, chunk.resample_config)
                writer.write(cut)
                count += 1
                pbar.set_postfix({"processed": count, "errors": errors})

            except Exception as e:
                errors += 1
                print(f"[ERROR] Failed to convert cut {cut_id}: {e}")
                pbar.set_postfix({"processed": count, "errors": errors})

    return count, errors, str(chunk_output_dir)


def convert_webdataset_to_shar_batched(
    shards_parent_dir: Path,
    output_dir: Path,
    shards_pattern: str = "*.tar*",
    recursive: bool = False,
    shard_size: int = 4000,
    audio_format: str = "flac",
    language: str = "und",
    num_workers: int | None = None,
    max_pending: int = 20000,
    resample: bool = False,
    orig_sr: int = 48000,
    target_sr: int = 16000,
) -> tuple[int, int]:
    """Convert WebDataset tar shards to Lhotse Shar format using multiprocessing.

    Args:
        shards_parent_dir: Directory containing WebDataset tar shards.
        output_dir: Output directory for Shar archives.
        shards_pattern: Glob pattern for shard files (default: "*.tar*").
        recursive: Search for shards recursively (default: False).
        shard_size: Number of cuts per shard file (default: 4000).
        audio_format: Audio format for Shar (default: "flac").
        language: Language code for supervision segments (default: "und").
        num_workers: Number of worker processes (default: cpu_count()).
        max_pending: Max pending samples for merging audio+metadata (default: 20000).
        resample: Whether to resample audio (default: False).
        orig_sr: Expected original sample rate (default: 48000).
        target_sr: Target sample rate when resampling (default: 16000).

    Returns:
        Tuple of (total cuts processed, total errors).
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Find all shard files
    shards = _find_shards(shards_parent_dir, shards_pattern, recursive)
    if not shards:
        raise SystemExit(f"No shards found under {shards_parent_dir} with pattern {shards_pattern!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Found %s shard files", len(shards))
    logger.info("Writing Shar to: %s (using %s workers)", output_dir, num_workers)

    # Create resample config
    resample_config = ResampleConfig(
        enabled=resample,
        orig_sr=orig_sr,
        target_sr=target_sr,
    )

    # Distribute shards across workers
    shards_per_worker = max(1, (len(shards) + num_workers - 1) // num_workers)
    chunks: list[WebDatasetChunkTask] = []
    for i in range(num_workers):
        start_idx = i * shards_per_worker
        end_idx = min(start_idx + shards_per_worker, len(shards))
        if start_idx >= len(shards):
            break
        worker_shards = shards[start_idx:end_idx]
        chunks.append(
            WebDatasetChunkTask(
                chunk_id=i,
                shards=worker_shards,
                language=language,
                resample_config=resample_config,
            )
        )

    logger.info(
        "Processing %s shards with %s workers (~%s shards/worker)",
        len(shards),
        len(chunks),
        shards_per_worker,
    )

    # Process chunks in parallel
    count = 0
    errors = 0
    chunk_outputs: list[str] = []

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {}
        for chunk in chunks:
            future = executor.submit(
                _process_webdataset_chunk,
                chunk,
                str(output_dir),
                audio_format,
                shard_size,
                max_pending,
            )
            futures[future] = chunk.chunk_id

        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    chunk_count, chunk_errors, chunk_path = future.result()
                    count += chunk_count
                    errors += chunk_errors
                    chunk_outputs.append(chunk_path)
                    pbar.update(1)
                    pbar.set_postfix({"processed": count, "errors": errors})
                except Exception as e:
                    logger.error("Chunk %s failed: %s", chunk_id, e)
                    errors += 1

    # Merge chunk outputs
    logger.info("Merging %s chunk outputs...", len(chunk_outputs))
    _merge_chunk_outputs(chunk_outputs, output_dir)

    logger.info(
        "Conversion complete: %s cuts processed, %s errors",
        count,
        errors,
    )

    return count, errors
