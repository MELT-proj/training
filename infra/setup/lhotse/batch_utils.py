"""Batch processing utilities for faster HuggingFace to Lhotse Shar conversion.

This module provides optional optimizations for converting HuggingFace datasets
to Lhotse Shar format using batched and parallel processing instead of streaming.

Key Optimizations:
------------------
1. Batched loading: Load data in chunks (e.g., 5000 items) instead of streaming
2. Parallel processing: Use multiprocessing for cut creation
3. Memory-efficient: Process batch, write to Shar, release memory before next batch
4. No full dataset download: Only loads one batch at a time into memory

Usage:
------
    from batch_utils import BatchedSharConverter

    converter = BatchedSharConverter(
        batch_size=5000,      # Items per batch
        num_workers=4,        # Parallel workers for cut creation
    )

    count, errors = converter.convert_to_shar(
        dataset_name="openslr/librispeech_asr",
        hf_config="all",
        hf_split="train.clean.100",
        output_dir=Path("/path/to/output"),
        audio_format="flac",
        shard_size=2000,
        language="en",
    )
"""

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def is_dataset_cached_locally(
    dataset_name: str,
    *,
    repo_type: str = "dataset",
    cache_dir: str | None = None,
) -> bool:
    """Return True if the HF dataset repo is already present in the local cache.

    Uses `snapshot_download(..., local_files_only=True)` which never fetches
    from the network. Any cache path resolution honors HF_HOME/HF_HUB_CACHE
    and the optional `cache_dir` argument.
    """

    try:
        snapshot_download(
            repo_id=dataset_name,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=["*"],
        )
        return True
    except LocalEntryNotFoundError:
        return False
    except Exception as err:  # pragma: no cover - defensive
        logger.debug("Cache check for %s failed: %s", dataset_name, err)
        return False


@dataclass
class CutData:
    """Intermediate data structure holding info needed to create a cut."""

    cut_id: str
    audio_bytes: bytes
    text: str
    language: str


def _create_cut_from_data(data: CutData) -> MonoCut | None:
    """Create a MonoCut from CutData. Designed to be called in parallel.

    This function is a module-level function (not a method) so it can be
    pickled and used with ProcessPoolExecutor.

    Args:
        data: CutData containing all info needed to create a cut.

    Returns:
        MonoCut if successful, None if an error occurred.
    """
    try:
        # Create Recording from raw bytes
        recording = Recording.from_bytes(
            data=data.audio_bytes, recording_id=data.cut_id
        )

        # Create Supervision
        supervision = SupervisionSegment(
            id=data.cut_id,
            recording_id=data.cut_id,
            start=0.0,
            duration=recording.duration,
            text=data.text,
            language=data.language,
        )

        # Create Cut
        cut = MonoCut(
            id=data.cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )

        return cut

    except Exception as e:
        logger.error(f"Failed to create cut {data.cut_id}: {e}")
        return None


class BatchedSharConverter:
    """Converter that processes HuggingFace datasets in batches with parallelization.

    This is an OPTIONAL optimization over the default streaming approach.
    Use this when you have sufficient RAM and want faster conversion.

    Memory usage is controlled by batch_size: only one batch is loaded at a time.
    """

    def __init__(
        self,
        batch_size: int = 5000,
        num_workers: int = 4,
        io_num_workers: int = 8,
        prefetch_batches: int = 1,
        use_temp_cache: bool = False,
        hf_num_proc: int = 4,
    ):
        """Initialize the batched converter.

        Args:
            batch_size: Number of items to load and process at once.
                       Higher = faster but more memory. Default: 5000
            num_workers: Number of parallel workers for cut creation.
                        Set to 1 to disable parallelization. Default: 4
            io_num_workers: Number of worker threads for IO-bound work when
                           materializing a loaded batch (e.g., reading audio
                           bytes from disk/cache). Default: 8
            prefetch_batches: Number of batches to prefetch ahead. Currently
                              supports 0 (disabled) or 1. Default: 1
            use_temp_cache: If True, use a temp directory for HF cache during
                           batch downloads (cleaned up after each batch).
            hf_num_proc: Number of processes for HuggingFace data loading.
                        Used when loading batches. Default: 4
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.io_num_workers = io_num_workers
        self.prefetch_batches = prefetch_batches
        self.use_temp_cache = use_temp_cache
        self.hf_num_proc = hf_num_proc

    def _materialize_batch(
        self,
        batch_dataset,
        *,
        io_num_workers: int,
    ) -> list[dict[str, Any]]:
        """Materialize an in-memory HF Dataset into a list of python dicts.

        This is where IO can happen for audio columns: with `Audio(decode=False)`
        HuggingFace will typically read bytes on access (e.g., `row['audio']['bytes']`).
        We parallelize *row materialization* with threads to overlap IO.
        """
        n_items = len(batch_dataset)
        if io_num_workers <= 1 or n_items <= 1:
            return list(batch_dataset)

        def _get_row(i: int) -> dict[str, Any]:
            row = batch_dataset[i]
            # Some HF datasets like HF leave row["audio"]["bytes"] to None when the column is casted with Audio(decode=False)
            # Check if this is the case, and if so, read the raw bytes manually using standard python IO
            if row.get("audio") and row["audio"].get("bytes") is None:
                audio_path = row["audio"].get("path")
                if audio_path:
                    with open(audio_path, "rb") as f:
                        row["audio"]["bytes"] = f.read()
            
            # print(row["audio"]["bytes"] == None)
            return row

        # Threads are intentional here: the hot path is file/network IO.
        # Using processes would require pickling/copying the dataset object.
        with ThreadPoolExecutor(max_workers=io_num_workers) as pool:
            return list(pool.map(_get_row, range(n_items)))

    def _load_batch_range(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
        start_idx: int,
        end_idx: int,
        cache_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load a specific range of items from the dataset.

        Uses HuggingFace's split slicing to load only the needed range.

        Args:
            dataset_name: HuggingFace dataset name.
            hf_config: HuggingFace config name.
            hf_split: HuggingFace split name.
            start_idx: Start index (inclusive).
            end_idx: End index (exclusive).
            cache_dir: Optional cache directory for downloads.

        Returns:
            List of dataset items.
        """
        # Use HF split slicing: "train[0:1000]" loads only indices 0-999
        split_slice = f"{hf_split}[{start_idx}:{end_idx}]"

        logger.debug(f"Loading batch: {split_slice}")

        # Load the batch (non-streaming to allow parallel processing)
        batch_dataset = load_dataset(
            dataset_name,
            hf_config,
            split=split_slice,
            num_proc=self.hf_num_proc,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # Don't decode audio - get raw bytes
        batch_dataset = batch_dataset.cast_column("audio", Audio(decode=False))

        # Materialize the batch as python dicts. This step can be IO-bound due
        # to lazy audio byte loading; we optionally parallelize it.
        return self._materialize_batch(
            batch_dataset,
            io_num_workers=self.io_num_workers,
        )

    def _get_dataset_size(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
        *,
        use_streaming: bool,
        cache_dir: str | None = None,
    ) -> int:
        """Get the total number of items in a dataset split.

        Uses streaming only when the dataset is not cached locally; otherwise
        loads locally with `download_mode="reuse_dataset_if_exists"`.
        """
        load_kwargs = {
            "path": dataset_name,
            "name": hf_config,
            "split": hf_split,
            "streaming": use_streaming,
            "trust_remote_code": True,
            "cache_dir": cache_dir,
        }
        if not use_streaming:
            load_kwargs["download_mode"] = "reuse_dataset_if_exists"
            load_kwargs["num_proc"] = self.hf_num_proc

        ds = load_dataset(**load_kwargs)

        # Get the dataset info if available
        if hasattr(ds, "info") and getattr(ds.info, "splits", None):  # type: ignore
            split_info = ds.info.splits.get(hf_split)  # type: ignore
            if split_info and getattr(split_info, "num_examples", None):
                return split_info.num_examples  # type: ignore

        # Fallback: count items (slow but works)
        logger.warning(
            f"Dataset size not in metadata for {hf_split}, counting items..."
        )
        count = 0
        for _ in tqdm(ds, desc="Counting items", unit="item"):  # type: ignore
            count += 1
        return count

    def _process_batch_parallel(
        self,
        items: list[dict[str, Any]],
        language: str,
        id_field: str = "id",
        text_field: str = "text",
        audio_field: str = "audio",
        batch_start_idx: int = 0,
    ) -> tuple[list[MonoCut], int]:
        """Process a batch of items in parallel to create cuts.

        Args:
            items: List of dataset items.
            language: Language code for supervisions.
            id_field: Field name containing the item ID.
            text_field: Field name containing the transcription.
            audio_field: Field name containing the audio data.
            batch_start_idx: Starting index for fallback IDs.

        Returns:
            Tuple of (list of cuts, error count).
        """
        # Prepare data for parallel processing
        cut_data_list = []
        for i, item in enumerate(items):
            hf_id = item.get(id_field, f"no_id_{batch_start_idx + i}")
            cut_id = str(hf_id)

            cut_data_list.append(
                CutData(
                    cut_id=cut_id,
                    audio_bytes=item[audio_field]["bytes"],
                    text=item[text_field],
                    language=language,
                )
            )

        cuts = []
        errors = 0

        if self.num_workers <= 1:
            # Sequential processing
            for data in tqdm(
                cut_data_list, desc="Creating cuts", unit="cut", leave=False
            ):
                cut = _create_cut_from_data(data)
                if cut is not None:
                    cuts.append(cut)
                else:
                    errors += 1
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_create_cut_from_data, data): data
                    for data in cut_data_list
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Creating cuts (parallel)",
                    unit="cut",
                    leave=False,
                ):
                    cut = future.result()
                    if cut is not None:
                        cuts.append(cut)
                    else:
                        errors += 1

        return cuts, errors

    def _iter_batches(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
        total_size: int,
        cache_dir: str | None = None,
    ) -> Iterator[tuple[list[dict[str, Any]], int]]:
        """Iterate over dataset in batches.

        Args:
            dataset_name: HuggingFace dataset name.
            hf_config: HuggingFace config name.
            hf_split: HuggingFace split name.
            total_size: Total number of items in the dataset.
            cache_dir: Optional cache directory.

        Yields:
            Tuple of (batch items, start index).
        """
        if self.prefetch_batches not in (0, 1):
            raise ValueError("prefetch_batches currently supports only 0 or 1")

        if self.prefetch_batches == 0:
            for start_idx in range(0, total_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_size)
                batch = self._load_batch_range(
                    dataset_name,
                    hf_config,
                    hf_split,
                    start_idx,
                    end_idx,
                    cache_dir,
                )
                yield batch, start_idx
            return

        # Prefetch one batch ahead: overlaps HF download/reading + audio byte IO
        # with CPU-bound cut creation/writing for the current batch.
        with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
            start_idx = 0
            end_idx = min(self.batch_size, total_size)
            next_future = prefetch_pool.submit(
                self._load_batch_range,
                dataset_name,
                hf_config,
                hf_split,
                start_idx,
                end_idx,
                cache_dir,
            )

            while start_idx < total_size:
                batch = next_future.result()

                next_start = start_idx + self.batch_size
                if next_start < total_size:
                    next_end = min(next_start + self.batch_size, total_size)
                    next_future = prefetch_pool.submit(
                        self._load_batch_range,
                        dataset_name,
                        hf_config,
                        hf_split,
                        next_start,
                        next_end,
                        cache_dir,
                    )
                else:
                    next_future = None

                yield batch, start_idx

                if next_future is None:
                    break
                start_idx = next_start

    def convert_to_shar(
        self,
        dataset_name: str,
        hf_config: str,
        hf_split: str,
        output_dir: Path,
        audio_format: str = "flac",
        shard_size: int = 2000,
        language: str = "en",
        id_field: str = "id",
        text_field: str = "text",
        audio_field: str = "audio",
    ) -> tuple[int, int]:
        """Convert a HuggingFace dataset split to Lhotse Shar format using batching.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "openslr/librispeech_asr").
            hf_config: HuggingFace config name.
            hf_split: HuggingFace split name.
            output_dir: Output directory for Shar archives.
            audio_format: Audio format for Shar (default: "flac").
            shard_size: Number of cuts per shard file (default: 2000).
            language: Language code for supervision segments.
            id_field: Field name in HF dataset containing item ID.
            text_field: Field name containing transcription text.
            audio_field: Field name containing audio data.

        Returns:
            Tuple of (total cuts processed, total errors).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup temp cache if requested (applies to size probing and batches)
        temp_dir = None
        cache_dir = None
        if self.use_temp_cache:
            temp_dir = tempfile.TemporaryDirectory(prefix="hf_batch_cache_")
            cache_dir = temp_dir.name
            logger.info(f"Using temporary cache directory: {cache_dir}")

        # Decide streaming based on local cache presence
        dataset_cached = is_dataset_cached_locally(dataset_name, cache_dir=cache_dir)
        use_streaming = not dataset_cached

        if not dataset_cached:
            logger.info(
                "Dataset %s not found in local HF cache; using streaming to compute size.",
                dataset_name,
            )

        # Get dataset size
        logger.info(f"Getting dataset size for {hf_split} (streaming={use_streaming})...")
        total_size = self._get_dataset_size(
            dataset_name,
            hf_config,
            hf_split,
            use_streaming=use_streaming,
            cache_dir=cache_dir,
        )
        logger.info(f"Total items: {total_size}")

        total_count = 0
        total_errors = 0
        num_batches = (total_size + self.batch_size - 1) // self.batch_size

        try:
            # Initialize SharWriter
            writer = SharWriter(
                output_dir=output_dir,
                fields={"recording": audio_format},
                shard_size=shard_size,
            )

            with writer:
                batch_pbar = tqdm(
                    self._iter_batches(
                        dataset_name, hf_config, hf_split, total_size, cache_dir
                    ),
                    total=num_batches,
                    desc="Processing batches",
                    unit="batch",
                )

                for batch_items, start_idx in batch_pbar:
                    batch_pbar.set_postfix(
                        {"items": f"{start_idx}-{start_idx + len(batch_items)}"}
                    )

                    # Process batch to create cuts
                    cuts, errors = self._process_batch_parallel(
                        batch_items,
                        language=language,
                        id_field=id_field,
                        text_field=text_field,
                        audio_field=audio_field,
                        batch_start_idx=start_idx,
                    )

                    # Write cuts to Shar
                    for cut in cuts:
                        writer.write(cut)

                    total_count += len(cuts)
                    total_errors += errors

                    # Clear memory
                    del batch_items
                    del cuts

        finally:
            # Clean up temp directory if used
            if temp_dir is not None:
                temp_dir.cleanup()
                logger.info("Cleaned up temporary cache directory")

        logger.info(
            f"Batched conversion complete: {total_count} cuts, {total_errors} errors"
        )
        return total_count, total_errors


def convert_subset_to_shar_batched(
    dataset_name: str,
    hf_config: str,
    hf_split: str,
    output_dir: Path,
    audio_format: str = "flac",
    shard_size: int = 2000,
    language: str = "en",
    batch_size: int = 5000,
    num_workers: int = 4,
    io_num_workers: int = 8,
    prefetch_batches: int = 1,
    hf_num_proc: int = 4,
    use_temp_cache: bool = False,
    id_field: str = "id",
    text_field: str = "text",
    audio_field: str = "audio",
) -> tuple[int, int]:
    """Convenience function for batched conversion with sensible defaults.

    This is a simpler interface to BatchedSharConverter for common use cases.

    Args:
        dataset_name: HuggingFace dataset name.
        hf_config: HuggingFace config name.
        hf_split: HuggingFace split name.
        output_dir: Output directory for Shar archives.
        audio_format: Audio format for Shar (default: "flac").
        shard_size: Number of cuts per shard file (default: 2000).
        language: Language code for supervision segments.
        batch_size: Items per batch (default: 5000).
        num_workers: Parallel workers for cut creation (default: 4).
        io_num_workers: Threads for IO-bound batch materialization (default: 8).
        prefetch_batches: Prefetch batches ahead (0 or 1; default: 1).
        hf_num_proc: HuggingFace loading processes (default: 4).
        use_temp_cache: Use temp directory for HF cache (default: False).
        id_field: Field name for item ID.
        text_field: Field name for transcription.
        audio_field: Field name for audio data.

    Returns:
        Tuple of (total cuts processed, total errors).
    """
    converter = BatchedSharConverter(
        batch_size=batch_size,
        num_workers=num_workers,
        io_num_workers=io_num_workers,
        prefetch_batches=prefetch_batches,
        use_temp_cache=use_temp_cache,
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
    )
