"""
Lhotse DataLoader utilities for MELT training.

This module provides functions to create Lhotse samplers and dataloaders
from configuration objects, following patterns from NeMo's dataloader utilities.

Key functions:
- get_lhotse_sampler_from_config: Creates a CutSampler from config
- get_lhotse_dataloader_from_config: Creates a full DataLoader from config
- compute_dataset_duration: Computes total dataset duration for epoch estimation
"""

import gzip
import json
import math
import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import (
    DynamicBucketingSampler,
    DynamicCutSampler,
    IterableDatasetWrapper,
    RoundRobinSampler,
    make_worker_init_fn,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import CutSampler, TimeConstraint
from lhotse.utils import fix_random_seed

from .....logging_utils import get_logger
from ....config import DataConfig, DatasetConfig, DataSourceConfig


logger = get_logger(__name__)


def _maybe_attach_set_epoch(dataloader: torch.utils.data.DataLoader, sampler: CutSampler) -> None:
    """Ensure the returned DataLoader exposes a ``set_epoch`` method.

    HF Trainer advances epoch state via ``epoch_dataloader.set_epoch(epoch)``.
    For iterable-style Lhotse pipelines, we want this to reach the
    ``IterableDatasetWrapper`` and/or underlying sampler.
    """

    if hasattr(dataloader, "set_epoch"):
        return

    def set_epoch(epoch: int) -> None:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            return

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        dataset_sampler = getattr(dataset, "sampler", None)
        if dataset_sampler is not None and hasattr(dataset_sampler, "set_epoch"):
            dataset_sampler.set_epoch(epoch)

    setattr(dataloader, "set_epoch", set_epoch)


# -----------------------------------------------------------------------------
# Iterable Dataset Wrappers
# -----------------------------------------------------------------------------


class InfiniteIterableDatasetWrapper(IterableDatasetWrapper):
    """Lhotse IterableDatasetWrapper with __len__ support for HF Trainer.

    This wrapper adds epoch length estimation to Lhotse's IterableDatasetWrapper,
    allowing HF Trainer to display progress bars and compute steps_in_epoch.

    The dataset still iterates infinitely (via sampler.repeat()), but __len__
    returns the estimated number of batches per epoch per rank for progress tracking.

    Args:
        dataset: PyTorch Dataset that processes CutSets.
        sampler: Lhotse CutSampler that yields batches of cuts.
        estimated_batches_per_epoch: Expected number of batches per rank per epoch.
        **kwargs: Additional arguments passed to IterableDatasetWrapper.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sampler: CutSampler,
        estimated_batches_per_epoch: int,
        **kwargs,
    ):
        super().__init__(dataset=dataset, sampler=sampler, **kwargs)
        self._estimated_batches = estimated_batches_per_epoch

    def __len__(self) -> int:
        """Return estimated number of batches per epoch for progress bars."""
        return self._estimated_batches


class FiniteIterableDatasetWrapper(torch.utils.data.IterableDataset):
    """Wrapper that iterates through sampler once and has known length.

    Unlike Lhotse's IterableDatasetWrapper, this does NOT loop infinitely.
    It's designed for evaluation where we want exactly one pass through the data.

    This wrapper:
    1. Iterates through the sampler exactly once (one epoch)
    2. Provides __len__ for progress bar support
    3. Handles multi-worker sharding via round-robin batch assignment

    Multi-worker handling:
    When num_workers > 0, each worker gets a copy of this dataset. To avoid
    duplicate processing, each worker only yields batches where:
        batch_idx % num_workers == worker_id

    This ensures all batches are processed exactly once across all workers.

    Args:
        dataset: PyTorch Dataset that processes CutSets.
        sampler: Lhotse CutSampler that yields batches of cuts.
        num_batches: Expected number of batches (for __len__).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sampler: CutSampler,
        num_batches: int | None = None,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self._num_batches = num_batches

    def __iter__(self):
        """Iterate through the sampler once and yield processed batches.

        With num_workers > 0, each worker only processes batches assigned
        to it via round-robin (batch_idx % num_workers == worker_id).
        """
        # Get worker info for proper sharding
        worker_info = torch.utils.data.get_worker_info()

        # Set epoch for reproducibility
        self.sampler.set_epoch(0)

        if worker_info is None:
            # Single-process: yield all batches
            for batch in self.sampler:
                result = self.dataset[batch]
                if result is not None:
                    yield result
        else:
            # Multi-process: each worker handles its shard via round-robin
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            for batch_idx, batch in enumerate(self.sampler):
                if batch_idx % num_workers == worker_id:
                    result = self.dataset[batch]
                    if result is not None:
                        yield result

    def __len__(self) -> int:
        """Return the expected number of batches.

        This enables progress bars in HF Trainer's evaluation loop.
        """
        if self._num_batches is not None:
            return self._num_batches
        # Fallback: try to get from sampler if it has length
        try:
            return len(self.sampler)
        except TypeError:
            raise TypeError(
                "FiniteIterableDatasetWrapper length unknown. Provide num_batches or use a sampler with __len__."
            )


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get a value from config, supporting both dataclass and dict access."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


def _read_shar_manifest_durations(
    shar_path: str | Path,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> tuple[float, int]:
    """Read total duration and cut count from SHAR manifest files.

    SHAR format stores cut manifests as gzipped JSONL files in the shar directory.
    This function reads only the manifest files (not audio) to extract durations.

    Args:
        shar_path: Path to the SHAR directory.
        min_duration: Minimum cut duration to include (default: 0.0).
        max_duration: Maximum cut duration to include (default: inf).

    Returns:
        Tuple of (total_duration_seconds, num_cuts).
    """
    shar_path = Path(shar_path)
    total_duration = 0.0
    num_cuts = 0

    # Find all cuts manifest files (cuts.*.jsonl.gz pattern)
    manifest_files = sorted(glob(str(shar_path / "cuts.*.jsonl.gz")))

    if not manifest_files:
        logger.warning(f"No manifest files found in {shar_path}")
        return 0.0, 0

    for manifest_file in manifest_files:
        try:
            with gzip.open(manifest_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cut_data = json.loads(line)
                        duration = cut_data.get("duration", 0.0)
                        # Apply duration filter
                        if min_duration <= duration <= max_duration:
                            total_duration += duration
                            num_cuts += 1
        except Exception as e:
            logger.warning(f"Error reading manifest {manifest_file}: {e}")
            continue

    return total_duration, num_cuts


def compute_dataset_duration(
    config: DatasetConfig | dict,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> tuple[float, int]:
    """Compute total dataset duration from configuration.

    Reads SHAR manifest files to compute total duration without loading audio.
    Applies duration filtering if min/max_duration are specified.

    Args:
        config: DatasetConfig with input_cfg specifying data sources.
        min_duration: Minimum cut duration filter (from config if None).
        max_duration: Maximum cut duration filter (from config if None).

    Returns:
        Tuple of (total_duration_seconds, num_cuts) after filtering.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        return 0.0, 0

    # Get duration filters from config if not explicitly provided
    if min_duration is None:
        min_duration = _get_config_value(config, "min_duration", 0.0)
    if max_duration is None:
        max_duration = _get_config_value(config, "max_duration", float("inf"))

    total_duration = 0.0
    num_cuts = 0

    for source_cfg in input_cfg:
        source_type = _get_config_value(source_cfg, "type", "lhotse_shar")

        if source_type == "lhotse_shar":
            shar_path = _get_config_value(source_cfg, "shar_path")
            if shar_path is None:
                continue

            # Expand environment variables in path
            shar_path = os.path.expandvars(str(shar_path))

            if not Path(shar_path).exists():
                logger.warning(f"Shar path not found: {shar_path}")
                continue

            # Use helper function to read manifest durations
            source_duration, source_cuts = _read_shar_manifest_durations(shar_path, min_duration, max_duration)
            total_duration += source_duration
            num_cuts += source_cuts

        elif source_type == "lhotse_cuts":
            cuts_path = _get_config_value(source_cfg, "cuts_path")
            if cuts_path is None:
                continue

            cuts_path = os.path.expandvars(str(cuts_path))

            if not Path(cuts_path).exists():
                logger.warning(f"Cuts path not found: {cuts_path}")
                continue

            # Load cuts and compute duration
            try:
                cuts = CutSet.from_file(cuts_path)
                for cut in cuts:
                    if min_duration <= cut.duration <= max_duration:
                        total_duration += cut.duration
                        num_cuts += 1
            except Exception as e:
                logger.warning(f"Error reading cuts {cuts_path}: {e}")
                continue

    return total_duration, num_cuts


def estimate_steps_per_epoch(
    config: DatasetConfig | dict,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> tuple[int, float, int, int, int]:
    """Estimate the number of training steps per epoch.

    Computes steps based on how data is sharded across ranks:
        steps_per_epoch = total_duration / (batch_duration * world_size * gradient_accumulation_steps)

    Each rank processes 1/world_size of the data, and each optimizer step
    requires gradient_accumulation_steps micro-batches.

    Args:
        config: DatasetConfig with batch_duration and data sources.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        world_size: Number of distributed processes.

    Returns:
        Tuple of (steps_per_epoch, total_duration_hours, num_cuts).
    """
    batch_duration = _get_config_value(config, "batch_duration")
    if batch_duration is None:
        # Fixed batch size mode - can't estimate without more info
        logger.warning("batch_duration not set, cannot estimate steps per epoch")
        return -1, 0.0, 0, 0, 0

    num_workers = _get_config_value(config, "num_workers", 1)
    total_hours = _get_config_value(config, "total_hours", None)
    total_cuts = _get_config_value(config, "total_cuts", None)
    force_estimate = _get_config_value(config, "force_estimate", None)
    if total_hours is None:
        if force_estimate:
            logger.info(
                "Users requested forced estimation of total_hours; proceeding with estimation (disable `force_estimate` if your training starts gets delayed too much)."
            )
            total_duration, total_cuts = compute_dataset_duration(config)

            if total_duration <= 0:
                return 0, 0.0, 0, 0, 0

            total_hours = total_duration / 3600.0
        else:
            logger.info("`total_hours` and `force_estimate` not set; cannot estimate steps per epoch")
            return -1, 0.0, 0, 0, 0
    else:
        total_duration = total_hours * 3600.0

    batches_per_epoch = math.ceil(total_duration / batch_duration)

    # We use data parallelism by setting split_for_loading=True in CutSet.from_shar()
    # The shard are hence divided to world_size * num_workers processes.
    batches_per_worker = batches_per_epoch / (world_size * num_workers)

    # The number of update steps is rescaled by gradient accumulation steps
    optimizer_steps_per_epoch = math.ceil(batches_per_worker / gradient_accumulation_steps)

    return optimizer_steps_per_epoch, total_hours, total_cuts, batches_per_epoch, batches_per_worker


def read_cutset_from_config(config: DatasetConfig | dict) -> tuple[CutSet, bool]:
    """Read CutSet(s) from configuration.

    Args:
        config: DatasetConfig with input_cfg specifying data sources.

    Returns:
        Tuple of (CutSet, use_iterable_dataset).
        use_iterable_dataset is True for tarred/shar data.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        raise ValueError("No data sources specified in input_cfg")

    cutsets = []
    use_iterable = False
    shard_seed = _get_config_value(config, "shard_seed", "randomized")

    # For SHAR/WebDataset-style streaming, each PyTorch DataLoader worker (and each DDP rank)
    # will host its own iterator/sampler replica.
    # Unless the underlying CutSet shards are explicitly split between node+worker
    # combinations, this can easily lead to duplicated data across workers/ranks.
    #
    # Lhotse provides a dedicated mechanism for this: CutSet.from_shar(..., split_for_dataloading=True).
    # split_for_dataloading_default = bool(_get_config_value(config, "split_for_dataloading", True))

    for source_cfg in input_cfg:
        source_type = _get_config_value(source_cfg, "type", "lhotse_shar")

        if source_type == "lhotse_shar":
            shar_path = _get_config_value(source_cfg, "shar_path")
            if shar_path is None:
                raise ValueError("shar_path must be specified for lhotse_shar type")

            # Expand environment variables in path
            shar_path = os.path.expandvars(str(shar_path))

            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")

            logger.info(f"Loading CutSet from shar: {shar_path} (seed: {shard_seed})")
            cuts = CutSet.from_shar(
                in_dir=shar_path,
                shuffle_shards=True,
                seed=config.seed,
                stateful_shuffle=True,
                split_for_dataloading=True,
            )
            use_iterable = True  # Shar always uses iterable dataset

        elif source_type == "lhotse_cuts":
            cuts_path = _get_config_value(source_cfg, "cuts_path")
            if cuts_path is None:
                raise ValueError("cuts_path must be specified for lhotse_cuts type")

            cuts_path = os.path.expandvars(str(cuts_path))

            if not Path(cuts_path).exists():
                raise FileNotFoundError(f"Cuts path not found: {cuts_path}")

            logger.info(f"Loading CutSet from cuts: {cuts_path}")
            cuts = CutSet.from_file(cuts_path)

        else:
            raise ValueError(f"Unknown data source type: {source_type}")

        # Add tags to cuts if specified
        tags = _get_config_value(source_cfg, "tags", {})
        if tags:
            tag_dict = dict(tags) if not isinstance(tags, dict) else tags
            cuts = cuts.map(partial(_add_tags_to_cut, tags=tag_dict), apply_fn=None)

        cutsets.append(cuts)

    # Combine multiple cutsets
    if len(cutsets) == 1:
        combined = cutsets[0]
    else:
        # Use mux for weighted combination if weights differ
        weights = []
        for cfg in input_cfg:
            weight = _get_config_value(cfg, "weight", 1.0)
            weights.append(float(weight))

        combined = CutSet.mux(*cutsets, weights=weights, seed=shard_seed)

    # Since we split cuts across data workers (split_for_dataloading=True),
    # to avoid deadlocks we force the combined CutSet to repeat infinitely.
    combined = combined.repeat()

    return combined, use_iterable


def _add_tags_to_cut(cut: Cut, tags: dict[str, str]) -> Cut:
    """Add metadata tags to a cut."""
    if cut.custom is None:
        cut.custom = {}

    cut.custom.update(tags)
    # Also store as attribute for easier access
    cut.tags = tags
    return cut


def get_lhotse_sampler_from_config(
    config: DatasetConfig | dict,
    global_rank: int = 0,
    world_size: int = 1,
    split_batches: bool = False,
) -> tuple[CutSampler, bool]:
    """Create a CutSampler from configuration.

    Args:
        config: DatasetConfig with sampling parameters.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (CutSampler, use_iterable_dataset).
    """
    # Load cutset from config. Since we are using Shar data, this is a lazy CutSet.
    # For now, it should always be use_iterable = True.
    cuts, use_iterable = read_cutset_from_config(config)

    # Apply duration filtering
    min_duration = _get_config_value(config, "min_duration")
    max_duration = _get_config_value(config, "max_duration")

    if min_duration is not None or max_duration is not None:
        min_dur = min_duration if min_duration is not None else 0.0
        max_dur = max_duration if max_duration is not None else float("inf")
        cuts = cuts.filter(lambda c: min_dur <= c.duration <= max_dur)
        logger.info(f"Applied duration filter: [{min_dur}, {max_dur}]")

    # Determine sampling constraint.
    # When training with an IterableDataset under Accelerate + `split_batches=True`,
    # the main process fetches a *global* batch and slices it across `world_size`.
    # To preserve the effective per-rank batch constraint, scale it by `world_size`.
    max_cuts = _get_config_value(config, "batch_size")
    batch_duration = _get_config_value(config, "batch_duration")
    quadratic_duration = _get_config_value(config, "quadratic_duration")

    if use_iterable and split_batches and world_size > 1:
        if max_cuts is not None:
            max_cuts = int(max_cuts) * int(world_size)
        if batch_duration is not None:
            batch_duration = float(batch_duration) * float(world_size)
        if quadratic_duration is not None:
            quadratic_duration = float(quadratic_duration) * float(world_size)

        logger.info(
            "IterableDataset + split_batches=True: scaling sampler constraints by world_size=%s ",
            world_size,
        )

    constraint = TimeConstraint(
        max_cuts=max_cuts,
        max_duration=batch_duration,
        quadratic_duration=quadratic_duration,
    )

    # Create sampler
    shuffle = _get_config_value(config, "shuffle", True)
    drop_last = _get_config_value(config, "drop_last", False)
    shuffle_buffer_size = _get_config_value(config, "shuffle_buffer_size", 10000)

    seed = resolve_seed(_get_config_value(config, "seed", 0))
    shard_seed = _get_config_value(config, "shard_seed", "randomized")

    # Important: do NOT resolve shard_seed='randomized' in the main process.
    # Lhotse's resolve_seed('randomized') is designed to run in DataLoader workers after
    # make_worker_init_fn has set LHOTSE_PROCESS_SEED.
    if isinstance(shard_seed, str) and shard_seed not in ("trng", "randomized"):
        raise ValueError(f"Unsupported shard_seed={shard_seed!r}. Supported values: int, 'trng', 'randomized'.")

    if _get_config_value(config, "use_bucketing", False):
        # Dynamic bucketing sampler for efficient batching
        num_buckets = _get_config_value(config, "num_buckets", 30)
        bucket_buffer_size = _get_config_value(config, "bucket_buffer_size", 10000)
        bucket_duration_bins = _get_config_value(config, "bucket_duration_bins")

        # Auto-estimate duration bins if not provided
        if bucket_duration_bins is None and batch_duration is not None:
            begin = min_duration if min_duration is not None and min_duration > 0 else 0.0
            end = max_duration if max_duration is not None and max_duration < float("inf") else 30.0
            bucket_duration_bins = np.linspace(begin, end, num_buckets + 1)[1:-1].tolist()

        logger.info(
            f"Creating DynamicBucketingSampler with "
            f"batch_duration={batch_duration}, "
            f"batch_size={max_cuts}, "
            f"num_buckets={num_buckets}"
        )

        sampler = DynamicBucketingSampler(
            cuts,
            constraint=constraint,
            shuffle=shuffle,
            drop_last=drop_last,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=shard_seed,
            num_buckets=num_buckets,
            duration_bins=bucket_duration_bins,
            buffer_size=bucket_buffer_size,
            rank=0 if use_iterable else global_rank,
            world_size=1 if use_iterable else world_size,
        )
    else:
        # Simple dynamic sampler (no bucketing)
        logger.info(f"Creating DynamicCutSampler with batch_duration={batch_duration}, batch_size={max_cuts}")

        sampler = DynamicCutSampler(
            cuts,
            constraint=constraint,
            shuffle=shuffle,
            drop_last=drop_last,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=shard_seed,
            rank=0 if use_iterable else global_rank,
            world_size=1 if use_iterable else world_size,
        )

    return sampler, use_iterable


def get_lhotse_dataloader_from_config(
    config: DatasetConfig | dict,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    split_batches: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader from configuration.

    Args:
        config: DatasetConfig with data loading parameters.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.
        dataset: PyTorch Dataset that processes CutSets.

    Returns:
        DataLoader configured for Lhotse data loading.
    """
    logger.info("Creating Lhotse DataLoader")

    # Set up CUDA expandable segments for better memory management
    _maybe_set_cuda_expandable_segments(enabled=True)

    # Resolve seed
    seed = resolve_seed(_get_config_value(config, "seed", 0))
    logger.info(f"Creating a lhotse dataloader with seed {seed}")
    fix_random_seed(seed)

    # Get sampler
    sampler, use_iterable = get_lhotse_sampler_from_config(
        config=config,
        global_rank=global_rank,
        world_size=world_size,
        split_batches=split_batches,
    )

    # Create dataloader
    num_workers = _get_config_value(config, "num_workers", 0)
    pin_memory = _get_config_value(config, "pin_memory", True)

    # Extract optional prefetch_factor for worker-level prefetching
    prefetch_factor_val = _get_config_value(config, "prefetch_factor", 2)
    prefetch_factor = int(prefetch_factor_val) if prefetch_factor_val is not None else 2

    if use_iterable:
        # For tarred/shar data, wrap dataset with sampler
        # This moves sampling to worker processes
        logger.info("Using InfiniteIterableDatasetWrapper for shar data")
        logger.info(f"Using world size: {world_size}, rank: {global_rank}")

        # Estimate batches per epoch for progress bars
        # This is the micro-batch count per rank per epoch
        gradient_accumulation_steps = 1  # We count micro-batches, not optimizer steps
        steps_per_epoch, _, _, _, _ = estimate_steps_per_epoch(
            config=config,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        logger.info(f"Estimated {steps_per_epoch} micro-batches per epoch per rank")

        dloader_kwargs = {
            "dataset": InfiniteIterableDatasetWrapper(
                dataset=dataset,
                sampler=sampler,
                estimated_batches_per_epoch=steps_per_epoch,
            ),
            "worker_init_fn": make_worker_init_fn(
                rank=global_rank,
                world_size=world_size,
                seed=seed,
                set_different_node_and_worker_seeds=True,
            ),
            "persistent_workers": num_workers > 0,
        }
    else:
        # For non-tarred data, sampler stays in main process
        logger.info("Using map-style dataset")
        dloader_kwargs = {
            "dataset": dataset,
            "sampler": sampler,
        }

    dataloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,  # Batching handled by sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else 0,
    )

    _maybe_attach_set_epoch(dataloader=dataloader, sampler=sampler)
    return dataloader


def get_train_dataloader_from_config(
    data_config: DataConfig | dict,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
    split_batches: bool = False,
) -> torch.utils.data.DataLoader:
    """Convenience function to create training dataloader.

    Args:
        data_config: Full DataConfig with train_ds settings.
        dataset: Dataset to use.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Training DataLoader.
    """
    train_ds = _get_config_value(data_config, "train_ds")
    return get_lhotse_dataloader_from_config(
        config=train_ds,
        global_rank=global_rank,
        world_size=world_size,
        dataset=dataset,
        split_batches=split_batches,
    )


def estimate_num_batches(
    config: DatasetConfig | dict,
    world_size: int = 1,
) -> int:
    """Estimate number of batches for a dataset configuration.

    This is useful for progress bars and epoch estimation.

    Note: When using DynamicBucketingSampler with multiple buckets, the actual
    number of batches may differ from this estimate because each bucket may
    produce a partial final batch when exhausted.

    Args:
        config: DatasetConfig with batch settings and data sources.
        world_size: Number of distributed processes (batches are split across ranks).

    Returns:
        Estimated number of batches per rank.
    """
    batch_size = _get_config_value(config, "batch_size")
    batch_duration = _get_config_value(config, "batch_duration")
    num_buckets = _get_config_value(config, "num_buckets", 10)

    total_duration, num_cuts = compute_dataset_duration(config)

    if num_cuts == 0:
        return 0

    if batch_size is not None:
        # Fixed batch size mode - use ceiling division
        num_batches = (num_cuts + batch_size - 1) // batch_size
    elif batch_duration is not None:
        # Dynamic batching by duration
        # Estimate average duration per cut
        avg_duration = total_duration / num_cuts if num_cuts > 0 else 10.0
        cuts_per_batch = max(1, int(batch_duration / avg_duration))
        num_batches = (num_cuts + cuts_per_batch - 1) // cuts_per_batch
    else:
        # Fallback: assume one cut per batch
        num_batches = num_cuts

    # Account for bucket overhead: each bucket may produce a partial final batch
    # This gives a more accurate estimate for DynamicBucketingSampler
    if num_buckets is not None and num_buckets > 1:
        bucket_overhead = num_buckets
        num_batches += bucket_overhead
        logger.warning(
            f"Batch estimate includes +{bucket_overhead} for dynamic bucketing overhead. "
            f"Actual batch count may vary slightly due to bucket boundaries."
        )

    # Divide by world_size since each rank sees a shard
    num_batches_per_rank = max(1, num_batches // world_size) if world_size > 1 else num_batches

    return num_batches_per_rank


def get_eval_sampler_from_config(
    config: DatasetConfig | dict,
    global_rank: int = 0,
    world_size: int = 1,
) -> tuple[CutSampler, CutSet, bool]:
    """Create a CutSampler optimized for evaluation.

    Unlike get_lhotse_sampler_from_config (for training), this:
    - Uses shuffle=False for deterministic evaluation
    - Uses DynamicBucketingSampler for efficient batching by duration
    - Properly sets rank/world_size for distributed evaluation

    Distributed evaluation:
    - Each GPU (rank) processes 1/world_size of the data
    - The sampler handles GPU sharding via rank/world_size
    - DataLoader workers within each GPU use round-robin (handled by FiniteIterableDatasetWrapper)

    Args:
        config: DatasetConfig with sampling parameters.
        global_rank: Global rank for distributed evaluation.
        world_size: Total number of processes (GPUs).

    Returns:
        Tuple of (CutSampler, CutSet, use_iterable_dataset).
    """
    # Load cutset from config
    cuts, use_iterable = read_cutset_from_config(config)

    # Apply duration filtering
    min_duration = _get_config_value(config, "min_duration")
    max_duration = _get_config_value(config, "max_duration")

    if min_duration is not None or max_duration is not None:
        min_dur = min_duration if min_duration is not None else 0.0
        max_dur = max_duration if max_duration is not None else float("inf")
        cuts = cuts.filter(lambda c: min_dur <= c.duration <= max_dur)
        logger.info(f"Applied duration filter: [{min_dur}, {max_dur}]")

    # Get batching constraints
    max_cuts = _get_config_value(config, "batch_size")
    batch_duration = _get_config_value(config, "batch_duration")
    quadratic_duration = _get_config_value(config, "quadratic_duration")

    constraint = TimeConstraint(
        max_cuts=max_cuts,
        max_duration=batch_duration,
        quadratic_duration=quadratic_duration,
    )

    # For eval, always use bucketing for efficiency (groups similar-length utterances)
    num_buckets = _get_config_value(config, "num_buckets", 10)
    seed = resolve_seed(_get_config_value(config, "seed", 0))

    # Auto-estimate duration bins if not provided
    bucket_duration_bins = _get_config_value(config, "bucket_duration_bins")
    if bucket_duration_bins is None and batch_duration is not None:
        begin = min_duration if min_duration is not None and min_duration > 0 else 0.0
        end = max_duration if max_duration is not None and max_duration < float("inf") else 30.0
        bucket_duration_bins = np.linspace(begin, end, num_buckets + 1)[1:-1].tolist()

    logger.info(
        f"Creating eval DynamicBucketingSampler: "
        f"batch_duration={batch_duration}, batch_size={max_cuts}, "
        f"num_buckets={num_buckets}, rank={global_rank}/{world_size}"
    )

    # For distributed evaluation, each rank (GPU) processes 1/world_size of data.
    # This applies to both iterable and non-iterable datasets.
    # Within each rank, DataLoader workers use round-robin via FiniteIterableDatasetWrapper.
    sampler = DynamicBucketingSampler(
        cuts,
        constraint=constraint,
        shuffle=False,  # Deterministic for evaluation
        drop_last=False,  # Keep all samples for eval
        seed=seed,
        num_buckets=num_buckets,
        duration_bins=bucket_duration_bins,
        rank=global_rank,
        world_size=world_size,
    )

    return sampler, cuts, use_iterable


def get_finite_dataloader_from_config(
    config: DatasetConfig | dict,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    """Create a FINITE DataLoader for evaluation.

    Unlike get_lhotse_dataloader_from_config (for training), this creates a dataloader that:
    1. Iterates exactly once through the data (one epoch)
    2. Has __len__ for progress bar support
    3. Uses DynamicBucketingSampler for efficient batching
    4. Supports num_workers > 0 with proper sharding

    This function can be imported and used by external evaluation scripts.

    Args:
        config: DatasetConfig with data loading parameters.
        global_rank: Global rank for distributed evaluation.
        world_size: Total number of processes.
        dataset: PyTorch Dataset that processes CutSets (e.g., SpeechToTextDataset).

    Returns:
        DataLoader that iterates once and has known length.
    """
    logger.info("Creating finite Lhotse DataLoader for evaluation")

    # Set up CUDA expandable segments for better memory management
    _maybe_set_cuda_expandable_segments(enabled=True)

    # Resolve seed and fix for reproducibility
    seed = resolve_seed(_get_config_value(config, "seed", 0))
    fix_random_seed(seed)

    # Get eval-specific sampler
    sampler, cuts, use_iterable = get_eval_sampler_from_config(
        config=config,
        global_rank=global_rank,
        world_size=world_size,
    )

    # Estimate number of batches for progress bar
    num_batches = estimate_num_batches(config, world_size=world_size)
    logger.info(f"Estimated {num_batches} batches for evaluation")

    # Get dataloader settings from config
    num_workers = _get_config_value(config, "num_workers", 0)
    pin_memory = _get_config_value(config, "pin_memory", True)
    prefetch_factor_val = _get_config_value(config, "prefetch_factor", 2)
    prefetch_factor = int(prefetch_factor_val) if prefetch_factor_val is not None else 2

    if use_iterable:
        # For shar/tarred data, use FiniteIterableDatasetWrapper
        # Worker sharding is handled via round-robin in FiniteIterableDatasetWrapper.__iter__
        logger.info(
            f"Using FiniteIterableDatasetWrapper for eval (num_workers={num_workers}, num_batches={num_batches})"
        )

        wrapped_dataset = FiniteIterableDatasetWrapper(
            dataset=dataset,
            sampler=sampler,
            num_batches=num_batches,
        )

        # Note: We don't use make_worker_init_fn here because:
        # 1. It's designed for infinite training iteration
        # 2. FiniteIterableDatasetWrapper handles worker sharding via round-robin
        dataloader = torch.utils.data.DataLoader(
            dataset=wrapped_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        # For non-tarred data, sampler handles sharding directly
        logger.info(f"Using map-style dataset for eval (num_workers={num_workers})")

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    return dataloader


def get_eval_dataloader_from_config(
    data_config: DataConfig | dict,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
) -> torch.utils.data.DataLoader:
    """Convenience function to create validation dataloader.

    This uses get_finite_dataloader_from_config to create a dataloader
    that iterates once (not infinitely) and has __len__ for progress bars.

    Args:
        data_config: Full DataConfig with validation_ds settings.
        dataset: Dataset to use (typically SpeechToTextDataset or FallbackDataset).
        global_rank: Global rank for distributed evaluation.
        world_size: Total number of processes.

    Returns:
        Validation DataLoader (finite, with progress bar support).
    """
    validation_ds = _get_config_value(data_config, "validation_ds")
    return get_finite_dataloader_from_config(
        config=validation_ds,
        global_rank=global_rank,
        world_size=world_size,
        dataset=dataset,
    )


def _maybe_set_cuda_expandable_segments(enabled: bool = True) -> None:
    """Configure PyTorch CUDA allocator for better memory management.

    This helps reduce memory fragmentation when batch sizes vary.
    """
    if not enabled or not torch.cuda.is_available():
        return

    try:
        current_settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" not in current_settings:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            logger.debug("Enabled CUDA expandable segments")
    except RuntimeError:
        logger.debug("Could not enable CUDA expandable segments")


__all__ = [
    "read_cutset_from_config",
    "get_lhotse_sampler_from_config",
    "get_eval_sampler_from_config",
    "get_lhotse_dataloader_from_config",
    "get_train_dataloader_from_config",
    "get_eval_dataloader_from_config",
    "get_finite_dataloader_from_config",
    "compute_dataset_duration",
    "estimate_steps_per_epoch",
    "estimate_num_batches",
    "InfiniteIterableDatasetWrapper",
    "FiniteIterableDatasetWrapper",
]
