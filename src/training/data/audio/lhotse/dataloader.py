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
import os
import warnings
from dataclasses import asdict
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

from ....config import DataConfig, DatasetConfig, DataSourceConfig
from .....logging_utils import get_logger

logger = get_logger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get a value from config, supporting both dataclass and dict access."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


def _config_to_dict(config: Any) -> dict:
    """Convert config to dict, handling dataclasses and dicts."""
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    elif isinstance(config, dict):
        return config
    return dict(config)


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
            source_duration, source_cuts = _read_shar_manifest_durations(
                shar_path, min_duration, max_duration
            )
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
) -> tuple[int, float, int]:
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
        return -1, 0.0, 0

    total_duration, num_cuts = compute_dataset_duration(config)
    if total_duration <= 0:
        return 0, 0.0, 0

    total_hours = total_duration / 3600.0

    # Each rank sees total_duration / world_size of data.
    # Each rank creates (total_duration / world_size) / batch_duration micro-batches.
    # One optimizer step = gradient_accumulation_steps micro-batches.
    # steps_per_epoch = total_duration / (batch_duration * world_size * gradient_accumulation_steps)
    steps_per_epoch = int(total_duration / (batch_duration * world_size * gradient_accumulation_steps))

    logger.info(
        f"Dataset stats: {num_cuts} cuts, {total_hours:.2f} hours total, "
        f"~{steps_per_epoch} steps/epoch (batch_duration={batch_duration}s, "
        f"grad_accum={gradient_accumulation_steps}, world_size={world_size})"
    )

    return steps_per_epoch, total_hours, num_cuts


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
    shuffle = _get_config_value(config, "shuffle", True)

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

            logger.info(f"Loading CutSet from shar: {shar_path}")
            cuts = CutSet.from_shar(in_dir=shar_path, shuffle_shards=bool(shuffle))
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
        if all(w == weights[0] for w in weights):
            combined = CutSet.mux(*cutsets)
        else:
            combined = CutSet.mux(*cutsets, weights=weights)

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
    shard_seed = _get_config_value(config, "shard_seed", "trng")
    if isinstance(shard_seed, str) and shard_seed != "trng":
        shard_seed = resolve_seed(shard_seed)

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
        logger.info(
            f"Creating DynamicCutSampler with "
            f"batch_duration={batch_duration}, "
            f"batch_size={max_cuts}"
        )

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
        logger.info("Using IterableDatasetWrapper for shar data")
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(
                rank=global_rank,
                world_size=world_size,
                seed=seed,
            ),
            persistent_workers=num_workers > 0,
        )
    else:
        # For non-tarred data, sampler stays in main process
        logger.info("Using map-style dataset")
        dloader_kwargs = dict(
            dataset=dataset,
            sampler=sampler,
        )

    dataloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,  # Batching handled by sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else 0,
    )

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


def get_eval_dataloader_from_config(
    data_config: DataConfig | dict,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
    split_batches: bool = False,
) -> torch.utils.data.DataLoader:
    """Convenience function to create validation dataloader.

    Args:
        data_config: Full DataConfig with validation_ds settings.
        dataset: Dataset to use.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Validation DataLoader.
    """
    validation_ds = _get_config_value(data_config, "validation_ds")
    return get_lhotse_dataloader_from_config(
        config=validation_ds,
        global_rank=global_rank,
        world_size=world_size,
        dataset=dataset,
        split_batches=split_batches,
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
    "get_lhotse_dataloader_from_config",
    "get_train_dataloader_from_config",
    "get_eval_dataloader_from_config",
    "compute_dataset_duration",
    "estimate_steps_per_epoch",
]
