"""
Lhotse DataLoader utilities for MELT training.

This module provides functions to create Lhotse samplers and dataloaders
from configuration objects, following patterns from NeMo's dataloader utilities.

Key functions:
- get_lhotse_sampler_from_config: Creates a CutSampler from config
- get_lhotse_dataloader_from_config: Creates a full DataLoader from config
"""

import os
import warnings
from dataclasses import asdict
from functools import partial
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

from src.config import DataConfig, DatasetConfig, DataSourceConfig
from src.logging_utils import get_logger

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
]
