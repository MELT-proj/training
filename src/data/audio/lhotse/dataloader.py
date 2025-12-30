"""
Lhotse DataLoader utilities for MELT training.

This module provides functions to create Lhotse samplers and dataloaders
from configuration objects, following patterns from NeMo's dataloader utilities.

Key functions:
- get_lhotse_sampler_from_config: Creates a CutSampler from config
- get_lhotse_dataloader_from_config: Creates a full DataLoader from config
"""

import logging
import os
import warnings
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
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _as_dictconfig(cfg: Any) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, dict):
        return OmegaConf.create(cfg)
    return OmegaConf.create(cfg)


def read_cutset_from_config(config: DictConfig) -> tuple[CutSet, bool]:
    """Read CutSet(s) from configuration.

    Args:
        config: DatasetConfig with input_cfg specifying data sources.

    Returns:
        Tuple of (CutSet, use_iterable_dataset).
        use_iterable_dataset is True for tarred/shar data.
    """
    if not config.get("input_cfg"):
        raise ValueError("No data sources specified in input_cfg")

    cutsets = []
    use_iterable = False

    for source_cfg in config.get("input_cfg"):
        source_cfg = _as_dictconfig(source_cfg)
        source_type = source_cfg.get("type")

        if source_type == "lhotse_shar":
            if source_cfg.get("shar_path") is None:
                raise ValueError("shar_path must be specified for lhotse_shar type")

            # Expand environment variables in path
            shar_path = os.path.expandvars(str(source_cfg.get("shar_path")))

            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")

            logger.info(f"Loading CutSet from shar: {shar_path}")
            cuts = CutSet.from_shar(in_dir=shar_path, shuffle_shards=bool(config.get("shuffle", True)))
            use_iterable = True  # Shar always uses iterable dataset

        elif source_type == "lhotse_cuts":
            if source_cfg.get("cuts_path") is None:
                raise ValueError("cuts_path must be specified for lhotse_cuts type")

            cuts_path = os.path.expandvars(str(source_cfg.get("cuts_path")))

            if not Path(cuts_path).exists():
                raise FileNotFoundError(f"Cuts path not found: {cuts_path}")

            logger.info(f"Loading CutSet from cuts: {cuts_path}")
            cuts = CutSet.from_file(cuts_path)

        else:
            raise ValueError(f"Unknown data source type: {source_type}")

        # Add tags to cuts if specified
        tags = source_cfg.get("tags")
        if tags:
            cuts = cuts.map(partial(_add_tags_to_cut, tags=dict(tags)), apply_fn=None)

        cutsets.append(cuts)

    # Combine multiple cutsets
    if len(cutsets) == 1:
        combined = cutsets[0]
    else:
        # Use mux for weighted combination if weights differ
        weights = []
        for cfg in config.get("input_cfg"):
            if isinstance(cfg, (DictConfig, dict)):
                weights.append(float(cfg.get("weight", 1.0)))
            else:
                weights.append(1.0)
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
    config: DictConfig,
    global_rank: int = 0,
    world_size: int = 1,
) -> tuple[CutSampler, bool]:
    """Create a CutSampler from configuration.

    Args:
        config: DatasetConfig with sampling parameters.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (CutSampler, use_iterable_dataset).
    """
    config = _as_dictconfig(config)

    # Load cutset from config
    cuts, use_iterable = read_cutset_from_config(config)

    # Apply duration filtering
    min_duration = config.get("min_duration", None)
    max_duration = config.get("max_duration", None)

    if min_duration is not None or max_duration is not None:
        min_dur = min_duration if min_duration is not None else 0.0
        max_dur = max_duration if max_duration is not None else float("inf")
        cuts = cuts.filter(lambda c: min_dur <= c.duration <= max_dur)
        logger.info(f"Applied duration filter: [{min_dur}, {max_dur}]")

    # Determine sampling constraint
    constraint = TimeConstraint(
        max_cuts=config.get("batch_size", None),
        max_duration=config.get("batch_duration", None),
        quadratic_duration=config.get("quadratic_duration", None),
    )

    # Create sampler
    shuffle = config.get("shuffle", True)
    drop_last = config.get("drop_last", False)
    shuffle_buffer_size = config.get("shuffle_buffer_size", 10000)
    seed = resolve_seed(config.get("seed", 0))
    shard_seed = config.get("shard_seed", "trng")
    if isinstance(shard_seed, str) and shard_seed != "trng":
        shard_seed = resolve_seed(shard_seed)

    if config.get("use_bucketing", False):
        # Dynamic bucketing sampler for efficient batching
        num_buckets = config.get("num_buckets", 30)
        bucket_buffer_size = config.get("bucket_buffer_size", 10000)
        bucket_duration_bins = config.get("bucket_duration_bins", None)

        # Auto-estimate duration bins if not provided
        if bucket_duration_bins is None and max_duration is not None:
            begin = min_duration if min_duration is not None and min_duration > 0 else 0.0
            end = max_duration if max_duration < float("inf") else 30.0
            bucket_duration_bins = np.linspace(begin, end, num_buckets + 1)[1:-1].tolist()

        logger.info(
            f"Creating DynamicBucketingSampler with "
            f"batch_duration={config.get('batch_duration')}, "
            f"batch_size={config.get('batch_size')}, "
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
            f"batch_duration={config.get('batch_duration')}, "
            f"batch_size={config.get('batch_size')}"
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
    config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
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
    config = _as_dictconfig(config)

    logger.info("Creating Lhotse DataLoader")

    # Set up CUDA expandable segments for better memory management
    _maybe_set_cuda_expandable_segments(enabled=True)

    # Resolve seed
    seed = resolve_seed(config.get("seed", 0))
    fix_random_seed(seed)

    # Get sampler
    sampler, use_iterable = get_lhotse_sampler_from_config(
        config=config,
        global_rank=global_rank,
        world_size=world_size,
    )

    # Create dataloader
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)

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
    )

    return dataloader


def get_train_dataloader_from_config(
    data_config: DictConfig,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
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
    return get_lhotse_dataloader_from_config(
        config=_as_dictconfig(data_config.get("train_ds")),
        global_rank=global_rank,
        world_size=world_size,
        dataset=dataset,
    )


def get_eval_dataloader_from_config(
    data_config: DictConfig,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
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
    return get_lhotse_dataloader_from_config(
        config=_as_dictconfig(data_config.get("validation_ds")),
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
    "get_lhotse_dataloader_from_config",
    "get_train_dataloader_from_config",
    "get_eval_dataloader_from_config",
]
