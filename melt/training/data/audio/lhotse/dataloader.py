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
import warnings
from copy import deepcopy
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
    BucketingSampler,
    DynamicCutSampler,
    SimpleCutSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig
from functools import partial


from .....logging_utils import get_logger


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

    IMPORTANT: Due to dynamic batching (e.g., DynamicBucketingSampler, DynamicCutSampler),
    the actual number of batches may differ from __len__. This is expected and normal:
    - Batches are formed by duration or count constraints, not fixed sizes
    - Shuffling and data splitting can cause the final batch in each bucket/shard to vary
    - The actual batch count typically varies by ±10-20% from the estimate

    This estimate is used only for progress bar display, not for data correctness.
    All data is processed exactly once per epoch regardless of batch count variations.

    Args:
        dataset: PyTorch Dataset that processes CutSets.
        sampler: Lhotse CutSampler that yields batches of cuts.
        estimated_batches_per_epoch: Expected number of batches per rank per epoch (approximate).
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
    config: DictConfig,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> tuple[float, int]:
    """Compute total dataset duration from configuration.

    Reads SHAR manifest files to compute total duration without loading audio.
    Applies duration filtering if min/max_duration are specified.

    Args:
        config: DictConfig with input_cfg specifying data sources.
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
    config: DictConfig,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> tuple[int, float, int, int, int]:
    """Estimate the number of training steps per epoch.

    Computes steps based on how data is sharded across ranks:
        steps_per_epoch = total_duration / (batch_duration * world_size * gradient_accumulation_steps)

    Each rank processes 1/world_size of the data, and each optimizer step
    requires gradient_accumulation_steps micro-batches.

    Args:
        config: DictConfig with batch_duration and data sources.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        world_size: Number of distributed processes.

    Returns:
        Tuple of (steps_per_epoch, total_duration_hours, num_cuts).
    """
    num_workers = _get_config_value(config, "num_workers", 1)
    total_hours = _get_config_value(config, "total_hours", None)
    total_cuts = _get_config_value(config, "total_cuts", None)
    force_estimate = _get_config_value(config, "force_estimate", None)
    batch_size = _get_config_value(config, "batch_size")
    batch_duration = _get_config_value(config, "batch_duration")

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

    # We need to decide whether to estimate steps based on total duration / batch_duration or
    # simply by dividing total cuts by batch_size.
    if batch_size is not None and batch_size > 0:
        batches_per_epoch = math.ceil(total_cuts / batch_size)
    elif batch_duration is not None and batch_duration > 0:
        batches_per_epoch = math.ceil(total_duration / batch_duration)
    else:
        logger.warning("Neither batch_size nor batch_duration is set; cannot estimate steps per epoch")
        return -1, 0.0, 0, 0, 0

    # We use data parallelism by setting split_for_loading=True in CutSet.from_shar()
    # The shard are hence divided to world_size * num_workers processes.
    batches_per_worker = batches_per_epoch / (world_size * num_workers) if num_workers > 0 else batches_per_epoch / world_size

    # The number of update steps is rescaled by gradient accumulation steps
    optimizer_steps_per_epoch = math.ceil(batches_per_worker / gradient_accumulation_steps)

    return optimizer_steps_per_epoch, total_hours, total_cuts, batches_per_epoch, batches_per_worker


def read_cutset_from_config(config: DictConfig, repeat: bool = True) -> tuple[CutSet, bool]:
    """Read CutSet(s) from configuration.

    Args:
        config: DictConfig with input_cfg specifying data sources.

    Returns:
        Tuple of (CutSet, use_iterable_dataset).
        use_iterable_dataset is True for tarred/shar data.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        raise ValueError("No data sources specified in input_cfg")

    cutsets = []
    use_iterable = False

    seed = config.seed
    shuffle = config.shuffle

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

            logger.info(f"Loading CutSet from shar: {shar_path} (seed: {seed})")
            
            # Some notes:
            # Training (repeat=True): split shards across workers for efficiency;
            # uneven shard counts don't matter because the CutSet repeats infinitely.
            # Eval (repeat=False): every rank reads all shards; the sampler handles
            # rank-based sharding to guarantee even batch counts and avoid deadlocks.
            cuts = CutSet.from_shar(
                in_dir=shar_path,
                shuffle_shards=shuffle,
                seed=seed,  
                stateful_shuffle=shuffle,
                split_for_dataloading=repeat,
            )
            use_iterable = True  # Shar always uses iterable dataset
        # elif source_type == "lhotse_cuts":
        #     cuts_path = _get_config_value(source_cfg, "cuts_path")
        #     if cuts_path is None:
        #         raise ValueError("cuts_path must be specified for lhotse_cuts type")

        #     cuts_path = os.path.expandvars(str(cuts_path))

        #     if not Path(cuts_path).exists():
        #         raise FileNotFoundError(f"Cuts path not found: {cuts_path}")

        #     logger.info(f"Loading CutSet from cuts: {cuts_path}")
        #     cuts = CutSet.from_file(cuts_path)
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

        logger.info(f"Mux-ing data sources. Weights: {weights}")

        combined = CutSet.mux(*cutsets, weights=weights, seed=config.shard_seed)

    # Since we split cuts across data workers (split_for_dataloading=True),
    # to avoid deadlocks we force the combined CutSet to repeat infinitely.
    if repeat:
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
    config: DictConfig,
    global_rank: int = 0,
    world_size: int = 1,
    repeat: bool = False,
) -> tuple[CutSampler, bool]:
    """Create a CutSampler from configuration.

    Args:
        config: DictConfig with sampling parameters.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.
        repeat: Whether to repeat the CutSet infinitely.

    Returns:
        Tuple of (CutSampler, use_iterable_dataset).
    """
    # Load cutset from config. Since we are using Shar data, this is a lazy CutSet.
    # For now, it should always be use_iterable = True.
    cuts, use_iterable = read_cutset_from_config(config, repeat=repeat)

    # Apply duration filtering
    min_duration = _get_config_value(config, "min_duration")
    max_duration = _get_config_value(config, "max_duration")

    if min_duration is not None or max_duration is not None:
        min_dur = min_duration if min_duration is not None else 0.0
        max_dur = max_duration if max_duration is not None else float("inf")
        cuts = cuts.filter(lambda c: min_dur <= c.duration <= max_dur)
        logger.info(f"Applied duration filter: [{min_dur}, {max_dur}]")

    # Apply max_samples subsampling if requested.
    # A shuffle + subset gives a random subsample that works lazily for both
    # shar (iterable) and cuts (map-style) CutSets without loading everything
    # into memory. The buffer_size controls how thoroughly the data is shuffled
    # before truncation; 4x max_samples is a reasonable trade-off between
    # randomness and memory usage.
    max_samples = _get_config_value(config, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        shuffle_buffer = max(max_samples * 4, 10_000)
        cuts = cuts.shuffle(buffer_size=shuffle_buffer).subset(max_cuts=max_samples)
        logger.info(
            f"Applied max_samples={max_samples} random subsampling "
            f"(shuffle buffer={shuffle_buffer})"
        )

    # Determine sampling constraint.
    # When training with an IterableDataset under Accelerate + `split_batches=True`,
    # the main process fetches a *global* batch and slices it across `world_size`.
    # To preserve the effective per-rank batch constraint, scale it by `world_size`.
    max_cuts = _get_config_value(config, "batch_size")
    batch_duration = _get_config_value(config, "batch_duration")
    quadratic_duration = _get_config_value(config, "quadratic_duration")

    # if use_iterable and split_batches and world_size > 1:
    #     if max_cuts is not None:
    #         max_cuts = int(max_cuts) * int(world_size)
    #     if batch_duration is not None:
    #         batch_duration = float(batch_duration) * float(world_size)
    #     if quadratic_duration is not None:
    #         quadratic_duration = float(quadratic_duration) * float(world_size)

    #     logger.info(
    #         "IterableDataset + split_batches=True: scaling sampler constraints by world_size=%s ",
    #         world_size,
    #     )

    # Create sampler
    shuffle = _get_config_value(config, "shuffle", True)
    drop_last = _get_config_value(config, "drop_last", False)
    buffer_size = _get_config_value(config, "buffer_size", 10000)
    shard_seed = _get_config_value(config, "shard_seed")

    # Important: do NOT resolve shard_seed='randomized' in the main process.
    # Lhotse's resolve_seed('randomized') is designed to run in DataLoader workers after
    # make_worker_init_fn has set LHOTSE_PROCESS_SEED.
    if isinstance(shard_seed, str) and shard_seed not in ("trng", "randomized"):
        raise ValueError(f"Unsupported shard_seed={shard_seed!r}. Supported values: int, 'trng', 'randomized'.")

    lhotse_sampler_type = _get_config_value(config, "lhotse_sampler_type", False) 
    if lhotse_sampler_type == "dynamic_bucketing":
        num_buckets = _get_config_value(config, "num_buckets", None)
        if num_buckets is None:
            raise ValueError(f"Using a `{lhotse_sampler_type}` sampler requires setting `num_buckets`.")

        bucket_duration_bins = _get_config_value(config, "bucket_duration_bins", None)

        # Auto-estimate duration bins if not provided
        # if bucket_duration_bins is None and batch_duration is not None:
        #     begin = min_duration if min_duration is not None and min_duration > 0 else 0.0
        #     end = max_duration if max_duration is not None and max_duration < float("inf") else 30.0
        #     bucket_duration_bins = np.linspace(begin, end, num_buckets + 1)[1:-1].tolist()

        logger.info(
            f"Creating DynamicBucketingSampler with "
            f"batch_duration={batch_duration}, "
            f"batch_size={max_cuts}, "
            f"num_buckets={num_buckets}"
        )

        # Training (repeat=True) with iterable data: sharding is at the CutSet level
        # via split_for_dataloading=True, so the sampler sees rank=0/world_size=1.
        # Eval (repeat=False): split_for_dataloading=False, so the sampler must handle
        # rank-based sharding to ensure each GPU processes the same number of batches.
        sampler_rank = 0 if (use_iterable and repeat) else global_rank
        sampler_world_size = 1 if (use_iterable and repeat) else world_size

        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=batch_duration,
            max_cuts=max_cuts,
            quadratic_duration=quadratic_duration,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=config.seed,
            num_buckets=num_buckets,
            duration_bins=bucket_duration_bins,
            buffer_size=buffer_size,
            rank=sampler_rank,
            world_size=sampler_world_size,
        )
    elif lhotse_sampler_type == "dynamic":
        # Simple dynamic sampler (no bucketing)
        logger.info(
            f"Creating DynamicCutSampler with batch_duration={batch_duration}, batch_size={max_cuts}, shuffle={shuffle}, drop_last={drop_last}, repeat={repeat}"
        )

        # Same logic as above for rank/world_size
        sampler_rank = 0 if (use_iterable and repeat) else global_rank
        sampler_world_size = 1 if (use_iterable and repeat) else world_size

        sampler = DynamicCutSampler(
            cuts,
            max_duration=batch_duration,
            max_cuts=max_cuts,
            quadratic_duration=quadratic_duration,
            shuffle=shuffle,
            drop_last=drop_last,
            buffer_size=buffer_size,
            rank=sampler_rank,
            world_size=sampler_world_size,
            seed=shard_seed,
        )
    elif lhotse_sampler_type == "bucketing":
        num_buckets = _get_config_value(config, "num_buckets", None)
        if num_buckets is None:
            raise ValueError(f"Using a `{lhotse_sampler_type}` sampler requires setting `num_buckets`.")

        sampler_rank = 0 if (use_iterable and repeat) else global_rank
        sampler_world_size = 1 if (use_iterable and repeat) else world_size
        sampler = BucketingSampler(
            cuts,
            sampler_type=SimpleCutSampler,
            num_buckets=num_buckets,
            drop_last=drop_last,
            seed=shard_seed,
            # kwargs below here are sent to the sampler_type object
            max_duration=batch_duration,
            max_cuts=max_cuts,
            shuffle=shuffle,
            world_size=sampler_world_size,
            rank=sampler_rank,
        )
    else:
        raise ValueError(f"Lhotse sampler type `{lhotse_sampler_type}` unknown.")

    return sampler, use_iterable


def get_lhotse_dataloader_from_config(
    config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    repeat: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader from configuration.

    Args:
        config: DictConfig with data loading parameters.
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
    config.seed = resolve_seed(config.seed)
    logger.info(f"Creating a lhotse dataloader with seed {config.seed}")
    fix_random_seed(config.seed)

    # Get sampler
    sampler, use_iterable = get_lhotse_sampler_from_config(
        config=config,
        global_rank=global_rank,
        world_size=world_size,
        repeat=repeat,
    )

    # Create dataloader
    num_workers = _get_config_value(config, "num_workers", 0)
    pin_memory = _get_config_value(config, "pin_memory", True)

    # For eval (repeat=False), cap num_workers to 1.
    # With split_for_dataloading=False (needed for even rank distribution), multiple
    # workers each read all shards and run identical samplers, causing duplicated data.
    # Eval is model-bound anyway, so multi-worker loading has minimal benefit.
    if not repeat and use_iterable and num_workers > 1:
        logger.warning(
            f"Eval mode: overriding num_workers from {num_workers} to 1. "
            "With sampler-level rank sharding (needed for even batch counts across GPUs), "
            "multiple workers would duplicate data. Eval throughput is model-bound, "
            "so num_workers=1 has negligible impact on speed."
        )
        num_workers = 1

    # Extract optional prefetch_factor for worker-level prefetching
    prefetch_factor_val = _get_config_value(config, "prefetch_factor", 2)
    prefetch_factor = int(prefetch_factor_val) if prefetch_factor_val is not None else None

    if use_iterable:
        # For tarred/shar data, wrap dataset with sampler
        # This moves sampling to worker processes
        # Note: The finite/infinite behavior is controlled by whether the CutSet was
        # .repeat()'ed in read_cutset_from_config(), not by the wrapper itself.
        logger.info(f"Using InfiniteIterableDatasetWrapper for shar data ({'infinite' if repeat else 'finite'} mode)")
        logger.info(f"Using world size: {world_size}, rank: {global_rank}")

        # Estimate batches per epoch for progress bars
        # We need micro-batches per worker, per rank, per epoch (not optimizer steps)
        # So we call estimate_steps_per_epoch with gradient_accumulation_steps=1
        _, total_duration_hours, total_cuts, _, batches_per_worker = estimate_steps_per_epoch(
            config=config,
            gradient_accumulation_steps=1,  # We want micro-batches, not optimizer steps
            world_size=world_size,
        )
        # Convert to int for __len__
        batches_per_worker_int = max(1, int(batches_per_worker))
        logger.info(
            "This dataloader will yield (approx):\n"
            + f" {total_cuts} total cuts (after filtering)\n"
            + f" with a total duration of {total_duration_hours:.2f} hours and\n"
            + f" with an estimated {batches_per_worker_int} micro-batches per epoch per rank"
        )
        if repeat:
            logger.info(
                "NOTE: With dynamic batching (by duration or count), the actual batch count may vary ±10-20% "
                "from this estimate. This is expected and normal. All data is processed exactly once per epoch. "
                "Minor warnings about batch count mismatches can be safely ignored."
            )

        # Suppress Lhotse's warning about using rank/world_size with IterableDatasetWrapper.
        # For eval (repeat=False), we intentionally use sampler-level sharding instead of
        # shard-level splitting to guarantee even batch counts across GPUs and avoid deadlocks.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*CutSampler with rank.*inside an IterableDatasetWrapper.*",
                category=UserWarning,
            )
            wrapped_dataset = InfiniteIterableDatasetWrapper(
                dataset=dataset,
                sampler=sampler,
                estimated_batches_per_epoch=batches_per_worker_int,
            )

        dloader_kwargs = {
            "dataset": wrapped_dataset,
            "worker_init_fn": make_worker_init_fn(
                rank=global_rank,
                world_size=world_size,
                seed=config.seed,
                set_different_node_and_worker_seeds=True,
            ),
            "persistent_workers": num_workers > 0 and repeat,  # Only persistent for training
        }
    else:
        # For non-tarred data, sampler stays in main process
        logger.info("Using map-style dataset")
        dloader_kwargs = {
            "dataset": dataset,
            "sampler": sampler,
        }

    # Suppress PyTorch's warning about iterable dataset length mismatch.
    # This warning is expected for dynamic batching: actual batch count varies due to
    # duration-based/count-based batching and shuffling, not due to misconfiguration.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Length of IterableDataset .* was reported to be .* but .* samples have been fetched",
            category=UserWarning,
        )
        dataloader = torch.utils.data.DataLoader(
            **dloader_kwargs,
            batch_size=None,  # Batching handled by sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    _maybe_attach_set_epoch(dataloader=dataloader, sampler=sampler)
    return dataloader


def get_train_dataloader_from_config(
    data_config: DictConfig,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
) -> torch.utils.data.DataLoader:
    """Convenience function to create training dataloader.

    Args:
        data_config: Full DictConfig with train_ds settings.
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
        repeat=True,
    )


def get_eval_dataloader_from_config(
    data_config: DictConfig,
    dataset: torch.utils.data.Dataset,
    global_rank: int = 0,
    world_size: int = 1,
) -> torch.utils.data.DataLoader:
    """Convenience function to create validation dataloader.

    This uses get_finite_dataloader_from_config to create a dataloader
    that iterates once (not infinitely) and has __len__ for progress bars.

    Args:
        data_config: Full DictConfig with validation_ds settings.
        dataset: Dataset to use (typically SpeechToTextDataset or FallbackDataset).
        global_rank: Global rank for distributed evaluation.
        world_size: Total number of processes.

    Returns:
        Validation DataLoader (finite, with progress bar support).
    """
    validation_ds = _get_config_value(data_config, "validation_ds")
    return get_lhotse_dataloader_from_config(
        config=validation_ds, global_rank=global_rank, world_size=world_size, dataset=dataset, repeat=False
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
    # "get_eval_sampler_from_config",
    "get_lhotse_dataloader_from_config",
    "get_train_dataloader_from_config",
    "get_eval_dataloader_from_config",
    # "get_finite_dataloader_from_config",
    "compute_dataset_duration",
    "estimate_steps_per_epoch",
    # "estimate_num_batches",
    "InfiniteIterableDatasetWrapper",
]
