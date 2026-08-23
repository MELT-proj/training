"""
Lhotse DataLoader utilities for MELT training.

This module provides functions to create Lhotse samplers and dataloaders
from configuration objects, following patterns from NeMo's dataloader utilities.

Key functions:
- get_lhotse_sampler_from_config: Creates a CutSampler from config
- get_lhotse_dataloader_from_config: Creates a full DataLoader from config
- compute_dataset_duration: Computes total dataset duration for epoch estimation
"""

import copy
import gzip
import json
import math
import os
import random
import warnings
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
from omegaconf import DictConfig, OmegaConf
from functools import partial
from torchdata.stateful_dataloader import StatefulDataLoader


from .....logging_utils import get_logger


logger = get_logger(__name__)


def _harden_rng_setstate() -> None:
    """Let ``random.setstate`` accept an RNG state whose tuples were flattened.

    ``random.getstate()`` returns a tuple of tuples and ``setstate`` rejects
    anything else with "state vector must be a tuple". Lhotse checkpoints several
    RNGs this way, and the state dict that comes back through the dataloader's
    worker transport in a multi-rank run has those tuples flattened to lists, so
    every worker died on the first batch after a resume.

    Patching ``setstate`` rather than each caller is deliberate. Lhotse restores
    RNG state at seven sites; two of them already route through its own
    ``_rng_state_from_json`` helper -- which does exactly this coercion, so
    upstream knows the state can arrive as lists -- and five do not. Fixing them
    one at a time is whack-a-mole: hardening the bucket sampler's restore simply
    moved the failure to the multiplexer's. This is the one point they all share.

    Normalising in the training process instead does not work: the restore runs
    inside a worker subprocess and the transport re-flattens the state after any
    earlier fix. Hence the patch is also re-applied in ``worker_init_fn``, for a
    spawned worker that re-imports rather than inheriting the patched class.

    The behaviour change is confined to input that would otherwise raise: a
    well-formed state is passed through untouched. Report upstream.
    """
    original = random.Random.setstate
    if getattr(original, "_melt_coerces_tuples", False):
        return

    def setstate(self, state):
        # random.getstate() -> (version, internalstate, gauss_next)
        if isinstance(state, (list, tuple)) and len(state) == 3:
            version, internalstate, gauss_next = state
            if isinstance(internalstate, list):
                state = (version, tuple(internalstate), gauss_next)
        return original(self, state)

    setstate._melt_coerces_tuples = True
    random.Random.setstate = setstate


_harden_rng_setstate()


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


def shar_manifest_files(shar_path: str | Path) -> list[Path]:
    """Cut manifests in a Shar directory, one per shard, either layout.

    A streaming collection stores ``cuts.000000.jsonl.gz``; an indexed one
    stores plain ``cuts.000000.jsonl`` beside a ``.idx`` of byte offsets, which
    is why it cannot stay compressed. Globbing only the gzipped form silently
    reports a fully indexed source as empty.

    A shard present in both forms is counted once, preferring the plain file:
    the two are the same records, and double-counting would inflate every
    duration and cut total derived from them.
    """
    shar_path = Path(shar_path)
    by_shard: dict[str, Path] = {}
    # Plain first so it wins the setdefault against its own .gz.
    for pattern in ("cuts.*.jsonl", "cuts.*.jsonl.gz"):
        for path in sorted(shar_path.glob(pattern)):
            stem = path.name[: path.name.index(".jsonl")]
            by_shard.setdefault(stem, path)
    return [by_shard[k] for k in sorted(by_shard)]


def _read_shar_manifest_durations(
    shar_path: str | Path,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> tuple[float, int]:
    """Read total duration and cut count from SHAR manifest files.

    Reads only the manifest files (not audio) to extract durations. Handles
    both Shar layouts: gzipped ``cuts.*.jsonl.gz`` and the plain
    ``cuts.*.jsonl`` that an indexed collection uses (the .idx sidecars hold
    byte offsets into the manifest, so it cannot be compressed).

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

    for manifest_file in shar_manifest_files(shar_path):
        opener = gzip.open if manifest_file.suffix == ".gz" else open
        try:
            with opener(manifest_file, "rt", encoding="utf-8") as f:
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

    if not num_cuts:
        logger.warning(f"No cuts read from manifests in {shar_path}")

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


def _effective_duration_inflation(config: DictConfig) -> float:
    """How many more batches an epoch holds than ``total_duration / batch_duration``.

    ``batch_duration`` is handed to lhotse as ``max_duration``, which budgets
    *effective* seconds, not audio seconds. With ``quadratic_duration`` set,
    lhotse charges a cut of length ``d``::

        cost(d) = d + d**2 / quadratic_duration        # sampling/base.py

    (the term that doubles the charge when ``d == quadratic_duration``, sized so
    that peak attention memory stays roughly flat across buckets). Summed over an
    epoch the sampler therefore spends ``sum(d) + sum(d^2)/q`` of budget to emit
    ``sum(d)`` of audio, so the batch count is inflated by::

        1 + (sum(d^2) / sum(d)) / q  =  1 + mean_weighted_duration / q

    where the mean is **duration-weighted** (``sum(d^2)/sum(d)``), not the plain
    mean -- long cuts carry proportionally more of the total hours.

    ``sum(d^2)`` is not in the config, but ``bucket_duration_bins`` is, and
    ``infra/bucket_bins.py`` builds those boundaries by *dividing total duration
    equally among the buckets*. Equal duration mass per bucket is exactly the
    weighting this mean needs, so averaging the bucket midpoints approximates
    ``sum(d^2)/sum(d)`` without touching the manifests.

    It is an approximation, and a known-low one: on ABL-MA-125-asr it returns
    1.63 where the run's own ``train_hours`` counter measured 1.87 (job
    44947472), because midpoints under-represent the mass in each bucket's upper
    half and because the shipped bins were estimated on a wider mixture than the
    125 h subset. Under-correcting is the safe direction -- it shortens an epoch
    rather than overrunning one -- but do not read the result as exact.

    Args:
        config: A ``train_ds``-style config. Read: ``quadratic_duration``,
            ``bucket_duration_bins``, ``min_duration``, ``max_duration``.

    Returns:
        Multiplier >= 1.0 for ``batches_per_epoch``. Exactly 1.0 when
        ``quadratic_duration`` is unset (no penalty, so no inflation).
    """
    quadratic_duration = _get_config_value(config, "quadratic_duration", None)
    if quadratic_duration is None or float(quadratic_duration) <= 0:
        return 1.0
    quadratic_duration = float(quadratic_duration)

    bins = _get_config_value(config, "bucket_duration_bins", None)
    if not bins:
        # Correction impossible, and silently returning 1.0 is what caused the
        # problem in the first place -- say so.
        logger.warning(
            "quadratic_duration=%.3g is set but bucket_duration_bins is not, so the "
            "steps-per-epoch estimate cannot be corrected for it. lhotse charges each cut "
            "d + d^2/%.3g, so batches hold LESS audio than batch_duration and this estimate "
            "is optimistic -- an epoch will cover less data than it claims, and a max_steps "
            "derived from it will rescale the LR schedule to match. Add bucket_duration_bins "
            "(infra/estimate_bucket_bins.py) to get a corrected estimate.",
            quadratic_duration,
            quadratic_duration,
        )
        return 1.0

    lo = float(_get_config_value(config, "min_duration", 0.0) or 0.0)
    hi = float(_get_config_value(config, "max_duration", 0.0) or 0.0)
    edges = [lo] + [float(b) for b in bins]
    # The open top bucket needs a right edge; max_duration is the only principled
    # one available, and the sampler will not emit anything longer anyway.
    if hi > edges[-1]:
        edges.append(hi)
    if len(edges) < 2:
        return 1.0

    midpoints = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
    mean_weighted_duration = sum(midpoints) / len(midpoints)
    inflation = 1.0 + mean_weighted_duration / quadratic_duration

    logger.info(
        "quadratic_duration=%.3g inflates the epoch's batch count by ~%.3fx "
        "(duration-weighted mean cut ~%.1f s over %d buckets): lhotse charges each cut "
        "d + d^2/%.3g, so a batch holds only ~%.0f%% of batch_duration in real audio. "
        "Steps per epoch are scaled up accordingly. This correction is approximate "
        "(measured ~13%% low on the ablation mix), so treat the step count as a close "
        "estimate rather than an exact epoch boundary.",
        quadratic_duration,
        inflation,
        mean_weighted_duration,
        len(midpoints),
        quadratic_duration,
        100.0 / inflation,
    )
    return inflation


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
        Tuple of (steps_per_epoch, total_duration_hours, num_cuts,
        batches_per_epoch, batches_per_rank).
    """
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
        # `batch_duration` is a budget of *effective* seconds, not audio seconds:
        # under `quadratic_duration` lhotse charges each cut `d + d^2/q`, so a
        # batch fills up well before it holds `batch_duration` seconds of audio.
        # Dividing total_duration by batch_duration therefore under-counts the
        # batches in an epoch -- by 87% on ABL-MA-125-asr, which is enough to
        # make a "1 epoch" run cover barely half the data and to rescale the LR
        # schedule with it. See _effective_duration_inflation for the factor.
        inflation = _effective_duration_inflation(config)
        batches_per_epoch = math.ceil(total_duration / batch_duration * inflation)
    else:
        logger.warning("Neither batch_size nor batch_duration is set; cannot estimate steps per epoch")
        return -1, 0.0, 0, 0, 0

    # `num_workers` deliberately does not appear in this divisor.
    #
    # lhotse's `make_worker_init_fn` gives worker `w` of rank `r` the partition
    # (rank = r * num_workers + w, world_size = world_size * num_workers), so a
    # single worker does emit only `batches_per_epoch / (world_size *
    # num_workers)` batches. But a rank's DataLoader interleaves all of its
    # workers round-robin into one stream, so the rank still consumes
    # `batches_per_epoch / world_size` batches per epoch however many workers
    # produce them -- and it is the rank's stream that `len(dataloader)` counts
    # and that the training loop steps through.
    #
    # Dividing by `num_workers` as well therefore under-reported epoch length by
    # exactly that factor, which propagated into `max_steps` (when derived from
    # `num_train_epochs`) and from there into the LR schedule. The extra factor
    # predates #52: it was a leftover from `split_for_dataloading=True`, which
    # was never actually passed. Now that indexed sources really are partitioned
    # the `world_size` term is correct on its own.
    batches_per_rank = batches_per_epoch / world_size

    # The number of update steps is rescaled by gradient accumulation steps
    optimizer_steps_per_epoch = math.ceil(batches_per_rank / gradient_accumulation_steps)

    return optimizer_steps_per_epoch, total_hours, total_cuts, batches_per_epoch, batches_per_rank


def read_cutset_from_config(config: DictConfig, repeat: bool = True) -> tuple[CutSet, bool]:
    """Read CutSet(s) from configuration.

    ``input_cfg`` is a list of sources.  An entry may also be ``type: group``
    with its own nested ``input_cfg``, in which case the group is muxed
    internally and then muxed against its siblings.  A corpus's sampling
    probability is therefore the product of the weights along its path, which
    is how a two-tier language/corpus mixture is expressed::

        input_cfg:
          - type: group          # a language
            weight: 0.21         # p_l
            input_cfg:
              - type: lhotse_shar
                shar_path: ...
                weight: 0.44     # p_c, so this corpus is drawn at p_l * p_c

    Within a level, either every entry sets ``weight`` or none does; see
    :func:`_resolve_weights`.

    Args:
        config: DictConfig with input_cfg specifying data sources.
        repeat: If True, the combined CutSet is repeated infinitely (for training).

    Returns:
        Tuple of (CutSet, use_iterable_dataset).
        use_iterable_dataset is True for tarred/shar data.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        raise ValueError("No data sources specified in input_cfg")

    seed = config.seed
    shard_seed = _get_config_value(config, "shard_seed", seed)
    shuffle = config.shuffle
    indexed = _get_config_value(config, "indexed", None)

    # How ranks and workers end up with different data depends on which reader
    # backs the sources, so resolve that first.
    #
    # Streaming (no .idx): nothing partitions the corpus, so every rank and
    # worker walks all of it. Separation is statistical, and comes from each
    # process traversing the shards in its own order -- which is why traversal
    # must be seeded from `shard_seed` ('randomized' resolves per rank+worker)
    # and not from `seed`, one integer shared by every process.
    #
    # Indexed (.idx present): lhotse partitions by sample index across the whole
    # (rank x worker) pool, so separation is exact and no longer needs a
    # per-process seed. There, `shard_seed` only sets shard *shuffle order*,
    # which should be the SAME everywhere -- and lhotse rejects
    # seed='randomized' outright once a multiplexer sits over partitioned
    # indexed sources, because each shard would draw a different permutation.
    using_indexed = _sources_are_indexed(input_cfg) if indexed is None else bool(indexed)
    if using_indexed and shard_seed == "randomized":
        logger.warning(
            "shard_seed='randomized' is incompatible with indexed Shar sources: "
            "partitioning already gives each rank and worker a disjoint slice, and "
            "lhotse refuses a randomized seed under a multiplexer over indexed "
            "sources. Falling back to shard_seed=%s (the config's `seed`). Set "
            "`shard_seed` to an integer explicitly to silence this.",
            seed,
        )
        shard_seed = seed

    combined, use_iterable, _ = _combine_entries(
        input_cfg, shuffle, shard_seed, repeat=repeat, indexed=indexed
    )

    return combined, use_iterable


def _iter_shar_paths(entries: list) -> list[str]:
    """Every `shar_path` under `input_cfg`, descending through `type: group`."""
    out: list[str] = []
    for entry in entries or []:
        if _get_config_value(entry, "type", None) == "group":
            out.extend(_iter_shar_paths(_get_config_value(entry, "input_cfg", [])))
        else:
            path = _get_config_value(entry, "shar_path", None)
            if path is not None:
                out.append(os.path.expandvars(str(path)))
    return out


def _sources_are_indexed(entries: list) -> bool:
    """Would lhotse read these sources through the indexed reader?

    Asks lhotse rather than looking for ``.idx`` ourselves, so this agrees with
    whatever ``from_shar(indexed=None)`` is about to decide. A collection part
    way through migration answers False, which is the safe reading: the mixed
    case behaves like streaming, and `shard_seed` should stay per-process.
    """
    paths = _iter_shar_paths(entries)
    if not paths:
        return False
    try:
        from lhotse.shar.readers.indexed import LazyIndexedSharIterator
    except ImportError:  # lhotse < 2.0 has no indexed reader at all
        return False
    return all(
        LazyIndexedSharIterator.supports_configuration(in_dir=p) for p in paths
    )


def _combine_entries(
    entries: list, shuffle: bool, shard_seed, repeat: bool, indexed: bool | None = None,
    _where: str = "input_cfg"
) -> tuple[CutSet, bool, int]:
    """Load one level of ``input_cfg`` and mux its entries together.

    An entry is either a leaf source or a ``type: group`` that carries its own
    nested ``input_cfg``.  Groups may nest arbitrarily deep; each level is muxed
    among its own children only, so the sampling probability of a corpus is the
    product of the weights along its path from the root.

    That product is what makes a two-tier language/corpus mixture expressible:
    put p_l on the language group and p_c on each corpus inside it, and the
    effective per-corpus probability is p_l * p_c.  It is also the schema NeMo
    Speech uses, so one weights file can drive both.

    Returns ``(cuts, use_iterable, n_cuts)``.  ``n_cuts`` is the total number of
    cuts underneath this level and is what auto-weighting uses one level up; it
    is 0 when the sources cannot report a length.

    When ``repeat`` is set, each leaf source is made infinite *before* being
    muxed.  This is what makes the weights mean anything: ``CutSet.mux`` draws
    from a source until it is exhausted and then drops it, so muxing finite
    sources and repeating the combination delivers 100% of every corpus per
    cycle -- the resulting mixture tracks corpus size and ignores the weights
    entirely.  Repeating each source first keeps every one of them available
    forever, so the draw probabilities hold for the whole run.  This matches
    what NeMo does in ``nemo/collections/common/data/lhotse/cutset.py::mux``.
    """
    cutsets: list[CutSet] = []
    explicit: list[float | None] = []
    sizes: list[int] = []
    use_iterable = False

    for idx, source_cfg in enumerate(entries):
        source_type = _get_config_value(source_cfg, "type", "lhotse_shar")
        where = f"{_where}[{idx}]"

        if source_type == "group":
            children = _get_config_value(source_cfg, "input_cfg", None)
            if not children:
                raise ValueError(
                    f"{where}: a 'group' entry must define a non-empty 'input_cfg'"
                )
            # The group's leaves are repeated inside the recursion, so the
            # group's own mux is already infinite.
            cuts, child_iterable, n_cuts = _combine_entries(
                children, shuffle, shard_seed, repeat, indexed, f"{where}.input_cfg"
            )
            use_iterable = use_iterable or child_iterable
        elif source_type == "lhotse_shar":
            shar_path = _get_config_value(source_cfg, "shar_path")
            if shar_path is None:
                raise ValueError("shar_path must be specified for lhotse_shar type")

            # Expand environment variables in path
            shar_path = os.path.expandvars(str(shar_path))

            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")

            logger.info(f"Loading CutSet from shar: {shar_path} (shard_seed: {shard_seed})")

            # `indexed` selects how the corpus is divided across the
            # (DP rank x DataLoader worker) pool:
            #
            #   True  -- .idx sidecars exist; lhotse partitions by SAMPLE index,
            #            so every cut is produced exactly once across the pool
            #            and a nominal epoch is 100% of the data.
            #   False -- streaming; every rank and worker walks the whole corpus
            #            in its own shuffled order, so streams overlap and an
            #            epoch is ~63.2% of the data with the rest repeats.
            #   None  -- auto-detect per source, which is lhotse's default.
            #
            # `split_for_dataloading` is deliberately not passed: it partitioned
            # by SHARD, which never worked here (median source has 6 shards
            # against 128+ ranks), and lhotse 2.0 accepts-but-ignores it.
            cuts = CutSet.from_shar(
                in_dir=shar_path,
                shuffle_shards=shuffle,
                seed=shard_seed,
                stateful_shuffle=shuffle,
                indexed=indexed,
            )
            use_iterable = True  # Shar always uses iterable dataset
            # Count before repeating: len() of a repeated CutSet is not the
            # corpus size, and auto-weighting needs the corpus size.
            n_cuts = _count_cuts(cuts, where)
            if repeat:
                cuts = cuts.repeat(preserve_id=True)
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

        # Add tags to cuts if specified.  On a group this applies to every cut
        # underneath it, and runs after the children have been tagged, so a
        # group tag overwrites a child's value for the same key.  Put on a group
        # only what is genuinely constant across it.
        tags = _get_config_value(source_cfg, "tags", {})
        if tags:
            tag_dict = dict(tags) if not isinstance(tags, dict) else tags
            cuts = cuts.map(partial(_add_tags_to_cut, tags=tag_dict), apply_fn=None)

        cutsets.append(cuts)
        explicit.append(_explicit_weight(source_cfg))
        sizes.append(n_cuts)

    total_cuts = sum(sizes)
    if len(cutsets) == 1:
        return cutsets[0], use_iterable, total_cuts

    weights = _resolve_weights(explicit, sizes, _where)
    logger.info(
        f"Mux-ing {len(cutsets)} sources at {_where}.  "
        f"Weights: {[f'{w:.4g}' for w in weights]}"
    )
    return CutSet.mux(*cutsets, weights=weights, seed=shard_seed), use_iterable, total_cuts


def _count_cuts(cuts: CutSet, where: str) -> int:
    """Number of cuts in a source, or 0 when it cannot report one.

    SHAR CutSets are lazy: len() only counts manifest lines, it does not load
    audio.
    """
    try:
        return len(cuts)
    except (TypeError, ValueError) as exc:
        logger.warning(f"  {where}: len() not supported ({exc}); counts as 0 cuts")
        return 0


def _explicit_weight(source_cfg) -> float | None:
    """The entry's ``weight`` if it sets one, else None."""
    has_explicit = (
        "weight" in source_cfg
        if isinstance(source_cfg, (dict, DictConfig))
        else False
    )
    if not has_explicit:
        return None
    return float(_get_config_value(source_cfg, "weight", 1.0))


def _resolve_weights(
    explicit: list[float | None], sizes: list[int], where: str
) -> list[float]:
    """Pick the mux weights for one level, either all explicit or all automatic.

    Mixing the two within a level is rejected rather than silently accepted: an
    explicit weight is a share of the level (values around 1), while an
    automatic one is a raw cut count (values in the millions).  Muxing them
    together normalises both onto the same scale, which starves every explicitly
    weighted source to approximately zero.
    """
    n_explicit = sum(w is not None for w in explicit)
    if n_explicit == 0:
        # Auto-weight by cut count so a cut from a large dataset and a cut from
        # a small one are sampled at the same per-cut rate.  Floor at 1 so an
        # empty or unmeasurable source is not weighted to zero.
        weights = [float(max(1, n)) for n in sizes]
        logger.info(f"  {where}: auto-weighting by cut count {sizes}")
        return weights
    if n_explicit != len(explicit):
        missing = [i for i, w in enumerate(explicit) if w is None]
        raise ValueError(
            f"{where}: {n_explicit} of {len(explicit)} entries set 'weight'. "
            f"Set it on all of them or none — entries at {missing} are missing "
            f"it. Mixing explicit weights with automatic cut counts would "
            f"reduce the explicitly weighted sources to near-zero probability."
        )
    return [float(w) for w in explicit]


def _add_tags_to_cut(cut: Cut, tags: dict[str, str]) -> Cut:
    """Add metadata tags to a cut.

    Merges into any tags a nested level already applied, with *tags* winning
    on shared keys -- matching ``cut.custom.update`` just above. A group
    calls this after its children, so a group tag overwrites a child's value
    for the same key (see the call site's docstring) while a child-only key,
    e.g. a per-source ``text_field``, survives the group's own pass. A plain
    ``cut.tags = tags`` would instead replace the dict outright and silently
    drop every child-only key.
    """
    if cut.custom is None:
        cut.custom = {}

    cut.custom.update(tags)
    # Also store as attribute for easier access
    existing_tags = cut.tags if getattr(cut, "tags", None) else {}
    cut.tags = {**existing_tags, **tags}
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
    # Validate shard_seed before anything consumes it: it now also seeds shard
    # traversal inside read_cutset_from_config, so an unsupported value would
    # otherwise surface as a lhotse error from deep inside the reader.
    #
    # Important: do NOT resolve shard_seed='randomized' in the main process.
    # Lhotse's resolve_seed('randomized') is designed to run in DataLoader workers after
    # make_worker_init_fn has set LHOTSE_PROCESS_SEED.
    # Falls back to `seed` when unset, matching read_cutset_from_config so the
    # reader and the sampler can never end up on different seeds.
    shard_seed = _get_config_value(config, "shard_seed", _get_config_value(config, "seed", 42))
    if isinstance(shard_seed, str) and shard_seed not in ("trng", "randomized"):
        raise ValueError(f"Unsupported shard_seed={shard_seed!r}. Supported values: int, 'trng', 'randomized'.")

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

    max_tokens = _get_config_value(config, "max_tokens", None)
    max_tps = _get_config_value(config, "max_tps", None)

    if max_tokens is not None or max_tps is not None:
        _max_tokens = int(max_tokens) if max_tokens is not None else None
        _max_tps = float(max_tps) if max_tps is not None else None

        filter_parts = []
        if _max_tokens is not None:
            filter_parts.append(f"max_tokens={_max_tokens}")
        if _max_tps is not None:
            filter_parts.append(f"max_tps={_max_tps}")
        logger.info(f"Applied token filter: {', '.join(filter_parts)}")

        # One line per cut lacking num_tokens is unusable: a source without the
        # field warns on every cut it yields, and because the train stream is
        # `.repeat()`ed it warns again on every pass. A small source can emit
        # the same warning millions of times in a long run and starve the
        # sampler on log I/O alone. Warn once, then count.
        missing_num_tokens = {"n": 0}

        def _warn_missing_once(cut_id: str) -> None:
            missing_num_tokens["n"] += 1
            if missing_num_tokens["n"] == 1:
                logger.warning(
                    f"Cut {cut_id} has no custom.num_tokens; the max_tokens/max_tps "
                    "filters cannot apply to it and it is kept. Further occurrences "
                    "are counted, not logged."
                )
            elif missing_num_tokens["n"] % 100_000 == 0:
                logger.warning(
                    f"{missing_num_tokens['n']:,} cuts so far had no "
                    "custom.num_tokens (repeats included)."
                )

        def _token_filter(c: Cut, max_tokens: int | None, max_tps: float | None) -> bool:
            custom = getattr(c, "custom", None) or {}
            num_tokens = custom.get("num_tokens") if isinstance(custom, dict) else None

            if num_tokens is None:
                if max_tokens is not None or max_tps is not None:
                    _warn_missing_once(c.id)
                return True

            if max_tokens is not None and num_tokens > max_tokens:
                return False

            if max_tps is not None and c.duration > 0 and (num_tokens / c.duration) > max_tps:
                return False

            return True

        cuts = cuts.filter(partial(_token_filter, max_tokens=_max_tokens, max_tps=_max_tps))

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
    # shard_seed was read and validated at the top of this function.

    # `use_bucketing` was retired in favour of naming the sampler outright. Now
    # that `lhotse_sampler_type` has a default, an unmigrated config would no
    # longer fail — it would quietly run a sampler its author never asked for
    # (notably `use_bucketing: false` getting bucketed). So say so instead.
    if _get_config_value(config, "use_bucketing", None) is not None:
        raise ValueError(
            "`use_bucketing` is retired. Replace it with `lhotse_sampler_type: "
            "dynamic_bucketing` (where it was true) or `lhotse_sampler_type: "
            "dynamic` (where it was false)."
        )

    lhotse_sampler_type = _get_config_value(config, "lhotse_sampler_type", None)
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

        # Training (repeat=True, iterable): split_for_dataloading=False so every
        # worker sees all shards.  Different shuffle seeds per worker (set by
        # make_worker_init_fn) provide uniqueness — the sampler just draws from
        # its local iterator with rank=0 / world_size=1.
        # Eval (repeat=False): split_for_dataloading=False, so the sampler must
        # handle rank-based partitioning to guarantee even batch counts across GPUs.
        sampler_rank = 0 if (use_iterable and repeat) else global_rank
        sampler_world_size = 1 if (use_iterable and repeat) else world_size

        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=batch_duration,
            max_cuts=max_cuts,
            quadratic_duration=quadratic_duration,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=shard_seed,
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
            shuffle_buffer_size=buffer_size,
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
        raise ValueError(
            f"Lhotse sampler type `{lhotse_sampler_type}` unknown; expected one "
            "of: dynamic_bucketing, dynamic, bucketing."
        )

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

    # Indexed partitioning is armed by `make_worker_init_fn`, which sets
    # LHOTSE_USE_WORKER_PARTITION -- and that only ever runs inside a DataLoader
    # worker subprocess. At num_workers=0 there is no subprocess, the partition
    # collapses to (0, 1), and every rank silently reads the entire corpus. That
    # is the exact duplication indexing was adopted to remove, so refuse it
    # rather than let a run look correct and train on world_size copies.
    if (
        repeat
        and use_iterable
        and world_size > 1
        and num_workers == 0
        and _sources_are_indexed(_get_config_value(config, "input_cfg", []))
    ):
        raise ValueError(
            f"num_workers=0 with world_size={world_size} and indexed Shar sources: "
            "cross-rank partitioning would not activate and every rank would read "
            "the whole corpus. Set num_workers >= 1."
        )

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

        # Estimate batches per epoch for progress bars.
        # We need micro-batches per rank per epoch (not optimizer steps), which
        # is what this DataLoader yields once its workers are interleaved, so we
        # call estimate_steps_per_epoch with gradient_accumulation_steps=1.
        _, total_duration_hours, total_cuts, _, batches_per_rank = estimate_steps_per_epoch(
            config=config,
            gradient_accumulation_steps=1,  # We want micro-batches, not optimizer steps
            world_size=world_size,
        )
        # Convert to int for __len__
        batches_per_rank_int = max(1, int(batches_per_rank))
        logger.info(
            "This dataloader will yield (approx):\n"
            + f" {total_cuts} total cuts (after filtering)\n"
            + f" with a total duration of {total_duration_hours:.2f} hours and\n"
            + f" with an estimated {batches_per_rank_int} micro-batches per epoch per rank"
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
                estimated_batches_per_epoch=batches_per_rank_int,
            )

        # This runs in every worker subprocess and is where LHOTSE_PROCESS_SEED
        # and LHOTSE_USE_WORKER_PARTITION get set -- the latter is what arms
        # indexed partitioning (issue #52). Without it every rank reads the whole
        # corpus. It used to be wrapped so a saved sampler state could be
        # fast-forwarded here; StatefulDataLoader restores each worker from its
        # own snapshot instead, so that part is gone.
        lhotse_worker_init = make_worker_init_fn(
            rank=global_rank,
            world_size=world_size,
            seed=config.seed,
            set_different_node_and_worker_seeds=True,
        )

        def _worker_init(worker_id: int) -> None:
            # Re-apply inside the worker. The module-level call already covers a
            # forked worker, which inherits the patched class, but a spawned one
            # re-imports and would not -- and the restore that needs it happens
            # here, in the worker, not in the training process.
            _harden_rng_setstate()
            lhotse_worker_init(worker_id)

        dloader_kwargs = {
            "dataset": wrapped_dataset,
            "worker_init_fn": _worker_init,
            "persistent_workers": num_workers > 0 and repeat,  # Only persistent for training
        }
    else:
        # For non-tarred data, sampler stays in main process
        logger.info("Using map-style dataset")
        dloader_kwargs = {
            "dataset": dataset,
            "sampler": sampler,
        }

    # Training uses StatefulDataLoader so the run can resume exactly where it
    # stopped. It snapshots each worker's position inside that worker, which is
    # the only place the position exists once num_workers > 0 -- the
    # main-process sampler is never iterated, so reading its state there yields
    # nothing to restore from (issues #46, #55).
    #
    # Eval stays on a plain DataLoader: it has no position worth resuming, and
    # the map-style branch is handed to accelerate's prepare_data_loader, which
    # builds its own loader anyway.
    loader_cls = StatefulDataLoader if (repeat and use_iterable) else torch.utils.data.DataLoader
    extra_kwargs = {}
    if loader_cls is StatefulDataLoader:
        # How often workers checkpoint their position. 1 means every batch: the
        # snapshot is small (an iterator position, not data) and anything coarser
        # would resume at the last multiple instead of the actual step.
        extra_kwargs["snapshot_every_n_steps"] = int(
            _get_config_value(config, "snapshot_every_n_steps", 1)
        )

    # Suppress PyTorch's warning about iterable dataset length mismatch.
    # This warning is expected for dynamic batching: actual batch count varies due to
    # duration-based/count-based batching and shuffling, not due to misconfiguration.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Length of IterableDataset .* was reported to be .* but .* samples have been fetched",
            category=UserWarning,
        )
        dataloader = loader_cls(
            **dloader_kwargs,
            **extra_kwargs,
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


# Keys declared at the ``data.`` level that decide the *sequence format* a batch
# is tokenised into.  The training path is handed the whole ``data`` block and so
# reads them; the eval path is handed ``data.validation_ds``, one level down, and
# before this resolver saw their defaults instead — formatting eval as a bare
# f"{audio_token}{text}" while training formatted a chat turn.  See issue #58.
_EVAL_FORMAT_KEYS = (
    "apply_chat_template",
    "prompt_template",
    "prompt_template_selection",
    "chat_template_config",
)


def _config_has_key(config, key: str) -> bool:
    """Whether ``config`` sets ``key``, as opposed to falling back to a default.

    Distinct from :func:`_get_config_value`, which cannot tell an explicit value
    from an absent one.  Inheritance has to know the difference: a
    ``validation_ds`` that names a key is overriding the parent deliberately.
    """
    if config is None:
        return False
    if isinstance(config, (dict, DictConfig)):
        return key in config
    return hasattr(config, key)


def _config_set(config, key: str, value) -> None:
    """Assign ``key`` on a config of any of the shapes ``_get_config_value`` reads."""
    if isinstance(config, DictConfig):
        # Struct mode rejects plain assignment of an undeclared key.
        OmegaConf.update(config, key, value, force_add=True)
    elif isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def resolve_eval_data_config(data_config: DictConfig):
    """Return ``validation_ds`` with the parent's formatting keys inherited.

    ``apply_chat_template`` and its companions are declared at ``data.``, which
    the training path receives whole.  Eval is built from ``data.validation_ds``,
    so it never saw them and silently used the defaults — a validation loss over
    a different sequence format than the training loss, and not comparable to it
    (issue #58).

    Inheritance is one-directional and ``validation_ds`` wins: a key set there is
    left alone, which keeps the documented workaround of setting these keys
    inside ``validation_ds`` working exactly as before.

    Args:
        data_config: The whole ``data`` config block.

    Returns:
        The ``validation_ds`` config, copied and extended when anything was
        inherited, or the original object when there was nothing to inherit.
        ``None`` if ``data_config`` has no ``validation_ds``.
    """
    validation_ds = _get_config_value(data_config, "validation_ds")
    if validation_ds is None:
        return None

    inherited = {
        key: _get_config_value(data_config, key)
        for key in _EVAL_FORMAT_KEYS
        if _config_has_key(data_config, key)
        and not _config_has_key(validation_ds, key)
    }
    if not inherited:
        return validation_ds

    resolved = copy.deepcopy(validation_ds)
    for key, value in inherited.items():
        _config_set(resolved, key, value)

    logger.info(
        "Eval formatting inherited from data.: %s",
        ", ".join(f"{k}={v!r}" for k, v in inherited.items()),
    )
    if inherited.get("apply_chat_template"):
        logger.warning(
            "Eval now applies the chat template, matching training (issue #58). "
            "Before this fix eval formatted text without it, so eval_loss here "
            "is not comparable to eval_loss from earlier runs of this config. "
            "Set apply_chat_template inside validation_ds to pin the old behaviour."
        )
    return resolved


def split_eval_config_by_name(config: DictConfig) -> dict[str, DictConfig] | None:
    """Split a ``validation_ds`` config into one sub-config per named eval set.

    Each ``input_cfg`` entry may carry an optional ``name``.  Sources sharing a
    name are evaluated together and reported under that name, which is how
    per-language / per-task validation loss is obtained: HF's Trainer loops over
    a dict of eval datasets and prefixes every metric with the key, giving
    ``eval_<name>_loss``.

    Naming is all-or-none, mirroring the mixture-weight rule in
    :func:`_resolve_weights`: a config where only some sources are named is
    almost certainly a mistake, and silently lumping the rest together would
    hide it.

    Args:
        config: A ``validation_ds`` DictConfig.

    Returns:
        A mapping of name -> sub-config (each a copy of ``config`` carrying only
        that name's sources), preserving first-appearance order.  ``None`` when
        no source is named, meaning the caller should build a single eval
        dataset exactly as before.

    Raises:
        ValueError: if some but not all sources declare a ``name``.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        return None

    names = [_get_config_value(s, "name", None) for s in input_cfg]
    named = [n for n in names if n]
    if not named:
        return None
    if len(named) != len(names):
        missing = [
            str(_get_config_value(s, "shar_path", "<no shar_path>"))
            for s, n in zip(input_cfg, names)
            if not n
        ]
        raise ValueError(
            "validation_ds.input_cfg mixes named and unnamed sources. Either "
            "name every source (to report per-set metrics as eval_<name>_loss) "
            f"or none of them. Unnamed: {missing}"
        )

    grouped: dict[str, list] = {}
    for source_cfg, name in zip(input_cfg, names):
        grouped.setdefault(str(name), []).append(source_cfg)

    sub_configs: dict[str, DictConfig] = {}
    for name, sources in grouped.items():
        sub = copy.deepcopy(config)
        # OmegaConf containers reject plain assignment of a foreign node list in
        # struct mode, so go through OmegaConf.update on the copy.
        OmegaConf.update(sub, "input_cfg", sources, force_add=True)
        sub_configs[name] = sub
    return sub_configs


def materialize_cuts_for_eval(config: DictConfig) -> list[Cut]:
    """Materialize Cut metadata from SHAR manifests into an in-memory list.

    Iterates over SHAR manifests (reads gzipped JSONL — NO audio loaded),
    applies duration/token/max_samples filters, and returns a plain
    ``list[Cut]``.  Multi-source configs are handled by simple concatenation
    (every cut appears exactly once).

    Args:
        config: DictConfig with ``input_cfg``, ``min_duration``, etc.

    Returns:
        List of Lhotse Cut objects.
    """
    input_cfg = _get_config_value(config, "input_cfg", [])
    if not input_cfg:
        raise ValueError("No data sources specified in input_cfg for eval")

    all_cuts: list[Cut] = []
    min_duration = _get_config_value(config, "min_duration")
    max_duration = _get_config_value(config, "max_duration")
    max_samples = _get_config_value(config, "max_samples", None)
    max_tokens = _get_config_value(config, "max_tokens", None)
    max_tps = _get_config_value(config, "max_tps", None)
    seed = int(_get_config_value(config, "seed", 0))

    for source_cfg in input_cfg:
        source_type = _get_config_value(source_cfg, "type", "lhotse_shar")
        if source_type != "lhotse_shar":
            raise ValueError(
                f"Unknown data source type for eval: {source_type}"
            )

        shar_path = os.path.expandvars(
            str(_get_config_value(source_cfg, "shar_path"))
        )
        if not Path(shar_path).exists():
            raise FileNotFoundError(f"Shar path not found: {shar_path}")

        logger.info(
            "Materialising cuts for eval from: %s", shar_path,
        )

        cuts = CutSet.from_shar(
            in_dir=shar_path,
            shuffle_shards=False,
            seed=seed,
            split_for_dataloading=False,
        )

        # --- Filters (mirrors read_cutset_from_config) ---
        if min_duration is not None or max_duration is not None:
            _min = float(min_duration) if min_duration is not None else None
            _max = float(max_duration) if max_duration is not None else None
            cuts = cuts.filter(
                lambda c: (_min is None or c.duration >= _min)
                and (_max is None or c.duration <= _max)
            )

        if max_tokens is not None:
            _mt = int(max_tokens)
            cuts = cuts.filter(
                lambda c: (
                    c.custom.get("num_tokens", 0)
                    if hasattr(c, "custom") and c.custom
                    else 0
                )
                <= _mt
            )

        if max_tps is not None and max_tps > 0:
            _mtps = float(max_tps)
            cuts = cuts.filter(
                lambda c: (
                    (
                        c.custom.get("num_tokens", 0) / max(c.duration, 1e-6)
                        if hasattr(c, "custom") and c.custom
                        else 0
                    )
                    <= _mtps
                )
            )

        # Attach tags (task, lang, dataset_id, etc.)
        tags = _get_config_value(source_cfg, "tags", {})
        tag_dict = dict(tags) if not isinstance(tags, dict) else tags
        if tag_dict:
            cuts = cuts.map(
                partial(_add_tags_to_cut, tags=tag_dict), apply_fn=None,
            )

        # Materialise
        for cut in cuts:
            all_cuts.append(cut)

    if max_samples is not None:
        import random as _random
        _random.Random(seed).shuffle(all_cuts)
        all_cuts = all_cuts[: int(max_samples)]

    logger.info(
        "Materialized %d cuts for eval from %d source(s).",
        len(all_cuts), len(input_cfg),
    )
    return all_cuts


def create_eval_dataloader(
    data_config: DictConfig,
    processor: "MELTProcessor",  # noqa: F821
    batch_size: int,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Create a standard PyTorch DataLoader for evaluation.

    Uses :class:`MELTMapDataset` + :class:`MELTDataCollator` instead of
    Lhotse's :class:`DynamicBucketingSampler` + :class:`IterableDatasetWrapper`.

    The returned DataLoader is suitable for HF Trainer's evaluation loop.
    Distributed sampling is handled by Accelerator via ``prepare_data_loader``.

    Args:
        data_config: Full DictConfig with ``validation_ds`` settings.
        processor: :class:`MELTProcessor` for audio/text processing.
        batch_size: Fixed number of cuts per batch.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Standard PyTorch DataLoader (NOT wrapped by Accelerator — the caller
        must pass it through ``self.accelerator.prepare_data_loader``).
    """
    from .collator import MELTDataCollator
    from .map_dataset import MELTMapDataset

    validation_ds = resolve_eval_data_config(data_config)

    # 1. Materialize cuts
    cuts = materialize_cuts_for_eval(validation_ds)

    # 2. Create map-style dataset
    dataset = MELTMapDataset(
        cuts=cuts,
        processor=processor,
        config=validation_ds,
        is_train=False,
        return_langs=True,
    )

    # 3. Create collator
    collator = MELTDataCollator(
        processor=processor,
        config=validation_ds,
        is_train=False,
    )

    # 4. Build DataLoader (no DistributedSampler — Accelerator adds it)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    logger.info(
        "Eval DataLoader: %d valid cuts, batch_size=%d, ~%d batches, "
        "num_workers=%d.",
        len(dataset),
        batch_size,
        math.ceil(len(dataset) / batch_size),
        num_workers,
    )

    return dataloader


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
    "materialize_cuts_for_eval",
    "create_eval_dataloader",
    "InfiniteIterableDatasetWrapper",
]
