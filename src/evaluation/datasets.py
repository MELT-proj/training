"""Dataset abstractions for evaluation.

This module provides a unified interface for iterating over test datasets.
Currently supports Lhotse CutSet (SHAR format). Designed to be extended
with HuggingFace ``datasets`` support in the future.

Each dataset backend yields :class:`EvalSample` named-tuples so downstream
code is decoupled from the storage format.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.utils.data
from omegaconf import DictConfig

from ..logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Common data container
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes:
        audio: Raw waveform tensor of shape ``(num_samples,)``, or ``None``
            when the dataset is text-only.
        reference: Ground-truth transcription / translation.
        sample_id: Unique identifier for the sample (e.g. Lhotse cut id).
        metadata: Arbitrary per-sample metadata (task, language, …).
    """

    audio: torch.Tensor | None
    reference: str
    sample_id: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EvalDataset(ABC):
    """Abstract base for evaluation dataset backends.

    Subclasses must implement :meth:`__iter__` which yields
    :class:`EvalSample` instances.
    """

    @abstractmethod
    def __iter__(self):
        """Yield :class:`EvalSample` objects one at a time."""
        ...

    @abstractmethod
    def __len__(self) -> int | None:
        """Return the number of samples, or ``None`` if unknown (lazy/streamed)."""
        ...


# ---------------------------------------------------------------------------
# Lhotse CutSet backend
# ---------------------------------------------------------------------------

class LhotseCutSetDataset(EvalDataset):
    """Evaluation dataset backed by a Lhotse CutSet (SHAR or manifest).

    Lazily loads cuts and yields :class:`EvalSample` instances.

    Args:
        ds_cfg: Per-dataset DictConfig node from the eval YAML. Expected keys:
            - ``type``: ``"lhotse_shar"`` or ``"lhotse_cuts"``
            - ``shar_path`` or ``cuts_path``: Path to data.
            - ``text_field``: Dot-separated path to the reference text
              inside the cut (default ``"supervisions.0.text"``).
            - ``sample_rate``: Expected sample rate (default 16000).
    """

    def __init__(self, ds_cfg: DictConfig) -> None:
        from lhotse import CutSet

        self._cfg = ds_cfg
        source_type = ds_cfg.get("type", "lhotse_shar")
        self._sample_rate = int(ds_cfg.get("sample_rate", 16000))
        self._text_field = ds_cfg.get("text_field", "supervisions.0.text")
        self._lang_field = ds_cfg.get("lang_field", None)

        if source_type == "lhotse_shar":
            shar_path = os.path.expandvars(str(ds_cfg.shar_path))
            if not Path(shar_path).exists():
                raise FileNotFoundError(f"Shar path not found: {shar_path}")
            logger.info("Loading eval CutSet from shar: %s", shar_path)
            self._cuts = CutSet.from_shar(in_dir=shar_path, shuffle_shards=False)
        elif source_type == "lhotse_cuts":
            cuts_path = os.path.expandvars(str(ds_cfg.cuts_path))
            if not Path(cuts_path).exists():
                raise FileNotFoundError(f"Cuts path not found: {cuts_path}")
            logger.info("Loading eval CutSet from manifest: %s", cuts_path)
            self._cuts = CutSet.from_file(cuts_path)
        else:
            raise ValueError(f"Unknown eval dataset type: {source_type}. Expected 'lhotse_shar' or 'lhotse_cuts'.")

    # -- helpers ---------------------------------------------------------------

    def _get_nested(self, obj, path: str, default=None):
        """Traverse *obj* along a dot-separated *path*.

        Supports attribute access, dict key access, and integer indexing
        (e.g. ``"supervisions.0.text"``).
        """
        for part in path.split("."):
            if obj is None:
                return default
            if part.isdigit():
                idx = int(part)
                try:
                    obj = obj[idx]
                except (IndexError, KeyError, TypeError):
                    return default
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default
        return obj if obj is not None else default

    # -- public API ------------------------------------------------------------

    def __len__(self) -> int | None:
        # CutSets backed by SHAR are lazy iterables — no length.
        try:
            return len(self._cuts)
        except TypeError:
            return None

    def __iter__(self):
        for cut in self._cuts:
            audio = torch.from_numpy(cut.load_audio()).squeeze(0)
            reference = self._get_nested(cut, self._text_field, default="")
            meta: dict = {}
            if cut.custom is not None:
                meta.update(dict(cut.custom) if not isinstance(cut.custom, dict) else cut.custom)
            if self._lang_field:
                lang = self._get_nested(cut, self._lang_field)
                if lang is not None:
                    meta["lang"] = lang
            yield EvalSample(
                audio=audio,
                reference=str(reference),
                sample_id=cut.id,
                metadata=meta,
            )


# ---------------------------------------------------------------------------
# Future: HuggingFace datasets backend (placeholder)
# ---------------------------------------------------------------------------
# class HFDataset(EvalDataset):
#     """Evaluation dataset backed by a HuggingFace ``datasets.Dataset``."""
#     ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_eval_dataset(ds_cfg: DictConfig) -> EvalDataset:
    """Instantiate the appropriate :class:`EvalDataset` from config.

    The ``type`` field selects the backend:
    - ``lhotse_shar``, ``lhotse_cuts`` → :class:`LhotseCutSetDataset`
    - (future) ``hf_dataset`` → HuggingFace datasets backend

    Args:
        ds_cfg: Per-dataset config node.

    Returns:
        An :class:`EvalDataset` instance.
    """
    source_type = ds_cfg.get("type", "lhotse_shar")

    if source_type in ("lhotse_shar", "lhotse_cuts"):
        return LhotseCutSetDataset(ds_cfg)
    # Future:
    # if source_type == "hf_dataset":
    #     return HFDataset(ds_cfg)
    raise ValueError(f"Unsupported eval dataset type: {source_type}")
