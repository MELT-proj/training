"""
Map-style Dataset for Lhotse-based data.

Unlike :class:`SpeechToTextDataset`, which receives a batch of cuts at a time,
:class:`MELTMapDataset` follows the standard ``torch.utils.data.Dataset``
pattern: ``__getitem__(idx)`` returns a single item dict with raw audio and
text.  Batching and processing are deferred to :class:`MELTDataCollator`.
"""

import torch

from lhotse.cut import Cut
from omegaconf import DictConfig

from .....logging_utils import get_logger
from .....modeling import MELTProcessor
from .helpers import (
    _get_config_value,
    get_tags_from_cut,
    get_text_from_cut,
    load_audio_from_cut,
)

logger = get_logger(__name__)


class MELTMapDataset(torch.utils.data.Dataset):
    """A generic map-style Dataset wrapping a list of Lhotse Cuts.

    Each ``__getitem__`` call returns a raw item dict (numpy audio array +
    text string).  Batching and featurisation happen later in
    :class:`MELTDataCollator`.

    Args:
        cuts: Pre-materialised list of Lhotse Cut objects.
        processor: :class:`MELTProcessor` instance.
        config: DictConfig with data processing settings
            (``text_field``, ``apply_chat_template``, …).
        is_train: Whether this is the training split (affects config key lookup).
        return_langs: Whether to include language codes in the output.
    """

    _SENTINEL = object()

    def __init__(
        self,
        cuts: list[Cut],
        processor: MELTProcessor,
        config: DictConfig,
        is_train: bool = False,
        return_langs: bool = True,
    ) -> None:
        super().__init__()
        self.cuts = cuts
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.return_langs = return_langs
        self.apply_chat_template = bool(
            _get_config_value(config, "apply_chat_template", False)
        )

        # Resolve the text_field. Every caller hands this the ds-level config
        # (``data.validation_ds``, or one per-name split of it), where the key
        # actually lives; the nested lookup is kept for a caller that passes the
        # whole ``data`` block instead. Looking only one level down meant a
        # `text_field` set in validation_ds was silently ignored and eval always
        # read plain `text`.
        ds_key = "train_ds" if is_train else "validation_ds"
        ds_config = _get_config_value(config, ds_key, None)
        source = ds_config if ds_config is not None else config
        self._text_field = str(_get_config_value(source, "text_field", "text"))

        # Build a *valid-indices* list once (cuts with missing text are skipped).
        # This is computed at construction time and reused across all eval calls.
        self._valid_indices: list[int] = []
        skipped = 0
        for idx, cut in enumerate(cuts):
            text = get_text_from_cut(cut, self._text_field)
            if text and text.strip():
                self._valid_indices.append(idx)
            else:
                skipped += 1

        # Sort valid indices by descending length proxy so that batches
        # produced by sequential iteration contain similarly-sized items,
        # minimising padding waste.  The longest items land in the first
        # batch, which also catches OOMs early.
        if self._valid_indices:
            self._valid_indices.sort(
                key=lambda idx: (
                    cuts[idx].duration
                    + (
                        cuts[idx].custom.get("num_tokens", 0)
                        if hasattr(cuts[idx], "custom") and cuts[idx].custom
                        else 0
                    )
                ),
                reverse=True,
            )

        logger.info(
            "MELTMapDataset: %d valid cuts out of %d total (skipped=%d).",
            len(self._valid_indices),
            len(cuts),
            skipped,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        cut_idx = self._valid_indices[idx]
        cut = self.cuts[cut_idx]

        # --- audio ---
        audio = load_audio_from_cut(cut)
        if audio is None:
            return {"__invalid__": True, "cut_id": cut.id}

        # --- text ---
        # Per-cut text_field override (tags.text_field takes precedence)
        text_field = self._text_field
        if hasattr(cut, "custom") and cut.custom:
            tags = cut.custom.get("tags", {})
            if isinstance(tags, dict) and tags.get("text_field"):
                text_field = tags["text_field"]

        text = get_text_from_cut(cut, text_field)
        if not text or not text.strip():
            return {"__invalid__": True, "cut_id": cut.id}
        text = text.strip().lower()

        # --- tags ---
        task, lang, src_lang, tgt_lang = get_tags_from_cut(cut)

        # --- dataset_id ---
        dataset_id = ""
        if hasattr(cut, "tags") and cut.tags:
            dataset_id = cut.tags.get("dataset_id", "")

        return {
            "audio": audio,
            "text": text,
            "task": task,
            "lang": lang,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "cut_id": cut.id,
            "dataset_id": dataset_id,
        }


__all__ = ["MELTMapDataset"]
