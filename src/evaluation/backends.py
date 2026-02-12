"""Model backend abstractions for evaluation.

Provides a uniform ``generate(batch) → list[str]`` interface so the eval
loop is decoupled from the specific model/inference API.

Currently supports:
- :class:`MELTBackend` — loads a local MELT checkpoint and calls
  ``MELTForCausalLM.generate()``.

Designed to be extended with:
- HuggingFace ``pipeline()`` backend
- vLLM / TGI backends
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig

from ..logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelBackend(ABC):
    """Abstract interface for a model that can generate text from audio."""

    @abstractmethod
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        generation_args: dict | None = None,
    ) -> list[str]:
        """Run generation on a pre-processed batch.

        Args:
            batch: Dictionary produced by the processor / collator containing
                at least ``input_features`` and ``attention_mask``.
            generation_args: Additional keyword arguments forwarded to the
                underlying generation method.

        Returns:
            List of decoded hypothesis strings (one per sample in batch).
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device this backend's model lives on."""
        ...


# ---------------------------------------------------------------------------
# MELT backend
# ---------------------------------------------------------------------------

class MELTBackend(ModelBackend):
    """Backend that wraps a local :class:`MELTForCausalLM` checkpoint.

    Args:
        model_cfg: The ``model`` section of the eval config.
        device: Torch device to place the model on.
    """

    def __init__(self, model_cfg: DictConfig, eval_device: torch.device | str = "cuda") -> None:
        from transformers import AutoFeatureExtractor, AutoTokenizer

        from ..modeling import MELTConfig, MELTForCausalLM, MELTProcessor

        self._device = torch.device(eval_device)

        model_path: str = model_cfg.path
        encoder_name: str = model_cfg.encoder.name
        decoder_name: str = model_cfg.decoder.name

        logger.info("Loading MELT model from %s", model_path)

        # Load processor constituents from the original encoder/decoder names
        feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
        tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=True)

        # Load model from local checkpoint
        self._model = MELTForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._model.to(self._device)
        self._model.eval()

        # Build processor — needs the model config for special tokens
        self._processor = MELTProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            config=model_cfg,
        )

        logger.info("MELT model loaded (%s parameters)", sum(p.numel() for p in self._model.parameters()))

    # -- public API ------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def processor(self) -> "MELTProcessor":  # noqa: F821
        return self._processor

    def generate(
        self,
        batch: dict[str, torch.Tensor],
        generation_args: dict | None = None,
    ) -> list[str]:
        generation_args = generation_args or {}

        # Move tensors to device
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(**batch, **generation_args)

        # Decode, skipping special tokens
        hypotheses = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return hypotheses


# ---------------------------------------------------------------------------
# Future: HuggingFace pipeline backend (placeholder)
# ---------------------------------------------------------------------------
# class HFPipelineBackend(ModelBackend):
#     """Backend that wraps ``transformers.pipeline("automatic-speech-recognition", ...)``.
#
#     Useful for evaluating non-MELT models with a single API.
#     """
#     ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(model_cfg: DictConfig, device: str = "cuda") -> ModelBackend:
    """Instantiate the appropriate :class:`ModelBackend` from config.

    The ``backend`` field in the model config selects the implementation:
    - ``"melt"`` (default) → :class:`MELTBackend`
    - (future) ``"hf_pipeline"`` → HuggingFace pipeline

    Args:
        model_cfg: The ``model`` section of the eval config.
        device: Torch device string.

    Returns:
        A :class:`ModelBackend` instance.
    """
    backend_type = model_cfg.get("backend", "melt")

    if backend_type == "melt":
        return MELTBackend(model_cfg, eval_device=device)
    # Future:
    # if backend_type == "hf_pipeline":
    #     return HFPipelineBackend(model_cfg, device=device)
    raise ValueError(f"Unsupported model backend: {backend_type}")
