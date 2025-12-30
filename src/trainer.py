"""
MELT Trainer with Lhotse-based data loading.

This module provides a custom Trainer that integrates Lhotse dataloaders
for efficient speech data loading with dynamic batching and bucketing.
"""

import logging
import os
import random
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer

import ddp
from src.data.audio.lhotse import (
    FallbackDataset,
    SpeechToTextDataset,
    get_eval_dataloader_from_config,
    get_train_dataloader_from_config,
)
from src.melt import MELTProcessor

logger = logging.getLogger(__name__)


def current_cpumem_usage():
    import psutil

    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024**2:.2f}"


class MELTTrainer(Trainer):
    """Custom Trainer for MELT models with Lhotse data loading.

    This trainer overrides the dataloader creation methods to use
    Lhotse samplers for dynamic batching and efficient speech data loading.

    Args:
        model: The model to train.
        args: Training arguments.
        data_config: DataConfig with Lhotse data loading settings.
        processor: MELTProcessor for audio/text processing.
        **kwargs: Additional arguments passed to Trainer.
    """

    def __init__(
        self,
        model=None,
        args=None,
        config: DictConfig | None = None,
        processor: MELTProcessor | None = None,
        **kwargs,
    ):
        # Store config and processor before calling super().__init__
        self.config = config
        self.processor = processor

        # Always use ddp.py for distributed information
        self._global_rank = ddp.get_global_rank()
        self._world_size = ddp.get_world_size()

        # Initialize parent (may set up distributed)
        super().__init__(model=model, args=args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader using Lhotse.

        Returns the training DataLoader configured with Lhotse sampler
        for dynamic batching based on audio duration.
        """
        if self.processor is None:
            raise ValueError("processor must be provided for Lhotse data loading")

        logger.info("Creating Lhotse training dataloader")

        # Create dataset
        dataset = SpeechToTextDataset(
            processor=self.processor,
            config=self.config.data,
            is_train=True,
        )

        # Wrap with fallback for fault tolerance
        dataset = FallbackDataset(dataset)

        # Create dataloader from config
        dataloader = get_train_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )

        dataloader = self.accelerator.prepare(dataloader)
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create evaluation dataloader using Lhotse.

        Args:
            eval_dataset: Ignored when using Lhotse (config specifies data).

        Returns the evaluation DataLoader configured with Lhotse sampler.
        """
        if self.processor is None:
            raise ValueError("processor must be provided for Lhotse data loading")

        # Check if validation data is configured
        if not self.data_config.validation_ds.input_cfg:
            logger.warning("No validation data configured, skipping eval dataloader")
            return None

        logger.info("Creating Lhotse evaluation dataloader")

        # Create dataset
        dataset = SpeechToTextDataset(
            processor=self.processor,
            config=self.data_config,
            is_train=False,
        )

        # Wrap with fallback for fault tolerance
        dataset = FallbackDataset(dataset)

        # Create dataloader from config
        dataloader = get_eval_dataloader_from_config(
            data_config=self.config,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )

        return dataloader

    @staticmethod
    def num_tokens(train_dl: DataLoader, max_steps: None | int = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            dataset = train_dl.dataset
            words_by_row = [
                len(t.split(" "))
                for t in tqdm(
                    dataset["text"], total=len(dataset), desc="Counting tokens"
                )
            ]
            train_tokens = sum(
                words_by_row
            )  # it's not tokens, but it's a good approximation
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")

        return train_tokens

    def create_optimizer(self):
        """Create optimizer groups respecting freeze flags and modular audio stack.

        This method prefers explicit attributes when available:
        - If the model has `audio_stack`, use `audio_stack.encoder` and `audio_stack.adapter`.
        - Fallback gracefully to legacy attributes or by inspecting parameter names.
        """

        decoder_module = getattr(self.model, "text_decoder", None)
        adapter_module = getattr(self.model.audio_stack, "adapter", None)
        encoder_module = getattr(self.model.audio_stack, "encoder", None)
        adapter_params = (
            list(adapter_module.parameters()) if adapter_module is not None else []
        )
        # Get optimization config if present, otherwise fall back to args
        opt_cfg = getattr(self.config, "optimization", None) if getattr(self, "config", None) is not None else None

        # encoder_module may itself wrap the underlying model (has .encoder)
        if encoder_module is not None:
            encoder_params = list(encoder_module.model.parameters())
        else:
            encoder_params = []

        decoder_params = (
            list(decoder_module.parameters()) if decoder_module is not None else []
        )

        # Apply freezes using provided freeze helpers when available
        if getattr(self.args, "freeze_adapter", False):
            if adapter_module is not None and hasattr(adapter_module, "freeze"):
                adapter_module.freeze()
            else:
                for p in adapter_params:
                    p.requires_grad = False

        if getattr(self.args, "freeze_encoder", False):
            if encoder_module is not None and hasattr(encoder_module, "freeze"):
                encoder_module.freeze()
            else:
                for p in encoder_params:
                    p.requires_grad = False

        if getattr(self.args, "freeze_decoder", False):
            if hasattr(self.model, "freeze_decoder"):
                self.model.freeze_decoder()
            else:
                for p in decoder_params:
                    p.requires_grad = False

        # Determine learning rates: prefer config.optimization values, otherwise fall back to args
        adapter_lr = getattr(opt_cfg, "adapter_lr", None) if opt_cfg is not None else None
        if adapter_lr is None:
            adapter_lr = getattr(self.args, "adapter_lr", 1e-4)
        encoder_lr = getattr(opt_cfg, "encoder_lr", None) if opt_cfg is not None else None
        if encoder_lr is None:
            encoder_lr = getattr(self.args, "encoder_lr", 1e-5)
        decoder_lr = getattr(opt_cfg, "decoder_lr", None) if opt_cfg is not None else None
        if decoder_lr is None:
            decoder_lr = getattr(self.args, "decoder_lr", 1e-3)

        groups = []
        if not getattr(self.args, "freeze_adapter", False) and adapter_params:
            groups.append({"params": adapter_params, "lr": adapter_lr})
        if not getattr(self.args, "freeze_encoder", False) and encoder_params:
            groups.append(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                }
            )
        if not getattr(self.args, "freeze_decoder", False) and decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "lr": decoder_lr,
                }
            )

        # If everything got frozen or no groups created, fall back to any remaining trainable params
        if len(groups) == 0:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable) == 0:
                raise ValueError(
                    "All model parameters are frozen; cannot create optimizer."
                )
            groups = [{"params": trainable, "lr": getattr(self.args, "lr", 1e-5)}]

        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )


def sanitize_model_name(x: str) -> str:
    return x.replace("/", "--")


def _format_param_count(count: int, precision: int = 2) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.{precision}f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.{precision}f}M"
    if count >= 1_000:
        return f"{count / 1_000:.{precision}f}K"
    return str(count)


def count_trainable_parameters(
    model: torch.nn.Module, precision: int = 2, return_int: bool = False
):
    """Return the number of trainable parameters, respecting any frozen modules.

    Args:
        model: A torch.nn.Module to inspect.
        precision: Decimal precision for the formatted string.
        return_int: If True, also return the raw integer count.

    Returns:
        str | tuple[int, str]: Formatted count (and optionally the raw integer).
    """

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted = _format_param_count(trainable, precision)
    if return_int:
        return trainable, formatted
    return formatted
