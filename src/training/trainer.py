"""
MELT Trainer with Lhotse-based data loading.

This module provides a custom Trainer that integrates Lhotse dataloaders
for efficient speech data loading with dynamic batching and bucketing.

Key features:
- Dynamic batching based on audio duration (batch_duration)
- Epoch estimation from total dataset duration
- Proper step/epoch tracking for Lhotse's infinite dataloaders
"""

import os
import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import ddp
from .config import TrainingConfig
from .data.audio.lhotse import (
    FallbackDataset,
    SpeechToTextDataset,
    estimate_num_batches,
    estimate_steps_per_epoch,
    get_eval_dataloader_from_config,
    get_train_dataloader_from_config,
)
from ..logging_utils import get_logger
from ..modeling import MELTProcessor
from transformers import Trainer
from transformers.trainer_utils import has_length

logger = get_logger(__name__)


def current_cpumem_usage():
    import psutil

    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024**2:.2f}"


class MELTTrainer(Trainer):
    """Custom Trainer for MELT models with Lhotse data loading.

    This trainer overrides the dataloader creation methods to use
    Lhotse samplers for dynamic batching and efficient speech data loading.
    It also provides epoch estimation for Lhotse's infinite dataloaders.

    Args:
        model: The model to train.
        args: Training arguments.
        config: TrainingConfig with Lhotse data loading settings.
        processor: MELTProcessor for audio/text processing.
        **kwargs: Additional arguments passed to Trainer.

    Attributes:
        steps_per_epoch: Estimated steps per epoch based on dataset duration.
        dataset_duration_hours: Total dataset duration in hours.
        dataset_num_cuts: Number of cuts in the training dataset.
        eval_num_cuts: Number of cuts in the evaluation dataset.
        eval_num_batches: Estimated number of eval batches.
    """

    def __init__(
        self,
        model=None,
        args=None,
        config: TrainingConfig | None = None,
        processor: MELTProcessor | None = None,
        **kwargs,
    ):
        # Store config and processor before calling super().__init__
        self.config = config
        self.processor = processor

        # Always use ddp.py for distributed information
        self._global_rank = ddp.get_global_rank()
        self._world_size = ddp.get_world_size()

        # Compute epoch estimation from dataset duration
        self.steps_per_epoch = -1
        self.dataset_duration_hours = 0.0
        self.dataset_num_cuts = 0
        self.eval_num_cuts = 0
        self.eval_num_batches = 0

        if config is not None and hasattr(config, "data"):
            # Training dataset stats
            if hasattr(config.data, "train_ds"):
                grad_accum = getattr(args, "gradient_accumulation_steps", 1) if args else 1
                self.steps_per_epoch, self.dataset_duration_hours, self.dataset_num_cuts = (
                    estimate_steps_per_epoch(
                        config=config.data.train_ds,
                        gradient_accumulation_steps=grad_accum,
                        world_size=self._world_size,
                    )
                )

            # Evaluation dataset stats
            if hasattr(config.data, "validation_ds") and config.data.validation_ds.input_cfg:
                from .data.audio.lhotse import compute_dataset_duration
                
                _, self.eval_num_cuts = compute_dataset_duration(config.data.validation_ds)
                self.eval_num_batches = estimate_num_batches(
                    config.data.validation_ds,
                    world_size=self._world_size,
                )
                logger.info(
                    f"Evaluation dataset: {self.eval_num_cuts} cuts, "
                    f"~{self.eval_num_batches} batches"
                )

            # Log epoch estimation info
            if self.steps_per_epoch > 0:
                logger.info(
                    f"Epoch estimation: {self.dataset_num_cuts} cuts, "
                    f"{self.dataset_duration_hours:.2f} hours, "
                    f"~{self.steps_per_epoch} steps/epoch"
                )

                # Get max_steps and num_train_epochs from args
                max_steps = getattr(args, "max_steps", None) if args else None
                num_train_epochs = getattr(args, "num_train_epochs", None) if args else None
                
                # Validate: at least one must be set
                if max_steps is None and num_train_epochs is None:
                    raise ValueError(
                        "Either max_steps or num_train_epochs must be set. "
                        "Both cannot be None."
                    )
                
                # Check if we should compute max_steps from epochs
                compute_from_epochs = (
                    hasattr(config, "trainer")
                    and getattr(config.trainer, "compute_max_steps_from_epochs", False)
                )

                if compute_from_epochs or max_steps is None:
                    # Compute max_steps from num_train_epochs
                    if num_train_epochs is None:
                        num_train_epochs = getattr(config.trainer, "num_train_epochs", 1)
                    computed_max_steps = int(self.steps_per_epoch * num_train_epochs)
                    logger.info(
                        f"Computing max_steps from epochs: "
                        f"{num_train_epochs} epochs * {self.steps_per_epoch} steps/epoch = {computed_max_steps} steps"
                    )
                    # Update args.max_steps so HF Trainer uses it
                    if args is not None:
                        args.max_steps = computed_max_steps
                elif max_steps is not None and max_steps > 0:
                    # If max_steps is set, compute how many epochs that represents
                    total_epochs = max_steps / self.steps_per_epoch
                    logger.info(
                        f"Training for {max_steps} steps = ~{total_epochs:.2f} epochs"
                    )

                # Compute and log checkpoint/logging/eval intervals in hours
                if self.dataset_duration_hours > 0 and self.steps_per_epoch > 0:
                    hours_per_step = self.dataset_duration_hours / self.steps_per_epoch
                    
                    logger.info("=" * 80)
                    logger.info("CHECKPOINT & LOGGING SCHEDULE (in wall-clock time):")
                    logger.info("=" * 80)
                    
                    # Logging frequency
                    if args and hasattr(args, "logging_steps") and args.logging_steps > 0:
                        logging_hours = args.logging_steps * hours_per_step
                        logging_minutes = logging_hours * 60
                        if logging_hours >= 1.0:
                            logger.info(f"  📊 Logging every {args.logging_steps} steps = ~{logging_hours:.2f} hours")
                        else:
                            logger.info(f"  📊 Logging every {args.logging_steps} steps = ~{logging_minutes:.1f} minutes")
                    
                    # Evaluation frequency
                    if args and hasattr(args, "eval_steps") and args.eval_steps > 0:
                        eval_hours = args.eval_steps * hours_per_step
                        if eval_hours >= 1.0:
                            logger.info(f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_hours:.2f} hours")
                        else:
                            eval_minutes = eval_hours * 60
                            logger.info(f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_minutes:.1f} minutes")
                    
                    # Save frequency
                    if args and hasattr(args, "save_steps") and args.save_steps > 0:
                        save_hours = args.save_steps * hours_per_step
                        if save_hours >= 1.0:
                            logger.info(f"  💾 Checkpoints every {args.save_steps} steps = ~{save_hours:.2f} hours")
                        else:
                            save_minutes = save_hours * 60
                            logger.info(f"  💾 Checkpoints every {args.save_steps} steps = ~{save_minutes:.1f} minutes")
                    
                    logger.info("=" * 80)
                    logger.info(
                        f"Note: Time estimates based on {self.dataset_duration_hours:.2f}h dataset, "
                        f"{self._world_size} GPUs, grad_accum={grad_accum}"
                    )
                    logger.info("=" * 80)

        # Create eval dataset before super().__init__() so HF Trainer can use it
        # Uses Lhotse's DynamicBucketingSampler for memory-efficient evaluation
        # (supports lazy CutSets from shar/webdataset without materialization)
        eval_dataset = None
        if (
            processor is not None
            and config is not None
            and hasattr(config, "data")
            and hasattr(config.data, "validation_ds")
            and config.data.validation_ds.input_cfg
        ):
            # Create SpeechToTextDataset for evaluation (same class as training)
            # The finite iteration is handled by the dataloader, not the dataset
            logger.info("Creating evaluation SpeechToTextDataset...")
            eval_dataset = SpeechToTextDataset(
                processor=processor,
                config=config.data,
                is_train=False,
            )
            # Wrap with fallback for fault tolerance
            eval_dataset = FallbackDataset(eval_dataset)
            logger.info(f"Eval dataset ready ({self.eval_num_cuts} cuts)")

        # Initialize parent (may set up distributed)
        super().__init__(model=model, args=args, eval_dataset=eval_dataset, **kwargs)

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
        split_batches = bool(getattr(getattr(self.args, "accelerator_config", None), "split_batches", False))
        dataloader = get_train_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
            split_batches=split_batches,
        )

        # We do not prepare the dataloader with the accelerator since rank and DDP allocation is already
        # taken care of by lhotse. Also, it seems the accelerator does not like setting batch_size to None.
        # dataloader = self.accelerator.prepare(dataloader)
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create evaluation dataloader using Lhotse.

        This overrides the default HF Trainer method to use Lhotse's
        distributed-aware samplers with proper bucketing for efficiency.

        Key characteristics:
        - Finite iteration: iterates exactly once through the eval data
        - Distributed: properly shards data across ranks
        - Efficient: uses DynamicBucketingSampler for batching by duration
        - Multi-worker: supports num_workers > 0 via Lhotse's worker_init_fn
        - Progress bars: has __len__ for proper progress tracking

        Args:
            eval_dataset: Ignored when using Lhotse (config specifies data).
                         If provided, uses self.eval_dataset instead.

        Returns:
            DataLoader for evaluation.
        """
        if self.processor is None:
            raise ValueError("processor must be provided for Lhotse data loading")

        # Check if validation data is configured
        if not self.config.data.validation_ds.input_cfg:
            raise ValueError("No validation data configured (validation_ds.input_cfg is empty)")

        logger.info("Creating Lhotse evaluation dataloader")

        # Use the eval_dataset created in __init__ (SpeechToTextDataset wrapped in FallbackDataset)
        dataset = self.eval_dataset if self.eval_dataset is not None else eval_dataset
        if dataset is None:
            # Create it now if not available
            dataset = SpeechToTextDataset(
                processor=self.processor,
                config=self.config.data,
                is_train=False,
            )
            dataset = FallbackDataset(dataset)

        # Create finite dataloader using Lhotse's distributed-aware sampler
        dataloader = get_eval_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )

        return dataloader

    def num_examples(self, dataloader: DataLoader) -> int:
        """Return the number of examples in the dataloader.

        For Lhotse dataloaders, we use the pre-computed dataset_num_cuts
        since the dataloader may be infinite or not have __len__.

        Args:
            dataloader: The dataloader to count examples from.

        Returns:
            Number of examples (cuts) in the dataset.
        """
        # First try to get length from the dataset directly
        if has_length(dataloader):
            try:
                dataset = dataloader.dataset
                if hasattr(dataset, "__len__"):
                    # For EvalCutSetDataset, __len__ returns num_cuts directly
                    return len(dataset)
            except (TypeError, AttributeError):
                pass

        # For Lhotse infinite dataloaders (training), use pre-computed count
        if self.dataset_num_cuts > 0:
            return self.dataset_num_cuts

        # Fallback: try to get from parent
        try:
            return super().num_examples(dataloader)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def num_tokens(train_dl: DataLoader, max_steps: None | int = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            dataset = train_dl.dataset
            words_by_row = [
                len(t.split(" ")) for t in tqdm(dataset["text"], total=len(dataset), desc="Counting tokens")
            ]
            train_tokens = sum(words_by_row)  # it's not tokens, but it's a good approximation
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
        audio_stack = getattr(self.model, "audio_stack", None)
        adapter_module = getattr(audio_stack, "adapter", None) if audio_stack is not None else None
        encoder_module = getattr(audio_stack, "encoder", None) if audio_stack is not None else None

        adapter_params = list(adapter_module.parameters()) if adapter_module is not None else []
        # Get optimization config if present, otherwise fall back to args
        opt_cfg = getattr(self.config, "optimization", None) if getattr(self, "config", None) is not None else None

        # encoder_module is expected to be MELTAudioEncoder; its underlying HF model is `encoder_module.encoder`
        if encoder_module is not None:
            inner = getattr(encoder_module, "encoder", None)
            encoder_params = list(inner.parameters()) if inner is not None else list(encoder_module.parameters())
        else:
            encoder_params = []

        decoder_params = list(decoder_module.parameters()) if decoder_module is not None else []

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

        # Filter out any frozen params before building optimizer groups.
        # DeepSpeed ZeRO will error out if any param group is empty.
        adapter_params = [p for p in adapter_params if p.requires_grad]
        encoder_params = [p for p in encoder_params if p.requires_grad]
        decoder_params = [p for p in decoder_params if p.requires_grad]

        # Determine learning rates: prefer config.optimization values, otherwise fall back to args
        adapter_lr = getattr(opt_cfg, "adapter_lr", None) if opt_cfg is not None else None
        if adapter_lr is None:
            adapter_lr = getattr(self.args, "adapter_lr", 1e-4)
        adapter_lr = float(adapter_lr)
        
        encoder_lr = getattr(opt_cfg, "encoder_lr", None) if opt_cfg is not None else None
        if encoder_lr is None:
            encoder_lr = getattr(self.args, "encoder_lr", 1e-5)
        encoder_lr = float(encoder_lr)
        
        decoder_lr = getattr(opt_cfg, "decoder_lr", None) if opt_cfg is not None else None
        if decoder_lr is None:
            decoder_lr = getattr(self.args, "decoder_lr", 1e-3)
        decoder_lr = float(decoder_lr)

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

        # Final safety: drop any accidentally-empty groups (defensive against config mistakes)
        groups = [g for g in groups if g.get("params")]

        # If everything got frozen or no groups created, fall back to any remaining trainable params
        if len(groups) == 0:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable) == 0:
                raise ValueError("All model parameters are frozen; cannot create optimizer.")
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


def count_trainable_parameters(model: torch.nn.Module, precision: int = 2, return_int: bool = False):
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
