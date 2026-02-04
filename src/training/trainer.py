"""
MELT Trainer with Lhotse-based data loading.

This module provides a custom Trainer that integrates Lhotse dataloaders
for efficient speech data loading with dynamic batching and bucketing.

Key features:
- Dynamic batching based on audio duration (batch_duration)
- Epoch estimation from total dataset duration
- Proper step/epoch tracking for Lhotse's infinite dataloaders
"""

import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import has_length

from .. import ddp
from ..logging_utils import get_logger
from ..modeling import MELTProcessor
from .config import TrainingConfig
from .data.audio.lhotse import (
    FallbackDataset,
    SpeechToTextDataset,
    estimate_num_batches,
    estimate_steps_per_epoch,
    get_eval_dataloader_from_config,
    get_train_dataloader_from_config,
)


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
        model,
        args: TrainingArguments,
        config: TrainingConfig,
        processor: MELTProcessor,
        **kwargs,
    ):
        # Store config and processor before calling super().__init__
        self.config = config
        self.processor = processor

        # Set seed
        # set_seed(config.trainer.seed)

        # Always use ddp.py for distributed information
        self._global_rank = ddp.get_global_rank()
        self._world_size = ddp.get_world_size()

        # Compute epoch estimation from dataset duration
        self.steps_per_epoch = -1
        self.dataset_duration_hours = 0.0
        self.dataset_num_cuts = 0
        self.eval_num_cuts = 0
        self.eval_num_batches = 0

        # # Training dataset stats
        # if hasattr(config.data, "train_ds"):
        #     grad_accum = getattr(args, "gradient_accumulation_steps", 1)
        #     self.steps_per_epoch, self.dataset_duration_hours, self.dataset_num_cuts = estimate_steps_per_epoch(
        #         config=config.data.train_ds,
        #         gradient_accumulation_steps=grad_accum,
        #         world_size=self._world_size,
        #     )

        # Evaluation dataset stats
        if hasattr(config.data, "validation_ds") and config.data.validation_ds.input_cfg:
            from .data.audio.lhotse import compute_dataset_duration

            _, self.eval_num_cuts = compute_dataset_duration(config.data.validation_ds)
            self.eval_num_batches = estimate_num_batches(
                config.data.validation_ds,
                world_size=self._world_size,
            )
            logger.info(f"Evaluation dataset: {self.eval_num_cuts} cuts, ~{self.eval_num_batches} batches")

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
        raise NotImplementedError("This method should not be used in this custom trainer.")

    def get_total_train_batch_size(self, args):
        """
        In the original class, this function returns
        self._train_batch_size * args.gradient_accumulation_steps * dp_world_size.
        Here, we override this and the next one to avoid using self._train_batch_size.
        """
        if args.per_device_train_batch_size == -1:
            return -1  # we are using lhotse's automatic batching
        else:
            return super().get_total_train_batch_size(args)

    def set_initial_training_values(
        self,
        args: TrainingArguments,
        dataloader: DataLoader,
        total_train_batch_size: int | None = None,
    ):
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`: number of optimization steps in an epoch.
        - `num_examples`: used only for logging.
        - `num_train_samples`: used for speed metrics.
        - `epoch_based`: used to scale num_train_tokens.
        - `len_dataloader`: used to compute `steps_in_epoch`. If not provided, falls back to max_steps * grad_accum
        - `max_steps`: used for scheduler setup, logging, and total optimization steps.

        Roughly:
        Parent Trainer._inner_training_loop:
        ├── for epoch in range(0, sys.maxsize):     # runs "forever"
        │   ├── steps_in_epoch = len(dataloader)    # = batches_per_worker (int)
        │   ├── for update_step in total_updates:   # = steps_in_epoch // grad_accum
        │   │   ├── for micro_batch in grad_accum:
        │   │   │   └── self.state.global_step += 1 (after grad_accum micro-batches)
        │   │   └── if global_step >= max_steps: break
        │   └── if should_training_stop: break
        """
        if args.per_device_train_batch_size != -1:
            return super().set_initial_training_values(args, dataloader, total_train_batch_size)

        num_workers = self.config.data.train_ds.num_workers
        grad_accum = args.gradient_accumulation_steps

        # # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        epoch_based = max_steps < 0

        (
            optimization_steps_per_epoch,
            self.dataset_duration_hours,
            self.dataset_num_cuts,
            batches_per_epoch,
            batches_per_worker,
        ) = estimate_steps_per_epoch(
            config=self.config.data.train_ds,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            world_size=self._world_size,
        )

        if epoch_based:
            max_steps = math.ceil(args.num_train_epochs * optimization_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        num_examples = self.dataset_num_cuts  # Just for logging, not critical for audio
        if args.max_steps > 0:
            num_train_epochs = max_steps // optimization_steps_per_epoch + int(
                max_steps % optimization_steps_per_epoch > 0
            )
            # num_train_samples is used for speed_metrics (samples_per_second)
            # For audio, we count micro-batches as "samples"
            num_train_samples = max_steps * args.gradient_accumulation_steps
        else:
            num_train_epochs = args.num_train_epochs
            num_train_samples = num_train_epochs * batches_per_epoch

        # len_dataloader MUST match len(dataloader.dataset) = batches_per_worker
        # This is what the InfiniteIterableDatasetWrapper.__len__ returns
        # It represents micro-batches per worker, per rank, per epoch
        len_dataloader = int(batches_per_worker)

        logger.info(
            f"Single epoch estimation: {self.dataset_num_cuts} cuts, \n"
            f"{self.dataset_duration_hours:.2f} hours, \n"
            f"~{len_dataloader} number of batches, "
            f"~{optimization_steps_per_epoch} optimization steps/epoch (world size: {self._world_size}, num workers: {num_workers}, grad acc steps: {grad_accum})"
        )

        # Compute and log checkpoint/logging/eval intervals in hours
        if self.dataset_duration_hours > 0 and optimization_steps_per_epoch > 0:
            hours_per_step = self.dataset_duration_hours / optimization_steps_per_epoch

            logger.info("=" * 80)
            logger.info("CHECKPOINT & LOGGING SCHEDULE (in input audio hours):")
            logger.info("=" * 80)

            # Logging frequency
            if args and hasattr(args, "logging_steps") and args.logging_steps > 0:
                logging_hours = args.logging_steps * hours_per_step
                logging_minutes = logging_hours * 60
                if logging_hours >= 1.0:
                    logger.info(f"  📊 Logging every {args.logging_steps} steps = ~{logging_hours:.2f} input hours")
                else:
                    logger.info(
                        f"  📊 Logging every {args.logging_steps} steps = ~{logging_minutes:.1f} input minutes"
                    )

            # Evaluation frequency
            if args and hasattr(args, "eval_steps") and args.eval_steps > 0:
                eval_hours = args.eval_steps * hours_per_step
                if eval_hours >= 1.0:
                    logger.info(f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_hours:.2f} input hours")
                else:
                    eval_minutes = eval_hours * 60
                    logger.info(f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_minutes:.1f} input minutes")

            # Save frequency
            if args and hasattr(args, "save_steps") and args.save_steps > 0:
                save_hours = args.save_steps * hours_per_step
                if save_hours >= 1.0:
                    logger.info(f"  💾 Checkpoints every {args.save_steps} steps = ~{save_hours:.2f} input hours")
                else:
                    save_minutes = save_hours * 60
                    logger.info(f"  💾 Checkpoints every {args.save_steps} steps = ~{save_minutes:.1f} input minutes")

            logger.info("=" * 80)
            logger.info(
                f"Note: Time estimates based on a {self.dataset_duration_hours:.2f}h dataset, "
                f"{self._world_size} world size, grad_accum={grad_accum}"
            )
            logger.info("=" * 80)

        # For step-based training with infinite dataloaders, we want:
        # - num_train_epochs = sys.maxsize (so the outer loop keeps running)
        # - epoch_based = False (we stop by max_steps, not epochs)
        # The training loop will terminate when global_step >= max_steps
        if epoch_based:
            logger.info(
                "User specified num_train_epochs. Converting to equivalent max_steps for Lhotse infinite dataloader."
            )

        # Always use step-based termination for infinite dataloaders
        # Set num_train_epochs to a large value so the outer loop keeps running
        # The inner loop will break when global_step >= max_steps
        num_train_epochs = sys.maxsize
        epoch_based = False

        return (
            num_train_epochs,
            optimization_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )

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
