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
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, nested_detach
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize, has_length, denumpify_detensorize
from typing import Optional
import numpy as np

from .. import ddp
from ..logging_utils import get_logger
from ..modeling import MELTProcessor
from .data.audio.lhotse import (
    FallbackDataset,
    SpeechToTextDataset,
    # estimate_num_batches,
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
        config: OmegaConf DictConfig with Lhotse data loading settings.
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
        config: DictConfig,
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
                return_labels=True,  # we need labels for evaluation metrics
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
            return_labels=True,
        )

        # Wrap with fallback for fault tolerance
        dataset = FallbackDataset(dataset)

        # Create dataloader from config
        split_batches = bool(
            getattr(
                getattr(self.args, "accelerator_config", None), "split_batches", False
            )
        )
        dataloader = get_train_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
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
            raise ValueError(
                "No validation data configured (validation_ds.input_cfg is empty)"
            )

        logger.info("Creating Lhotse evaluation dataloader")

        # Use the eval_dataset created in __init__ (SpeechToTextDataset wrapped in FallbackDataset)
        dataset = self.eval_dataset if self.eval_dataset is not None else eval_dataset
        if dataset is None:
            # Create it now if not available
            dataset = SpeechToTextDataset(
                processor=self.processor,
                config=self.config.data,
                is_train=False,
                return_labels=True,
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
            return super().set_initial_training_values(
                args, dataloader, total_train_batch_size
            )

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
                    logger.info(
                        f"  📊 Logging every {args.logging_steps} steps = ~{logging_hours:.2f} input hours"
                    )
                else:
                    logger.info(
                        f"  📊 Logging every {args.logging_steps} steps = ~{logging_minutes:.1f} input minutes"
                    )

            # Evaluation frequency
            if args and hasattr(args, "eval_steps") and args.eval_steps > 0:
                eval_hours = args.eval_steps * hours_per_step
                if eval_hours >= 1.0:
                    logger.info(
                        f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_hours:.2f} input hours"
                    )
                else:
                    eval_minutes = eval_hours * 60
                    logger.info(
                        f"  📈 Evaluation every {args.eval_steps} steps = ~{eval_minutes:.1f} input minutes"
                    )

            # Save frequency
            if args and hasattr(args, "save_steps") and args.save_steps > 0:
                save_hours = args.save_steps * hours_per_step
                if save_hours >= 1.0:
                    logger.info(
                        f"  💾 Checkpoints every {args.save_steps} steps = ~{save_hours:.2f} input hours"
                    )
                else:
                    save_minutes = save_hours * 60
                    logger.info(
                        f"  💾 Checkpoints every {args.save_steps} steps = ~{save_minutes:.1f} input minutes"
                    )

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
        adapter_module = (
            getattr(audio_stack, "adapter", None) if audio_stack is not None else None
        )
        encoder_module = (
            getattr(audio_stack, "encoder", None) if audio_stack is not None else None
        )

        adapter_params = (
            list(adapter_module.parameters()) if adapter_module is not None else []
        )
        # Get optimization config if present, otherwise fall back to args
        opt_cfg = (
            getattr(self.config, "optimization", None)
            if getattr(self, "config", None) is not None
            else None
        )

        # encoder_module is expected to be MELTAudioEncoder; its underlying HF model is `encoder_module.encoder`
        if encoder_module is not None:
            inner = getattr(encoder_module, "encoder", None)
            encoder_params = (
                list(inner.parameters())
                if inner is not None
                else list(encoder_module.parameters())
            )
        else:
            encoder_params = []

        decoder_params = (
            list(decoder_module.parameters()) if decoder_module is not None else []
        )

        # Filter out any frozen params before building optimizer groups.
        # DeepSpeed ZeRO will error out if any param group is empty.
        adapter_params = [p for p in adapter_params if p.requires_grad]
        encoder_params = [p for p in encoder_params if p.requires_grad]
        decoder_params = [p for p in decoder_params if p.requires_grad]

        # Determine learning rates: prefer config.optimization values, otherwise fall back to args
        adapter_lr = (
            getattr(opt_cfg, "adapter_lr", None) if opt_cfg is not None else None
        )
        if adapter_lr is None:
            adapter_lr = getattr(self.args, "adapter_lr", 1e-4)
        adapter_lr = float(adapter_lr)

        encoder_lr = (
            getattr(opt_cfg, "encoder_lr", None) if opt_cfg is not None else None
        )
        if encoder_lr is None:
            encoder_lr = getattr(self.args, "encoder_lr", 1e-5)
        encoder_lr = float(encoder_lr)

        decoder_lr = (
            getattr(opt_cfg, "decoder_lr", None) if opt_cfg is not None else None
        )
        if decoder_lr is None:
            decoder_lr = getattr(self.args, "decoder_lr", 1e-3)
        decoder_lr = float(decoder_lr)

        groups = []
        if len(adapter_params) > 0:
            logger.info(
                f"Optimizer group: adapter ({_format_param_count(len(adapter_params))} params, lr={adapter_lr})"
            )
            groups.append({"params": adapter_params, "lr": adapter_lr})
        if len(encoder_params) > 0:
            logger.info(
                f"Optimizer group: encoder ({_format_param_count(len(encoder_params))} params, lr={encoder_lr})"
            )
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if len(decoder_params) > 0:
            logger.info(
                f"Optimizer group: decoder ({_format_param_count(len(decoder_params))} params, lr={decoder_lr})"
            )
            groups.append({"params": decoder_params, "lr": decoder_lr})

        # Final safety: drop any accidentally-empty groups (defensive against config mistakes)
        groups = [g for g in groups if g.get("params")]

        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Override evaluation_loop to handle Lhotse dynamic batching.

        Key differences from the base implementation:
        - Uses observed_batch_size per step instead of a fixed batch_size for loss repeating
        - Skips gather_function calls that deadlock when the dataloader isn't
          prepared by accelerator (Lhotse handles distribution internally)
        """
        args = self.args
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        )

        # Model prep (same as base)
        if self.is_deepspeed_enabled and self.deepspeed is None:
            from transformers.integrations.deepspeed import deepspeed_init
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            import time as _time
            start_time = _time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(_time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info("  Batch size = dynamic (Lhotse)")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers — no gather, just accumulate locally
        all_losses: list[float] = []
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            losses, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
            )
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name])
                if "inputs" in args.include_for_metrics
                else None
            )

            # --- Collect losses without gather / repeat ---
            if losses is not None:
                # losses is a scalar tensor from prediction_step (.mean() already applied)
                all_losses.append(losses.item())

            if logits is not None:
                logits = nested_detach(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = nested_detach(labels) if isinstance(labels, torch.Tensor) else labels
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
            if inputs_decode is not None:
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control,
            )

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    batch_kwargs: dict = {}
                    batch_kwargs["losses"] = (
                        losses if "loss" in args.include_for_metrics else None
                    )
                    batch_kwargs["inputs"] = (
                        inputs if "inputs" in args.include_for_metrics else None
                    )
                    # Always accumulate; never compute_result inside the loop.
                    # Final computation happens after the loop exits.
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=False,
                    )
                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            elif (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # Loop is done — trigger final metric computation for batch_eval_metrics
        if self.args.batch_eval_metrics and self.compute_metrics is not None:
            # Pass empty tensors so no new data is accumulated, only the result
            # is computed from what was already buffered inside compute_metrics.
            empty = torch.zeros(0)
            metrics = self.compute_metrics(
                EvalPrediction(predictions=empty, label_ids=empty),
                compute_result=True,
            )

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # Finalise containers
        all_preds_arr = all_preds.get_arrays()
        all_labels_arr = all_labels.get_arrays()
        all_inputs_arr = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        else:
            num_samples = observed_num_examples

        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Compute metrics
        if (
            self.compute_metrics is not None
            and all_preds_arr is not None
            and all_labels_arr is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs: dict = {}
            eval_set_kwargs["losses"] = (
                np.array(all_losses) if "loss" in args.include_for_metrics else None
            )
            eval_set_kwargs["inputs"] = (
                all_inputs_arr if "inputs" in args.include_for_metrics else None
            )
            metrics = self.compute_metrics(
                EvalPrediction(
                    predictions=all_preds_arr,
                    label_ids=all_labels_arr,
                    **eval_set_kwargs,
                )
            )
        elif metrics is None:
            metrics = {}

        metrics = denumpify_detensorize(metrics)

        # if all_losses:
        #     metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses).item()
        # At the end of the loop, after computing mean loss:
        if all_losses:
            if torch.distributed.is_initialized():
                loss_tensor = torch.tensor([np.sum(all_losses), len(all_losses)], device=args.device)
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                global_mean_loss = (loss_tensor[0] / loss_tensor[1]).item()
                metrics[f"{metric_key_prefix}_loss"] = global_mean_loss
            else:
                metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses).item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        logger.info(f"***** RANK: {self._global_rank} -- {description} results *****")
        logger.info(f"  Num samples = {num_samples}")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")

        return EvalLoopOutput(
            predictions=all_preds_arr,
            label_ids=all_labels_arr,
            metrics=metrics,
            num_samples=num_samples,
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
