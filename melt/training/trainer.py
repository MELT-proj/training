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
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, nested_detach
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PREFIX_CHECKPOINT_DIR,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
)

from .. import ddp
from ..logging_utils import force_print, get_logger
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

        # Sampler state restoration flag (set in train(), consumed in get_train_dataloader())
        self._lhotse_resume_from: str | None = None
        # Reference to the training dataloader (for sampler access during checkpoint saving)
        self._train_dataloader_ref: DataLoader | None = None

        # Initialize parent (may set up distributed)
        super().__init__(model=model, args=args, eval_dataset=eval_dataset, **kwargs)

        # Start CUDA memory history recording if memory_profiling is enabled.
        # Must happen after super().__init__() so the CUDA device is already initialised.
        self._memory_profiling = bool(
            config is not None and config.get("run", {}).get("memory_profiling", False)
        )
        if self._memory_profiling:
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                force_print(
                    f"[MemProf] rank={self._global_rank} — CUDA memory history recording enabled "
                    "(snapshot will be written to output_dir on OOM)"
                )
            else:
                force_print(
                    f"[MemProf] rank={self._global_rank} — memory_profiling=true but CUDA is not "
                    "available; skipping."
                )
                self._memory_profiling = False

        self._memory_preallocation = bool(
            config is not None and config.get("run", {}).get("memory_preallocation", False)
        )
        self._preallocation_done = False

    # ------------------------------------------------------------------
    # Training entry point override: capture resume path for sampler
    # ------------------------------------------------------------------

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", dict[str, Any], None] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        """Override to handle lhotse sampler state restoration on resume."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # Resolve bool → path (same logic as parent)
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        # Check whether the checkpoint contains a saved sampler state for this rank
        self._lhotse_resume_from = None
        if resume_from_checkpoint is not None:
            sampler_file = os.path.join(
                resume_from_checkpoint,
                "sampler",
                f"sampler_state_rank{self._global_rank}.pt",
            )
            if os.path.isfile(sampler_file):
                self._lhotse_resume_from = sampler_file
                logger.info(
                    f"Found lhotse sampler checkpoint at {sampler_file}; "
                    "sampler state will be restored when the dataloader is created."
                )

        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dataloader creation
    # ------------------------------------------------------------------

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

        # Keep a reference for sampler access during checkpoint saving
        self._train_dataloader_ref = dataloader

        # Restore sampler state if we are resuming from a checkpoint
        if self._lhotse_resume_from is not None:
            self._restore_sampler_state(dataloader)

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

    # ------------------------------------------------------------------
    # Lhotse sampler state: save / restore helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sampler_from_dataloader(dataloader: DataLoader):
        """Extract the lhotse CutSampler from a DataLoader.

        The sampler lives on ``dataloader.dataset.sampler`` when using
        :class:`InfiniteIterableDatasetWrapper`.
        """
        dataset = getattr(dataloader, "dataset", None)
        return getattr(dataset, "sampler", None)

    def _save_sampler_state(self, output_dir: str) -> None:
        """Save the lhotse sampler state dict into *output_dir*/sampler/.

        Each rank saves its own file so that restoration is rank-aware.
        """
        dataloader = self._train_dataloader_ref
        if dataloader is None:
            logger.warning("No training dataloader reference — skipping sampler state save.")
            return

        sampler = self._get_sampler_from_dataloader(dataloader)
        if sampler is None:
            logger.warning("No sampler found on the training dataloader — skipping sampler state save.")
            return

        sampler_dir = os.path.join(output_dir, "sampler")
        os.makedirs(sampler_dir, exist_ok=True)

        # --- Build the payload ---------------------------------------------------
        state: dict[str, Any] = {}

        # 1) Native sampler state_dict (accurate when num_workers == 0).
        try:
            sampler_sd = sampler.state_dict()
        except Exception as exc:
            logger.warning(f"sampler.state_dict() failed ({exc}); saving training-progress only.")
            sampler_sd = None

        # 2) When dataloader workers > 0, the main-process sampler is never
        #    iterated, so its diagnostics are empty.  We patch them with values
        #    derived from the training progress so that ``_fast_forward`` on
        #    restore replays the right number of batches.
        if sampler_sd is not None:
            diag = sampler_sd.get("diagnostics", {})
            stats = diag.get("stats_per_epoch", {})
            epoch = sampler_sd.get("epoch", 0)

            has_real_stats = any(
                s.get("kept_batches", 0) + s.get("discarded_batches", 0) > 0
                for s in stats.values()
            )

            if not has_real_stats:
                # Compute the number of micro-batches consumed in the current epoch.
                total_microbatches = (
                    self.state.global_step * self.args.gradient_accumulation_steps
                )
                batches_per_epoch = (
                    len(dataloader)
                    if hasattr(dataloader, "__len__")
                    else total_microbatches
                )
                batches_in_epoch = (
                    total_microbatches % batches_per_epoch
                    if batches_per_epoch > 0
                    else 0
                )
                sampler_sd["diagnostics"] = {
                    "current_epoch": epoch,
                    "stats_per_epoch": {
                        epoch: {
                            "epoch": epoch,
                            "kept_batches": batches_in_epoch,
                            "kept_cuts": 0,
                            "discarded_batches": 0,
                            "discarded_cuts": 0,
                        }
                    },
                }
                logger.info(
                    f"Augmented sampler diagnostics for epoch {epoch}: "
                    f"{batches_in_epoch} micro-batches (num_workers > 0 detected)."
                )

            state["sampler_state_dict"] = sampler_sd

        # 3) Always include training-progress metadata for safety / debugging.
        state["global_step"] = self.state.global_step
        state["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps

        save_path = os.path.join(
            sampler_dir, f"sampler_state_rank{self._global_rank}.pt"
        )
        torch.save(state, save_path)
        logger.info(f"Saved lhotse sampler state to {save_path}")

    def _restore_sampler_state(self, dataloader: DataLoader) -> None:
        """Load a previously saved sampler state into *dataloader*'s sampler."""
        sampler_file = self._lhotse_resume_from
        self._lhotse_resume_from = None  # consumed

        sampler = self._get_sampler_from_dataloader(dataloader)
        if sampler is None:
            logger.warning(
                "Could not locate sampler on the dataloader — "
                "sampler state will NOT be restored."
            )
            return

        logger.info(f"Loading lhotse sampler state from {sampler_file}")
        state = torch.load(sampler_file, map_location="cpu", weights_only=False)

        sampler_sd = state.get("sampler_state_dict")
        if sampler_sd is None:
            logger.warning(
                "Checkpoint does not contain 'sampler_state_dict' — "
                "sampler state will NOT be restored."
            )
            return

        num_workers = getattr(dataloader, "num_workers", 0)
        if num_workers > 0:
            # With num_workers > 0, the sampler runs exclusively in worker processes.
            # _fast_forward() inside load_state_dict() must execute AFTER
            # make_worker_init_fn sets LHOTSE_PROCESS_SEED, otherwise the
            # CutSet shard assignment (split_for_dataloading=True) differs from
            # the original run and all resumed batches are wrong.
            # We defer the restore: store the state dict on the dataset wrapper;
            # _make_lhotse_worker_init_fn picks it up and calls load_state_dict()
            # in the worker after LHOTSE_PROCESS_SEED is set.
            dataset = dataloader.dataset
            if hasattr(dataset, "_pending_lhotse_state"):
                dataset._pending_lhotse_state = sampler_sd
                logger.info(
                    f"Deferred lhotse sampler restoration to worker process "
                    f"(num_workers={num_workers}); _fast_forward will run after "
                    "LHOTSE_PROCESS_SEED is set by make_worker_init_fn."
                )
            else:
                logger.warning(
                    "DataLoader has num_workers > 0 but dataset does not support "
                    "deferred sampler state. Falling back to main-process restore — "
                    "shard assignment may not match the original run."
                )
                logger.info(
                    "Restoring lhotse sampler state (this may take a while "
                    "as the sampler fast-forwards through already-seen data)…"
                )
                sampler.load_state_dict(deepcopy(sampler_sd))
                logger.info("Lhotse sampler state restored successfully.")
        else:
            logger.info(
                "Restoring lhotse sampler state (this may take a while "
                "as the sampler fast-forwards through already-seen data)…"
            )
            sampler.load_state_dict(deepcopy(sampler_sd))
            logger.info("Lhotse sampler state restored successfully.")

        # The sampler now handles data positioning internally, so we
        # disable HF Trainer's own batch-skipping to avoid double-skipping.
        self.args.ignore_data_skip = True
        logger.info(
            "Set ignore_data_skip=True — the lhotse sampler handles "
            "data resumption natively."
        )

    # ------------------------------------------------------------------
    # Checkpoint saving override
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model, trial):
        """Extend parent to also persist the lhotse sampler state."""
        super()._save_checkpoint(model, trial)

        # Reconstruct the same output directory the parent just wrote to.
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        self._save_sampler_state(output_dir)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Memory preallocation
    # ------------------------------------------------------------------

    def _build_max_length_batch(self, model: torch.nn.Module, duration_per_utt: float | None = None) -> dict:
        """Build a synthetic batch for a given per-utterance audio duration.

        Tensor shapes are derived entirely from config and the feature extractor
        so that no actual audio needs to be loaded.

        Args:
            duration_per_utt: Audio duration in seconds for each item in the batch.
                Defaults to train_ds.max_duration (encoder worst-case: longest
                sequences, fewest items). Pass train_ds.min_duration to get the
                decoder worst-case: shortest sequences, most items, maximum total
                text tokens in one forward pass.

        Audio frame count:
            audio_frames = int(duration_per_utt / effective_frame_duration_s)
            feature_dim  = feature_size * fe_stride

            where effective_frame_duration_s = (hop_length / sampling_rate) * fe_stride,
            i.e. the time between consecutive *output* frames after stride-stacking.
            Example: wav2vec-bert-2.0 with stride=2, hop=160, sr=16000 →
              effective_frame_duration_s=0.02s, feature_dim=160, 90s → 4500 frames.

        Batch size:
            n_utts = max(1, int(batch_duration / duration_per_utt))
        """
        train_ds_cfg   = self.config.data.train_ds
        if duration_per_utt is None:
            duration_per_utt = float(train_ds_cfg.max_duration)
        batch_duration = float(train_ds_cfg.batch_duration) # seconds per batch

        # --- derive frame resolution from the feature extractor ---
        fe = self.processor.feature_extractor
        hop_length    = getattr(fe, "hop_length", None)
        sampling_rate = getattr(fe, "sampling_rate", 16_000)

        # SeamlessM4TFeatureExtractor (and similar) stack `stride` consecutive
        # frames: output shape is (T // stride, feature_size * stride).
        fe_stride   = getattr(fe, "stride", 1)
        feature_dim = getattr(fe, "feature_size", 80) * fe_stride

        # effective_frame_duration_s is the time between consecutive *output* frames,
        # i.e. after any stride-stacking applied by the feature extractor.
        # - When hop_length is available it is the raw hop; multiply by fe_stride to
        #   get the output frame duration (e.g. hop=160/16000=0.01 s × stride=2 → 0.02 s).
        # - The fallback 0.02 s is already the effective output frame duration for
        #   wav2vec-bert-2.0 (raw hop 10 ms × stride 2); do NOT multiply by fe_stride
        #   again or the frame count will be halved.
        if hop_length is not None:
            effective_frame_duration_s = (hop_length / sampling_rate) * fe_stride
        else:
            effective_frame_duration_s = 0.02  # 20 ms effective output frame (wav2vec-bert-2.0 default)
            logger.warning(
                "[Preallocation] feature_extractor has no hop_length attribute; "
                "falling back to 20 ms effective output frame duration."
            )

        max_audio_frames = int(duration_per_utt / effective_frame_duration_s)
        n_utts           = max(1, int(batch_duration / duration_per_utt))
        # batch_size in config maps to Lhotse's max_cuts: a hard cap on items per batch.
        max_cuts = getattr(train_ds_cfg, "batch_size", None)
        if max_cuts is not None:
            n_utts = min(n_utts, int(max_cuts))

        # Use max_tokens from config if set, otherwise fall back to a fixed realistic
        # worst-case (model_max_length on large LMs can be 128k; we don't need that).
        cfg_max_tokens = getattr(train_ds_cfg, "max_tokens", None)
        max_text_len = int(cfg_max_tokens) if cfg_max_tokens is not None else 1024

        device = self.args.device
        dtype  = (
            torch.bfloat16 if self.args.bf16
            else torch.float16 if self.args.fp16
            else torch.float32
        )

        logger.warning(
            f"[Preallocation] rank={self._global_rank} — synthetic batch: "
            f"n_utts={n_utts}, max_audio_frames={max_audio_frames} "
            f"({duration_per_utt:.1f}s / {effective_frame_duration_s*1000:.0f}ms output frame), "
            f"feature_dim={feature_dim}, max_text_len={max_text_len}, dtype={dtype}"
        )

        input_features        = torch.zeros(n_utts, max_audio_frames, feature_dim, device=device, dtype=dtype)
        features_attention_mask = torch.ones(n_utts, max_audio_frames, device=device, dtype=torch.long)
        input_ids             = torch.zeros(n_utts, max_text_len, device=device, dtype=torch.long)
        attention_mask        = torch.ones(n_utts, max_text_len, device=device, dtype=torch.long)
        # All -100 except the first token per row to produce a valid (non-NaN) loss.
        labels                = torch.full((n_utts, max_text_len), fill_value=-100, device=device, dtype=torch.long)
        labels[:, 0]          = 0

        return {
            "input_features":          input_features,
            "features_attention_mask": features_attention_mask,
            "input_ids":               input_ids,
            "attention_mask":          attention_mask,
            "labels":                  labels,
        }

    def _run_preallocation_pass(self, model: torch.nn.Module, duration_per_utt: float, label: str) -> bool:
        """Run a single forward+backward warmup pass for the given per-utterance duration.

        Returns True if the pass succeeded, False on OOM (caller may choose to abort).
        """
        mem_before_gb = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )
        try:
            batch = self._build_max_length_batch(model, duration_per_utt=duration_per_utt)
            logger.warning(
                f"[Preallocation/{label}] rank={self._global_rank} — batch shapes: "
                f"input_features={tuple(batch['input_features'].shape)}, "
                f"input_ids={tuple(batch['input_ids'].shape)}"
            )
            loss = self.compute_loss(model, batch)
            self.accelerator.backward(loss)
        except torch.OutOfMemoryError:
            logger.warning(
                f"[Preallocation/{label}] rank={self._global_rank} — OOM during warmup pass "
                "(batch exceeds available memory). Training will proceed but may OOM later."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        finally:
            model.zero_grad(set_to_none=True)

        mem_after_gb = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )
        logger.warning(
            f"[Preallocation/{label}] rank={self._global_rank} — pass complete. "
            f"Peak CUDA memory: {mem_before_gb:.2f} GB → {mem_after_gb:.2f} GB"
        )
        return True

    def _run_preallocation(self, model: torch.nn.Module) -> None:
        """Run forward+backward warmup passes to preallocate CUDA memory buffers.

        Follows the PyTorch performance guide for variable-length inputs:
        allocate the largest buffers that will appear during training so the
        caching allocator never needs to release and re-request memory for a
        longer sequence mid-run.

        Two passes cover the two memory extremes:
          1. max_duration items  — encoder worst-case: O(T²) conformer attention.
          2. min_duration items  — decoder worst-case: maximum number of items in one
             batch, hence maximum total text tokens and decoder hidden-state memory.

        The optimizer and LR scheduler are intentionally NOT stepped.
        Gradients are cleared with set_to_none=True after each pass.
        """
        if not self._memory_preallocation:
            return

        train_ds_cfg = self.config.data.train_ds
        max_duration = float(train_ds_cfg.max_duration)
        min_duration = float(train_ds_cfg.min_duration)

        logger.warning(
            f"[Preallocation] rank={self._global_rank} — starting warmup passes "
            f"(max_duration={max_duration}s, min_duration={min_duration}s) …"
        )

        # Pass 1: few long items — stresses the conformer's quadratic self-attention.
        self._run_preallocation_pass(model, duration_per_utt=max_duration, label="max_duration")

        # Pass 2: many short items — stresses the decoder with maximum total tokens.
        # Total audio tokens per batch are constant (batch_duration × token_rate), but
        # each item adds its own full text sequence, so more items → more decoder memory.
        if min_duration < max_duration:
            self._run_preallocation_pass(model, duration_per_utt=min_duration, label="min_duration")

    # ------------------------------------------------------------------
    # OOM diagnostics
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Override to run memory preallocation on the first step and log OOM diagnostics."""
        if not self._preallocation_done:
            self._run_preallocation(model)
            self._preallocation_done = True

        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError:
            self._log_oom_batch_info(inputs)
            if self._memory_profiling:
                self._dump_memory_snapshot()
            raise

    def _log_oom_batch_info(self, inputs: dict) -> None:
        """Log tensor shapes, model info, and GPU memory state on OOM to identify the offending batch.

        Uses force_print (direct stderr write) instead of logger so that the
        diagnostic is visible on every rank, not just rank 0.
        """
        force_print(
            f"[OOM] rank={self._global_rank} step={self.state.global_step} — "
            "dumping batch info to help identify the offending batch"
        )

        # --- Training context ---
        force_print(
            f"[OOM] Training context: "
            f"fp16={self.args.fp16}, bf16={self.args.bf16}, "
            f"grad_accum={self.args.gradient_accumulation_steps}, "
            f"grad_checkpointing={self.args.gradient_checkpointing}"
        )

        # --- Model dtypes and memory footprint ---
        model = self.model
        param_dtypes = {p.dtype for p in model.parameters()}
        param_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buf_mem_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        force_print(
            f"[OOM] Model: dtypes={[str(d) for d in param_dtypes]}, "
            f"param_mem={param_mem_mb:.1f} MB, buffer_mem={buf_mem_mb:.1f} MB"
        )

        # --- Batch tensors ---
        force_print("[OOM] Batch tensors:")
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                mem_mb = val.numel() * val.element_size() / 1024**2
                force_print(
                    f"[OOM]   {key}: shape={list(val.shape)}, dtype={val.dtype}, "
                    f"device={val.device}, mem={mem_mb:.2f} MB"
                )
            elif isinstance(val, list):
                force_print(f"[OOM]   {key}: list of {len(val)} items")
                if val and isinstance(val[0], str):
                    # e.g. cut IDs — log all of them so the exact cuts can be reproduced
                    force_print(f"[OOM]     values={val}")
                elif val and isinstance(val[0], float):
                    # e.g. audio_lengths — print each value in seconds for easy inspection
                    formatted = ", ".join(f"{v:.2f}s" for v in val)
                    force_print(f"[OOM]     values=[{formatted}]")
            else:
                force_print(f"[OOM]   {key}: {type(val).__name__} = {val!r}")

        # --- GPU memory ---
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                force_print(
                    f"[OOM]   GPU {i}: allocated={allocated:.2f} GB, "
                    f"reserved={reserved:.2f} GB, peak={max_allocated:.2f} GB"
                )

    def _dump_memory_snapshot(self) -> None:
        """Dump a PyTorch CUDA memory snapshot to disk and stop recording.

        The snapshot is a pickle file that can be inspected interactively at
        https://pytorch.org/memory_viz (runs entirely client-side, no upload).

        Recording is stopped after the dump so that repeated OOMs in the same
        run don't accumulate unbounded history.
        """
        import pathlib
        import pickle

        try:
            snapshot = torch.cuda.memory._snapshot()
            out = (
                pathlib.Path(self.args.output_dir)
                / f"oom_memory_snapshot_rank{self._global_rank}_step{self.state.global_step}.pkl"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(pickle.dumps(snapshot))
            force_print(f"[MemProf] rank={self._global_rank} — memory snapshot written to {out}")
        except Exception as exc:
            force_print(f"[MemProf] rank={self._global_rank} — failed to write snapshot: {exc}")
        finally:
            # Stop recording to avoid unbounded accumulation if training continues.
            torch.cuda.memory._record_memory_history(None)
            self._memory_profiling = False

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
