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
from typing import Any, Optional, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import (
    EvalPrediction,
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
)

from .. import ddp
from ..logging_utils import force_print, get_logger
from ..modeling import MELTProcessor
from .data.audio.lhotse import (
    FallbackDataset,
    MELTDataCollator,
    MELTMapDataset,
    SpeechToTextDataset,
    # estimate_num_batches,
    estimate_steps_per_epoch,
    get_train_dataloader_from_config,
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)

logger = get_logger(__name__)


def current_cpumem_usage():
    import psutil

    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024**2:.2f}"


def _shutdown_dataloader_workers(dataloader: DataLoader | None) -> None:
    """Shut down *dataloader*'s worker processes and pin-memory thread now.

    Called from the main thread so the teardown is ordered and observable
    instead of being left to ``__del__`` on whichever thread the cyclic garbage
    collector happens to run on. See ``MELTTrainer._shutdown_dataloaders`` for
    why that matters (issue #63).

    Safe to call on anything: a loader with no live iterator, a single-process
    iterator with no workers to stop, or an already shut-down iterator (torch's
    ``_shutdown_workers`` is idempotent).
    """
    if dataloader is None:
        return

    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return

    # Clear the loader's handle first, so nothing can hand this iterator out
    # again — notably StatefulDataLoader.state_dict(), which would otherwise
    # build a fresh iterator and start a new set of workers.
    try:
        dataloader._iterator = None
    except AttributeError:  # pragma: no cover - defensive
        pass

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None:
        # Single-process iterator: no workers, no pin-memory thread.
        return

    try:
        shutdown()
    except Exception as exc:  # pragma: no cover - defensive
        # Teardown must never be the reason a finished run fails.
        logger.warning(f"Ignoring error while shutting down dataloader workers: {exc}")


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

        # Seeding happens in train.py, before the model is built -- HF's
        # Trainer.__init__ seeds too, but by then our model already exists.

        # Always use ddp.py for distributed information
        self._global_rank = ddp.get_global_rank()
        self._world_size = ddp.get_world_size()

        # Compute epoch estimation from dataset duration
        self.steps_per_epoch = -1
        self.dataset_duration_hours = 0.0
        self.dataset_num_cuts = 0
        self.eval_num_cuts = 0
        self.eval_num_batches = 0

        # Create eval dataset before super().__init__() so HF Trainer can use it.
        # Uses the new map-style MELTMapDataset + MELTDataCollator pattern
        # instead of Lhotse's DynamicBucketingSampler.
        eval_dataset = None
        self._eval_collator = None
        if (
            processor is not None
            and config is not None
            and hasattr(config, "data")
            and hasattr(config.data, "validation_ds")
            and config.data.validation_ds.input_cfg
        ):
            logger.info("Creating MELTMapDataset for evaluation...")

            # The formatting keys live at `data.`, not under `validation_ds`, so
            # eval has to inherit them or it scores a different sequence format
            # than training produced (issue #58).
            eval_data_config = resolve_eval_data_config(config.data)

            def _build(ds_config) -> MELTMapDataset:
                return MELTMapDataset(
                    cuts=materialize_cuts_for_eval(ds_config),
                    processor=processor,
                    config=ds_config,
                    is_train=False,
                    return_langs=True,
                )

            # A `name` on each validation source splits eval into separately
            # reported sets; HF prefixes each one's metrics, giving
            # eval_<name>_loss.  Unnamed sources keep the old single-set
            # behaviour, so existing configs are unaffected.
            named = split_eval_config_by_name(eval_data_config)
            if named is None:
                eval_dataset = _build(eval_data_config)
                logger.info("Eval dataset ready: %d valid cuts", len(eval_dataset))
            else:
                eval_dataset = {name: _build(sub) for name, sub in named.items()}
                logger.info(
                    "Eval datasets ready: %s",
                    ", ".join(f"{n}={len(d)} cuts" for n, d in eval_dataset.items()),
                )

            self._eval_collator = MELTDataCollator(
                processor=processor,
                config=eval_data_config,
                is_train=False,
            )

        # Sampler state restoration flag (set in train(), consumed in get_train_dataloader())
        self._lhotse_resume_from: str | None = None
        # Reference to the training dataloader (for sampler access during checkpoint saving)
        self._train_dataloader_ref: DataLoader | None = None

        # Buffer for per-sample language codes collected during evaluation.
        # Populated by prediction_step(), consumed by the compute_metrics wrapper below.
        self._eval_langs_buffer: list[str] = []

        # Prepared eval dataloaders kept alive across evaluate() calls, keyed by
        # eval-dataset name.  Only populated when persistent workers are enabled
        # (see get_eval_dataloader).
        self._prepared_eval_dataloaders: dict[str, tuple[Any, DataLoader]] = {}

        # Initialize parent (may set up distributed)
        super().__init__(model=model, args=args, eval_dataset=eval_dataset, **kwargs)

        # Wrap compute_metrics so that language codes buffered during
        # prediction_step are attached to EvalPrediction before the real
        # metric function runs.  This avoids the need to override the entire
        # evaluation_loop just for langs plumbing.
        _original_compute_metrics = self.compute_metrics
        if _original_compute_metrics is not None:

            def _compute_metrics_with_langs(
                eval_prediction: EvalPrediction, **kwargs
            ) -> dict:
                if self._eval_langs_buffer:
                    eval_prediction.langs = list(self._eval_langs_buffer)
                result = _original_compute_metrics(eval_prediction, **kwargs)
                # Clear the buffer on the *final* call (batch_eval_metrics mode
                # makes per-batch calls with compute_result=False).
                if kwargs.get("compute_result", True):
                    self._eval_langs_buffer = []
                return result

            self.compute_metrics = _compute_metrics_with_langs

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

        try:
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs,
            )
        finally:
            # Every checkpoint save, including the final one, happens inside
            # super().train(), so by here the sampler state is already on disk
            # and the workers have nothing left to report (issue #63).
            self._shutdown_dataloaders()

    # ------------------------------------------------------------------
    # Dataloader teardown
    # ------------------------------------------------------------------

    def _shutdown_dataloaders(self) -> None:
        """Stop dataloader workers and pin-memory threads, from this thread.

        Without this the run finishes its work and then aborts at exit with
        ``RuntimeError: cannot join current thread`` out of
        ``_StatefulMultiProcessingDataLoaderIter.__del__``, which one rank turns
        into ``terminate called without an active exception`` — SIGABRT, so
        SLURM records a completed run as FAILED (issue #63).

        The chain is: ``StatefulDataLoader.__iter__`` stores the iterator on the
        loader, and the loader outlives training because both this trainer and
        HF's ``callback_handler.train_dataloader`` hold it. The trainer is only
        reachable through reference cycles, so the whole chain is freed by the
        *cyclic* collector rather than by refcounting — and the cyclic collector
        runs on whichever thread trips its allocation threshold. The pin-memory
        thread is still alive and still pinning prefetched batches, so it is a
        candidate; when it wins, ``__del__`` runs there and ``_shutdown_workers``
        joins the thread it is executing on.

        Dropping our own reference is not enough, because HF's is never cleared.
        Shutting the iterator down explicitly is: it joins the pin-memory thread
        here on the main thread and sets the iterator's ``_shutdown`` flag, so
        any later ``__del__`` — on any thread — is a no-op.
        """
        _shutdown_dataloader_workers(self._train_dataloader_ref)
        self._train_dataloader_ref = None

        # Cached eval loaders are only kept when persistent workers are on, and
        # they pin memory too, so they can strand a pin-memory thread the same
        # way. Clearing the cache means a later evaluate() builds a fresh one.
        for _dataset, loader in self._prepared_eval_dataloaders.values():
            _shutdown_dataloader_workers(loader)
        self._prepared_eval_dataloaders.clear()

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
        """Create evaluation dataloader using standard PyTorch Dataset + DataLoader.

        Uses MELTMapDataset + MELTDataCollator instead of Lhotse's
        DynamicBucketingSampler.  Distributed sampling is handled by
        Accelerator via ``prepare_data_loader``.

        Args:
            eval_dataset: If provided, used instead of self.eval_dataset.

        Returns:
            DataLoader for evaluation (prepared by Accelerator for DDP).
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval dataset configured")

        if self._eval_collator is None:
            raise ValueError("No eval collator configured — was __init__ called correctly?")

        # `-1` is the "batching is handled elsewhere" sentinel used throughout our
        # YAMLs (the train path honours it in get_total_train_batch_size).  Eval
        # uses a stock DataLoader, which rejects it, so normalise here.
        batch_size = self.args.per_device_eval_batch_size
        if batch_size is not None and batch_size < 0:
            batch_size = 1

        # dataloader_num_workers is only read by this method — the train path takes
        # its worker count from data.train_ds.num_workers — so it is in effect an
        # eval-only knob and is passed through unclamped.
        num_workers = max(int(self.args.dataloader_num_workers or 0), 0)

        # prefetch_factor is per worker and MUST be None when num_workers == 0,
        # otherwise DataLoader raises.  It only smooths jitter; it cannot lift
        # steady-state throughput above what the workers produce.
        prefetch_factor = None
        persistent_workers = False
        if num_workers > 0:
            prefetch_factor = int(
                getattr(self.args, "dataloader_prefetch_factor", None) or 8
            )
            persistent_workers = bool(
                getattr(self.args, "dataloader_persistent_workers", False)
            )

        # evaluate() calls this on every eval, so without persistence the workers
        # are re-forked each time.  Cache the *prepared* loader: accelerate builds
        # a fresh DataLoader in prepare_data_loader, and re-preparing would discard
        # the live worker pool that persistent_workers exists to keep.
        cache_key = eval_dataset if isinstance(eval_dataset, str) else "eval"

        # With a dict of named eval sets, HF's evaluate() loop hands the *key*
        # back here rather than the dataset, so resolve it before use.
        if isinstance(eval_dataset, str):
            if not isinstance(self.eval_dataset, dict):
                raise ValueError(
                    f"Eval dataset {eval_dataset!r} requested by name, but "
                    "eval_dataset is not a dict of named sets."
                )
            try:
                eval_dataset = self.eval_dataset[eval_dataset]
            except KeyError:
                raise ValueError(
                    f"Unknown eval dataset {cache_key!r}. "
                    f"Available: {sorted(self.eval_dataset)}"
                ) from None

        if persistent_workers:
            cached_dataset, cached_loader = self._prepared_eval_dataloaders.get(
                cache_key, (None, None)
            )
            if cached_loader is not None and cached_dataset is eval_dataset:
                return cached_loader

        logger.info(
            "Eval dataloader: batch_size=%s num_workers=%d prefetch_factor=%s "
            "persistent_workers=%s",
            batch_size,
            num_workers,
            prefetch_factor,
            persistent_workers,
        )

        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=self._eval_collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Let Accelerator handle DistributedSampler
        prepared = self.accelerator.prepare_data_loader(dataloader)
        if persistent_workers:
            self._prepared_eval_dataloaders[cache_key] = (eval_dataset, prepared)
        return prepared

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

        # estimate_steps_per_epoch signals "could not estimate" with 0 (sources
        # read but nothing measurable) or -1 (no basis to estimate from). Both
        # are useless as a divisor below, and 0 raises ZeroDivisionError several
        # frames away from the cause, so fail here with the reason.
        if optimization_steps_per_epoch <= 0:
            raise ValueError(
                "Could not estimate optimization steps per epoch "
                f"(got {optimization_steps_per_epoch}) from "
                f"{self.dataset_num_cuts} cuts / {self.dataset_duration_hours:.2f} h. "
                "Either the configured sources hold no readable cut manifests, or "
                "neither batch_size nor batch_duration is set. Set "
                "data.train_ds.total_hours and total_cuts explicitly to skip "
                "estimation."
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
    # Lhotse dataloader state: save / restore helpers
    # ------------------------------------------------------------------

    def _save_sampler_state(self, output_dir: str) -> None:
        """Save the training dataloader state into *output_dir*/sampler/.

        Each rank saves its own file so that restoration is rank-aware.

        The payload is ``StatefulDataLoader.state_dict()``: a per-worker snapshot
        of where each worker's iterator actually is. It is collected from inside
        the workers, which is the only place that position exists once
        ``num_workers > 0``.
        """
        dataloader = self._train_dataloader_ref
        if dataloader is None:
            logger.warning("No training dataloader reference — skipping dataloader state save.")
            return

        if not hasattr(dataloader, "state_dict"):
            # Map-style/eval-shaped loaders have no resumable position.
            logger.warning(
                f"Training dataloader is a {type(dataloader).__name__}, which has no "
                "state_dict() — skipping dataloader state save. Resume will restart "
                "the data stream."
            )
            return

        sampler_dir = os.path.join(output_dir, "sampler")
        os.makedirs(sampler_dir, exist_ok=True)

        state: dict[str, Any] = {
            "dataloader_state_dict": dataloader.state_dict(),
            # Kept for debugging and for spotting a checkpoint written by a run
            # with a different parallelism layout.
            "global_step": self.state.global_step,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "num_workers": getattr(dataloader, "num_workers", 0),
            "world_size": self._world_size,
        }

        save_path = os.path.join(
            sampler_dir, f"sampler_state_rank{self._global_rank}.pt"
        )
        torch.save(state, save_path)
        logger.info(f"Saved lhotse dataloader state to {save_path}")

    def _restore_sampler_state(self, dataloader: DataLoader) -> None:
        """Load a previously saved dataloader state into *dataloader*."""
        sampler_file = self._lhotse_resume_from
        self._lhotse_resume_from = None  # consumed

        if not hasattr(dataloader, "load_state_dict"):
            logger.warning(
                f"Training dataloader is a {type(dataloader).__name__}, which has no "
                "load_state_dict() — dataloader state will NOT be restored."
            )
            return

        logger.info(f"Loading lhotse dataloader state from {sampler_file}")
        state = torch.load(sampler_file, map_location="cpu", weights_only=False)

        dl_sd = state.get("dataloader_state_dict")
        if dl_sd is None:
            logger.warning(
                "Checkpoint does not contain 'dataloader_state_dict' — it predates "
                "the StatefulDataLoader migration (issue #55). Dataloader state will "
                "NOT be restored; the data stream restarts from the beginning."
            )
            return

        # A worker's snapshot only means anything to the worker that wrote it, so
        # a changed layout silently maps state onto the wrong stream. Refuse
        # rather than resume against a corpus slice the checkpoint never saw.
        saved_workers = state.get("num_workers")
        saved_world = state.get("world_size")
        now_workers = getattr(dataloader, "num_workers", 0)
        if saved_workers is not None and saved_workers != now_workers:
            raise ValueError(
                f"Checkpoint was written with num_workers={saved_workers} but this run "
                f"has num_workers={now_workers}. Per-worker dataloader state cannot be "
                "remapped across a different worker count — resume with the original "
                "value, or delete the sampler state to restart the data stream."
            )
        if saved_world is not None and saved_world != self._world_size:
            raise ValueError(
                f"Checkpoint was written with world_size={saved_world} but this run has "
                f"world_size={self._world_size}. Indexed partitioning assigns each rank a "
                "different slice at a different world size, so the saved position refers "
                "to data this rank no longer reads."
            )

        dataloader.load_state_dict(dl_sd)
        logger.info(
            f"Restored dataloader state (num_workers={now_workers}, "
            f"world_size={self._world_size}) — each worker resumes at its own position."
        )

        # The dataloader now handles data positioning internally, so we
        # disable HF Trainer's own batch-skipping to avoid double-skipping.
        self.args.ignore_data_skip = True
        logger.info(
            "Set ignore_data_skip=True — the lhotse dataloader handles "
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
        max_text_len = int(cfg_max_tokens) + 32 if cfg_max_tokens is not None else 1024

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
        # Place the audio token in the synthetic batch so the audio-injection path
        # (and its CUDA memory) is exercised during preallocation.
        audio_token_id = model.config.audio_token_id
        input_ids             = torch.full((n_utts, max_text_len), fill_value=0, device=device, dtype=torch.long)
        input_ids[:, 1]       = audio_token_id
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

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Pop ``langs`` from inputs before delegating to the base implementation.

        Language codes are buffered on the trainer so that the
        ``compute_metrics`` wrapper can attach them to the
        :class:`EvalPrediction` later.
        """
        langs = inputs.pop("langs", None)
        if langs is not None:
            # Buffer is initialised in __init__ and cleared by the
            # compute_metrics wrapper on the final call.
            self._eval_langs_buffer.extend(langs)
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
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
