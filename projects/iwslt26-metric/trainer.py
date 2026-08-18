"""Regression trainer for IWSLT 2026 speech translation metric task.

This module provides MELTTrainerForRegression, a subclass of MELTTrainer
that swaps the SpeechToTextDataset for SpeechTextQEDataset so that the
model is trained with scalar quality scores as labels rather than token
sequences.

Only the ``__init__``, ``get_train_dataloader``, and ``get_eval_dataloader``
methods are overridden; all other training logic is inherited unchanged from
MELTTrainer.
"""

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import EvalLoopOutput

from melt.training.data.audio.lhotse import (
    FallbackDataset,
    SpeechTextQEDataset,
    get_eval_dataloader_from_config,
    get_train_dataloader_from_config,
)
from melt.logging_utils import get_logger
from melt.modeling import MELTProcessor
from melt.training.trainer import MELTTrainer

logger = get_logger(__name__)


class MELTTrainerForRegression(MELTTrainer):
    """MELTTrainer variant that uses SpeechTextQEDataset for regression training.

    Inherits all training logic from MELTTrainer. The only difference is that
    the training and evaluation datasets are instances of SpeechTextQEDataset,
    which produces scalar float labels (quality scores in [0, 1]) instead of
    token-sequence labels.

    Args:
        model: The model to train.
        args: HuggingFace Seq2SeqTrainingArguments.
        config: OmegaConf DictConfig with Lhotse data loading settings.
        processor: MELTProcessor for audio processing.
        **kwargs: Additional arguments forwarded to MELTTrainer / Trainer.
    """

    def __init__(
        self,
        model,
        args: Seq2SeqTrainingArguments,
        config: DictConfig,
        processor: MELTProcessor,
        **kwargs,
    ):
        # We need to build the eval dataset with SpeechTextQEDataset before
        # MELTTrainer.__init__ runs (it passes eval_dataset to super().__init__).
        # To do so we temporarily monkey-patch the eval dataset construction
        # by calling MELTTrainer.__init__ and then rebuilding where needed.
        #
        # The cleanest approach is to reproduce only the __init__ body here,
        # replacing the dataset class, and then delegate to Trainer.__init__
        # directly (skipping MELTTrainer.__init__ entirely).

        from transformers import Trainer
        from melt import ddp

        # Mirror MELTTrainer.__init__ preamble
        self.config = config
        self.processor = processor

        self._global_rank = ddp.get_global_rank()
        self._world_size = ddp.get_world_size()

        self.steps_per_epoch = -1
        self.dataset_duration_hours = 0.0
        self.dataset_num_cuts = 0
        self.eval_num_cuts = 0
        self.eval_num_batches = 0

        # Build eval dataset using SpeechTextQEDataset
        eval_dataset = None
        if (
            processor is not None
            and config is not None
            and hasattr(config, "data")
            and hasattr(config.data, "validation_ds")
            and config.data.validation_ds.input_cfg
        ):
            logger.info("Creating evaluation SpeechTextQEDataset...")
            eval_dataset = SpeechTextQEDataset(
                processor=processor,
                config=config.data,
                is_train=False,
                return_labels=True,
            )
            eval_dataset = FallbackDataset(eval_dataset)
            logger.info(f"Eval QE dataset ready ({self.eval_num_cuts} cuts)")

        # Initialise attributes that MELTTrainer.__init__ would normally set but
        # that we must set ourselves because we skip it and call Trainer.__init__
        # directly.
        self._lhotse_resume_from: str | None = None
        self._train_dataloader_ref = None

        # Skip MELTTrainer.__init__ and call Trainer.__init__ directly so that
        # we supply our own eval_dataset without the parent re-creating it.
        Trainer.__init__(self, model=model, args=args, eval_dataset=eval_dataset, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader backed by SpeechTextQEDataset.

        Returns:
            DataLoader configured with Lhotse sampler and QE scalar labels.
        """
        if self.processor is None:
            raise ValueError("processor must be provided for Lhotse data loading")

        logger.info("Creating Lhotse training dataloader (QE regression)")

        dataset = SpeechTextQEDataset(
            processor=self.processor,
            config=self.config.data,
            is_train=True,
            return_labels=True,
        )
        dataset = FallbackDataset(dataset)

        dataloader = get_train_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )

        # Mirror MELTTrainer.get_train_dataloader: keep a reference for sampler
        # state saving during checkpointing, and restore state on resume.
        self._train_dataloader_ref = dataloader
        if self._lhotse_resume_from is not None:
            self._restore_sampler_state(dataloader)

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create evaluation dataloader backed by SpeechTextQEDataset.

        Args:
            eval_dataset: Ignored; the dataset is taken from ``self.eval_dataset``
                or created fresh from config.

        Returns:
            DataLoader for evaluation.
        """
        if self.processor is None:
            raise ValueError("processor must be provided for Lhotse data loading")

        if not self.config.data.validation_ds.input_cfg:
            raise ValueError(
                "No validation data configured (validation_ds.input_cfg is empty)"
            )

        logger.info("Creating Lhotse evaluation dataloader (QE regression)")

        dataset = self.eval_dataset if self.eval_dataset is not None else eval_dataset
        if dataset is None:
            dataset = SpeechTextQEDataset(
                processor=self.processor,
                config=self.config.data,
                is_train=False,
                return_labels=True,
            )
            dataset = FallbackDataset(dataset)

        dataloader = get_eval_dataloader_from_config(
            data_config=self.config.data,
            dataset=dataset,
            global_rank=self._global_rank,
            world_size=self._world_size,
        )
        return dataloader

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Extend parent evaluation loop to log label and prediction distributions to wandb."""
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if not self.is_world_process_zero():
            return output

        try:
            import wandb
        except ImportError:
            return output

        if wandb.run is None:
            return output

        preds = output.predictions
        labels = output.label_ids

        if preds is None and labels is None:
            return output

        log_payload: dict = {}

        if labels is not None:
            labels_flat = np.array(labels).flatten().tolist()
            log_payload[f"{metric_key_prefix}/label_distribution"] = wandb.Histogram(labels_flat)
            log_payload[f"{metric_key_prefix}/label_table"] = wandb.Table(
                columns=["label"],
                data=[[v] for v in labels_flat],
            )

        if preds is not None:
            preds_flat = np.array(preds).flatten().tolist()
            log_payload[f"{metric_key_prefix}/pred_distribution"] = wandb.Histogram(preds_flat)
            log_payload[f"{metric_key_prefix}/pred_table"] = wandb.Table(
                columns=["score"],
                data=[[v] for v in preds_flat],
            )

        wandb.log(log_payload, step=self.state.global_step)

        return output
