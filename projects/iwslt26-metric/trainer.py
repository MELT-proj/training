"""Regression trainer for IWSLT 2026 speech translation metric task.

This module provides MELTTrainerForRegression, a subclass of MELTTrainer
that swaps the SpeechToTextDataset for SpeechTextQEDataset so that the
model is trained with scalar quality scores as labels rather than token
sequences.

Only the ``__init__``, ``get_train_dataloader``, and ``get_eval_dataloader``
methods are overridden; all other training logic is inherited unchanged from
MELTTrainer.
"""

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import TrainingArguments

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
        args: HuggingFace TrainingArguments.
        config: OmegaConf DictConfig with Lhotse data loading settings.
        processor: MELTProcessor for audio processing.
        **kwargs: Additional arguments forwarded to MELTTrainer / Trainer.
    """

    def __init__(
        self,
        model,
        args: TrainingArguments,
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
