"""Lhotse dataset loaders module.

This module provides a unified interface for loading Lhotse CutSets
from prepared Shar archives on disk.

Directory Structure Convention:
    The Shar archives follow different structures depending on the dataset type:

    - Monolingual datasets (e.g., People's Speech):
        {shar_dir}/{config}/{split}/
        Example: peoples_speech/clean/train/

    - Multilingual datasets without configs (e.g., Common Voice):
        {shar_dir}/{lang}/{split}/
        Example: common_voice/en/train/

    - Multilingual datasets with configs:
        {shar_dir}/{config}/{lang}/{split}/
        Example: mls/default/en/train/

Example usage:
    >>> from src.data.audio.lhotse import (
    ...     SpeechToTextDataset,
    ...     get_lhotse_dataloader_from_config,
    ... )
    >>> processor = MELTProcessor(feature_extractor, tokenizer)
    >>> dataset = SpeechToTextDataset(processor, data_config)
    >>> dataloader = get_lhotse_dataloader_from_config(
    ...     config=data_config.train_ds,
    ...     global_rank=0,
    ...     world_size=1,
    ...     dataset=dataset,
    ... )
"""

from .base import BaseLhotseDataset
from .collator import MELTDataCollator
from .dataloader import (
    # FiniteIterableDatasetWrapper,
    compute_dataset_duration,
    create_eval_dataloader,
    # estimate_num_batches,
    estimate_steps_per_epoch,
    get_eval_dataloader_from_config,
    # get_eval_sampler_from_config,
    # get_finite_dataloader_from_config,
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
    get_train_dataloader_from_config,
    materialize_cuts_for_eval,
    read_cutset_from_config,
)
from .dataset import FallbackDataset, SpeechTextQEDataset, SpeechToTextDataset
from .map_dataset import MELTMapDataset

__all__ = [
    # Base classes
    "BaseLhotseDataset",
    # Dataset classes
    "SpeechToTextDataset",
    "SpeechTextQEDataset",
    "FallbackDataset",
    "MELTMapDataset",
    # Collator
    "MELTDataCollator",
    # Sampler/dataloader functions
    "read_cutset_from_config",
    "get_lhotse_sampler_from_config",
    "get_eval_sampler_from_config",
    "get_lhotse_dataloader_from_config",
    "get_train_dataloader_from_config",
    "get_eval_dataloader_from_config",
    "get_finite_dataloader_from_config",
    "FiniteIterableDatasetWrapper",
    "materialize_cuts_for_eval",
    "create_eval_dataloader",
    # Epoch estimation utilities
    "compute_dataset_duration",
    "estimate_steps_per_epoch",
    "estimate_num_batches",
]
