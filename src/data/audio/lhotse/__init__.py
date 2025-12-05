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
    >>> from src.data_utils.lhotse import get_lhotse_dataset, PeoplesSpeechLhotse
    >>>
    >>> # Monolingual dataset (People's Speech)
    >>> dataset = get_lhotse_dataset("peoples_speech", "/path/to/shar/peoples_speech")
    >>> cuts = dataset.load_cuts(split="train", config="clean")
    >>>
    >>> # Or using the class directly with convenience methods
    >>> dataset = PeoplesSpeechLhotse("/path/to/shar/peoples_speech")
    >>> cuts = dataset.load_train(config="clean", lazy=True)
    >>>
    >>> # Multilingual dataset (hypothetical example)
    >>> dataset = get_lhotse_dataset("common_voice", "/path/to/shar/common_voice")
    >>> cuts = dataset.load_cuts(split="train", lang="en")

Adding a new dataset:
    1. Create a new file: src/data_utils/lhotse/{dataset_name}.py
    2. Implement a class inheriting from BaseLhotseDataset
    3. Set is_multilingual=True/False and supported_languages accordingly
    4. Register it in LHOTSE_DATASET_REGISTRY below
    5. Add it to __all__ and import it
"""

from .base import BaseLhotseDataset
from .librispeech import LibriSpeechLhotse
from .peoples_speech import PeoplesSpeechLhotse

__all__ = [
    "BaseLhotseDataset",
    "LibriSpeechLhotse",
    "PeoplesSpeechLhotse",
    "get_lhotse_dataset",
    "list_available_datasets",
    "LHOTSE_DATASET_REGISTRY",
]

# Registry mapping dataset nicknames to their loader classes
LHOTSE_DATASET_REGISTRY: dict[str, type[BaseLhotseDataset]] = {
    "librispeech": LibriSpeechLhotse,
    "peoples_speech": PeoplesSpeechLhotse,
}


def get_lhotse_dataset(dataset_name: str, shar_dir: str) -> BaseLhotseDataset:
    """Factory function to get a Lhotse dataset loader by name.

    Args:
        dataset_name: The nickname of the dataset (e.g., "peoples_speech").
        shar_dir: Path to the Shar archive directory for this dataset.

    Returns:
        An instance of the appropriate dataset loader class.

    Raises:
        ValueError: If the dataset name is not recognized.

    Example:
        >>> dataset = get_lhotse_dataset("peoples_speech", "/data/shar/peoples_speech")
        >>> cuts = dataset.load_cuts(split="train", config="clean")
    """
    if dataset_name not in LHOTSE_DATASET_REGISTRY:
        available = list(LHOTSE_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )

    dataset_class = LHOTSE_DATASET_REGISTRY[dataset_name]
    return dataset_class(shar_dir)


def list_available_datasets() -> list[str]:
    """Return a list of all available Lhotse dataset loaders.

    Returns:
        List of dataset nicknames that can be used with get_lhotse_dataset().
    """
    return list(LHOTSE_DATASET_REGISTRY.keys())
