"""Lhotse dataset loader for People's Speech.

People's Speech is a large-scale English speech recognition dataset
from MLCommons, available under CC-BY-SA and CC-BY 4.0 licenses.

Reference: https://huggingface.co/datasets/MLCommons/peoples_speech

Directory Structure:
    {shar_dir}/{config}/{split}/
    Example: peoples_speech/clean/train/

Note: This is a monolingual (English-only) dataset, so no language
parameter is needed when loading cuts.
"""

from pathlib import Path
from typing import Union

from .base import BaseLhotseDataset


class PeoplesSpeechLhotse(BaseLhotseDataset):
    """Lhotse CutSet loader for the People's Speech dataset.

    The People's Speech Dataset is among the world's largest English speech
    recognition corpus with 30,000+ hours of transcribed speech.

    Available configurations:
        - clean: CC-BY licensed clean data
        - dirty: CC-BY licensed noisy data
        - clean_sa: CC-BY-SA licensed clean data
        - dirty_sa: CC-BY-SA licensed noisy data
        - microset: Small subset for testing

    Example:
        >>> from src.data_utils.lhotse import PeoplesSpeechLhotse
        >>> dataset = PeoplesSpeechLhotse("/path/to/shar/peoples_speech")
        >>> cuts = dataset.load_cuts(split="train", config="clean")
        >>> for cut in cuts:
        ...     print(cut.id, cut.supervisions[0].text)
    """

    nickname = "peoples_speech"
    is_multilingual = False
    default_language = "en"
    supported_languages = ["en"]
    task = "transcribe"

    supported_configs = ["clean", "dirty", "clean_sa", "dirty_sa", "microset"]

    supported_splits = {
        "clean": ["train", "validation", "test"],
        "dirty": ["train", "validation", "test"],
        "clean_sa": ["train", "validation", "test"],
        "dirty_sa": ["train", "validation", "test"],
        "microset": ["train"],
    }

    def __init__(self, shar_dir: Union[str, Path]):
        """Initialize the People's Speech dataset loader.

        Args:
            shar_dir: Base directory containing the Shar archives.
                      Expected structure: {shar_dir}/{config}/{split}/
                      e.g., /data/shar/peoples_speech/clean/train/
        """
        super().__init__(shar_dir)

    def load_train(
        self,
        config: str = "clean",
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """Convenience method to load training data.

        Args:
            config: Dataset configuration to use.
            shuffle_shards: Whether to shuffle the shards.
            seed: Random seed for shuffling.

        Returns:
            CutSet containing training cuts (always lazy).
        """
        return self.load_cuts(
            split="train",
            config=config,
            shuffle_shards=shuffle_shards,
            seed=seed,
        )

    def load_validation(
        self,
        config: str = "clean",
        shuffle_shards: bool = False,
        seed: int = 42,
    ):
        """Convenience method to load validation data.

        Args:
            config: Dataset configuration to use.
            shuffle_shards: Whether to shuffle the shards.
            seed: Random seed for shuffling.

        Returns:
            CutSet containing validation cuts (always lazy).
        """
        return self.load_cuts(
            split="validation",
            config=config,
            shuffle_shards=shuffle_shards,
            seed=seed,
        )

    def load_test(
        self,
        config: str = "clean",
        shuffle_shards: bool = False,
        seed: int = 42,
    ):
        """Convenience method to load test data.

        Args:
            config: Dataset configuration to use.
            shuffle_shards: Whether to shuffle the shards.
            seed: Random seed for shuffling.

        Returns:
            CutSet containing test cuts (always lazy).
        """
        return self.load_cuts(
            split="test",
            config=config,
            shuffle_shards=shuffle_shards,
            seed=seed,
        )
