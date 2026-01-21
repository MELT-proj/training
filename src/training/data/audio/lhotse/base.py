"""Base class for Lhotse dataset loaders.

This module provides the abstract base class that all Lhotse dataset loaders
should inherit from, ensuring a consistent interface for loading CutSets.

Directory Structure Convention:
    - Monolingual datasets (is_multilingual=False):
        {shar_dir}/{config}/{split}/
        Example: peoples_speech/clean/train/

    - Multilingual datasets (is_multilingual=True):
        {shar_dir}/{lang}/{split}/           (if no configs)
        {shar_dir}/{config}/{lang}/{split}/  (if has configs)
        Example: common_voice/en/train/ or mls/default/en/train/
"""

from abc import ABC
from pathlib import Path
from typing import Optional, Union

from lhotse import CutSet

from .....logging_utils import get_logger

logger = get_logger(__name__)


class BaseLhotseDataset(ABC):
    """Abstract base class for loading Lhotse CutSets from disk.

    All dataset-specific loaders should inherit from this class and implement
    the required class attributes and methods.

    Class Attributes:
        nickname: A short identifier for the dataset (e.g., "peoples_speech").
        is_multilingual: Whether the dataset supports multiple languages.
        supported_languages: List of supported language codes (ISO 639-1).
        supported_configs: List of available configurations/subsets (can be None).
        supported_splits: Dictionary mapping configs (or "default") to their splits.
        default_language: Default language for monolingual datasets.
        task: The task this dataset is intended for (e.g., "transcribe").
    """

    nickname: Optional[str] = None
    is_multilingual: bool = False
    supported_languages: Optional[list[str]] = None
    supported_configs: Optional[list[str]] = None
    supported_splits: Optional[dict[str, list[str]]] = None
    default_language: Optional[str] = None
    task: str = "transcribe"

    def __init__(self, shar_dir: Union[str, Path]):
        """Initialize the dataset loader.

        Args:
            shar_dir: Base directory containing the Shar archives for this dataset.
        """
        self.shar_dir = Path(shar_dir)
        if not self.shar_dir.exists():
            raise FileNotFoundError(f"Shar directory not found: {self.shar_dir}")

    def _get_split_dir(
        self,
        split: str,
        lang: Optional[str] = None,
        config: Optional[str] = None,
    ) -> Path:
        """Get the directory path for a specific config/lang/split combination.

        Directory structure depends on dataset type:
            - Monolingual: {shar_dir}/{config}/{split}/
            - Multilingual without configs: {shar_dir}/{lang}/{split}/
            - Multilingual with configs: {shar_dir}/{config}/{lang}/{split}/
        """
        if self.is_multilingual:
            if lang is None:
                raise ValueError(
                    f"Language must be specified for multilingual dataset {self.nickname}"
                )
            if config is not None:
                return self.shar_dir / config / lang / split
            return self.shar_dir / lang / split
        else:
            if config is not None:
                return self.shar_dir / config / split
            return self.shar_dir / split

    def _validate_params(
        self,
        split: str,
        lang: Optional[str] = None,
        config: Optional[str] = None,
    ) -> None:
        """Validate that the config, lang, and split are supported."""
        # Validate language for multilingual datasets
        if self.is_multilingual:
            if lang is None:
                raise ValueError(
                    f"Language must be specified for multilingual dataset {self.nickname}. "
                    f"Available languages: {self.supported_languages}"
                )
            if (
                self.supported_languages is not None
                and lang not in self.supported_languages
            ):
                raise ValueError(
                    f"Language '{lang}' not supported for {self.nickname}. "
                    f"Available languages: {self.supported_languages}"
                )

        # Validate config if provided and supported_configs is defined
        if self.supported_configs is not None:
            if config is None:
                raise ValueError(
                    f"Config must be specified for {self.nickname}. "
                    f"Available configs: {self.supported_configs}"
                )
            if config not in self.supported_configs:
                raise ValueError(
                    f"Config '{config}' not supported for {self.nickname}. "
                    f"Available configs: {self.supported_configs}"
                )
            splits_key = config
        else:
            splits_key = "default"

        # Validate split
        if self.supported_splits is None:
            raise ValueError(f"No supported splits defined for {self.nickname}")
        available_splits = self.supported_splits.get(splits_key, [])
        if split not in available_splits:
            raise ValueError(
                f"Split '{split}' not supported for {self.nickname} "
                f"(config={config}). Available splits: {available_splits}"
            )

    def load_cuts(
        self,
        split: str,
        lang: Optional[str] = None,
        config: Optional[str] = None,
        shuffle_shards: bool = False,
        seed: int = 42,
    ) -> CutSet:
        """Load a CutSet for the specified configuration, language, and split.

        Note: CutSet.from_shar() always returns a lazy CutSet that streams data
        on-demand via LazySharIterator. This is memory-efficient for large datasets.

        Args:
            split: The data split (e.g., "train", "validation", "test").
            lang: Language code (required for multilingual datasets).
            config: The dataset configuration (e.g., "clean", "dirty").
            shuffle_shards: Whether to shuffle the shards (not individual cuts).
                           For per-cut shuffling, use .shuffle() on the returned CutSet.
            seed: Random seed for shard shuffling.

        Returns:
            A lazy CutSet that loads data on-demand from Shar archives.
        """
        self._validate_params(split, lang, config)
        split_dir = self._get_split_dir(split, lang, config)

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Make sure the data has been prepared."
            )

        logger.info(f"Loading CutSet from {split_dir}")
        cuts = CutSet.from_shar(
            in_dir=split_dir,
            shuffle_shards=shuffle_shards,
            seed=seed,
        )

        return cuts

    @classmethod
    def get_available_configs(cls) -> Optional[list[str]]:
        """Return a list of available configurations, or None if not applicable."""
        return cls.supported_configs

    @classmethod
    def get_available_languages(cls) -> list[str]:
        """Return a list of available languages."""
        if cls.supported_languages is not None:
            return cls.supported_languages
        if cls.default_language is not None:
            return [cls.default_language]
        return []

    @classmethod
    def get_available_splits(cls, config: Optional[str] = None) -> list[str]:
        """Return a list of available splits for a given configuration."""
        if cls.supported_splits is None:
            return []
        if cls.supported_configs is not None:
            if config is None:
                raise ValueError(
                    f"Config must be specified. Available configs: {cls.supported_configs}"
                )
            if config not in cls.supported_configs:
                raise ValueError(
                    f"Config '{config}' not supported. "
                    f"Available configs: {cls.supported_configs}"
                )
            return cls.supported_splits.get(config, [])
        return cls.supported_splits.get("default", [])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nickname='{self.nickname}', "
            f"shar_dir='{self.shar_dir}')"
        )
