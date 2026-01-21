"""Lhotse dataset loader for LibriSpeech.

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech,
derived from audiobooks from the LibriVox project.

Reference: https://huggingface.co/datasets/openslr/librispeech_asr

Directory Structure:
    {shar_dir}/{config}/{split}/
    Example: librispeech/clean/train.100/

Note: This is a monolingual (English-only) dataset, so no language
parameter is needed when loading cuts.

The dataset is split by quality level:
    - clean: Higher quality recordings (clearer speech, less noise)
    - other: More challenging recordings (varied accents, recording quality)
"""

from pathlib import Path
from typing import Union

from .base import BaseLhotseDataset


class LibriSpeechLhotse(BaseLhotseDataset):
    """Lhotse CutSet loader for the LibriSpeech dataset.

    LibriSpeech contains ~1000 hours of English read speech at 16kHz,
    carefully segmented and aligned from LibriVox audiobooks.

    Available configurations:
        - clean: ~460 hours of higher quality recordings
            - train.100: 100 hours of training data
            - train.360: 360 hours of training data
            - validation: Validation set
            - test: Test set
        - other: ~500 hours of more challenging recordings
            - train.500: 500 hours of training data
            - validation: Validation set
            - test: Test set

    Example:
        >>> from src.data_utils.lhotse import LibriSpeechLhotse
        >>> dataset = LibriSpeechLhotse("/path/to/shar/librispeech")
        >>> cuts = dataset.load_cuts(split="train.100", config="clean")
        >>> for cut in cuts:
        ...     print(cut.id, cut.supervisions[0].text)
    """

    nickname = "librispeech"
    is_multilingual = False
    default_language = "en"
    supported_languages = ["en"]
    task = "transcribe"

    supported_configs = ["clean", "other"]

    supported_splits = {
        "clean": ["train.100", "train.360", "validation", "test"],
        "other": ["train.500", "validation", "test"],
    }

    def __init__(self, shar_dir: Union[str, Path]):
        """Initialize the LibriSpeech dataset loader.

        Args:
            shar_dir: Base directory containing the Shar archives.
                      Expected structure: {shar_dir}/{config}/{split}/
                      e.g., /data/shar/librispeech/clean/train.100/
        """
        super().__init__(shar_dir)

    def load_train(
        self,
        config: str = "clean",
        subset: str = "all",
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """Convenience method to load training data.

        Note: CutSet.from_shar() always returns a lazy CutSet that streams
        data on-demand. This is memory-efficient for large training sets.

        Args:
            config: Dataset configuration ("clean" or "other").
            subset: Which training subset to load:
                - For "clean": "100", "360", or "all" (combines both)
                - For "other": "500" or "all"
            shuffle_shards: Whether to shuffle the shards.
            seed: Random seed for shuffling.

        Returns:
            CutSet containing training cuts (always lazy).
        """
        from lhotse import CutSet

        if config == "clean":
            if subset == "100":
                return self.load_cuts(
                    split="train.100",
                    config=config,
                    shuffle_shards=shuffle_shards,
                    seed=seed,
                )
            elif subset == "360":
                return self.load_cuts(
                    split="train.360",
                    config=config,
                    shuffle_shards=shuffle_shards,
                    seed=seed,
                )
            elif subset == "all":
                cuts_100 = self.load_cuts(
                    split="train.100", config=config, shuffle_shards=False, seed=seed
                )
                cuts_360 = self.load_cuts(
                    split="train.360", config=config, shuffle_shards=False, seed=seed
                )
                combined = CutSet.mux(cuts_100, cuts_360)
                return combined
            else:
                raise ValueError(
                    f"Invalid subset '{subset}' for config 'clean'. "
                    f"Use '100', '360', or 'all'."
                )
        elif config == "other":
            if subset in ("500", "all"):
                return self.load_cuts(
                    split="train.500",
                    config=config,
                    shuffle_shards=shuffle_shards,
                    seed=seed,
                )
            else:
                raise ValueError(
                    f"Invalid subset '{subset}' for config 'other'. "
                    f"Use '500' or 'all'."
                )
        else:
            raise ValueError(f"Invalid config '{config}'. Use 'clean' or 'other'.")

    def load_validation(
        self,
        config: str = "clean",
        shuffle_shards: bool = False,
        seed: int = 42,
    ):
        """Convenience method to load validation data.

        Args:
            config: Dataset configuration ("clean" or "other").
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
            config: Dataset configuration ("clean" or "other").
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

    def load_all_train(
        self,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """Load all training data from both clean and other configurations.

        This combines train.100 + train.360 (clean) + train.500 (other)
        for a total of ~960 hours of training data.

        Args:
            shuffle_shards: Whether to shuffle the shards.
            seed: Random seed for shuffling.

        Returns:
            CutSet containing all training cuts (~960 hours, always lazy).
        """
        from lhotse import CutSet

        cuts_clean_100 = self.load_cuts(
            split="train.100", config="clean", shuffle_shards=False, seed=seed
        )
        cuts_clean_360 = self.load_cuts(
            split="train.360", config="clean", shuffle_shards=False, seed=seed
        )
        cuts_other_500 = self.load_cuts(
            split="train.500", config="other", shuffle_shards=False, seed=seed
        )

        combined = CutSet.mux(cuts_clean_100, cuts_clean_360, cuts_other_500)
        return combined
