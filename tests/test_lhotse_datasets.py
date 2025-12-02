"""Test suite for Lhotse dataset loaders.

This module tests that the Lhotse dataset loaders can:
1. Load CutSets from disk correctly
2. Be iterated with a BucketingSampler
3. Provide valid audio and supervision data

To run tests:
    pytest tests/test_lhotse_datasets.py -v

Note: These tests require the Shar archives to be present on disk.
      They are integration tests that verify the full data loading pipeline.
"""

from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler

from src.data_utils.lhotse import (
    LHOTSE_DATASET_REGISTRY,
    LibriSpeechLhotse,
    get_lhotse_dataset,
    list_available_datasets,
)

# --- Configuration ---
SHAR_BASE_DIR = Path("/mnt/home/giuseppe/myscratch/melt-data/shar")
MAX_DURATION = 120.0  # seconds
NUM_BUCKETS = 10
MAX_CUTS_TO_CHECK = 100  # Limit for detailed checks to keep tests fast


class TestLhotseRegistry:
    """Tests for the dataset registry and factory function."""

    def test_list_available_datasets(self):
        """Test that list_available_datasets returns expected datasets."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "librispeech" in datasets

    def test_registry_contains_librispeech(self):
        """Test that LibriSpeech is in the registry."""
        assert "librispeech" in LHOTSE_DATASET_REGISTRY
        assert LHOTSE_DATASET_REGISTRY["librispeech"] == LibriSpeechLhotse

    def test_get_lhotse_dataset_invalid_name(self):
        """Test that invalid dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_lhotse_dataset("nonexistent_dataset", "/fake/path")


class TestLibriSpeechLoader:
    """Tests for LibriSpeech dataset loading and iteration."""

    @pytest.fixture
    def librispeech_shar_dir(self) -> Path:
        """Path to LibriSpeech Shar archives."""
        return SHAR_BASE_DIR / "librispeech"

    @pytest.fixture
    def librispeech_dataset(self, librispeech_shar_dir: Path) -> LibriSpeechLhotse:
        """Load LibriSpeech dataset loader."""
        if not librispeech_shar_dir.exists():
            pytest.skip(f"LibriSpeech Shar directory not found: {librispeech_shar_dir}")
        return get_lhotse_dataset("librispeech", str(librispeech_shar_dir))

    def test_dataset_attributes(self, librispeech_dataset: LibriSpeechLhotse):
        """Test that dataset has correct attributes."""
        assert librispeech_dataset.nickname == "librispeech"
        assert librispeech_dataset.is_multilingual is False
        assert librispeech_dataset.default_language == "en"
        assert "clean" in librispeech_dataset.supported_configs
        assert "other" in librispeech_dataset.supported_configs
        assert "train.100" in librispeech_dataset.supported_splits["clean"]
        assert "train.360" in librispeech_dataset.supported_splits["clean"]
        assert "train.500" in librispeech_dataset.supported_splits["other"]

    @pytest.mark.parametrize(
        "config,split",
        [
            ("clean", "train.100"),
            ("clean", "train.360"),
            ("other", "train.500"),
        ],
    )
    def test_load_train_splits(
        self,
        librispeech_dataset: LibriSpeechLhotse,
        config: str,
        split: str,
    ):
        """Test loading different training splits."""
        split_dir = librispeech_dataset.shar_dir / config / split
        if not split_dir.exists():
            pytest.skip(f"Split directory not found: {split_dir}")

        # Note: load_cuts() always returns a lazy CutSet from Shar archives
        cuts = librispeech_dataset.load_cuts(
            split=split,
            config=config,
            shuffle_shards=False,
        )

        assert cuts is not None
        # Check that we can iterate (lazy loading)
        first_cut = next(iter(cuts))
        assert first_cut is not None
        assert first_cut.id is not None
        assert first_cut.duration > 0

    def test_load_train_clean_100_with_bucketing_sampler(
        self,
        librispeech_dataset: LibriSpeechLhotse,
    ):
        """Test iterating over train.clean.100 with BucketingSampler."""
        split_dir = librispeech_dataset.shar_dir / "clean" / "train.100"
        if not split_dir.exists():
            pytest.skip(f"train.clean.100 not found: {split_dir}")

        # Load cuts (always lazy from Shar archives)
        cuts = librispeech_dataset.load_cuts(
            split="train.100",
            config="clean",
            shuffle_shards=True,
            seed=42,
        )

        # Create BucketingSampler
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        # Iterate through batches
        batch_count = 0
        total_cuts = 0
        total_duration = 0.0

        for batch_cuts in sampler:
            batch_count += 1
            batch_duration = sum(c.duration for c in batch_cuts)
            total_cuts += len(batch_cuts)
            total_duration += batch_duration

            # Verify batch doesn't exceed max duration
            assert (
                batch_duration <= MAX_DURATION + 30.0
            ), f"Batch {batch_count} exceeded max duration: {batch_duration:.1f}s > {MAX_DURATION}s"

            # Verify each cut in batch
            for cut in batch_cuts:
                assert cut.id is not None
                assert cut.duration > 0
                assert len(cut.supervisions) > 0
                assert cut.supervisions[0].text is not None
                assert cut.supervisions[0].language == "en"

            # Limit iteration for test speed
            if batch_count >= 10:
                break

        assert batch_count > 0, "No batches were produced by the sampler"
        assert total_cuts > 0, "No cuts were loaded"

        print(
            f"\n  Processed {batch_count} batches, {total_cuts} cuts, "
            f"{total_duration:.1f}s total duration"
        )

    def test_load_train_other_500_with_bucketing_sampler(
        self,
        librispeech_dataset: LibriSpeechLhotse,
    ):
        """Test iterating over train.other.500 with BucketingSampler."""
        split_dir = librispeech_dataset.shar_dir / "other" / "train.500"
        if not split_dir.exists():
            pytest.skip(f"train.other.500 not found: {split_dir}")

        # Load cuts (always lazy from Shar archives)
        cuts = librispeech_dataset.load_cuts(
            split="train.500",
            config="other",
            shuffle_shards=True,
            seed=42,
        )

        # Create BucketingSampler
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        # Iterate through some batches
        batch_count = 0
        for batch_cuts in sampler:
            batch_count += 1

            # Verify batch structure
            assert len(batch_cuts) > 0
            for cut in batch_cuts:
                assert cut.id is not None
                assert cut.duration > 0
                assert len(cut.supervisions) > 0

            if batch_count >= 5:
                break

        assert batch_count > 0, "No batches were produced by the sampler"

    def test_load_all_train_clean(self, librispeech_dataset: LibriSpeechLhotse):
        """Test loading combined train.100 + train.360 using load_train()."""
        # Check if both splits exist
        clean_100_dir = librispeech_dataset.shar_dir / "clean" / "train.100"
        clean_360_dir = librispeech_dataset.shar_dir / "clean" / "train.360"

        if not clean_100_dir.exists() or not clean_360_dir.exists():
            pytest.skip(
                "Both train.clean.100 and train.clean.360 required for this test"
            )

        # Load all clean training data
        cuts = librispeech_dataset.load_train(
            config="clean",
            subset="all",
            shuffle_shards=True,
            seed=42,
        )

        assert cuts is not None

        # Iterate through a few cuts to verify
        cut_count = 0
        for cut in cuts:
            cut_count += 1
            assert cut.id is not None
            assert cut.duration > 0

            if cut_count >= MAX_CUTS_TO_CHECK:
                break

        assert cut_count > 0, "No cuts loaded from combined training set"

    def test_cut_audio_loading(self, librispeech_dataset: LibriSpeechLhotse):
        """Test that audio can be loaded from cuts."""
        split_dir = librispeech_dataset.shar_dir / "clean" / "train.100"
        if not split_dir.exists():
            pytest.skip(f"train.clean.100 not found: {split_dir}")

        cuts = librispeech_dataset.load_cuts(
            split="train.100",
            config="clean",
            shuffle_shards=False,
        )

        # Load audio for a few cuts
        cuts_checked = 0
        for cut in cuts:
            # Load audio
            audio = cut.load_audio()

            # Verify audio shape
            assert audio is not None
            assert len(audio.shape) == 2  # (channels, samples)
            assert audio.shape[0] == 1  # mono
            assert audio.shape[1] > 0  # has samples

            # Verify audio length matches duration
            expected_samples = int(cut.duration * cut.sampling_rate)
            # Allow 1% tolerance for rounding
            assert abs(audio.shape[1] - expected_samples) < expected_samples * 0.01

            cuts_checked += 1
            if cuts_checked >= 5:
                break

        assert cuts_checked > 0, "No cuts were checked for audio loading"

    def test_supervision_text_content(self, librispeech_dataset: LibriSpeechLhotse):
        """Test that supervision text is valid."""
        split_dir = librispeech_dataset.shar_dir / "clean" / "train.100"
        if not split_dir.exists():
            pytest.skip(f"train.clean.100 not found: {split_dir}")

        cuts = librispeech_dataset.load_cuts(
            split="train.100",
            config="clean",
            shuffle_shards=False,
        )

        cuts_checked = 0
        for cut in cuts:
            assert (
                len(cut.supervisions) == 1
            ), "Expected exactly one supervision per cut"

            supervision = cut.supervisions[0]
            assert supervision.text is not None
            assert len(supervision.text) > 0, "Supervision text should not be empty"
            assert supervision.language == "en"
            assert supervision.duration == cut.duration
            assert supervision.start == 0.0

            cuts_checked += 1
            if cuts_checked >= MAX_CUTS_TO_CHECK:
                break

        assert cuts_checked > 0


class TestBucketingSamplerIntegration:
    """Integration tests for BucketingSampler with loaded datasets."""

    @pytest.fixture
    def librispeech_cuts(self) -> CutSet:
        """Load LibriSpeech cuts for integration tests."""
        shar_dir = SHAR_BASE_DIR / "librispeech"
        split_dir = shar_dir / "clean" / "train.100"

        if not split_dir.exists():
            pytest.skip(f"LibriSpeech train.clean.100 not found: {split_dir}")

        dataset = LibriSpeechLhotse(shar_dir)
        return dataset.load_cuts(
            split="train.100",
            config="clean",
            shuffle_shards=True,
            seed=42,
        )

    def test_sampler_batch_duration_constraint(self, librispeech_cuts: CutSet):
        """Test that sampler respects max_duration constraint."""
        sampler = DynamicBucketingSampler(
            librispeech_cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        batch_durations = []
        for i, batch in enumerate(sampler):
            duration = sum(c.duration for c in batch)
            batch_durations.append(duration)

            # Allow some tolerance (last cut might exceed slightly)
            assert duration <= MAX_DURATION + 30.0

            if i >= 20:
                break

        # Verify batches are reasonably filled
        avg_duration = sum(batch_durations) / len(batch_durations)
        assert avg_duration > MAX_DURATION * 0.5, (
            f"Average batch duration too low: {avg_duration:.1f}s "
            f"(expected > {MAX_DURATION * 0.5:.1f}s)"
        )

    def test_sampler_reproducibility(self, librispeech_cuts: CutSet):
        """Test that sampler produces reproducible results with same seed."""
        sampler1 = DynamicBucketingSampler(
            librispeech_cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        sampler2 = DynamicBucketingSampler(
            librispeech_cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        # Compare first few batches
        for i, (batch1, batch2) in enumerate(zip(sampler1, sampler2)):
            ids1 = [c.id for c in batch1]
            ids2 = [c.id for c in batch2]
            assert ids1 == ids2, f"Batch {i} differs between samplers"

            if i >= 5:
                break


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
