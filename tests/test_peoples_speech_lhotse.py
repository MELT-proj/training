"""Test suite for People's Speech Lhotse dataset loader.

This module tests that the People's Speech dataset loader can:
1. Load CutSets from disk correctly
2. Be iterated with a BucketingSampler
3. Provide valid audio and supervision data

To run tests:
    pytest tests/test_peoples_speech_lhotse.py -v

Note: These tests require the Shar archives to be present on disk.
      They are integration tests that verify the full data loading pipeline.
"""

import sys
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler

sys.path.append(".")

from src.data_utils.lhotse import (
    LHOTSE_DATASET_REGISTRY,
    PeoplesSpeechLhotse,
    get_lhotse_dataset,
    list_available_datasets,
)

# --- Configuration ---
SHAR_BASE_DIR = Path("/mnt/home/giuseppe/myscratch/melt-data/shar")
MAX_DURATION = 120.0  # seconds
NUM_BUCKETS = 10
MAX_CUTS_TO_CHECK = 100  # Limit for detailed checks to keep tests fast


class TestPeoplesSpeechRegistry:
    """Tests for the People's Speech registry entry."""

    def test_list_available_datasets_includes_peoples_speech(self):
        """Test that list_available_datasets includes peoples_speech."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)
        assert "peoples_speech" in datasets

    def test_registry_contains_peoples_speech(self):
        """Test that People's Speech is in the registry."""
        assert "peoples_speech" in LHOTSE_DATASET_REGISTRY
        assert LHOTSE_DATASET_REGISTRY["peoples_speech"] == PeoplesSpeechLhotse


class TestPeoplesSpeechLoader:
    """Tests for People's Speech dataset loading and iteration."""

    @pytest.fixture
    def peoples_speech_shar_dir(self) -> Path:
        """Path to People's Speech Shar archives."""
        return SHAR_BASE_DIR / "peoples_speech"

    @pytest.fixture
    def peoples_speech_dataset(
        self, peoples_speech_shar_dir: Path
    ) -> PeoplesSpeechLhotse:
        """Load People's Speech dataset loader."""
        if not peoples_speech_shar_dir.exists():
            pytest.skip(
                f"People's Speech Shar directory not found: {peoples_speech_shar_dir}"
            )
        return get_lhotse_dataset("peoples_speech", str(peoples_speech_shar_dir))

    def test_dataset_attributes(self, peoples_speech_dataset: PeoplesSpeechLhotse):
        """Test that dataset has correct attributes."""
        assert peoples_speech_dataset.nickname == "peoples_speech"
        assert peoples_speech_dataset.is_multilingual is False
        assert peoples_speech_dataset.default_language == "en"
        assert "clean" in peoples_speech_dataset.supported_configs
        assert "dirty" in peoples_speech_dataset.supported_configs
        assert "clean_sa" in peoples_speech_dataset.supported_configs
        assert "dirty_sa" in peoples_speech_dataset.supported_configs
        assert "microset" in peoples_speech_dataset.supported_configs
        assert "train" in peoples_speech_dataset.supported_splits["clean"]
        assert "validation" in peoples_speech_dataset.supported_splits["clean"]
        assert "test" in peoples_speech_dataset.supported_splits["clean"]

    @pytest.mark.parametrize(
        "config,split",
        [
            ("clean", "train"),
            ("clean", "validation"),
            ("clean", "test"),
            ("microset", "train"),
        ],
    )
    def test_load_splits(
        self,
        peoples_speech_dataset: PeoplesSpeechLhotse,
        config: str,
        split: str,
    ):
        """Test loading different splits."""
        split_dir = peoples_speech_dataset.shar_dir / config / split
        if not split_dir.exists():
            pytest.skip(f"Split directory not found: {split_dir}")

        # Note: load_cuts() always returns a lazy CutSet from Shar archives
        cuts = peoples_speech_dataset.load_cuts(
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

    def test_load_train_clean_with_bucketing_sampler(
        self,
        peoples_speech_dataset: PeoplesSpeechLhotse,
    ):
        """Test iterating over clean/train with BucketingSampler."""
        split_dir = peoples_speech_dataset.shar_dir / "clean" / "train"
        if not split_dir.exists():
            pytest.skip(f"clean/train not found: {split_dir}")

        # Load cuts (always lazy from Shar archives)
        cuts = peoples_speech_dataset.load_cuts(
            split="train",
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

            # Verify batch doesn't exceed max duration (with tolerance)
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

    def test_load_validation_clean(
        self,
        peoples_speech_dataset: PeoplesSpeechLhotse,
    ):
        """Test loading validation split with convenience method."""
        split_dir = peoples_speech_dataset.shar_dir / "clean" / "validation"
        if not split_dir.exists():
            pytest.skip(f"clean/validation not found: {split_dir}")

        cuts = peoples_speech_dataset.load_validation(
            config="clean",
            shuffle_shards=False,
        )

        assert cuts is not None

        # Iterate through a few cuts
        cut_count = 0
        for cut in cuts:
            cut_count += 1
            assert cut.id is not None
            assert cut.duration > 0
            assert len(cut.supervisions) > 0

            if cut_count >= MAX_CUTS_TO_CHECK:
                break

        assert cut_count > 0, "No cuts loaded from validation set"

    def test_load_test_clean(
        self,
        peoples_speech_dataset: PeoplesSpeechLhotse,
    ):
        """Test loading test split with convenience method."""
        split_dir = peoples_speech_dataset.shar_dir / "clean" / "test"
        if not split_dir.exists():
            pytest.skip(f"clean/test not found: {split_dir}")

        cuts = peoples_speech_dataset.load_test(
            config="clean",
            shuffle_shards=False,
        )

        assert cuts is not None

        # Iterate through a few cuts
        cut_count = 0
        for cut in cuts:
            cut_count += 1
            assert cut.id is not None
            assert cut.duration > 0

            if cut_count >= MAX_CUTS_TO_CHECK:
                break

        assert cut_count > 0, "No cuts loaded from test set"

    def test_cut_audio_loading(self, peoples_speech_dataset: PeoplesSpeechLhotse):
        """Test that audio can be loaded from cuts."""
        split_dir = peoples_speech_dataset.shar_dir / "clean" / "train"
        if not split_dir.exists():
            pytest.skip(f"clean/train not found: {split_dir}")

        cuts = peoples_speech_dataset.load_cuts(
            split="train",
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

    def test_supervision_text_content(
        self, peoples_speech_dataset: PeoplesSpeechLhotse
    ):
        """Test that supervision text is valid."""
        split_dir = peoples_speech_dataset.shar_dir / "clean" / "train"
        if not split_dir.exists():
            pytest.skip(f"clean/train not found: {split_dir}")

        cuts = peoples_speech_dataset.load_cuts(
            split="train",
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


class TestPeoplesSpeechBucketingSamplerIntegration:
    """Integration tests for BucketingSampler with People's Speech."""

    @pytest.fixture
    def peoples_speech_cuts(self) -> CutSet:
        """Load People's Speech cuts for integration tests."""
        shar_dir = SHAR_BASE_DIR / "peoples_speech"
        split_dir = shar_dir / "clean" / "train"

        if not split_dir.exists():
            pytest.skip(f"People's Speech clean/train not found: {split_dir}")

        dataset = PeoplesSpeechLhotse(shar_dir)
        return dataset.load_cuts(
            split="train",
            config="clean",
            shuffle_shards=True,
            seed=42,
        )

    def test_sampler_batch_duration_constraint(self, peoples_speech_cuts: CutSet):
        """Test that sampler respects max_duration constraint."""
        sampler = DynamicBucketingSampler(
            peoples_speech_cuts,
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

    def test_sampler_reproducibility(self, peoples_speech_cuts: CutSet):
        """Test that sampler produces reproducible results with same seed."""
        sampler1 = DynamicBucketingSampler(
            peoples_speech_cuts,
            max_duration=MAX_DURATION,
            num_buckets=NUM_BUCKETS,
            shuffle=True,
            seed=42,
        )

        sampler2 = DynamicBucketingSampler(
            peoples_speech_cuts,
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
