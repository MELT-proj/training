"""Tests for Lhotse-based data loading.

These tests focus on the dataclass-based config approach:
- Configs are Python dataclasses defined in src/config.py
- CLI parsing is handled by tyro
- Lhotse CutSet/sampler/dataloader creation works with dataclasses/dicts
"""

from pathlib import Path

import pytest
import torch
import yaml


# Skip all tests if lhotse is not installed
pytest.importorskip("lhotse")


def get_librispeech_shar_base_path() -> str | None:
    """Get the base path to LibriSpeech shar data, checking multiple locations."""

    paths = [
        "/mnt/scratch-artemis/giuseppe/melt-data/shar/librispeech",
        "/mnt/home/giuseppe/myscratch/melt-data/shar/librispeech",
    ]
    for path in paths:
        if Path(path).exists():
            return path
    return None


@pytest.fixture(scope="module")
def librispeech_shar_base() -> str:
    path = get_librispeech_shar_base_path()
    if path is None:
        pytest.skip("LibriSpeech shar data not found at any known location")
    return path


@pytest.fixture(scope="module")
def librispeech_train100_path(librispeech_shar_base: str) -> str:
    path = f"{librispeech_shar_base}/clean/train.100"
    if not Path(path).exists():
        pytest.skip(f"LibriSpeech train.100 not found at {path}")
    return path


class TestConfigIO:
    def test_load_config_from_yaml(self):
        from src.config import load_config_from_yaml

        cfg = load_config_from_yaml("config/train/LS_asr.yaml")
        assert cfg.model.encoder.name
        assert cfg.model.decoder.name
        assert cfg.model.adapter is not None

    def test_cli_overrides(self):
        """Test that tyro can parse CLI args (simulated via dataclass instantiation)."""
        from src.config import TrainingConfig

        cfg = TrainingConfig()
        # Override via direct attribute setting (simulating CLI override)
        cfg.trainer.max_steps = 123
        cfg.model.adapter.freeze = True

        assert cfg.trainer.max_steps == 123
        assert cfg.model.adapter.freeze is True

    def test_save_config_roundtrip(self, tmp_path: Path):
        from src.config import load_config_from_yaml, save_config, TrainingConfig

        cfg = load_config_from_yaml("config/train/LS_asr.yaml")
        cfg.trainer.max_steps = 321

        out = tmp_path / "cfg.yaml"
        save_config(cfg, str(out))
        assert out.exists()

        # Load back and verify
        with open(out) as f:
            loaded = yaml.safe_load(f)
        assert loaded["trainer"]["max_steps"] == 321


class TestCutSetLoading:
    def test_read_cutset_from_config(self, librispeech_train100_path: str):
        from src.data.audio.lhotse.dataloader import read_cutset_from_config
        from src.config import DatasetConfig, DataSourceConfig

        config = DatasetConfig(
            input_cfg=[
                DataSourceConfig(
                    type="lhotse_shar",
                    shar_path=librispeech_train100_path,
                    tags={"task": "asr", "lang": "en"},
                )
            ],
            shuffle=False,
        )

        cuts, use_iterable = read_cutset_from_config(config)
        assert use_iterable is True

        first_cut = next(iter(cuts))
        assert hasattr(first_cut, "duration")
        assert first_cut.custom is not None
        assert first_cut.custom.get("task") == "asr"
        assert first_cut.custom.get("lang") == "en"


class TestSamplerAndDataloader:
    def test_sampler_creation(self, librispeech_train100_path: str):
        from src.data.audio.lhotse.dataloader import get_lhotse_sampler_from_config
        from src.config import DatasetConfig, DataSourceConfig

        config = DatasetConfig(
            input_cfg=[
                DataSourceConfig(
                    type="lhotse_shar",
                    shar_path=librispeech_train100_path,
                    tags={"task": "asr", "lang": "en"},
                )
            ],
            batch_duration=60.0,
            use_bucketing=True,
            shuffle=True,
            min_duration=0.5,
            max_duration=20.0,
        )

        sampler, use_iterable = get_lhotse_sampler_from_config(config=config, global_rank=0, world_size=1)
        assert sampler is not None
        assert use_iterable is True

    def test_dataloader_creation(self, librispeech_train100_path: str):
        from src.data.audio.lhotse.dataloader import get_lhotse_dataloader_from_config
        from src.config import DatasetConfig, DataSourceConfig

        class DummyDataset(torch.utils.data.Dataset):
            def __getitem__(self, cuts):
                return {"input_features": torch.randn(1, 10, 80)}

        config = DatasetConfig(
            input_cfg=[
                DataSourceConfig(
                    type="lhotse_shar",
                    shar_path=librispeech_train100_path,
                    tags={"task": "asr", "lang": "en"},
                )
            ],
            batch_duration=10.0,
            use_bucketing=False,
            shuffle=False,
            min_duration=0.5,
            max_duration=10.0,
            num_workers=1,
        )

        dataloader = get_lhotse_dataloader_from_config(
            config=config,
            global_rank=0,
            world_size=1,
            dataset=DummyDataset(),
        )

        batch = next(iter(dataloader))
        assert "input_features" in batch


class TestFallbackDataset:
    def test_fallback_returns_last_good_batch(self):
        from unittest.mock import MagicMock

        from src.data.audio.lhotse.dataset import FallbackDataset

        inner_dataset = MagicMock()
        good_batch = {"input_features": torch.randn(2, 100, 80)}

        inner_dataset.__getitem__.side_effect = [good_batch, None]
        fallback = FallbackDataset(inner_dataset)
        assert fallback[None] == good_batch
        assert fallback[None] == good_batch
