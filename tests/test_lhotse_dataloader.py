"""Tests for Lhotse-based data loading.

These tests focus on the simplified config approach:
- YAML is loaded as an OmegaConf DictConfig
- CLI dotlist overrides are supported
- Lhotse CutSet/sampler/dataloader creation works with DictConfig/dicts
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


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
        from src.config import load_config

        cfg = load_config("config/train/LS_asr.yaml")
        assert cfg.model.encoder.name
        assert cfg.model.decoder.name
        assert "adapter" in cfg.model

    def test_dotlist_overrides(self):
        from src.config import load_config

        cfg = load_config(
            "config/train/LS_asr.yaml",
            dotlist_overrides=["trainer.max_steps=123", "model.adapter.add_adapter=true"],
        )
        assert int(cfg.trainer.max_steps) == 123
        assert bool(cfg.model.adapter.add_adapter) is True

    def test_save_config_roundtrip(self, tmp_path: Path):
        from src.config import load_config, save_config

        cfg = load_config("config/train/LS_asr.yaml", dotlist_overrides=["trainer.max_steps=321"])
        out = tmp_path / "cfg.yaml"
        save_config(cfg, str(out))
        assert out.exists()

        cfg2 = OmegaConf.load(str(out))
        assert int(cfg2.trainer.max_steps) == 321


class TestCutSetLoading:
    def test_read_cutset_from_config(self, librispeech_train100_path: str):
        from src.data.audio.lhotse.dataloader import read_cutset_from_config

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "lhotse_shar",
                        "shar_path": librispeech_train100_path,
                        "tags": {"task": "asr", "lang": "en"},
                    }
                ],
                "shuffle": False,
            }
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

        config = {
            "input_cfg": [
                {
                    "type": "lhotse_shar",
                    "shar_path": librispeech_train100_path,
                    "tags": {"task": "asr", "lang": "en"},
                }
            ],
            "batch_duration": 60.0,
            "use_bucketing": True,
            "shuffle": True,
            "min_duration": 0.5,
            "max_duration": 20.0,
        }

        sampler, use_iterable = get_lhotse_sampler_from_config(config=config, global_rank=0, world_size=1)
        assert sampler is not None
        assert use_iterable is True

    def test_dataloader_creation(self, librispeech_train100_path: str):
        from src.data.audio.lhotse.dataloader import get_lhotse_dataloader_from_config

        class DummyDataset(torch.utils.data.Dataset):
            def __getitem__(self, cuts):
                return {"input_features": torch.randn(1, 10, 80)}

        config = {
            "input_cfg": [
                {
                    "type": "lhotse_shar",
                    "shar_path": librispeech_train100_path,
                    "tags": {"task": "asr", "lang": "en"},
                }
            ],
            "batch_duration": 10.0,
            "use_bucketing": False,
            "shuffle": False,
            "min_duration": 0.5,
            "max_duration": 10.0,
            "num_workers": 0,
        }

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
