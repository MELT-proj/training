"""Tests for Lhotse-based data loading.

These tests focus on the OmegaConf-based config approach:
- Configs are OmegaConf DictConfigs loaded from YAML or created programmatically
- CLI parsing is handled by OmegaConf's from_dotlist
- Lhotse CutSet/sampler/dataloader creation works with DictConfig objects
"""

import os
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf


# Skip all tests if lhotse is not installed
pytest.importorskip("lhotse")


def get_librispeech_shar_base_path() -> str | None:
    """Get the base path to LibriSpeech shar data from LOCAL_DATASETS_DIR."""
    base = os.environ.get("LOCAL_DATASETS_DIR")
    if base is None:
        return None
    path = Path(base) / "librispeech"
    return str(path) if path.exists() else None


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
    @pytest.mark.skipif(
        os.getenv("LOCAL_DATASETS_DIR") is None,
        reason="LOCAL_DATASETS_DIR environment variable not set",
    )
    def test_load_config_from_yaml(self):
        from melt.training.config import load_config

        cfg = load_config("config/train/SFT-v1.2.7.yaml")
        assert cfg.model.encoder.name
        assert cfg.model.decoder.name
        assert cfg.model.adapter is not None

    def test_cli_overrides(self):
        """Test that OmegaConf can apply CLI overrides via dotlist."""
        from melt.training.config import load_config

        cfg = load_config(cli_args=["trainer.max_steps=123", "model.adapter.freeze=true"])
        assert cfg.trainer.max_steps == 123
        assert cfg.model.adapter.freeze is True

    @pytest.mark.skipif(
        os.getenv("LOCAL_DATASETS_DIR") is None,
        reason="LOCAL_DATASETS_DIR environment variable not set",
    )
    def test_save_config_roundtrip(self, tmp_path: Path, monkeypatch):
        from melt.training.config import load_config, save_config

        # Set env vars that asr.yaml interpolates via ${oc.env:...}
        monkeypatch.setenv("LOCAL_DATASETS_DIR", "/tmp/fake_datasets")
        monkeypatch.setenv("OUTPUT_DIR", "/tmp/fake_output")

        cfg = load_config("config/train/SFT-v1.2.7.yaml")
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
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config

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
                "seed": 42,
                "shard_seed": 0,
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
        from melt.training.data.audio.lhotse.dataloader import get_lhotse_sampler_from_config

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "lhotse_shar",
                        "shar_path": librispeech_train100_path,
                        "tags": {"task": "asr", "lang": "en"},
                    }
                ],
                "batch_duration": 60.0,
                "lhotse_sampler_type": "dynamic_bucketing",
                "num_buckets": 10,
                "shuffle": True,
                "min_duration": 0.5,
                "max_duration": 20.0,
                "seed": 42,
                "shard_seed": "randomized",
            }
        )

        sampler, use_iterable = get_lhotse_sampler_from_config(config=config, global_rank=0, world_size=1)
        assert sampler is not None
        assert use_iterable is True

    def test_dataloader_creation(self, librispeech_train100_path: str):
        from melt.training.data.audio.lhotse.dataloader import get_lhotse_dataloader_from_config

        class DummyDataset(torch.utils.data.Dataset):
            def __getitem__(self, cuts):
                return {"input_features": torch.randn(1, 10, 80)}

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "lhotse_shar",
                        "shar_path": librispeech_train100_path,
                        "tags": {"task": "asr", "lang": "en"},
                    }
                ],
                "batch_duration": 10.0,
                "lhotse_sampler_type": "dynamic",
                "shuffle": False,
                "min_duration": 0.5,
                "max_duration": 10.0,
                "num_workers": 1,
                "seed": 42,
                "shard_seed": 0,
            }
        )

        dataloader = get_lhotse_dataloader_from_config(
            config=config,
            global_rank=0,
            world_size=1,
            dataset=DummyDataset(),
        )

        batch = next(iter(dataloader))
        assert "input_features" in batch


@pytest.fixture(scope="module")
def synthetic_shar(tmp_path_factory) -> dict[str, str]:
    """Three tiny shar sources whose cut IDs identify the source they came from.

    Synthetic rather than real data so the mixture tests assert on proportions
    without depending on LOCAL_DATASETS_DIR.
    """
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    root = tmp_path_factory.mktemp("group_shar")
    out = {}
    for name in ("a", "b", "c"):
        cuts = DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True)
        cuts = CutSet.from_cuts(c.with_id(f"{name}{i}") for i, c in enumerate(cuts))
        d = root / name
        d.mkdir(parents=True, exist_ok=True)  # to_shar does not create it
        cuts.to_shar(str(d), fields={"recording": "wav"}, shard_size=10)
        out[name] = str(d)
    return out


def _source_of(cut_id: str) -> str:
    return cut_id[0]


class TestWeightResolution:
    """`_resolve_weights` decides one level's weights; it is all or nothing."""

    def test_all_explicit_are_used_verbatim(self):
        from melt.training.data.audio.lhotse.dataloader import _resolve_weights

        assert _resolve_weights([0.7, 0.3], [100, 900], "x") == [0.7, 0.3]

    def test_no_explicit_falls_back_to_cut_counts(self):
        from melt.training.data.audio.lhotse.dataloader import _resolve_weights

        assert _resolve_weights([None, None], [100, 900], "x") == [100.0, 900.0]

    def test_unmeasurable_source_floors_at_one(self):
        from melt.training.data.audio.lhotse.dataloader import _resolve_weights

        assert _resolve_weights([None, None], [0, 900], "x") == [1.0, 900.0]

    def test_mixing_explicit_and_automatic_raises(self):
        from melt.training.data.audio.lhotse.dataloader import _resolve_weights

        # 0.7 against a raw count of 900 would starve the weighted source, so
        # this is rejected rather than silently normalised.
        with pytest.raises(ValueError, match="Set it on all of them or none"):
            _resolve_weights([0.7, None], [100, 900], "x")


class TestNestedGroups:
    def test_group_is_muxed_at_the_product_of_weights(self, synthetic_shar):
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config

        # Group 1 holds a and b at 0.25/0.75; group 2 holds c alone.
        # Both groups carry weight 0.5, so the effective shares are
        # a=0.125, b=0.375, c=0.5.
        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "group",
                        "weight": 0.5,
                        "input_cfg": [
                            {"type": "lhotse_shar", "shar_path": synthetic_shar["a"],
                             "weight": 0.25},
                            {"type": "lhotse_shar", "shar_path": synthetic_shar["b"],
                             "weight": 0.75},
                        ],
                    },
                    {
                        "type": "group",
                        "weight": 0.5,
                        "input_cfg": [
                            {"type": "lhotse_shar", "shar_path": synthetic_shar["c"],
                             "weight": 1.0},
                        ],
                    },
                ],
                "shuffle": False,
                "seed": 42,
                "shard_seed": 0,
            }
        )

        cuts, use_iterable = read_cutset_from_config(config, repeat=True)
        assert use_iterable is True

        counts = {"a": 0, "b": 0, "c": 0}
        for i, cut in enumerate(cuts):
            if i >= 3000:
                break
            counts[_source_of(cut.id)] += 1

        total = sum(counts.values())
        shares = {k: v / total for k, v in counts.items()}
        # Loose bounds: this asserts the product is applied, not the RNG.
        assert shares["a"] == pytest.approx(0.125, abs=0.05)
        assert shares["b"] == pytest.approx(0.375, abs=0.06)
        assert shares["c"] == pytest.approx(0.500, abs=0.06)

    def test_group_tags_reach_every_cut_beneath(self, synthetic_shar):
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "group",
                        "weight": 1.0,
                        "tags": {"lang": "de"},
                        "input_cfg": [
                            {
                                "type": "lhotse_shar",
                                "shar_path": synthetic_shar["a"],
                                "weight": 1.0,
                                "tags": {"task": "asr", "region_code": "de_de"},
                            }
                        ],
                    }
                ],
                "shuffle": False,
                "seed": 42,
                "shard_seed": 0,
            }
        )

        cuts, _ = read_cutset_from_config(config, repeat=False)
        first = next(iter(cuts))
        # The leaf's own tags survive, and the group's tag is added on top.
        assert first.custom.get("task") == "asr"
        assert first.custom.get("region_code") == "de_de"
        assert first.custom.get("lang") == "de"

    def test_flat_config_still_loads(self, synthetic_shar):
        """A config with no groups must behave exactly as before."""
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {"type": "lhotse_shar", "shar_path": synthetic_shar["a"]},
                    {"type": "lhotse_shar", "shar_path": synthetic_shar["b"]},
                ],
                "shuffle": False,
                "seed": 42,
                "shard_seed": 0,
            }
        )

        cuts, use_iterable = read_cutset_from_config(config, repeat=False)
        assert use_iterable is True
        seen = {_source_of(c.id) for c in cuts}
        assert seen == {"a", "b"}

    def test_empty_group_raises(self, synthetic_shar):
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config

        config = OmegaConf.create(
            {
                "input_cfg": [{"type": "group", "weight": 1.0, "input_cfg": []}],
                "shuffle": False,
                "seed": 42,
                "shard_seed": 0,
            }
        )

        with pytest.raises(ValueError, match="must define a non-empty 'input_cfg'"):
            read_cutset_from_config(config)


class TestFallbackDataset:
    def test_fallback_returns_last_good_batch(self):
        from unittest.mock import MagicMock

        from melt.training.data.audio.lhotse.dataset import FallbackDataset

        inner_dataset = MagicMock()
        good_batch = {"input_features": torch.randn(2, 100, 80)}

        inner_dataset.__getitem__.side_effect = [good_batch, None]
        fallback = FallbackDataset(inner_dataset)
        assert fallback[None] == good_batch
        assert fallback[None] == good_batch
