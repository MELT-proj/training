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


def _shar_writable(cut):
    """Strip the in-memory extras `DummyManifest(with_data=True)` attaches.

    Alongside the recording, it hangs `features` and four `custom_*` arrays off
    each cut, all backed by raw bytes in memory. `to_shar` exports only the
    fields it is given, and leaves the rest on the cut to be written into
    `cuts.*.jsonl.gz` — where the bytes are not JSON serializable and the whole
    write dies. Dropping them keeps the audio, which is the only part these
    tests want, and the shar dirs still come out shaped like real ones.
    """
    cut.features = None
    cut.custom = None
    return cut


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
        cuts = CutSet.from_cuts(
            _shar_writable(c.with_id(f"{name}{i}"))
            for i, c in enumerate(DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True))
        )
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
        # Same, on the `cut.tags` attribute -- this is what get_tags_from_cut
        # and the text_field/target_field readers actually consume, and
        # `_add_tags_to_cut` used to replace it outright on the group's pass
        # (`cut.tags = tags`) rather than merge, silently dropping every
        # child-only key even though `cut.custom` merged correctly.
        assert first.tags.get("task") == "asr"
        assert first.tags.get("region_code") == "de_de"
        assert first.tags.get("lang") == "de"

    def test_group_tags_do_not_clobber_a_child_only_text_field(self, synthetic_shar):
        """A group wrapping a source with its own `tags.text_field` (e.g. a
        two-tier language/corpus mix with a per-corpus text_field override,
        as used for `cv22_sidon`) must not lose that override to the group's
        own tagging pass.
        """
        from melt.training.data.audio.lhotse.dataloader import read_cutset_from_config
        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        config = OmegaConf.create(
            {
                "input_cfg": [
                    {
                        "type": "group",
                        "weight": 1.0,
                        "tags": {"task": "asr", "lang": "en"},
                        "input_cfg": [
                            {
                                "type": "lhotse_shar",
                                "shar_path": synthetic_shar["a"],
                                "weight": 1.0,
                                "tags": {
                                    "task": "asr",
                                    "lang": "en",
                                    "text_field": "custom.metadata.sentence",
                                },
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
        assert first.tags.get("text_field") == "custom.metadata.sentence"

        ds_cfg = OmegaConf.create({"input_cfg": [], "text_field": "text"})
        ds = MELTMapDataset(cuts=[first], processor=None, config=ds_cfg, is_train=False)
        assert ds._resolve_text_field(first) == "custom.metadata.sentence"

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


@pytest.fixture(scope="module")
def indexed_synthetic_shar(tmp_path_factory) -> str:
    """A tiny shar source converted to the indexed layout.

    `to_shar` writes gzipped cut manifests, and the indexer skips those outright
    because an .idx is a table of byte offsets into a plain file. So this does
    what MELT-proj/preprocessing's `data-utils/index_shar.py` does to a real
    source: gunzip the manifests, then write the sidecars.
    """
    import gzip
    import shutil

    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    pytest.importorskip("lhotse.indexing")
    from lhotse.indexing import create_shar_index

    d = tmp_path_factory.mktemp("indexed_shar") / "src"
    d.mkdir(parents=True, exist_ok=True)
    cuts = CutSet.from_cuts(
        _shar_writable(c.with_id(f"cut{i:04d}"))
        for i, c in enumerate(DummyManifest(CutSet, begin_id=0, end_id=40, with_data=True))
    )
    cuts.to_shar(str(d), fields={"recording": "wav"}, shard_size=10)

    for gz in sorted(d.glob("cuts.*.jsonl.gz")):
        plain = gz.with_suffix("")  # drop .gz
        with gzip.open(gz, "rb") as fin, open(plain, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz.unlink()
    create_shar_index(d)
    return str(d)


def _loader_config(shar_path: str, **overrides):
    config = {
        "input_cfg": [
            {"type": "lhotse_shar", "shar_path": shar_path,
             "tags": {"task": "asr", "lang": "en"}},
        ],
        "batch_duration": 10.0,
        "lhotse_sampler_type": "dynamic",
        "shuffle": False,
        "min_duration": 0.0,
        "max_duration": 100.0,
        "num_workers": 2,
        "seed": 42,
        "shard_seed": 0,
        "prefetch_factor": 2,
    }
    config.update(overrides)
    return OmegaConf.create(config)


class _IdDataset(torch.utils.data.Dataset):
    """Returns the cut IDs of each batch, so batches can be compared by identity."""

    def __getitem__(self, cuts):
        return {"cut_ids": [c.id for c in cuts]}


class TestIndexedSharDetection:
    def test_indexed_source_is_detected(self, indexed_synthetic_shar: str):
        from melt.training.data.audio.lhotse.dataloader import _sources_are_indexed

        entries = [{"type": "lhotse_shar", "shar_path": indexed_synthetic_shar}]
        assert _sources_are_indexed(entries) is True

    def test_plain_source_is_not_detected(self, synthetic_shar: dict[str, str]):
        from melt.training.data.audio.lhotse.dataloader import _sources_are_indexed

        entries = [{"type": "lhotse_shar", "shar_path": synthetic_shar["a"]}]
        assert _sources_are_indexed(entries) is False


class TestStatefulDataLoader:
    """Training resumes from a per-worker snapshot rather than a replayed count.

    The bug this guards against (#46) is invisible at num_workers <= 1, where the
    rank-wide count and the per-worker count coincide, so everything here runs at
    num_workers=2.
    """

    def test_train_loader_is_stateful(self, indexed_synthetic_shar: str):
        from torchdata.stateful_dataloader import StatefulDataLoader

        from melt.training.data.audio.lhotse.dataloader import (
            get_lhotse_dataloader_from_config,
        )

        dl = get_lhotse_dataloader_from_config(
            config=_loader_config(indexed_synthetic_shar),
            global_rank=0,
            world_size=1,
            dataset=_IdDataset(),
            repeat=True,
        )
        assert isinstance(dl, StatefulDataLoader)

    def test_eval_loader_is_not_stateful(self, indexed_synthetic_shar: str):
        """Eval has no position worth resuming, and is handed to accelerate."""
        from torchdata.stateful_dataloader import StatefulDataLoader

        from melt.training.data.audio.lhotse.dataloader import (
            get_lhotse_dataloader_from_config,
        )

        dl = get_lhotse_dataloader_from_config(
            config=_loader_config(indexed_synthetic_shar),
            global_rank=0,
            world_size=1,
            dataset=_IdDataset(),
            repeat=False,
        )
        assert not isinstance(dl, StatefulDataLoader)

    def test_state_resumes_at_the_cut_point(self, indexed_synthetic_shar: str):
        from melt.training.data.audio.lhotse.dataloader import (
            get_lhotse_dataloader_from_config,
        )

        def build():
            return get_lhotse_dataloader_from_config(
                config=_loader_config(indexed_synthetic_shar),
                global_rank=0,
                world_size=1,
                dataset=_IdDataset(),
                repeat=True,
            )

        dl = build()
        it = iter(dl)
        consumed = [next(it)["cut_ids"] for _ in range(4)]
        state = dl.state_dict()
        tail = [next(it)["cut_ids"] for _ in range(6)]

        # The training CutSet is .repeat()ed, so this stream never ends: take a
        # fixed number of batches rather than materialising it.
        resumed_dl = build()
        resumed_dl.load_state_dict(state)
        resumed_it = iter(resumed_dl)
        resumed = [next(resumed_it)["cut_ids"] for _ in range(6)]

        assert consumed, "loader produced nothing to snapshot"
        assert resumed == tail, (
            "resumed stream diverged from the uninterrupted one: "
            f"{resumed[:2]} vs {tail[:2]}"
        )


class TestRngSetstateHardening:
    """RNG state must survive a state dict whose tuples were flattened to lists.

    `random.getstate()` returns tuples and `random.setstate()` rejects anything
    else. Lhotse checkpoints several RNGs this way, and the state dict that
    returns through the dataloader's worker transport in a multi-rank run has
    those tuples flattened, which killed every worker on the first batch after a
    resume.

    The hardening patches `setstate` rather than individual lhotse call sites:
    lhotse restores RNG state at seven places and only two route through its own
    coercion helper. Patching one at a time just moves the failure to the next --
    which is exactly what happened before this landed.
    """

    @staticmethod
    def _flatten(state):
        return [state[0], list(state[1]), state[2]]

    def test_flattened_state_is_accepted_and_resumes_identically(self):
        import random

        import melt.training.data.audio.lhotse.dataloader  # noqa: F401  (applies the patch)

        reference = random.Random(1234)
        good = reference.getstate()
        expected = [reference.random() for _ in range(5)]

        target = random.Random()
        target.setstate(self._flatten(good))
        assert [target.random() for _ in range(5)] == expected

    def test_well_formed_state_is_untouched(self):
        import random

        import melt.training.data.audio.lhotse.dataloader  # noqa: F401

        reference = random.Random(7)
        good = reference.getstate()
        target = random.Random()
        target.setstate(good)
        assert target.random() == reference.random()

    def test_genuinely_invalid_state_still_raises(self):
        """The patch must not turn a real error into silent corruption."""
        import random

        import melt.training.data.audio.lhotse.dataloader  # noqa: F401

        with pytest.raises((TypeError, ValueError)):
            random.Random().setstate(["not", "an", "rng"])
        with pytest.raises((TypeError, ValueError)):
            random.Random().setstate("nonsense")

    def test_patch_is_idempotent(self):
        import random

        import melt.training.data.audio.lhotse.dataloader as dl

        before = random.Random.setstate
        dl._harden_rng_setstate()
        assert random.Random.setstate is before

    def test_every_rng_in_a_nested_state_survives(self):
        """Lhotse stores RNG state under several names; all of them must work."""
        import random

        import melt.training.data.audio.lhotse.dataloader  # noqa: F401

        names = ["_bucket_rng", "rng_state", "bucket_rng_state", "_rng_state"]
        refs = {n: random.Random(i) for i, n in enumerate(names)}
        flat = {n: self._flatten(r.getstate()) for n, r in refs.items()}

        for n in names:
            target = random.Random()
            target.setstate(flat[n])
            assert target.random() == refs[n].random(), f"{n} did not resume"


class TestNamedEvalSets:
    """`name` on a validation source splits eval into separately reported sets.

    HF's Trainer loops over a dict of eval datasets and prefixes every metric
    with the key, so named sets are what produce per-language / per-task
    `eval_<name>_loss` without any change to the metric plumbing.
    """

    @staticmethod
    def _cfg(entries):
        from omegaconf import OmegaConf

        return OmegaConf.create({"input_cfg": entries, "max_samples": 8})

    def test_unnamed_sources_keep_the_single_set_behaviour(self):
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        cfg = self._cfg([{"shar_path": "/a"}, {"shar_path": "/b"}])
        assert split_eval_config_by_name(cfg) is None

    def test_empty_input_cfg_is_not_an_error(self):
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        assert split_eval_config_by_name(self._cfg([])) is None

    def test_sources_sharing_a_name_are_grouped(self):
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        cfg = self._cfg(
            [
                {"shar_path": "/de1", "name": "asr_de"},
                {"shar_path": "/nl", "name": "asr_nl"},
                {"shar_path": "/de2", "name": "asr_de"},
            ]
        )
        groups = split_eval_config_by_name(cfg)

        assert list(groups) == ["asr_de", "asr_nl"]  # first-appearance order
        assert [s.shar_path for s in groups["asr_de"].input_cfg] == ["/de1", "/de2"]
        assert [s.shar_path for s in groups["asr_nl"].input_cfg] == ["/nl"]

    def test_sibling_keys_are_carried_into_every_sub_config(self):
        """Filters like max_samples must apply per set, not once globally."""
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        cfg = self._cfg(
            [{"shar_path": "/a", "name": "x"}, {"shar_path": "/b", "name": "y"}]
        )
        groups = split_eval_config_by_name(cfg)

        assert all(sub.max_samples == 8 for sub in groups.values())

    def test_sub_configs_do_not_alias_the_original(self):
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        cfg = self._cfg([{"shar_path": "/a", "name": "x"}])
        groups = split_eval_config_by_name(cfg)
        groups["x"].max_samples = 999

        assert cfg.max_samples == 8

    def test_partially_named_sources_raise(self):
        """Naming only some sources is a mistake, not a request to lump the rest."""
        from melt.training.data.audio.lhotse.dataloader import (
            split_eval_config_by_name,
        )

        cfg = self._cfg([{"shar_path": "/a", "name": "x"}, {"shar_path": "/b"}])
        with pytest.raises(ValueError, match="mixes named and unnamed"):
            split_eval_config_by_name(cfg)


class TestEvalFormatInheritance:
    """Eval must score the sequence format training produced.

    `apply_chat_template` and its companions are declared at `data.`, which the
    training path receives whole. Eval is built from `data.validation_ds`, one
    level down, so it read their defaults instead — training formatted a chat
    turn and masked everything outside the assistant span, while eval formatted
    a bare `{audio_token}{text}`. The two losses were computed over different
    formats and were never comparable. See issue #58.
    """

    @staticmethod
    def _data(parent: dict, validation: dict | None = None):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {**parent, "validation_ds": {"input_cfg": [], **(validation or {})}}
        )

    def test_apply_chat_template_is_inherited(self):
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        resolved = resolve_eval_data_config(self._data({"apply_chat_template": True}))

        assert resolved.apply_chat_template is True

    def test_all_formatting_keys_are_inherited(self):
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        resolved = resolve_eval_data_config(
            self._data(
                {
                    "apply_chat_template": True,
                    "prompt_template": "{audio_token}{t}",
                    "prompt_template_selection": "with_language",
                    "chat_template_config": "chatml",
                }
            )
        )

        assert resolved.prompt_template == "{audio_token}{t}"
        assert resolved.prompt_template_selection == "with_language"
        assert resolved.chat_template_config == "chatml"

    def test_validation_ds_overrides_the_parent(self):
        """The documented workaround must keep working, or configs relying on it move."""
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        resolved = resolve_eval_data_config(
            self._data(
                {"apply_chat_template": True, "prompt_template_selection": "random"},
                {"apply_chat_template": False, "prompt_template_selection": "custom"},
            )
        )

        assert resolved.apply_chat_template is False
        assert resolved.prompt_template_selection == "custom"

    def test_a_false_parent_is_still_inherited_rather_than_defaulted(self):
        """`False` is a value, not an absence — inheriting it must not be skipped."""
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        resolved = resolve_eval_data_config(self._data({"apply_chat_template": False}))

        assert resolved.apply_chat_template is False

    def test_nothing_declared_leaves_the_config_untouched(self):
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        data = self._data({})
        assert resolve_eval_data_config(data) is data.validation_ds

    def test_the_original_config_is_not_mutated(self):
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        data = self._data({"apply_chat_template": True})
        resolve_eval_data_config(data)

        assert "apply_chat_template" not in data.validation_ds

    def test_missing_validation_ds_is_not_an_error(self):
        from omegaconf import OmegaConf

        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        assert resolve_eval_data_config(OmegaConf.create({})) is None

    def test_resolved_config_drives_the_eval_collator(self):
        """The wiring, not just the dict: the collator must come out chat-formatting."""
        from melt.training.data.audio.lhotse.collator import MELTDataCollator
        from melt.training.data.audio.lhotse.dataloader import (
            resolve_eval_data_config,
        )

        class _FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        class _FakeProcessor:
            tokenizer = _FakeTokenizer()
            audio_token = "<|audio|>"

        resolved = resolve_eval_data_config(
            self._data(
                {
                    "apply_chat_template": True,
                    "prompt_template_selection": "with_language",
                }
            )
        )
        collator = MELTDataCollator(
            processor=_FakeProcessor(), config=resolved, is_train=False
        )

        assert collator.apply_chat_template is True
        assert collator.prompt_template_selection == "with_language"


class TestEvalTextFieldResolution:
    """`text_field` set on validation_ds must reach the eval dataset.

    `MELTMapDataset` looked the key up one level down (`config.validation_ds
    .text_field`) while every caller already hands it the ds-level config, so
    the lookup always missed and eval silently read plain `text`. Invisible in
    the shipped configs, which all set `text_field: text` anyway, but it would
    make eval score the wrong field the moment one of them didn't.
    """

    def test_ds_level_text_field_is_read(self):
        from omegaconf import OmegaConf

        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        cfg = OmegaConf.create({"input_cfg": [], "text_field": "custom.pnc_text"})
        ds = MELTMapDataset(cuts=[], processor=None, config=cfg, is_train=False)

        assert ds._text_field == "custom.pnc_text"

    def test_whole_data_block_still_resolves_through_the_nested_key(self):
        from omegaconf import OmegaConf

        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        cfg = OmegaConf.create(
            {"validation_ds": {"text_field": "custom.metadata.sentence"}}
        )
        ds = MELTMapDataset(cuts=[], processor=None, config=cfg, is_train=False)

        assert ds._text_field == "custom.metadata.sentence"

    def test_absent_text_field_still_defaults_to_text(self):
        from omegaconf import OmegaConf

        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        cfg = OmegaConf.create({"input_cfg": []})
        ds = MELTMapDataset(cuts=[], processor=None, config=cfg, is_train=False)

        assert ds._text_field == "text"


class _StubSupervision:
    def __init__(self, text, language="es"):
        self.text = text
        self.language = language


class _StubCut:
    """Minimal stand-in for a Cut, exercising only the text-resolution path."""

    def __init__(self, cut_id="c0", text="hola mundo", custom=None, duration=1.0, tags=None):
        self.id = cut_id
        self.supervisions = [_StubSupervision(text)] if text is not None else []
        self.custom = custom or {}
        self.duration = duration
        # Real cuts carry per-cut tags on the `tags` attribute (set by
        # `_add_tags_to_cut`), not nested under `custom["tags"]`.
        self.tags = tags or {}


class TestStrictTextField:
    """A configured `text_field` that resolves to nothing must not fall back.

    ST sources point `text_field` at `custom.translation_en`, which holds
    *different content* from the supervision text. Under the default fallback a
    cut with a null translation quietly becomes an ASR pair wearing an ST label
    and an inverted language pair — which is exactly the defect that shipped in
    SFT-v1.3.0. Strict mode makes that state loud.
    """

    def test_strict_raises_when_configured_field_is_absent(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        cut = _StubCut(custom={"translation_en": None})

        with pytest.raises(ValueError, match="translation_en"):
            get_text_from_cut(cut, "custom.translation_en", strict=True)

    def test_strict_raises_when_configured_field_is_blank(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        cut = _StubCut(custom={"translation_en": "   "})

        with pytest.raises(ValueError):
            get_text_from_cut(cut, "custom.translation_en", strict=True)

    def test_non_strict_preserves_the_existing_fallback(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        cut = _StubCut(text="hola mundo", custom={"translation_en": None})

        # The historical behaviour, which shipped configs still depend on.
        assert get_text_from_cut(cut, "custom.translation_en") == "hola mundo"

    def test_strict_is_satisfied_by_a_present_translation(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        cut = _StubCut(custom={"translation_en": "hello world"})

        assert (
            get_text_from_cut(cut, "custom.translation_en", strict=True) == "hello world"
        )

    def test_plain_text_field_is_unaffected_by_strict(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        cut = _StubCut(text="hola mundo")

        assert get_text_from_cut(cut, "text", strict=True) == "hola mundo"

    def test_strict_returns_none_when_no_text_exists_anywhere(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        # No supervisions (text=None), no custom.text, and the configured
        # field is absent too. There is nothing here to mislabel — a fallback
        # was never on the table — so strict mode must skip the cut like the
        # non-strict path instead of raising.
        cut = _StubCut(text=None, custom={"translation_en": None})

        assert get_text_from_cut(cut, "custom.translation_en", strict=True) is None

    def test_strict_returns_none_when_only_fallback_is_whitespace(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        # The supervision text exists but is whitespace-only, which counts as
        # absent for fallback purposes. There is still no real fallback to
        # protect against, so this must skip rather than raise.
        cut = _StubCut(text="   ", custom={"translation_en": None})

        assert get_text_from_cut(cut, "custom.translation_en", strict=True) is None

    def test_strict_still_raises_when_a_custom_text_fallback_exists(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        # No supervisions, but `custom.text` is populated — that is a real
        # fallback that falling back to would silently mislabel the sample,
        # so strict mode must still raise even though the supervision list
        # is empty.
        cut = _StubCut(text=None, custom={"translation_en": None, "text": "hola mundo"})

        with pytest.raises(ValueError, match="translation_en"):
            get_text_from_cut(cut, "custom.translation_en", strict=True)

    def test_non_strict_returns_none_for_a_textless_cut(self):
        from melt.training.data.audio.lhotse.helpers import get_text_from_cut

        # Same textless cut as the strict "skip, don't raise" case above:
        # non-strict mode already returned None here and this refactor must
        # not change that.
        cut = _StubCut(text=None, custom={"translation_en": None})

        assert get_text_from_cut(cut, "custom.translation_en", strict=False) is None


class TestEvalValidityScanHonoursPerCutOverride:
    """The validity scan and the fetch must resolve `text_field` identically.

    The scan used the ds-level field while `__getitem__` applied the per-cut
    `tags.text_field` override. An ST cut carrying a transcript but no
    translation therefore passed the scan on `text` and then came back
    `__invalid__` on `custom.translation_en`, silently shrinking the eval set
    with no counter to show for it.
    """

    def test_scan_applies_the_override_under_strict_mode(self):
        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        # Transcript present, translation absent. The scan must resolve the
        # per-cut override and fail there, rather than counting the cut valid on
        # `text` and discovering the problem only at fetch time.
        cut = _StubCut(
            text="hola mundo",
            custom={"translation_en": None},
            tags={"text_field": "custom.translation_en"},
        )
        cfg = OmegaConf.create(
            {"input_cfg": [], "text_field": "text", "strict_text_field": True}
        )

        with pytest.raises(ValueError, match="translation_en"):
            MELTMapDataset(cuts=[cut], processor=None, config=cfg, is_train=False)

    def test_override_resolution_matches_between_scan_and_fetch(self):
        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        cut = _StubCut(
            text="hola mundo",
            custom={"translation_en": "hello world"},
            tags={"text_field": "custom.translation_en"},
        )
        cfg = OmegaConf.create({"input_cfg": [], "text_field": "text"})

        ds = MELTMapDataset(cuts=[cut], processor=None, config=cfg, is_train=False)

        assert len(ds) == 1
        assert ds._resolve_text_field(cut) == "custom.translation_en"

    def test_strict_scan_skips_a_textless_cut_instead_of_raising(self):
        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        # A cut with no supervision text and no custom.text under the
        # configured text_field has nothing a fallback could mislabel, so
        # construction must complete and simply exclude the cut rather than
        # raising out of the constructor.
        cut = _StubCut(text=None, custom={"translation_en": None})
        cfg = OmegaConf.create(
            {"input_cfg": [], "text_field": "custom.translation_en", "strict_text_field": True}
        )

        ds = MELTMapDataset(cuts=[cut], processor=None, config=cfg, is_train=False)

        assert len(ds) == 0


class TestTagWriteReadRoundTrip:
    """The tag writer and the tag readers must agree on where tags live.

    `_add_tags_to_cut` (the writer, run over every `input_cfg` source/group)
    puts per-cut tags on the `cut.tags` attribute. `MELTMapDataset
    ._resolve_text_field`, `SpeechToTextDataset._get_text`, and
    `SpeechTextQEDataset._get_score` used to look for a nested
    `cut.custom["tags"]` dict instead, which the writer never created --
    so a `tags.text_field`/`tags.target_field` override silently never
    applied and fell back to the ds-level default. For a source with no
    real supervision text (e.g. CommonVoice via cv22_sidon, whose transcript
    lives only at `custom.metadata.sentence`) that fallback resolves to
    nothing and the cut is dropped as empty -- exactly the symptom that
    exposed this. `get_tags_from_cut` and the `dataset_id` lookup already
    read the correct `cut.tags` attribute, which is why task/lang tagging
    looked fine while text_field silently didn't apply. These tests exercise
    the writer and each reader together, rather than a reader against a
    hand-built cut shape that only matched the reader's (wrong) expectation.
    """

    def test_written_text_field_tag_resolves_in_map_dataset(self):
        from melt.training.data.audio.lhotse.dataloader import _add_tags_to_cut
        from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset

        cut = _StubCut(text=None, custom={"metadata": {"sentence": "bonjour le monde"}})
        _add_tags_to_cut(
            cut, {"task": "asr", "lang": "fr", "text_field": "custom.metadata.sentence"}
        )

        cfg = OmegaConf.create({"input_cfg": [], "text_field": "text"})
        ds = MELTMapDataset(cuts=[cut], processor=None, config=cfg, is_train=False)

        assert ds._resolve_text_field(cut) == "custom.metadata.sentence"
        # Under the old bug this cut has no supervision text and no working
        # override, so the validity scan drops it and len(ds) == 0.
        assert len(ds) == 1

    def test_written_text_field_tag_resolves_in_speech_to_text_dataset(self):
        from melt.training.data.audio.lhotse.dataloader import _add_tags_to_cut
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        cut = _StubCut(text=None, custom={"metadata": {"sentence": "bonjour le monde"}})
        _add_tags_to_cut(
            cut, {"task": "asr", "lang": "fr", "text_field": "custom.metadata.sentence"}
        )

        cfg = OmegaConf.create({"train_ds": {"text_field": "text"}})
        ds = SpeechToTextDataset(processor=None, config=cfg, is_train=True)

        assert ds._get_text(cut) == "bonjour le monde"

    def test_written_target_field_tag_resolves_in_qe_dataset(self):
        from melt.training.data.audio.lhotse.dataloader import _add_tags_to_cut
        from melt.training.data.audio.lhotse.dataset import SpeechTextQEDataset

        cut = _StubCut(custom={"score": 84.0})
        _add_tags_to_cut(
            cut, {"task": "speechqe", "target_field": "custom.score", "normalize_factor": 100.0}
        )

        cfg = OmegaConf.create({"train_ds": {}})
        ds = SpeechTextQEDataset(processor=None, config=cfg, is_train=True)

        assert ds._get_score(cut) == pytest.approx(0.84)


class TestSharManifestDiscovery:
    """Manifests must be found in either Shar layout.

    An indexed collection stores plain `cuts.*.jsonl` beside `.idx` byte
    offsets, so it cannot stay compressed. Globbing only `cuts.*.jsonl.gz`
    reports a fully indexed source as empty, which surfaces as a
    ZeroDivisionError several frames away in the trainer.
    """

    @staticmethod
    def _write(d, name, text, gzipped):
        import gzip as _gzip

        path = d / name
        opener = _gzip.open if gzipped else open
        with opener(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _cut(self, cut_id, duration):
        import json as _json

        return _json.dumps({"id": cut_id, "duration": duration}) + "\n"

    def test_plain_jsonl_manifests_are_found(self, tmp_path):
        from melt.training.data.audio.lhotse.dataloader import (
            _read_shar_manifest_durations,
        )

        self._write(tmp_path, "cuts.000000.jsonl", self._cut("a", 2.0), False)
        self._write(tmp_path, "cuts.000001.jsonl", self._cut("b", 3.0), False)

        duration, n = _read_shar_manifest_durations(tmp_path)
        assert (duration, n) == (5.0, 2)

    def test_gzipped_manifests_still_work(self, tmp_path):
        from melt.training.data.audio.lhotse.dataloader import (
            _read_shar_manifest_durations,
        )

        self._write(tmp_path, "cuts.000000.jsonl.gz", self._cut("a", 4.0), True)

        assert _read_shar_manifest_durations(tmp_path) == (4.0, 1)

    def test_a_shard_in_both_forms_is_counted_once(self, tmp_path):
        """A half-migrated source must not double its measured duration."""
        from melt.training.data.audio.lhotse.dataloader import (
            _read_shar_manifest_durations,
        )

        self._write(tmp_path, "cuts.000000.jsonl", self._cut("a", 7.0), False)
        self._write(tmp_path, "cuts.000000.jsonl.gz", self._cut("a", 7.0), True)

        assert _read_shar_manifest_durations(tmp_path) == (7.0, 1)

    def test_empty_directory_reports_nothing(self, tmp_path):
        from melt.training.data.audio.lhotse.dataloader import (
            _read_shar_manifest_durations,
        )

        assert _read_shar_manifest_durations(tmp_path) == (0.0, 0)

    def test_discovery_orders_shards_and_prefers_plain(self, tmp_path):
        from melt.training.data.audio.lhotse.dataloader import shar_manifest_files

        self._write(tmp_path, "cuts.000001.jsonl.gz", "", True)
        self._write(tmp_path, "cuts.000000.jsonl.gz", "", True)
        self._write(tmp_path, "cuts.000000.jsonl", "", False)

        names = [p.name for p in shar_manifest_files(tmp_path)]
        assert names == ["cuts.000000.jsonl", "cuts.000001.jsonl.gz"]


class TestStepsPerEpochEstimate:
    """`estimate_steps_per_epoch` counts a rank's batches, not one worker's.

    A rank's DataLoader interleaves all of its workers into a single stream, so
    the epoch length the training loop sees -- and therefore `max_steps` when it
    is derived from `num_train_epochs`, and the LR schedule derived from that --
    must not depend on `num_workers`.

    It used to. The divisor carried an extra `num_workers` factor left over from
    `split_for_dataloading=True`, which was never actually passed, so every run
    at the packaged default of 2 workers under-reported its epoch length by 2x.
    lhotse's `make_worker_init_fn` does give worker `w` of rank `r` the
    partition (r * num_workers + w, world_size * num_workers), but that splits
    the rank's stream among its workers -- it does not shorten it.
    """

    # 100 h at 200 s/batch is 1800 micro-batches over the whole mixture, before
    # any rank or grad-accum division. `total_hours` is set explicitly so the
    # estimator never touches disk and `shar_path` is never read.
    TOTAL_HOURS = 100.0
    BATCH_DURATION = 200.0
    BATCHES_PER_EPOCH = 1800

    @staticmethod
    def _config(**overrides):
        config = {
            "input_cfg": [{"type": "lhotse_shar", "shar_path": "/nonexistent"}],
            "total_hours": TestStepsPerEpochEstimate.TOTAL_HOURS,
            "batch_duration": TestStepsPerEpochEstimate.BATCH_DURATION,
            "num_workers": 2,
        }
        config.update(overrides)
        return OmegaConf.create(config)

    @pytest.mark.parametrize("num_workers", [0, 1, 2, 4, 8])
    def test_epoch_length_is_independent_of_num_workers(self, num_workers):
        from melt.training.data.audio.lhotse.dataloader import estimate_steps_per_epoch

        steps, _, _, batches_per_epoch, batches_per_rank = estimate_steps_per_epoch(
            config=self._config(num_workers=num_workers),
            gradient_accumulation_steps=1,
            world_size=1,
        )

        assert batches_per_epoch == self.BATCHES_PER_EPOCH
        assert batches_per_rank == self.BATCHES_PER_EPOCH
        assert steps == self.BATCHES_PER_EPOCH

    def test_num_workers_does_not_change_steps_at_scale(self):
        """The same run at 1 vs 8 workers must schedule the same number of steps."""
        from melt.training.data.audio.lhotse.dataloader import estimate_steps_per_epoch

        def steps_at(num_workers):
            return estimate_steps_per_epoch(
                config=self._config(num_workers=num_workers),
                gradient_accumulation_steps=8,
                world_size=4,
            )[0]

        assert steps_at(1) == steps_at(2) == steps_at(8)

    def test_matches_the_documented_formula(self):
        """The estimate is the formula in the module docstring and in
        `projects/ablation-campaign/build_campaign_config.py`:

            steps = total_hours * 3600 / (batch_duration * world_size * grad_accum)
        """
        import math

        from melt.training.data.audio.lhotse.dataloader import estimate_steps_per_epoch

        world_size, grad_accum = 4, 8
        steps, hours, _, _, _ = estimate_steps_per_epoch(
            config=self._config(num_workers=2),
            gradient_accumulation_steps=grad_accum,
            world_size=world_size,
        )

        expected = math.ceil(
            math.ceil(self.TOTAL_HOURS * 3600 / self.BATCH_DURATION)
            / (world_size * grad_accum)
        )
        assert hours == self.TOTAL_HOURS
        assert steps == expected == 57

    def test_world_size_and_grad_accum_still_divide(self):
        from melt.training.data.audio.lhotse.dataloader import estimate_steps_per_epoch

        _, _, _, _, one_rank = estimate_steps_per_epoch(
            config=self._config(), gradient_accumulation_steps=1, world_size=1
        )
        _, _, _, _, four_ranks = estimate_steps_per_epoch(
            config=self._config(), gradient_accumulation_steps=1, world_size=4
        )
        four_ranks_ga2, *_ = estimate_steps_per_epoch(
            config=self._config(), gradient_accumulation_steps=2, world_size=4
        )

        assert one_rank == self.BATCHES_PER_EPOCH
        assert four_ranks == self.BATCHES_PER_EPOCH / 4
        assert four_ranks_ga2 == self.BATCHES_PER_EPOCH / 8

    def test_dataloader_len_is_per_rank_not_per_worker(self, indexed_synthetic_shar):
        """`len(dataloader)` is what HF Trainer uses as `steps_in_epoch`.

        It has to be the number of batches the rank's loader yields, which is
        the same however many workers feed it.
        """
        from melt.training.data.audio.lhotse.dataloader import (
            get_lhotse_dataloader_from_config,
        )

        def loader_len(num_workers):
            dl = get_lhotse_dataloader_from_config(
                config=_loader_config(
                    indexed_synthetic_shar,
                    total_hours=self.TOTAL_HOURS,
                    batch_duration=self.BATCH_DURATION,
                    num_workers=num_workers,
                ),
                global_rank=0,
                world_size=2,
                dataset=_IdDataset(),
            )
            return len(dl)

        assert loader_len(1) == loader_len(2) == self.BATCHES_PER_EPOCH // 2
