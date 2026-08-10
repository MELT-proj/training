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
    what infra/index_shar.py does to a real source: gunzip the manifests, then
    write the sidecars.
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
