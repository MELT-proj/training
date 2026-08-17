"""Tests for infra/check_training_config.py.

The checker mirrors semantics that live in the training loader -- the tag-merge
order of nested groups, the all-or-none weight rule, the validation naming rule.
A mirror that drifts is worse than no check at all, so where the real function is
importable these tests assert the two agree rather than just asserting the
mirror's own behaviour.

Manifests here are written as plain JSONL by hand rather than through lhotse:
the checker only ever reads the manifests, so the tests do not need lhotse and
run in environments where it is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "infra"))

pytest.importorskip("numpy")

import bucket_bins  # noqa: E402
import check_training_config as ctc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def write_source(
    directory: Path,
    durations: list[float],
    *,
    pnc_text: str | None = None,
    supervision_text: str | None = "hello world",
    extra_custom: dict | None = None,
    shards: int = 1,
    gzipped: bool = False,
) -> Path:
    """Write a minimal shar-shaped source and return its path."""
    import gzip

    directory.mkdir(parents=True, exist_ok=True)
    per_shard = max(1, (len(durations) + shards - 1) // shards)
    for shard in range(shards):
        chunk = durations[shard * per_shard:(shard + 1) * per_shard]
        if not chunk and shard:
            break
        lines = []
        for i, duration in enumerate(chunk):
            custom: dict = dict(extra_custom or {})
            if pnc_text is not None:
                custom["pnc_text"] = pnc_text
            record = {
                "id": f"{directory.name}-{shard}-{i}",
                "duration": duration,
                "supervisions": [{"id": f"s{i}", "text": supervision_text or ""}],
            }
            if custom:
                record["custom"] = custom
            lines.append(json.dumps(record))
        name = f"cuts.{shard:06d}.jsonl" + (".gz" if gzipped else "")
        payload = "\n".join(lines) + "\n"
        if gzipped:
            with gzip.open(directory / name, "wt", encoding="utf-8") as fh:
                fh.write(payload)
        else:
            (directory / name).write_text(payload, encoding="utf-8")
    return directory


def flatten_yaml(text: str, where: str = "train_ds"):
    """Parse an input_cfg fragment and flatten it the way the checker does."""
    import yaml

    data = yaml.safe_load(text)
    index = ctc.build_line_index(text)
    return ctc.flatten(data["input_cfg"], where, index, "input_cfg")


# ---------------------------------------------------------------------------
# Flattening and tag precedence
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_group_tag_overrides_a_leaf_tag(self):
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: group
                tags: {task: st, src_lang: en, tgt_lang: de}
                input_cfg:
                  - type: lhotse_shar
                    shar_path: /d/a
                    tags: {task: st, src_lang: de, tgt_lang: en}
            """
        )
        # The loader applies the group's tags last, so the group wins and this
        # source trains as en->de despite the leaf saying de->en.
        assert leaves[0].eff_tags["src_lang"] == "en"
        assert leaves[0].eff_tags["tgt_lang"] == "de"
        assert leaves[0].leaf_tags["src_lang"] == "de"

    def test_child_only_key_survives_the_group_pass(self):
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: group
                tags: {task: asr, lang: en}
                input_cfg:
                  - type: lhotse_shar
                    shar_path: /d/a
                    tags: {text_field: custom.pnc_text}
            """
        )
        assert leaves[0].eff_tags["text_field"] == "custom.pnc_text"
        assert leaves[0].eff_tags["lang"] == "en"

    def test_outermost_group_wins_when_nested_three_deep(self):
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: group
                tags: {lang: outer}
                input_cfg:
                  - type: group
                    tags: {lang: inner}
                    input_cfg:
                      - type: lhotse_shar
                        shar_path: /d/a
                        tags: {lang: leaf}
            """
        )
        # Each level tags after its children return, so the last writer is the
        # outermost group.
        assert leaves[0].eff_tags["lang"] == "outer"

    def test_group_own_weight_is_recorded_on_its_level(self):
        _, levels = flatten_yaml(
            """
            input_cfg:
              - type: group
                weight: 0.25
                input_cfg:
                  - type: lhotse_shar
                    shar_path: /d/a
                    weight: 1.0
            """
        )
        group_levels = [lv for lv in levels if lv.kind == "group"]
        assert len(group_levels) == 1
        assert group_levels[0].own_weight == 0.25

    def test_line_index_points_at_the_entry(self):
        text = "input_cfg:\n  - type: lhotse_shar\n    shar_path: /d/a\n"
        leaves, _ = flatten_yaml(text)
        assert leaves[0].line == 2

    def test_suggested_names_come_from_effective_tags(self):
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                tags: {task: asr, lang: zh-CN}
              - type: lhotse_shar
                shar_path: /d/b
                tags: {task: st, src_lang: it, tgt_lang: en}
            """
        )
        # Locale codes fold, so zh-CN and zh-TW would share one curve.
        assert leaves[0].suggested_name() == "asr_zh"
        assert leaves[1].suggested_name() == "st_it_en"


class TestMirrorsTheLoader:
    """The checker restates loader rules; assert they still agree."""

    def test_all_or_none_matches_resolve_weights(self):
        pytest.importorskip("torch")
        pytest.importorskip("lhotse")
        from melt.training.data.audio.lhotse.dataloader import _resolve_weights

        # The loader raises on a partially weighted level...
        with pytest.raises(ValueError):
            _resolve_weights([0.7, None], [1, 1], "x")
        # ...and the checker reports W1 for the same shape.
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                weight: 0.7
              - type: lhotse_shar
                shar_path: /d/b
            """
        )
        ctc.check_weights(report, leaves, levels, {}, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W1") == ctc.FAIL

    def test_naming_rule_matches_split_eval_config_by_name(self):
        pytest.importorskip("torch")
        pytest.importorskip("lhotse")
        from omegaconf import OmegaConf

        from melt.training.data.audio.lhotse.dataloader import split_eval_config_by_name

        partial = OmegaConf.create(
            {"input_cfg": [{"shar_path": "/d/a", "name": "x"}, {"shar_path": "/d/b"}]}
        )
        with pytest.raises(ValueError):
            split_eval_config_by_name(partial)

        # Unnamed everywhere is silently accepted by the loader, which is
        # precisely why the checker flags it.
        none_named = OmegaConf.create({"input_cfg": [{"shar_path": "/d/a"}]})
        assert split_eval_config_by_name(none_named) is None


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

class TestWeights:
    def _two_group_config(self, w_group_a: float, w_child: float) -> str:
        return f"""
        input_cfg:
          - type: group
            weight: {w_group_a}
            tags: {{task: asr, lang: en}}
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/en1
                weight: {w_child}
                tags: {{task: asr, lang: en}}
              - type: lhotse_shar
                shar_path: /d/en2
                weight: {1 - w_child:.8f}
                tags: {{task: asr, lang: en}}
          - type: group
            weight: {1 - w_group_a:.8f}
            tags: {{task: asr, lang: de}}
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/de1
                weight: 1.0
                tags: {{task: asr, lang: de}}
        """

    def test_level_weights_must_sum_to_one(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(self._two_group_config(0.4, 0.5).replace(
            "weight: 0.60000000", "weight: 0.30000000"))
        ctc.check_weights(report, leaves, levels, {}, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W2") == ctc.FAIL

    def test_matching_the_policy_passes(self):
        # Hours chosen so the policy's own output is what the config carries.
        hours = {"/d/en1": 900.0, "/d/en2": 100.0, "/d/de1": 400.0}
        sources = [
            {"path": p, "lang_key": "en" if "en" in p else "de", "hours": h}
            for p, h in hours.items()
        ]
        ctc.cmw.compute_weights(sources, 0.5, 0.5)
        by_path = {s["path"]: s for s in sources}

        text = f"""
        input_cfg:
          - type: group
            weight: {by_path['/d/en1']['p_l']:.8f}
            tags: {{task: asr, lang: en}}
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/en1
                weight: {by_path['/d/en1']['p_c']:.8f}
                tags: {{task: asr, lang: en}}
              - type: lhotse_shar
                shar_path: /d/en2
                weight: {by_path['/d/en2']['p_c']:.8f}
                tags: {{task: asr, lang: en}}
          - type: group
            weight: {by_path['/d/de1']['p_l']:.8f}
            tags: {{task: asr, lang: de}}
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/de1
                weight: {by_path['/d/de1']['p_c']:.8f}
                tags: {{task: asr, lang: de}}
        """
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(text)
        cache = {p: {"hours": h, "cuts": 10, "hist": {"100": 10}} for p, h in hours.items()}
        ctc.check_weights(report, leaves, levels, cache, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W3") == ctc.PASS, [f.message for f in report.findings]

    def test_wrong_weights_are_reported_with_the_expected_value(self):
        hours = {"/d/en1": 900.0, "/d/en2": 100.0, "/d/de1": 400.0}
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(self._two_group_config(0.5, 0.5))
        cache = {p: {"hours": h, "cuts": 10, "hist": {"100": 10}} for p, h in hours.items()}
        ctc.check_weights(report, leaves, levels, cache, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W3") == ctc.FAIL
        finding = next(f for f in report.findings if f.check == "W3")
        assert "expected p_c" in " ".join(finding.detail)

    def test_unweighted_mixture_skips_the_numeric_checks(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                tags: {task: asr, lang: en}
              - type: lhotse_shar
                shar_path: /d/b
                tags: {task: asr, lang: de}
            """
        )
        ctc.check_weights(report, leaves, levels, {}, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W1") == ctc.PASS
        assert report.status("W3") == ctc.SKIP
        assert any("auto-weight" in n for n in report.notes)

    def test_a_group_mixing_two_languages_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, levels = flatten_yaml(
            """
            input_cfg:
              - type: group
                weight: 1.0
                input_cfg:
                  - type: lhotse_shar
                    shar_path: /d/a
                    weight: 0.5
                    tags: {task: asr, lang: en}
                  - type: lhotse_shar
                    shar_path: /d/b
                    weight: 0.5
                    tags: {task: asr, lang: de}
            """
        )
        ctc.check_weights(report, leaves, levels, {}, 0.5, 0.5, 1e-6, "cmd")
        assert report.status("W4") == ctc.FAIL


# ---------------------------------------------------------------------------
# Validation naming
# ---------------------------------------------------------------------------

class TestNames:
    def test_no_names_fails_and_suggests_every_one(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                tags: {task: asr, lang: en}
              - type: lhotse_shar
                shar_path: /d/b
                tags: {task: st, src_lang: it, tgt_lang: en}
            """,
            where="validation_ds",
        )
        ctc.check_names(report, leaves)
        assert report.status("N1") == ctc.FAIL
        finding = next(f for f in report.findings if f.check == "N1")
        assert any("asr_en" in d for d in finding.detail)
        assert any("st_it_en" in d for d in finding.detail)

    def test_partial_naming_is_reported_as_the_loader_error(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                name: asr_en
                tags: {task: asr, lang: en}
              - type: lhotse_shar
                shar_path: /d/b
                tags: {task: asr, lang: de}
            """,
            where="validation_ds",
        )
        ctc.check_names(report, leaves)
        assert report.status("N2") == ctc.FAIL

    def test_fully_named_passes_and_reports_the_curves(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /d/a
                name: asr_en
                tags: {task: asr, lang: en}
              - type: lhotse_shar
                shar_path: /d/b
                name: asr_en
                tags: {task: asr, lang: en}
              - type: lhotse_shar
                shar_path: /d/c
                name: asr_de
                tags: {task: asr, lang: de}
            """,
            where="validation_ds",
        )
        ctc.check_names(report, leaves)
        assert report.status("N1") == ctc.PASS
        assert report.status("N2") == ctc.PASS
        # Two sources share a name, so they form one curve, not two.
        assert any("2 named eval sets" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

class TestTotals:
    def _leaf(self, path: str, where: str = "train_ds") -> ctc.Leaf:
        return ctc.Leaf(
            where=where, yaml_path="x", raw_path=path, path=path, leaf_tags={},
            eff_tags={"task": "asr", "lang": "en"}, weight=None, has_weight=False,
            name=None, line=1, group_yaml_path=None,
        )

    def test_stale_total_hours_is_reported_with_the_measured_value(self):
        report = ctc.Report(Path("cfg.yaml"))
        cache = {"/d/a": {"hours": 100.0, "cuts": 1000, "hist": {"100": 1000},
                          "res": 0.01, "filtered": {"0.5:120.0": {"hours": 100.0, "cuts": 1000}}}}
        split = {"total_hours": 80.0, "total_cuts": 1000,
                 "min_duration": 0.5, "max_duration": 120.0}
        ctc.check_totals(report, "train_ds", split, [self._leaf("/d/a")], cache, {}, ("H1", "H2"))
        assert report.status("H1") == ctc.FAIL
        finding = next(f for f in report.findings if f.check == "H1")
        assert "100.0" in " ".join(finding.fix)

    def test_correct_totals_pass(self):
        report = ctc.Report(Path("cfg.yaml"))
        cache = {"/d/a": {"hours": 100.0, "cuts": 1000, "hist": {"100": 1000},
                          "res": 0.01, "filtered": {"0.5:120.0": {"hours": 100.0, "cuts": 1000}}}}
        split = {"total_hours": 100.0, "total_cuts": 1000,
                 "min_duration": 0.5, "max_duration": 120.0}
        ctc.check_totals(report, "train_ds", split, [self._leaf("/d/a")], cache, {}, ("H1", "H2"))
        assert report.status("H1") == ctc.PASS
        assert report.status("H2") == ctc.PASS

    def test_a_duplicated_source_counts_twice(self):
        report = ctc.Report(Path("cfg.yaml"))
        cache = {"/d/a": {"hours": 100.0, "cuts": 1000, "hist": {"100": 1000},
                          "res": 0.01, "filtered": {"0.5:120.0": {"hours": 100.0, "cuts": 1000}}}}
        # The same leaf listed under two tasks, as the yodas ast sources are.
        leaves = [self._leaf("/d/a"), self._leaf("/d/a")]
        split = {"total_hours": 200.0, "total_cuts": 2000,
                 "min_duration": 0.5, "max_duration": 120.0}
        ctc.check_totals(report, "train_ds", split, leaves, cache, {}, ("H1", "H2"))
        assert report.status("H1") == ctc.PASS
        assert any("distinct audio is 100.0 h" in n for n in report.notes)

    def test_unmeasured_source_skips_rather_than_counting_zero(self):
        report = ctc.Report(Path("cfg.yaml"))
        split = {"total_hours": 100.0, "total_cuts": 1000,
                 "min_duration": 0.5, "max_duration": 120.0}
        ctc.check_totals(report, "train_ds", split, [self._leaf("/d/a")], {}, {}, ("H1", "H2"))
        assert report.status("H1") == ctc.SKIP
        assert report.status("H2") == ctc.SKIP


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

class TestBinEstimator:
    def test_histogram_matches_the_exact_estimator(self):
        import random

        random.seed(7)
        durations = [random.uniform(0.5, 30.0) for _ in range(5000)]
        exact = bucket_bins.estimate_bins_from_durations(durations, 20)
        hist = bucket_bins.histogram_of(durations, 0.01)
        approx = bucket_bins.estimate_bins_from_histogram(hist, 0.01, 20)
        assert len(approx) == len(exact)
        # One resolution step is the guarantee; configs are written at 2 dp.
        assert max(abs(a - b) for a, b in zip(exact, approx)) <= 0.01

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No durations"):
            bucket_bins.estimate_bins_from_histogram({}, 0.01, 5)

    def test_more_buckets_than_cuts_raises(self):
        with pytest.raises(ValueError, match="must be <="):
            bucket_bins.estimate_bins_from_histogram({100: 2}, 0.01, 5)


class TestBinChecks:
    def _split(self, **over) -> dict:
        split = {
            "lhotse_sampler_type": "dynamic_bucketing", "num_buckets": 4,
            "bucket_duration_bins": [1.0, 2.0, 3.0],
            "min_duration": 0.5, "max_duration": 30.0, "batch_duration": 90,
        }
        split.update(over)
        return split

    def test_wrong_number_of_bins_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_bins_structure(report, "train_ds", self._split(num_buckets=10), {})
        assert report.status("B2") == ctc.FAIL
        assert "needs 9" in next(f for f in report.findings if f.check == "B2").message

    def test_non_monotonic_bins_are_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_bins_structure(report, "train_ds",
                                 self._split(bucket_duration_bins=[1.0, 3.0, 2.0]), {})
        assert report.status("B2") == ctc.FAIL

    def test_bins_outside_the_duration_filter_are_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_bins_structure(report, "train_ds",
                                 self._split(bucket_duration_bins=[1.0, 2.0, 99.0]), {})
        assert report.status("B2") == ctc.FAIL

    def test_missing_num_buckets_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        assert ctc.check_bins_structure(report, "train_ds", self._split(num_buckets=None), {}) is None
        assert report.status("B1") == ctc.FAIL

    def test_a_non_bucketing_sampler_skips_the_bin_checks(self):
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_bins_structure(report, "train_ds", self._split(lhotse_sampler_type="dynamic"), {})
        for check in ("B1", "B2", "B3"):
            assert report.status(check) == ctc.SKIP

    def test_stale_bins_are_reported_with_the_measured_replacement(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path="/d/a", path="/d/a", leaf_tags={},
            eff_tags={}, weight=None, has_weight=False, name=None, line=1,
            group_yaml_path=None,
        )
        # A uniform spread over 1..10 s cannot produce bins at 1/2/3 s.
        hist = {int(round(d / 0.01)): 10 for d in [1.0, 3.0, 5.0, 7.0, 9.0]}
        cache = {"/d/a": {"hours": 1.0, "cuts": 50, "hist": {str(k): v for k, v in hist.items()},
                          "res": 0.01, "filtered": {}}}
        ctc.check_bins_values(report, "train_ds", self._split(), [leaf], cache, {},
                              4, 0.05, 0.02, 2)
        assert report.status("B3") == ctc.FAIL
        finding = next(f for f in report.findings if f.check == "B3")
        assert "bucket_duration_bins:" in " ".join(finding.fix)

    def test_a_top_bin_below_the_long_tail_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path="/d/a", path="/d/a", leaf_tags={},
            eff_tags={}, weight=None, has_weight=False, name=None, line=1,
            group_yaml_path=None,
        )
        # Mostly short cuts, with a long tail well past the top bin at 3 s.
        hist = {100: 1000, 2500: 50}
        cache = {"/d/a": {"hours": 1.0, "cuts": 1050,
                          "hist": {str(k): v for k, v in hist.items()},
                          "res": 0.01, "filtered": {}}}
        ctc.check_bins_values(report, "train_ds", self._split(), [leaf], cache, {},
                              4, 0.05, 0.02, 2)
        assert report.status("C4") == ctc.FAIL or report.status("C4") == ctc.WARN
        assert any(f.check == "C4" for f in report.findings)


# ---------------------------------------------------------------------------
# text_field
# ---------------------------------------------------------------------------

class TestTextField:
    def _leaf(self, path: Path, text_field: str | None, task: str = "asr") -> ctc.Leaf:
        tags: dict = {"task": task}
        if task == "st":
            tags.update({"src_lang": "it", "tgt_lang": "en"})
        else:
            tags["lang"] = "it"
        if text_field:
            tags["text_field"] = text_field
        return ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path=str(path), path=str(path),
            leaf_tags=tags, eff_tags=tags, weight=None, has_weight=False, name=None,
            line=1, group_yaml_path=None,
        )

    def test_a_resolving_text_field_passes(self, tmp_path):
        source = write_source(tmp_path / "mls" / "it" / "train", [2.0], pnc_text="Ciao.")
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_text_fields(report, [self._leaf(source, "custom.pnc_text")],
                              "train_ds", probe=True, strict_text_field=True)
        assert report.status("T4") == ctc.PASS

    def test_an_unresolvable_text_field_is_reported(self, tmp_path):
        source = write_source(tmp_path / "vp" / "it" / "train", [2.0], pnc_text=None)
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_text_fields(report, [self._leaf(source, "custom.pnc_text")],
                              "train_ds", probe=True, strict_text_field=False)
        assert report.status("T4") == ctc.WARN
        finding = next(f for f in report.findings if f.check == "T4")
        assert "training#66" in " ".join(finding.fix)

    def test_nested_text_field_resolves(self, tmp_path):
        source = write_source(
            tmp_path / "cv22_sidon" / "cs" / "train", [2.0],
            supervision_text="", extra_custom={"metadata": {"sentence": "Ahoj."}},
        )
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_text_fields(report, [self._leaf(source, "custom.metadata.sentence")],
                              "train_ds", probe=True, strict_text_field=True)
        assert report.status("T4") == ctc.PASS

    def test_st_without_a_text_field_is_reported_when_a_translation_exists(self, tmp_path):
        source = write_source(
            tmp_path / "yodas-granary" / "Italian" / "ast", [2.0],
            supervision_text="testo italiano",
            extra_custom={"translation_en": "italian text"},
        )
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_text_fields(report, [self._leaf(source, None, task="st")],
                              "train_ds", probe=True, strict_text_field=True)
        assert report.status("T4") == ctc.WARN
        finding = next(f for f in report.findings if f.check == "T4")
        assert "custom.translation_en" in " ".join(finding.fix)

    def test_a_textless_source_is_reported(self, tmp_path):
        source = write_source(tmp_path / "x" / "y" / "train", [2.0], supervision_text="")
        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_text_fields(report, [self._leaf(source, None)],
                              "train_ds", probe=True, strict_text_field=True)
        assert report.status("T4") == ctc.WARN

    def test_corpus_disagreement_on_text_field_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaves, _ = flatten_yaml(
            """
            input_cfg:
              - type: lhotse_shar
                shar_path: /shar/cv22_sidon/cs/train
                tags: {task: asr, lang: cs, text_field: custom.metadata.sentence}
              - type: lhotse_shar
                shar_path: /shar/cv22_sidon/de/train
                tags: {task: asr, lang: de, text_field: custom.metadata.sentence}
              - type: lhotse_shar
                shar_path: /shar/cv22_sidon/fr/train
                tags: {task: asr, lang: fr, text_field: custom.pnc_text}
            """
        )
        ctc.check_text_field_consistency(report, leaves, "train_ds")
        assert report.status("T5") == ctc.WARN
        finding = next(f for f in report.findings if f.check == "T5")
        assert "custom.metadata.sentence" in " ".join(finding.fix)


# ---------------------------------------------------------------------------
# Disk and runtime
# ---------------------------------------------------------------------------

class TestDisk:
    def test_a_missing_path_is_reported(self, tmp_path):
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path="/nope", path=str(tmp_path / "nope"),
            leaf_tags={}, eff_tags={}, weight=None, has_weight=False, name=None, line=1,
            group_yaml_path=None,
        )
        ctc.check_disk(report, {"train_ds": [leaf]})
        assert report.status("D1") == ctc.WARN

    def test_a_directory_without_manifests_is_reported(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path="/empty", path=str(empty),
            leaf_tags={}, eff_tags={}, weight=None, has_weight=False, name=None, line=1,
            group_yaml_path=None,
        )
        ctc.check_disk(report, {"train_ds": [leaf]})
        finding = next(f for f in report.findings if f.check == "D1")
        assert "no cuts" in finding.message

    def test_a_stale_idx_sidecar_is_reported(self, tmp_path):
        source = write_source(tmp_path / "src", [1.0, 2.0])
        manifest = source / "cuts.000000.jsonl"
        sidecar = manifest.with_suffix(".jsonl.idx")
        sidecar.write_text("0 10\n")
        import os as _os
        # Sidecar older than the manifest it indexes.
        _os.utime(sidecar, (1, 1))
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="train_ds", yaml_path="x", raw_path="/src", path=str(source),
            leaf_tags={}, eff_tags={}, weight=None, has_weight=False, name=None, line=1,
            group_yaml_path=None,
        )
        ctc.check_disk(report, {"train_ds": [leaf]})
        assert any(f.check == "D2" and "older than" in f.message for f in report.findings)

    def test_train_validation_overlap_is_reported(self, tmp_path):
        source = write_source(tmp_path / "shared", [1.0])

        def leaf(where: str) -> ctc.Leaf:
            return ctc.Leaf(
                where=where, yaml_path="x", raw_path="/shared", path=str(source),
                leaf_tags={}, eff_tags={}, weight=None, has_weight=False, name=None,
                line=1, group_yaml_path=None,
            )

        report = ctc.Report(Path("cfg.yaml"))
        ctc.check_disk(report, {"train_ds": [leaf("train_ds")],
                                "validation_ds": [leaf("validation_ds")]})
        assert report.status("D3") == ctc.WARN


class TestRuntimeTraps:
    def _cfg(self, **train) -> dict:
        split = {"total_hours": 100.0, "total_cuts": 1000, "batch_duration": 90,
                 "input_cfg": [], "force_estimate": False}
        split.update(train)
        return {"data": {"train_ds": split}, "trainer": {}, "model": {}}

    def test_force_estimate_on_a_grouped_input_cfg_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(total_hours=None, force_estimate=True)
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": True}, {})
        assert report.status("E1") == ctc.WARN
        assert "dataloader.py:277" in " ".join(
            next(f for f in report.findings if f.check == "E1").fix
        )

    def test_force_estimate_on_a_flat_input_cfg_passes(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(total_hours=None, force_estimate=True)
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("E1") == ctc.PASS

    def test_total_cuts_null_with_batch_size_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(total_cuts=None, batch_size=16, batch_duration=None)
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("E2") == ctc.WARN

    def test_total_cuts_null_is_fine_when_force_estimate_fills_it_in(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(total_hours=None, total_cuts=None, batch_size=16,
                        batch_duration=None, force_estimate=True)
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("E2") == ctc.PASS

    def test_unestimable_steps_per_epoch_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(total_hours=None, force_estimate=False)
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("E3") == ctc.WARN

    def test_eval_batch_size_minus_one_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg()
        cfg["trainer"] = {"do_eval": True, "eval_strategy": "steps",
                          "per_device_eval_batch_size": -1}
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("C2") == ctc.WARN

    def test_max_duration_beyond_the_encoder_window_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        cfg = self._cfg(max_duration=120.0)
        cfg["model"] = {"encoder": {"max_audio_seq_len": 1500}}
        ctc.check_runtime(report, cfg, {"train_ds": []}, {"train_ds": False}, {})
        assert report.status("C3") == ctc.WARN
        assert "30" in next(f for f in report.findings if f.check == "C3").message

    def test_strict_text_field_not_inherited_by_eval_is_reported(self):
        report = ctc.Report(Path("cfg.yaml"))
        leaf = ctc.Leaf(
            where="validation_ds", yaml_path="x", raw_path="/d/a", path="/d/a",
            leaf_tags={}, eff_tags={"text_field": "custom.pnc_text"}, weight=None,
            has_weight=False, name=None, line=1, group_yaml_path=None,
        )
        cfg = self._cfg()
        cfg["data"]["strict_text_field"] = True
        cfg["data"]["validation_ds"] = {"input_cfg": [], "total_hours": 1.0,
                                        "total_cuts": 1, "batch_size": 4}
        ctc.check_runtime(report, cfg, {"validation_ds": [leaf]},
                          {"validation_ds": False}, {})
        assert report.status("C1") == ctc.WARN


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _config_text(self, root: Path, *, total_hours, total_cuts, names: bool) -> str:
        name_a = "\n        name: asr_en" if names else ""
        name_b = "\n        name: asr_de" if names else ""
        return f"""
run:
  exp_name: test
model:
  encoder: {{max_audio_seq_len: 1500}}
data:
  strict_text_field: true
  train_ds:
    input_cfg:
      - type: lhotse_shar
        shar_path: {root}/corpus/en/train
        tags: {{task: asr, lang: en, text_field: custom.pnc_text}}
      - type: lhotse_shar
        shar_path: {root}/corpus/de/train
        tags: {{task: asr, lang: de, text_field: custom.pnc_text}}
    total_hours: {total_hours}
    total_cuts: {total_cuts}
    force_estimate: false
    min_duration: 0.5
    max_duration: 30.0
    batch_duration: 90
    lhotse_sampler_type: dynamic
    num_workers: 1
  validation_ds:
    input_cfg:
      - type: lhotse_shar
        shar_path: {root}/corpus/en/validation
        tags: {{task: asr, lang: en, text_field: custom.pnc_text}}{name_a}
      - type: lhotse_shar
        shar_path: {root}/corpus/de/validation
        tags: {{task: asr, lang: de, text_field: custom.pnc_text}}{name_b}
    total_hours: null
    total_cuts: null
    force_estimate: true
    batch_size: 4
    min_duration: 0.5
    max_duration: 30.0
    lhotse_sampler_type: dynamic
trainer:
  do_eval: true
  eval_strategy: steps
  per_device_eval_batch_size: 4
"""

    def _build_tree(self, tmp_path: Path) -> tuple[Path, float, int]:
        durations = [1.0, 2.0, 3.0, 4.0]
        total_seconds = 0.0
        total_cuts = 0
        for lang in ("en", "de"):
            for split in ("train", "validation"):
                write_source(tmp_path / "corpus" / lang / split, durations,
                             pnc_text="Hello.")
                if split == "train":
                    total_seconds += sum(durations)
                    total_cuts += len(durations)
        return tmp_path, total_seconds / 3600.0, total_cuts

    def test_a_consistent_config_exits_zero(self, tmp_path, monkeypatch, capsys):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        config.write_text(self._config_text(root, total_hours=round(hours, 6),
                                            total_cuts=cuts, names=True))
        code = self._run(monkeypatch, config, tmp_path)
        assert code == 0, capsys.readouterr().out

    def _run(self, monkeypatch, config: Path, tmp_path: Path, *extra: str) -> int:
        monkeypatch.setattr(
            sys, "argv",
            ["check_training_config.py", "--config", str(config),
             "--datasets-root", str(tmp_path),
             "--cache", str(tmp_path / "cache.json"),
             "--seed-cache", str(tmp_path / "absent.json"),
             *extra],
        )
        return ctc.main()

    def test_stale_totals_and_missing_names_fail(self, tmp_path, monkeypatch, capsys):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        config.write_text(self._config_text(root, total_hours=999.0,
                                            total_cuts=cuts, names=False))
        code = self._run(monkeypatch, config, tmp_path)
        out = capsys.readouterr().out
        assert code == 1
        assert "H1" in out and "N1" in out
        # The measured value has to appear, so the fix is copy-pasteable.
        assert f"{hours:.1f}" in out

    def test_offline_needs_no_datasets_root(self, tmp_path, monkeypatch, capsys):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        config.write_text(self._config_text(root, total_hours=round(hours, 6),
                                            total_cuts=cuts, names=True))
        monkeypatch.delenv("LOCAL_DATASETS_DIR", raising=False)
        monkeypatch.setattr(
            sys, "argv",
            ["check_training_config.py", "--config", str(config), "--offline"],
        )
        assert ctc.main() == 0
        out = capsys.readouterr().out
        assert "--offline" in out

    def test_missing_datasets_root_exits_two(self, tmp_path, monkeypatch, capsys):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        config.write_text(self._config_text(root, total_hours=1.0, total_cuts=1, names=True))
        monkeypatch.delenv("LOCAL_DATASETS_DIR", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["check_training_config.py", "--config", str(config)]
        )
        assert ctc.main() == 2
        assert "LOCAL_DATASETS_DIR" in capsys.readouterr().err

    def test_json_report_is_written(self, tmp_path, monkeypatch):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        config.write_text(self._config_text(root, total_hours=round(hours, 6),
                                            total_cuts=cuts, names=True))
        out = tmp_path / "report.json"
        self._run(monkeypatch, config, tmp_path, "--json", str(out))
        data = json.loads(out.read_text())
        assert data["checks"]["H1"]["status"] == ctc.PASS
        assert data["failed"] is False

    def test_strict_promotes_warnings_to_failure(self, tmp_path, monkeypatch):
        root, hours, cuts = self._build_tree(tmp_path)
        config = tmp_path / "cfg.yaml"
        # max_duration beyond the encoder window is a warning; --strict fails it.
        text = self._config_text(root, total_hours=round(hours, 6), total_cuts=cuts,
                                 names=True).replace("max_duration: 30.0",
                                                     "max_duration: 120.0")
        config.write_text(text)
        assert self._run(monkeypatch, config, tmp_path) == 0
        assert self._run(monkeypatch, config, tmp_path, "--strict") == 1
