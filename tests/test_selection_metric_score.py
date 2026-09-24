"""Tests for the selection-metric scoring script, on small fake melt-eval JSON logs."""

import json
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "selection-metric"))

import score  # noqa: E402

FLEURS_LANGS = sorted(
    {lang for g in ("OOD-train", "OOD-related", "OOD-latin", "OOD-script") for lang in score.GROUPS[g][0]}
)


def _log(set_name: str, cer: dict[str, float], **overrides) -> dict:
    """The parts of a melt-eval JSON log that score.py reads."""
    metrics = {"corpus_cer": {"value": 0.5}, **{f"cer_{k}": {"value": v} for k, v in cer.items()}}
    log = {
        "status": "success",
        "eval": {"task_args": {"frozen_set": f"/frozen/{set_name}", "task_filter": "asr"}},
        "results": {"total_samples": 10, "completed_samples": 10, "scores": [{"metrics": metrics}]},
    }
    log.update(overrides)
    return log


def _write_checkpoint(root: Path, name: str, id_cer=0.1, fleurs_cer=0.2, fleurs_over=None, id_over=None) -> Path:
    """Write a full set of logs for one checkpoint, one sub-directory per set."""
    ckpt = root / name
    fleurs = {lang: fleurs_cer for lang in FLEURS_LANGS} | (fleurs_over or {})
    logs = {score.FLEURS_SET: fleurs}
    for lang in score.TRAIN_LANGS:
        logs[score.ID_SET_TEMPLATE.format(lang=lang)] = {lang: (id_over or {}).get(lang, id_cer)}
    for set_name, cer in logs.items():
        (ckpt / set_name).mkdir(parents=True)
        (ckpt / set_name / "log.json").write_text(json.dumps(_log(set_name, cer)))
    return ckpt


def _score(ckpt: Path) -> dict:
    return score.score_checkpoint(score.load_checkpoint(ckpt))


def test_groups_partition_fleurs_and_match_the_readme():
    assert len(FLEURS_LANGS) == 26
    assert sum(len(score.GROUPS[g][0]) for g in ("OOD-related", "OOD-latin", "OOD-script")) == 21
    weighted = {g: w for g, (_, _, w) in score.GROUPS.items() if w is not None}
    assert weighted == {"ID": 1.0, "OOD-train": 1.5, "OOD-related": 0.6}


def test_weighted_score(tmp_path):
    ckpt = _write_checkpoint(
        tmp_path, "a", id_cer=0.1,
        fleurs_over={**{l: 0.2 for l in score.TRAIN_LANGS}, **{l: 0.5 for l in score.GROUPS["OOD-related"][0]}},
    )
    result = _score(ckpt)
    assert result["medians"]["ID"] == pytest.approx(0.1)
    assert result["medians"]["OOD-train"] == pytest.approx(0.2)
    assert result["medians"]["OOD-related"] == pytest.approx(0.5)
    assert result["score"] == pytest.approx((1.0 * 0.1 + 1.5 * 0.2 + 0.6 * 0.5) / 3.1)


def test_reported_only_groups_do_not_enter_the_score(tmp_path):
    base = _score(_write_checkpoint(tmp_path, "a"))
    worse = _score(
        _write_checkpoint(
            tmp_path, "b",
            fleurs_over={l: 0.9 for g in ("OOD-latin", "OOD-script") for l in score.GROUPS[g][0]},
        )
    )
    assert worse["medians"]["OOD-latin"] == pytest.approx(0.9)
    assert worse["score"] == pytest.approx(base["score"])


def test_cer_is_clipped_at_one(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a", fleurs_over={"pt": 1.86, "ro": 2.4, "nl": 0.3})
    result = _score(ckpt)
    assert result["languages"]["OOD-related"]["pt"] == (1.86, 1.0)
    assert result["languages"]["OOD-related"]["nl"] == (0.3, 0.3)
    assert result["medians"]["OOD-related"] == pytest.approx(0.3)


def test_clip_changes_the_median_when_it_should(tmp_path):
    # three failed languages: raw median is 1.86, clipped median must be exactly 1.0
    over = {"pt": 1.86, "ro": 2.4, "nl": 3.0, "da": 0.1, "sv": 0.1}
    result = _score(_write_checkpoint(tmp_path, "a", fleurs_over=over))
    assert result["medians"]["OOD-related"] == 1.0
    assert statistics.median(over.values()) == 1.86


def test_median_of_an_even_sized_group(tmp_path):
    # OOD-latin has 12 languages; the median is the mean of the middle two
    latin = score.GROUPS["OOD-latin"][0]
    over = {lang: 0.1 * (i + 1) for i, lang in enumerate(latin)}  # 0.1 .. 1.2 -> clipped at 1.0
    result = _score(_write_checkpoint(tmp_path, "a", fleurs_over=over))
    expected = statistics.median(min(v, 1.0) for v in over.values())
    assert result["medians"]["OOD-latin"] == pytest.approx(expected) == pytest.approx(0.65)


def test_id_uses_the_in_domain_sets_and_ood_train_uses_fleurs(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a", id_cer=0.05, fleurs_over={l: 0.4 for l in score.TRAIN_LANGS})
    medians = _score(ckpt)["medians"]
    assert medians["ID"] == pytest.approx(0.05)
    assert medians["OOD-train"] == pytest.approx(0.4)


def test_missing_fleurs_language_fails_loudly(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")
    path = ckpt / score.FLEURS_SET / "log.json"
    log = json.loads(path.read_text())
    del log["results"]["scores"][0]["metrics"]["cer_pt"]
    path.write_text(json.dumps(log))
    with pytest.raises(score.ScoringError, match="OOD-related.*pt"):
        _score(ckpt)


def test_missing_in_domain_language_fails_loudly(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")
    (ckpt / "selection-id-it" / "log.json").unlink()
    with pytest.raises(score.ScoringError, match="selection-id-it"):
        _score(ckpt)


def test_in_domain_log_without_its_language_fails(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")
    path = ckpt / "selection-id-de" / "log.json"
    path.write_text(json.dumps(_log("selection-id-de", {"en": 0.1})))
    with pytest.raises(score.ScoringError, match="ID.*de"):
        _score(ckpt)


def test_duplicate_log_for_a_set_fails(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")
    (ckpt / "again").mkdir()
    (ckpt / "again" / "log.json").write_text(json.dumps(_log(score.FLEURS_SET, {"en": 0.1})))
    with pytest.raises(score.ScoringError, match="two logs"):
        _score(ckpt)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"status": "error"}, "status"),
        ({"results": {"total_samples": 10, "completed_samples": 7, "scores": [{"metrics": {}}]}}, "completed"),
    ],
)
def test_unusable_logs_fail(tmp_path, change, match):
    ckpt = _write_checkpoint(tmp_path, "a")
    path = ckpt / "selection-id-en" / "log.json"
    path.write_text(json.dumps({**_log("selection-id-en", {"en": 0.1}), **change}))
    with pytest.raises(score.ScoringError, match=match):
        _score(ckpt)


def test_log_scored_with_the_plumbing_scorer_fails(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")
    path = ckpt / "selection-id-en" / "log.json"
    path.write_text(json.dumps(_log("selection-id-en", {}) | {"results": {
        "total_samples": 1, "completed_samples": 1, "scores": [{"metrics": {"accuracy": {"value": 0.0}}}]}}))
    with pytest.raises(score.ScoringError, match="task_filter=asr"):
        _score(ckpt)


def test_main_prints_a_table_sorted_by_score_and_exits_zero(tmp_path, capsys):
    good = _write_checkpoint(tmp_path, "good", id_cer=0.05, fleurs_cer=0.1)
    bad = _write_checkpoint(tmp_path, "bad", id_cer=0.5, fleurs_cer=0.6)
    assert score.main([str(bad), str(good)]) == 0
    table = capsys.readouterr().out.split("checkpoint")[-1]
    assert table.index("good") < table.index("bad")


def test_main_exits_nonzero_on_a_missing_language(tmp_path, capsys):
    ckpt = _write_checkpoint(tmp_path, "a")
    (ckpt / "selection-id-fr" / "log.json").unlink()
    assert score.main([str(ckpt)]) == 1
    assert "selection-id-fr" in capsys.readouterr().err


# --- runaway generations (report only) -------------------------------------


def _sample(lang: str, errors: int, chars: int) -> dict:
    return {
        "metadata": {"lang": lang},
        "scores": {"asr_scorer": {"value": {"cer_errors": errors, "ref_chars": chars}}},
    }


def test_runaway_counts_share_and_cer_without():
    samples = [_sample("de", 5, 100), _sample("de", 900, 60), _sample("de", 5, 100), _sample("en", 3, 50)]
    stats = score.runaway_stats(samples)
    assert stats["de"] == {
        "samples": 3, "runaways": 1, "errors": 910, "runaway_errors": 900,
        "ref_chars": 260, "runaway_ref_chars": 60,
    }
    assert stats["en"]["runaways"] == 0


def test_a_sample_at_exactly_one_is_not_a_runaway():
    assert score.runaway_stats([_sample("de", 60, 60)])["de"]["runaways"] == 0
    assert score.runaway_stats([_sample("de", 61, 60)])["de"]["runaways"] == 1


def test_runaway_stats_is_none_without_per_sample_scores():
    assert score.runaway_stats(None) is None
    assert score.runaway_stats([]) is None
    assert score.runaway_stats([{"metadata": {"lang": "de"}, "scores": {"s": {"value": 0.5}}}]) is None


def _ckpt_with_samples(root: Path, name: str, id_de_samples, fleurs_samples) -> Path:
    ckpt = _write_checkpoint(root, name)
    for set_name, samples in ((score.ID_SET_TEMPLATE.format(lang="de"), id_de_samples), (score.FLEURS_SET, fleurs_samples)):
        path = ckpt / set_name / "log.json"
        log = json.loads(path.read_text())
        log["samples"] = samples
        path.write_text(json.dumps(log))
    for lang in score.TRAIN_LANGS:
        if lang == "de":
            continue
        path = ckpt / score.ID_SET_TEMPLATE.format(lang=lang) / "log.json"
        log = json.loads(path.read_text())
        log["samples"] = [_sample(lang, 1, 100)]
        path.write_text(json.dumps(log))
    return ckpt


def test_runaway_report_sums_id_sets_and_leaves_the_score_alone(tmp_path):
    plain = _score(_write_checkpoint(tmp_path, "plain"))
    ckpt = _ckpt_with_samples(
        tmp_path, "looping",
        id_de_samples=[_sample("de", 900, 60), _sample("de", 5, 100)],
        fleurs_samples=[_sample("pt", 400, 100), _sample("pt", 2, 100), _sample("ro", 1, 100)],
    )
    summaries = score.load_summaries(ckpt)
    report = score.runaway_report(summaries)
    assert (report["ID"]["runaways"], report["ID"]["samples"]) == (1, 6)
    assert set(report["ID"]["languages"]) == {"de"}
    n, total, share, cer_without = report["ID"]["languages"]["de"]
    assert (n, total) == (1, 2)
    assert share == pytest.approx(900 / 905)
    assert cer_without == pytest.approx(5 / 100)
    assert (report["FLEURS"]["runaways"], report["FLEURS"]["samples"]) == (1, 3)
    assert score.score_checkpoint({n: s["cer"] for n, s in summaries.items()})["score"] == pytest.approx(plain["score"])


def test_runaway_report_is_none_when_any_log_lacks_samples(tmp_path):
    ckpt = _write_checkpoint(tmp_path, "a")  # fixture logs carry no samples
    assert score.runaway_report(score.load_summaries(ckpt)) is None


def test_main_prints_runaways_and_marks_missing_samples_na(tmp_path, capsys):
    with_samples = _ckpt_with_samples(
        tmp_path, "with", [_sample("de", 900, 60)], [_sample("pt", 2, 100)]
    )
    without = _write_checkpoint(tmp_path, "without")
    assert score.main([str(with_samples), str(without)]) == 0
    out = capsys.readouterr().out
    assert "runaway-ID" in out and "n/a" in out
    assert "de=1/1 (100% of errors" in out
