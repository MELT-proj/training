"""Smoke tests of the FLEURS WER figure on a tiny master built through master.py.

    python -m pytest projects/ablation-campaign/charts/test_fig_fleurs_asr_vs_baselines.py -q

Needs matplotlib and numpy (skipped without them); the figure script itself needs nothing else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


pytest.importorskip("matplotlib")

import fig_fleurs_asr_vs_baselines as fig  # noqa: E402
import master  # noqa: E402


SETS = (
    "set,fmt,task,split,n_per_unit,metrics,description\n"
    "fleurs-reported,native,asr,reported,,wer,published\n"
    "fleurs24-asr-dev,repeatfirst,asr,dev,100,wer cer,ours\n"
)
OURS = "fleurs24-asr-dev.repeatfirst.wer"
PUB = "fleurs-reported.native.wer"


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    """Two baselines (en, de, uk), our model 'Mid' (en, de, pl), a cut-worthy 'Bad', and a start model."""
    (tmp_path / master.SETS_NAME).write_text(SETS, encoding="utf-8")
    path = tmp_path / master.MASTER_NAME

    def add(model, kind, order, recipe, family, cells, prefix):
        meta = {"label": f"{model} label", "kind": kind, "order": str(order), "recipe": recipe, "family": family}
        master.upsert(
            [(model, f"{prefix}.{lang}", v) for lang, v in cells.items()],
            who="test", note="tiny", meta={model: {**meta, "source": "test"}}, master=path,
        )  # fmt: skip

    add("Zed", "baseline", 2, "public", "x", {"en": 10.0, "de": 20.0, "uk": 30.0}, PUB)
    add("Alpha", "baseline", 1, "public", "y", {"en": 5.0, "de": 15.0, "uk": 99.0}, PUB)
    add("Mid", "ours", 11, "adapter-sd", "qwen3.5-2b", {"en": 6.0, "de": 7.0, "pl": 8.0}, OURS)
    add("Bad", "ours", 12, "adapter-sd-sift", "eurollm-1.7b", {"en": 80.0, "de": 7.5}, OURS)
    add("Start", "ours", 10, "start", "qwen3.5-2b", {"en": 9.0, "de": 9.5}, OURS)
    return path


def run(master_path: Path, out: Path, *extra: str) -> None:
    assert fig.main(["--master", str(master_path), "--out-dir", str(out), *extra]) == 0


def test_the_figure_is_drawn_from_the_master_alone_and_names_what_it_left_out(tiny, tmp_path, capsys):
    run(tiny, tmp_path / "out")
    printed = capsys.readouterr().out
    assert (tmp_path / "out" / "fig_fleurs_asr_vs_baselines.pdf").stat().st_size > 1000
    assert (tmp_path / "out" / "fig_fleurs_asr_vs_baselines.png").stat().st_size > 1000
    rows = {line.split("  ")[0].strip(): line for line in printed.splitlines() if " label" in line}
    assert list(rows) == ["Alpha label", "Zed label", "Mid label", "Bad label"]  # by `order`; the start model is out
    assert rows["Alpha label"].split()[-2:] == ["5.0", "15.0"] and rows["Bad label"].split()[-2:] == ["80.0", "7.5"]
    assert "left out, the baseline table has them but none of our models does: uk" in printed
    assert "pl" not in printed.split("model", 1)[1].splitlines()[0]  # pl has no baseline, so it is not a column


def test_start_checkpoints_join_only_when_asked(tiny, tmp_path, capsys):
    run(tiny, tmp_path / "a", "--include-start")
    assert "Start label" in capsys.readouterr().out
    run(tiny, tmp_path / "b", "--ours", "Mid", "Start")
    assert "Start label" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no ours data under .* for: Nope"):
        run(tiny, tmp_path / "c", "--ours", "Mid", "Nope")


def test_the_output_is_reproducible_and_carries_the_hash_of_its_master(tiny, tmp_path):
    run(tiny, tmp_path / "one")
    run(tiny, tmp_path / "two")
    first = (tmp_path / "one" / "fig_fleurs_asr_vs_baselines.pdf").read_bytes()
    assert first == (tmp_path / "two" / "fig_fleurs_asr_vs_baselines.pdf").read_bytes()
    assert hashlib.sha256(tiny.read_bytes()).hexdigest()[:16].encode() in first


def test_a_deprecated_model_is_not_drawn(tiny, tmp_path, capsys):
    master.upsert(
        [], who="t", note="n", meta={"Mid": {"status": "deprecated"}}, master=tiny, overwrite=True, reason="r"
    )
    run(tiny, tmp_path / "out")
    assert "Mid label" not in capsys.readouterr().out


def test_a_master_with_conflict_markers_or_ragged_rows_is_refused(tiny, tmp_path):
    good = tiny.read_text()
    tiny.write_text("<<<<<<< HEAD\n" + good, encoding="utf-8")
    with pytest.raises(SystemExit, match="git conflict marker"):
        fig.load_master(tiny)
    tiny.write_text(good + "Extra,label,ours\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="header has"):
        fig.load_master(tiny)


def test_cut_bars_keep_their_value_and_the_poster_context_renders(tiny, tmp_path, capsys):
    run(tiny, tmp_path / "poster", "--context", "poster", "--formats", "png", "svg", "--ymax", "40")
    assert (tmp_path / "poster" / "fig_fleurs_asr_vs_baselines.png").stat().st_size > 1000
    assert "bars above 40 are cut" in capsys.readouterr().out
