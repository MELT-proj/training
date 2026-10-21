"""Tests for projects/self-distill/eval/tier1_joint.py: the EU24 joint ASR + ST score of the MA-utility arms, on synthetic scores."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "eval"))

import tier1_joint as tj  # noqa: E402
import tier1_tables as t1  # noqa: E402


LANGS = tj.EU24
NON_EN = [lang for lang in LANGS if lang != "en"]
OUTPUTS = "melt//mnt/scratch-nyx/giuseppe/melt/outputs-sft"


def cell(cer: float | dict, xen: float | dict, enx: float | dict) -> dict:
    """One (arm, step) cell as ``tier1_tables.load`` hands it over; a score is one number for every language or a dict."""

    def per_lang(value, langs, key):
        return {lang: {"n": 100, key: value[lang] if isinstance(value, dict) else value} for lang in langs}

    return {
        "asr": {"lang": per_lang(cer, LANGS, "cer")},
        "xen": {"lang": per_lang(xen, NON_EN, "chrf")},
        "enx": {"lang": per_lang(enx, NON_EN, "chrf")},
    }


def grid_rows() -> dict:
    """What ``build`` returns for the ten arms at the four steps: all improve with the step, and the arms differ a little."""
    rows: dict = {}
    for s, step in enumerate(tj.STEPS):
        for a, arm in enumerate(t1.ARMS):
            agg = tj.aggregates(cell(0.08 - 0.005 * s - 0.0005 * a, 48.0 + s + 0.1 * a, 40.0 + s + 0.05 * a))
            agg["selection"], agg["id_ma5"] = 0.06 - 0.003 * s - 0.0002 * a, 6.0 - 0.3 * s
            rows.setdefault(step, {})[arm.key] = agg
    return rows


def test_the_joint_score_is_the_mean_of_the_cer_complement_and_the_two_chrf_means():
    agg = tj.aggregates(cell(0.04, 50.0, 40.0))
    assert agg["asr24"] == pytest.approx(4.0)
    assert (agg["xen23"], agg["enx23"], agg["st46"]) == pytest.approx((50.0, 40.0, 45.0))
    assert agg["joint"] == pytest.approx((100 - 4.0 + 50.0 + 40.0) / 3)
    assert tj.joint_of(agg, (0.5, 0.25, 0.25)) == pytest.approx(0.5 * 96.0 + 0.25 * 50.0 + 0.25 * 40.0)


def test_asr_is_a_geometric_mean_of_clipped_cer_so_one_runaway_language_cannot_swamp_the_rest():
    agg = tj.aggregates(cell(dict.fromkeys(LANGS, 0.04) | {"ga": 3.0}, 50.0, 40.0))  # ga: 3x as long as the reference
    assert agg["asr24"] == pytest.approx(100 * math.exp((23 * math.log(0.04) + math.log(1.0)) / 24))
    assert agg["asr24_arith"] == pytest.approx(100 * (23 * 0.04 + 1.0) / 24)
    assert agg["asr24_med"] == pytest.approx(4.0)


def test_a_perfect_language_keeps_the_geometric_mean_finite():
    agg = tj.aggregates(cell(dict.fromkeys(LANGS, 0.04) | {"mt": 0.0}, 50.0, 40.0))
    assert agg["asr24"] == pytest.approx(100 * math.exp((23 * math.log(0.04) + math.log(tj.CER_FLOOR)) / 24))


def test_tiers_and_the_five_ma_languages_are_means_over_their_own_languages():
    agg = tj.aggregates(cell(dict.fromkeys(LANGS, 0.10) | dict.fromkeys(tj.MA_LANGS, 0.02), 50.0, 40.0))
    assert [len(tj.TIERS[tier]) for tier in ("high", "mid", "low")] == [7, 6, 11]
    assert agg["cer_ma5"] == pytest.approx(2.0)
    assert agg["asr_high"] == pytest.approx(100 * (5 * 0.02 + 2 * 0.10) / 7)  # en de es fr it, and nl pt
    assert agg["asr_mid"] == pytest.approx(10.0)
    assert agg["asr_low"] == pytest.approx(10.0)


def test_st_directions_split_into_the_ones_the_eu24_mix_trains_and_the_ones_it_never_saw():
    zero_xen, zero_enx = set(NON_EN) - tj.TRAINED["xen"], set(NON_EN) - tj.TRAINED["enx"]
    assert zero_xen == {"ga", "mt"}
    assert zero_enx == {"bg", "cs", "da", "el", "fi", "ga", "hr", "hu", "lt", "mt", "sk"}
    xen = {lang: 20.0 if lang in zero_xen else 50.0 for lang in NON_EN}
    enx = {lang: 10.0 if lang in zero_enx else 40.0 for lang in NON_EN}
    agg = tj.aggregates(cell(0.04, xen, enx))
    assert (agg["xen_trained"], agg["xen_zero"]) == pytest.approx((50.0, 20.0))
    assert (agg["enx_trained"], agg["enx_zero"]) == pytest.approx((40.0, 10.0))
    assert agg["xen23"] == pytest.approx((21 * 50.0 + 2 * 20.0) / 23)


def test_a_cell_with_a_family_not_scored_yet_has_no_aggregates_and_a_missing_language_is_an_error():
    scored = cell(0.04, 50.0, 40.0)
    assert tj.aggregates({**scored, "enx": None}) is None
    assert tj.aggregates({"asr": scored["asr"]}) is None
    del scored["asr"]["lang"]["mt"]
    with pytest.raises(ValueError, match="incomplete language set"):
        tj.aggregates(scored)


def test_the_frontier_keeps_the_arms_nobody_beats_on_both_axes_and_the_bands_must_be_exceeded():
    points = {"a": (5.0, 50.0), "b": (6.0, 49.0), "c": (5.5, 51.0), "d": (5.0, 50.0)}  # (CER, chrF)
    assert tj.pareto(points) == {"a", "c", "d"}  # a and d tie, so neither is beaten
    assert tj.clearly_dominated(points, 0.5, 0.5) == {"b"}  # a beats b by 1.0 on both axes
    assert tj.clearly_dominated(points, 1.0, 1.0) == set()  # a gap equal to the band is not beyond it
    assert tj.clearly_dominated(points, 0.4, 1.5) == {
        "b"
    }  # only c beats b by enough on chrF, and by more than 0.4 on CER


def test_sign_counts_leave_ties_out_and_the_p_value_is_an_exact_two_sided_binomial():
    better, worse = tj.aggregates(cell(0.04, 50.0, 40.0)), tj.aggregates(cell(0.05, 49.0, 40.0))
    counts = tj.sign_counts(better, worse)
    assert (counts["wins"], counts["losses"]) == (47, 0)  # 24 ASR + 23 X->en cells; the 23 en->X cells tie
    assert counts["p"] == pytest.approx(2 / 2**47)
    assert tj.sign_counts(worse, better)["wins"] == 0
    split = tj.sign_counts(tj.aggregates(cell(0.04, 51.0, 39.0)), tj.aggregates(cell(0.04, 50.0, 40.0)))
    assert (split["wins"], split["losses"], split["p"]) == (23, 23, 1.0)


def test_the_advantage_is_signed_so_that_positive_means_the_ma_start_is_better():
    ma, no_ma = tj.aggregates(cell(0.04, 51.0, 41.0)), tj.aggregates(cell(0.05, 50.0, 40.0))
    assert tj.advantage(ma, no_ma, "asr24") == pytest.approx(1.0)  # a lower CER is better
    assert tj.advantage(ma, no_ma, "xen23") == pytest.approx(1.0)
    assert tj.advantage(no_ma, ma, "joint") < 0


def test_the_seed_spread_is_the_mean_absolute_gap_between_an_arm_and_its_replicate():
    rows = {
        718: {"T0": {"joint": 60.0}, "T0'": {"joint": 60.5}, "A": {"joint": 61.0}, "A'": {"joint": 60.0}},
        1436: {"T0": {"joint": 62.0}, "T0'": {"joint": 61.9}},
    }
    gaps = tj.replicate_gaps(rows, ("joint",))
    assert gaps["joint"] == pytest.approx([0.5, 1.0, 0.1])
    assert tj.seed_spread(gaps)["joint"] == pytest.approx((0.5 + 1.0 + 0.1) / 3)


def test_the_report_has_every_section_and_a_row_per_arm_and_step():
    text = tj.render(grid_rows())
    for heading in (
        "### Step 718",
        "### Step 2,872",
        "## MA against no MA, on identical batches",
        "## Under other definitions",
        "## Which aggregate resolves the arms best",
        "## What the aggregates are made of",
    ):
        assert heading in text
    assert text.count("| T5 | 3,700 |") == 4
    assert (
        text.count("| frontier |") + text.count("| within the spread |") + text.count("beaten beyond the spread |")
        == 40
    )


def test_the_three_figures_are_written_as_png_and_pdf(tmp_path):
    rows = grid_rows()
    written = tj.pareto_figure(rows, tmp_path / "pareto") + tj.hours_figure(rows, tmp_path / "hours")
    written += tj.trajectory_figure(rows, tmp_path / "trajectories")
    assert sorted(path.suffix for path in written) == [".pdf"] * 3 + [".png"] * 3
    assert all(path.stat().st_size > 0 for path in written)


def test_the_command_line_writes_the_report_and_the_json_and_joins_the_old_score(tmp_path):
    summaries = []
    for init, seed in (("noma", 1337), ("noma", 1338), ("goldw", 1337), ("goldw", 1338)):
        for s, step in enumerate(tj.STEPS):
            scores = cell(
                (0.06 if init == "noma" else 0.05) - 0.004 * s + (0.001 if seed == 1338 else 0.0),
                48.0 + s + (0.5 if init == "goldw" else 0.0),
                40.0 + s,
            )
            model = f"{OUTPUTS}/SFT-EU24-from-{init}-bd60-ga4-8g-s{seed}/checkpoint-{step}"
            summaries += [{"model": model, "set": t1.SETS[f], "status": "success", **scores[f]} for f in scores]
    (tmp_path / "summaries.json").write_text(json.dumps(summaries))
    selection = [
        {"arm": "A", "step": 718, "result": {"score": 0.05, "medians": {"ID": 0.067}}},
        {"arm": "T0", "step": 718, "result": None},
    ]
    (tmp_path / "selection_tier1.json").write_text(json.dumps(selection))
    out = tmp_path / "out"
    args = [str(tmp_path / "summaries.json"), "--selection", str(tmp_path / "selection_tier1.json")]
    assert tj.main([*args, "--out-dir", str(out), "--no-figure"]) == 0
    report = (out / "joint_report.md").read_text()
    assert "| 718 | **pooled** (A, A') − (T0, T0') | **+1.00** | **+0.50** | **+0.00** | **+0.50** | |" in report
    tables = json.loads((out / "joint_tables.json").read_text())
    assert set(tables) == {"718", "1436", "2154", "2872"}
    assert tables["718"]["A"]["selection"] == 0.05
    assert tables["718"]["A"]["id_ma5"] == pytest.approx(6.7)
    assert tables["718"]["T0"]["selection"] is None
    assert not any(key.startswith("_") for key in tables["718"]["A"])
