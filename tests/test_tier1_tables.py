"""Tests for projects/self-distill/eval/tier1_tables.py: the MA-utility arms' tables and scatter, on synthetic score summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "eval"))

import tier1_tables as t1  # noqa: E402


ALL24 = [
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
]
HR5 = ["en", "de", "fr", "es", "it"]
OUTPUTS = "melt//mnt/scratch-nyx/giuseppe/melt/outputs-sft"


def run_name(init: str, seed: int = 1337) -> str:
    return f"SFT-EU24-from-{init}-bd60-ga4-8g-s{seed}"


def model_path(init: str, step: int | None, seed: int = 1337) -> str:
    base = f"{OUTPUTS}/{run_name(init, seed)}"
    return base if step is None else f"{base}/checkpoint-{step}"


def rows_for(
    init: str,
    step: int,
    *,
    wer: float,
    cer: float,
    xen: float,
    enx: float,
    on: int,
    runaway: float = 0.0,
    seed: int = 1337,
):
    """The three summary rows and the language-share rows one checkpoint contributes."""
    model = model_path(init, step, seed)
    asr = {
        "model": model, "set": "fleurs24-asr-dev", "status": "success",
        "lang": {lang: {"n": 100, "wer": wer if lang in HR5 else 0.9, "cer": 0.5 if lang in HR5 else cer, "runaway": runaway} for lang in ALL24},
    }  # fmt: skip
    st = lambda name, value: {  # noqa: E731
        "model": model, "set": name, "status": "success",
        "lang": {lang: {"n": 100, "chrf": value if lang in HR5 else value / 2} for lang in ALL24 if lang != "en"},
    }  # fmt: skip
    share = [
        {"model": model, "set": "fleurs24-st-enx-dev", "key": lang, "n": 100, "on": on if lang in HR5 else on // 2}
        for lang in ALL24
        if lang != "en"
    ]
    return [asr, st("fleurs24-st-xen-dev", xen), st("fleurs24-st-enx-dev", enx)], share


@pytest.fixture
def data(tmp_path):
    summaries, shares = [], []
    for init, seed, step, kw in (
        ("noma", 1337, 718, {"wer": 0.30, "cer": 0.20, "xen": 10.0, "enx": 20.0, "on": 60, "runaway": 0.02}),
        ("noma", 1338, 718, {"wer": 0.32, "cer": 0.22, "xen": 12.0, "enx": 21.0, "on": 62}),
        ("goldw", 1337, 718, {"wer": 0.10, "cer": 0.10, "xen": 40.0, "enx": 30.0, "on": 90}),
        ("goldw", 1338, 718, {"wer": 0.11, "cer": 0.12, "xen": 41.0, "enx": 31.0, "on": 92}),
        ("goldw", 1337, 2872, {"wer": 0.07, "cer": 0.08, "xen": 50.0, "enx": 35.0, "on": 95}),
        ("sd-azeros", 1337, 718, {"wer": 0.09, "cer": 0.11, "xen": 45.0, "enx": 33.0, "on": 91}),
        ("ce24", 1337, 718, {"wer": 0.08, "cer": 0.10, "xen": 15.0, "enx": 12.0, "on": 40}),
    ):
        r, s = rows_for(init, step, seed=seed, **kw)
        summaries += r
        shares += s
    other = {
        "model": "melt//elsewhere/SFT-HR5-from-goldw-bd60-ga16-2g-s1337/checkpoint-718",
        "set": "fleurs24-asr-dev",
        "status": "success",
        "lang": {},
    }
    failed = {**rows_for("goldw", 1436, wer=0.0, cer=0.0, xen=0.0, enx=0.0, on=0)[0][0], "status": "failed"}
    (tmp_path / "summaries.json").write_text(json.dumps([*summaries, other, failed]))
    (tmp_path / "langshare.json").write_text(json.dumps({"lowercase": True, "rows": shares}))
    return tmp_path


def test_models_are_classified_into_arms_steps_and_seeds():
    assert t1.classify_arm(model_path("noma", 718)) == {
        "cond": "EU24",
        "init": "noma",
        "seed": 1337,
        "step": 718,
        "final": False,
    }
    assert t1.classify_arm(model_path("sd-azeros", 2872, 1337))["init"] == "sd-azeros"
    assert t1.classify_arm(model_path("goldw", None, 1338)) == {
        "cond": "EU24",
        "init": "goldw",
        "seed": 1338,
        "step": 11499,
        "final": True,
    }
    assert t1.classify_arm(model_path("ce24", 718, 4242)) is None  # a seed no arm has
    assert t1.classify_arm("melt//x/SFT-HR5-from-goldw-bd60-ga16-2g-s1337/checkpoint-718") is None  # another condition
    assert (
        t1.classify_arm(
            "melt//x/GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g"
        )
        is None
    )


def test_every_arm_has_a_distinct_start_and_seed_and_the_replicates_exist():
    assert len({(a.init, a.seed) for a in t1.ARMS}) == len(t1.ARMS)
    keys = {a.key for a in t1.ARMS}
    assert all(first in keys and second in keys for first, second in t1.REPLICATES.items())
    assert [a.ma_hours for a in t1.ARMS if a.key in ("T0", "T1", "T2", "A", "B", "T3")] == [
        0,
        120,
        360,
        1000,
        1050,
        3700,
    ]


def test_the_metrics_follow_the_step5_definitions(data):
    cells = t1.load(str(data / "summaries.json"), str(data / "langshare.json"))
    assert set(cells) == {("T0", 718), ("T0'", 718), ("A", 718), ("A'", 718), ("A", 2872), ("B", 718), ("T3", 718)}
    m = t1.metrics_of(cells[("T0", 718)], selection=0.0564)
    assert m["wer_hr5"] == pytest.approx(30.0)  # mean WER over the five HR languages, in percent
    assert m["cer_other"] == pytest.approx(20.0)  # mean CER over the other 19
    assert m["xen_hr"] == pytest.approx(10.0) and m["xen_other"] == pytest.approx(
        5.0
    )  # chrF: HR4 sources, the other 19
    assert m["enx_hr"] == pytest.approx(20.0) and m["enx_other"] == pytest.approx(10.0)
    assert m["share_hr"] == pytest.approx(60.0) and m["share_other"] == pytest.approx(30.0)
    assert (m["runaways"], m["clips"]) == (48, 2400)  # 2% of 24 x 100 clips
    assert m["selection"] == 0.0564


def test_an_arm_with_nothing_scored_yet_has_no_numbers_not_zeros(data):
    cells = t1.load(str(data / "summaries.json"), None)  # no language shares
    m = t1.metrics_of(cells[("A", 718)])
    assert m["share_hr"] is None and m["selection"] is None
    assert t1.metrics_of({})["wer_hr5"] is None


def test_build_orders_steps_and_measures_the_seed_replicates(data):
    result = t1.build(t1.load(str(data / "summaries.json"), str(data / "langshare.json")), {})
    assert result["steps"] == [718, 2872]
    assert set(result["rows"][718]) == {"T0", "T0'", "A", "A'", "B", "T3"}
    gap = result["replicate_gaps"][718]
    assert gap["T0 vs T0'"]["wer_hr5"] == pytest.approx(2.0)
    assert gap["A vs A'"]["wer_hr5"] == pytest.approx(1.0)
    assert 2872 not in result["replicate_gaps"]  # A' has no step-2,872 row yet


def test_selection_scores_join_by_arm_and_step(tmp_path, data):
    path = tmp_path / "selection_tier1.json"
    path.write_text(
        json.dumps(
            [{"arm": "A", "step": 718, "result": {"score": 0.0701}}, {"arm": "T0", "step": 718, "result": None}]
        )
    )
    scores = t1.load_selection(str(path))
    assert scores == {("A", 718): 0.0701}
    result = t1.build(t1.load(str(data / "summaries.json"), None), scores)
    assert result["rows"][718]["A"]["selection"] == 0.0701
    assert result["rows"][718]["B"]["selection"] is None
    assert t1.load_selection(None) == {}


def test_the_markdown_lists_the_arms_in_registry_order_with_their_ma_hours(data):
    result = t1.build(t1.load(str(data / "summaries.json"), str(data / "langshare.json")), {})
    text = t1.render(result)
    assert "## Step 718" in text and "## Step 2872" in text
    order = [
        text.index(label)
        for label in ("T0 no MA |", "A CE 1,000 h |", "B CE 1,000 + SD-mix 50 h |", "T3 CE 1,000 + CE 2,700 h |")
    ]
    assert order == sorted(order)
    assert "| T3 CE 1,000 + CE 2,700 h | 3,700 |" in text
    assert "48/2400" in text  # T0's runaways
    assert "## Seed replicates" in text and "T0 vs T0'" in text


def test_the_scatter_figure_is_written_as_png_and_pdf(tmp_path, data):
    pytest.importorskip("matplotlib")
    result = t1.build(t1.load(str(data / "summaries.json"), str(data / "langshare.json")), {})
    written = t1.figure(result, tmp_path / "scatter", ("wer_hr5", "xen_hr"))
    assert [p.suffix for p in written] == [".png", ".pdf"]
    assert all(p.stat().st_size > 1000 for p in written)


def test_the_command_line_writes_the_tables_and_a_json(tmp_path, data):
    out = tmp_path / "out"
    assert (
        t1.main([str(data / "summaries.json"), str(data / "langshare.json"), "--out-dir", str(out), "--no-figure"])
        == 0
    )
    assert (out / "tier1_tables.md").read_text().startswith("# MA utility, Tier 1")
    saved = json.loads((out / "tier1_tables.json").read_text())
    assert [a["key"] for a in saved["arms"]][:3] == ["T0", "T0'", "T1"]
    assert saved["rows"]["718"]["A"]["wer_hr5"] == pytest.approx(10.0)
