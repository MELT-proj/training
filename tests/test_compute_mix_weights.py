"""Tests for infra/compute_mix_weights.py.

lhotse's multiplexer draws one *cut* per pick, so a mux ``weight`` is a cut
probability and a source supplies ``weight x mean cut length`` of audio. The
policy this tool implements is stated in hours; ``--weights cuts`` (the default)
converts it into the cut probabilities that deliver those hours, and
``--weights hours`` keeps writing the policy's p_c / p_l unchanged, byte for byte.

Manifests are written as plain JSONL by hand, and the process pool is replaced by
an inline executor: nyx enforces strict commit accounting, so forking inside a
test is the expensive part, not the arithmetic.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import compute_mix_weights as cmw  # noqa: E402


class _InlineExecutor:
    """Stands in for ProcessPoolExecutor so tests never fork."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable, chunksize=1):
        return map(fn, iterable)


@pytest.fixture(autouse=True)
def _no_fork(monkeypatch):
    monkeypatch.setattr(cmw, "ProcessPoolExecutor", _InlineExecutor)


# ---------------------------------------------------------------------------
# Measuring cut counts
# ---------------------------------------------------------------------------


def test_shard_stats_counts_cuts_and_seconds_and_skips_bad_lines(tmp_path):
    shard = tmp_path / "cuts.000000.jsonl"
    shard.write_text(
        json.dumps({"duration": 2.5, "supervisions": [{"duration": 99.0}]}) + "\n"
        "\n"
        "not json\n" + json.dumps({"id": "no-duration"}) + "\n" + json.dumps({"duration": 4.0}) + "\n",
        encoding="utf-8",
    )
    # Supervision durations are not counted, and unparsable lines are neither summed nor counted.
    assert cmw.shard_stats(str(shard)) == (6.5, 2)
    assert cmw.shard_seconds(str(shard)) == 6.5


def test_shard_stats_reads_gzip(tmp_path):
    shard = tmp_path / "cuts.000000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"duration": 3.0}) + "\n" + json.dumps({"duration": 1.0}) + "\n")
    assert cmw.shard_stats(str(shard)) == (4.0, 2)


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

# group -> [(target hours, mean cut seconds)]: equal hours, but b's cuts are short.
TARGETS = [[60.0, 40.0], [100.0]]
MEAN_CUT = [[30.0, 10.0], [5.0]]


def test_cut_weights_deliver_the_target_audio():
    group_w, leaf_w = cmw.cut_mux_weights(TARGETS, MEAN_CUT)

    n_a = 60 * 3600 / 30 + 40 * 3600 / 10
    n_b = 100 * 3600 / 5
    assert group_w == pytest.approx([n_a / (n_a + n_b), n_b / (n_a + n_b)])
    assert sum(group_w) == pytest.approx(1.0)
    assert all(sum(row) == pytest.approx(1.0) for row in leaf_w)

    shares = cmw.implied_hour_shares(group_w, leaf_w, MEAN_CUT)
    assert shares[0] == pytest.approx([0.3, 0.2])
    assert shares[1] == pytest.approx([0.5])
    cmw.check_hour_shares(TARGETS, group_w, leaf_w, MEAN_CUT)  # no error


def test_cut_weights_match_a_simulated_per_cut_draw():
    """Independent of the formula: draw cuts the way lhotse's mux does and add up the audio."""
    group_w, leaf_w = cmw.cut_mux_weights(TARGETS, MEAN_CUT)
    leaves = [(g, i, gw * lw) for g, (gw, lws) in enumerate(zip(group_w, leaf_w)) for i, lw in enumerate(lws)]
    rng = random.Random(0)
    picks = rng.choices(range(len(leaves)), weights=[p for _, _, p in leaves], k=200_000)
    seconds = {(g, i): 0.0 for g, i, _ in leaves}
    for k in picks:
        g, i, _ = leaves[k]
        seconds[(g, i)] += MEAN_CUT[g][i]
    total = sum(seconds.values())
    assert seconds[(0, 0)] / total == pytest.approx(0.3, abs=0.01)
    assert seconds[(0, 1)] / total == pytest.approx(0.2, abs=0.01)
    assert seconds[(1, 0)] / total == pytest.approx(0.5, abs=0.01)


def test_hour_share_weights_would_over_draw_long_cuts():
    """The behaviour being replaced: weights equal to the hour shares give long-cut sources too much audio."""
    group_w, leaf_w = [0.5, 0.5], [[0.6, 0.4], [1.0]]
    shares = cmw.implied_hour_shares(group_w, leaf_w, MEAN_CUT)
    assert sum(shares[0]) > 0.8  # group a was meant to get 50%
    with pytest.raises(ValueError, match="would supply"):
        cmw.check_hour_shares(TARGETS, group_w, leaf_w, MEAN_CUT)


def test_cut_weights_equal_hour_weights_when_every_cut_is_the_same_length():
    same = [[12.0, 12.0], [12.0]]
    group_w, leaf_w = cmw.cut_mux_weights(TARGETS, same)
    assert group_w == pytest.approx([0.5, 0.5])
    assert leaf_w[0] == pytest.approx([0.6, 0.4])
    assert leaf_w[1] == pytest.approx([1.0])


def test_a_group_with_no_target_is_never_drawn():
    group_w, leaf_w = cmw.cut_mux_weights([[50.0], [0.0, 0.0]], [[10.0], [10.0, 10.0]])
    assert group_w == pytest.approx([1.0, 0.0])
    assert leaf_w[1] == pytest.approx([0.5, 0.5])  # moot, but a valid distribution
    with pytest.raises(ValueError, match="every target is zero"):
        cmw.cut_mux_weights([[0.0]], [[10.0]])


def test_mean_cut_needs_a_measured_cut_count():
    assert cmw.mean_cut_seconds(2.0, 720, "x") == pytest.approx(10.0)
    with pytest.raises(ValueError, match="No cut count measured for x/train"):
        cmw.mean_cut_seconds(2.0, None, "x/train")
    with pytest.raises(ValueError, match="No cut count measured"):
        cmw.mean_cut_seconds(0.0, 0, "x/train")


def _src(path, lang, hours, cuts, task="asr", extra=None):
    tags = {"task": task, "lang": lang}
    if task == "st":
        tags = {"task": "st", "src_lang": lang.split("-")[0], "tgt_lang": lang.split("-")[1]}
    tags.update(extra or {})
    return {
        "path": f"/data/{path}",
        "raw_path": f"${{oc.env:LOCAL_DATASETS_DIR}}/{path}",
        "tags": tags,
        "task": task,
        "lang_key": lang,
        "hours": hours,
        "cuts": cuts,
    }


def _policy_sources(with_cuts: bool = True):
    """Nine sources in four entries; cut lengths differ ~25x between corpora."""
    cuts = lambda hours, mean: round(hours * 3600 / mean) if with_cuts else None  # noqa: E731
    return [
        _src("cv/en", "en", 1800.0, cuts(1800.0, 5.7)),
        _src("mls/en", "en", 39000.0, cuts(39000.0, 14.9), extra={"text_field": "custom.pnc_text"}),
        _src("vox/en", "en", 500.0, cuts(500.0, 10.3)),
        _src("cv/de", "de", 960.0, cuts(960.0, 5.7)),
        _src("mls/de", "de", 1900.0, cuts(1900.0, 15.1)),
        _src("vox/de", "de", 260.0, cuts(260.0, 8.8)),
        _src("cv/it", "it", 250.0, cuts(250.0, 5.3)),
        _src("mls/it", "it", 240.0, cuts(240.0, 14.9)),
        _src(
            "yodas/de-en",
            "de-en",
            9000.0,
            cuts(9000.0, 24.0),
            task="st",
            extra={"text_field": "custom.translation_en"},
        ),
    ]


def test_to_cut_probabilities_delivers_the_policys_audio_shares():
    sources = cmw.compute_weights(_policy_sources(), 0.5, 0.5)
    policy = {s["path"]: s["p_cl"] for s in sources}
    cmw.to_cut_probabilities(sources)

    assert sum(s["p_cl"] for s in sources) == pytest.approx(1.0)
    audio = {s["path"]: s["p_cl"] * s["mean_cut"] for s in sources}
    total = sum(audio.values())
    for path, share in policy.items():
        assert audio[path] / total == pytest.approx(share, rel=1e-9)
    # The policy's own numbers are kept for reporting, and the weights moved the way the lengths say.
    by_path = {s["path"]: s for s in sources}
    assert by_path["/data/cv/en"]["p_cl_hours"] == pytest.approx(policy["/data/cv/en"])
    assert by_path["/data/cv/en"]["p_cl"] > by_path["/data/cv/en"]["p_cl_hours"]  # short cuts need more draws
    assert by_path["/data/yodas/de-en"]["p_cl"] < by_path["/data/yodas/de-en"]["p_cl_hours"]


def test_to_cut_probabilities_needs_cut_counts():
    sources = cmw.compute_weights(_policy_sources(with_cuts=False), 0.5, 0.5)
    with pytest.raises(ValueError, match="No cut count measured for /data/cv/en"):
        cmw.to_cut_probabilities(sources)


def test_a_source_with_no_hours_gets_no_weight_and_needs_no_cuts():
    sources = _policy_sources()
    sources.append(_src("empty/fr", "fr", 0.0, None))
    cmw.compute_weights(sources, 0.5, 0.5)
    cmw.to_cut_probabilities(sources)
    empty = sources[-1]
    assert empty["p_cl"] == 0.0 and empty["mean_cut"] is None


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------

# Built from the same nine sources with the code as it was before `--weights`
# existed. Hours mode must keep reproducing these exactly, so older configs can
# be re-emitted unchanged.
GOLDEN = {
    "header": "8fd7a8eef9e85b48ff0479676358138e0c22875d948a55dc9f434cebc45ac6ed",
    "block": "ecf6546dc8b525485248e2e910018743e8d57b11a0d5bb4569f85eb41952b46e",
    "block_indented": "83cfd154edcec6852fe1a5e93712ba594d7453ed2cf151149d0dd917e3a2971b",
    "config": "0f05fa8ed355015a5e90d2829693080d0cbbf5769855ef5285a4f25b727246ad",
    "nemo": "f507cb17ddc599caf0979fe0db8071517fbc172e91e80756a031cd946865804c",
}

TEMPLATE = "run:\n  exp_name: old\ndata:\n  train_ds:\n    total_hours: 1.0\n    input_cfg:\n      - placeholder\n  validation_ds:\n    total_hours: 2.0\n"


def _digest(text: str | bytes) -> str:
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


def test_hours_mode_output_is_byte_identical_to_before(tmp_path):
    sources = cmw.compute_weights(_policy_sources(), 0.5, 0.5)
    template = tmp_path / "tmpl.yaml"
    template.write_text(TEMPLATE, encoding="utf-8")
    cmw.emit_training_config(sources, template, tmp_path / "out.yaml", 0.5, 0.5, "exp1")
    cmw.emit_nemo_group_yaml(sources, tmp_path / "nemo.yaml", 0.5, 0.5)

    assert _digest("\n".join(cmw.header_comment(sources, 0.5, 0.5))) == GOLDEN["header"]
    assert _digest("\n".join(cmw.build_group_block(sources))) == GOLDEN["block"]
    assert _digest("\n".join(cmw.build_group_block(sources, indent=4))) == GOLDEN["block_indented"]
    assert _digest((tmp_path / "out.yaml").read_bytes()) == GOLDEN["config"]
    assert _digest((tmp_path / "nemo.yaml").read_bytes()) == GOLDEN["nemo"]


def test_cut_mode_output_explains_itself_and_keeps_the_policy_visible(tmp_path):
    sources = cmw.compute_weights(_policy_sources(), 0.5, 0.5)
    cmw.to_cut_probabilities(sources)
    cmw.emit_nemo_group_yaml(sources, tmp_path / "nemo.yaml", 0.5, 0.5, weights="cuts")
    text = (tmp_path / "nemo.yaml").read_text(encoding="utf-8")

    assert "CUT probabilities" in text
    assert "Group weight   = p_l" not in text
    assert "% of the cuts, " in text and "% of the audio" in text
    assert "mean cut 5.7 s" in text and "mean cut 24.0 s" in text


def test_cut_mode_nemo_file_delivers_the_policy(tmp_path):
    yaml = pytest.importorskip("yaml")
    sources = cmw.compute_weights(_policy_sources(), 0.5, 0.5)
    policy = {s["raw_path"]: s["p_cl"] for s in sources}
    mean_cut = {s["raw_path"]: s["hours"] * 3600 / s["cuts"] for s in sources}
    cmw.to_cut_probabilities(sources)
    cmw.emit_nemo_group_yaml(sources, tmp_path / "nemo.yaml", 0.5, 0.5, weights="cuts")

    groups = yaml.safe_load(
        (tmp_path / "nemo.yaml").read_text(encoding="utf-8").replace("${oc.env:LOCAL_DATASETS_DIR}", "ROOT")
    )["input_cfg"]
    audio = {}
    for group in groups:
        for leaf in group["input_cfg"]:
            raw = "${oc.env:LOCAL_DATASETS_DIR}/" + leaf["shar_path"][len("ROOT/") :]
            audio[raw] = group["weight"] * leaf["weight"] * mean_cut[raw]
    total = sum(audio.values())
    for raw, share in policy.items():
        assert audio[raw] / total == pytest.approx(share, abs=1e-6)


# ---------------------------------------------------------------------------
# End to end: the CLI over a small fake collection
# ---------------------------------------------------------------------------

CONFIG = """\
data:
  train_ds:
    input_cfg:
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/cv/en
        tags: {task: asr, lang: en}
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/yt/en
        tags: {task: asr, lang: en}
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/cv/de
        tags: {task: asr, lang: de}
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/yt/de
        tags: {task: asr, lang: de}
"""

# The same collection with cv/en listed a second time, as English's top-up and its
# yodas-granary entry are in the campaign config.
CONFIG_WITH_REPEAT = (
    CONFIG
    + """\
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/cv/en
        tags: {task: asr, lang: en}
"""
)

# path -> (total hours, cut length): CommonVoice-like short cuts, YouTube-like long ones.
COLLECTION = {
    "cv/en": (1.0, 6.0),
    "yt/en": (2.0, 30.0),
    "cv/de": (1.0, 6.0),
    "yt/de": (4.0, 30.0),
}


def _make_collection(root: Path) -> None:
    for rel, (hours, cut) in COLLECTION.items():
        directory = root / rel
        directory.mkdir(parents=True)
        lines = [json.dumps({"id": f"c{i}", "duration": cut}) for i in range(round(hours * 3600 / cut))]
        (directory / "cuts.000000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(tmp_path, monkeypatch, *flags: str, config_text: str = CONFIG):
    root = tmp_path / "shar"
    if not root.exists():
        _make_collection(root)
    # load_sources() writes these into the process environment; let monkeypatch undo it.
    for var, value in (("LOCAL_DATASETS_DIR", str(root)), ("OUTPUT_DIR", "/unused"), ("HF_HOME", "/unused")):
        monkeypatch.setenv(var, value)
    config = tmp_path / "cfg.yaml"
    config.write_text(config_text, encoding="utf-8")
    argv = [
        "compute_mix_weights.py",
        "--config",
        str(config),
        "--datasets-root",
        str(root),
        "--cache",
        str(tmp_path / "cache.json"),
        "--emit-nemo",
        str(tmp_path / "nemo.yaml"),
        *flags,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cmw.main()
    yaml = pytest.importorskip("yaml")
    text = (tmp_path / "nemo.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text.replace("${oc.env:LOCAL_DATASETS_DIR}", "ROOT"))["input_cfg"], text


def _policy_shares() -> dict[str, float]:
    """The alpha = beta = 0.5 policy over the collection's true hours, computed without the tool's conversion."""
    sources = [{"path": rel, "lang_key": rel.split("/")[1], "hours": hours} for rel, (hours, _) in COLLECTION.items()]
    cmw.compute_weights(sources, 0.5, 0.5)
    return {s["path"]: s["p_cl"] for s in sources}


def test_cli_cut_mode_gives_each_source_the_audio_the_policy_assigns(tmp_path, monkeypatch, capsys):
    groups, text = _run(tmp_path, monkeypatch)  # --weights cuts is the default
    audio = {}
    for group in groups:
        for leaf in group["input_cfg"]:
            rel = leaf["shar_path"][len("ROOT/") :]
            audio[rel] = group["weight"] * leaf["weight"] * COLLECTION[rel][1]
    total = sum(audio.values())
    for rel, share in _policy_shares().items():
        assert audio[rel] / total == pytest.approx(share, abs=1e-6), rel
    assert "CUT probabilities" in text
    assert "WARNING: --weights hours" not in capsys.readouterr().out


def test_cli_hours_mode_writes_the_policys_weights_and_warns(tmp_path, monkeypatch, capsys):
    groups, text = _run(tmp_path, monkeypatch, "--weights", "hours")
    sources = [{"path": rel, "lang_key": rel.split("/")[1], "hours": hours} for rel, (hours, _) in COLLECTION.items()]
    cmw.compute_weights(sources, 0.5, 0.5)
    p_l = {s["lang_key"]: s["p_l"] for s in sources}
    p_c = {s["path"]: s["p_c"] for s in sources}
    for group in groups:
        assert group["weight"] == pytest.approx(p_l[group["tags"]["lang"]], abs=1e-8)
        for leaf in group["input_cfg"]:
            assert leaf["weight"] == pytest.approx(p_c[leaf["shar_path"][len("ROOT/") :]], abs=1e-8)
    assert "CUT probabilities" not in text
    assert "WARNING: --weights hours" in capsys.readouterr().out


def test_cli_records_cuts_and_reads_a_path_listed_twice_only_once(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, config_text=CONFIG_WITH_REPEAT)
    cache = json.loads((tmp_path / "cache.json").read_text())
    entry = cache[str(tmp_path / "shar" / "cv/en")]  # listed twice in the config
    assert entry["hours"] == pytest.approx(1.0)
    assert entry["cuts"] == 600


def test_cli_re_reads_a_cache_that_has_hours_but_no_cuts(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    stale = {str(tmp_path / "shar" / rel): {"hours": 999.0, "shards": 1, "sample": None} for rel in COLLECTION}
    cache_path.write_text(json.dumps(stale), encoding="utf-8")

    _run(tmp_path, monkeypatch, "--weights", "hours")  # hours are enough here, so the stale numbers stand
    assert json.loads(cache_path.read_text())[str(tmp_path / "shar" / "cv/de")]["hours"] == 999.0

    _run(tmp_path, monkeypatch)  # cut weights need cut counts, so every source is read again
    fresh = json.loads(cache_path.read_text())[str(tmp_path / "shar" / "cv/de")]
    assert fresh["hours"] == pytest.approx(1.0) and fresh["cuts"] == 600
