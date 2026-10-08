"""Tests for projects/ablation-campaign/build_campaign_config.py.

lhotse's multiplexer draws one *cut* per pick, so a mux ``weight`` is a cut
probability, not an hours share: a source supplies ``weight x mean cut length``
of audio per draw. The builder used to write hour shares as weights, which
over-draws sources with long cuts (English got 11.9% of the audio instead of
20% on the 2,100 h config). ``--weights cuts`` converts hours to cut
probabilities; ``--weights hours`` keeps the old behaviour byte for byte.

Manifests are written as plain JSONL by hand, and the process pool is replaced
by an inline executor: nyx enforces strict commit accounting, so forking inside
a test is the expensive part, not the arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import build_campaign_config as bcc  # noqa: E402


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
    monkeypatch.setattr(bcc, "ProcessPoolExecutor", _InlineExecutor)


def write_manifest(directory: Path, durations: list[float], name: str = "cuts.000000.jsonl") -> None:
    """One plain JSONL cut manifest; only the top-level ``duration`` matters here."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"id": f"c{i}", "duration": d, "supervisions": [{"duration": d}]}) for i, d in enumerate(durations)
    ]
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Measuring cut counts
# ---------------------------------------------------------------------------


def test_measure_records_hours_and_cuts_together(tmp_path):
    root = tmp_path / "shar"
    write_manifest(root / "a/train", [10.0] * 360)  # 1 h, 360 cuts
    cache: dict = {}
    hours, cuts = bcc.measure(["a/train"], root, cache, None, None, 1)
    assert hours == {"a/train": pytest.approx(1.0)}
    assert cuts == {"a/train": 360}
    assert cache["a/train"] == {"hours": pytest.approx(1.0), "cuts": 360}


def test_measure_reads_a_legacy_hours_float_but_only_when_cuts_are_needed(tmp_path):
    root = tmp_path / "shar"
    write_manifest(root / "a/train", [10.0] * 360)
    cache = {"a/train": 123.0}  # the bare float older caches held; it disagrees with the data

    hours, cuts = bcc.measure(["a/train"], root, cache, None, None, 1)
    assert hours == {"a/train": 123.0} and cuts == {}  # not needed, so not re-read

    hours, cuts = bcc.measure(["a/train"], root, cache, None, None, 1, need_cuts={"a/train"})
    assert hours == {"a/train": pytest.approx(1.0)} and cuts == {"a/train": 360}


def test_measure_reads_a_directory_listed_twice_only_once(tmp_path):
    """English's top-up and its yodas-granary entry are one directory; it must not be counted twice."""
    root = tmp_path / "shar"
    write_manifest(root / "a/train", [10.0] * 360)
    hours, cuts = bcc.measure(["a/train", "a/train"], root, {}, None, None, 1, need_cuts={"a/train"})
    assert hours == {"a/train": pytest.approx(1.0)}
    assert cuts == {"a/train": 360}


def test_measure_extrapolates_cuts_with_hours_when_sampling(tmp_path):
    root = tmp_path / "shar"
    for i in range(4):
        write_manifest(root / "a/train", [10.0] * 90, name=f"cuts.{i:06d}.jsonl")
    cache: dict = {}
    hours, cuts = bcc.measure(["a/train"], root, cache, None, 2, 1)
    assert hours["a/train"] == pytest.approx(1.0)  # 2 of 4 shards read, scaled by 2
    assert cuts["a/train"] == 360


# ---------------------------------------------------------------------------
# The weights
# ---------------------------------------------------------------------------


def _groups(spec: dict[str, list[tuple[str, float, float]]]):
    """spec: group -> [(path, hours, mean cut seconds)] -> (groups, hours, cuts)."""
    groups, hours, cuts = [], {}, {}
    for name, leaves in spec.items():
        for path, h, d in leaves:
            hours[path] = h * 10  # the pool is bigger than the draw; only the ratio matters
            cuts[path] = round(hours[path] * 3600 / d)
        groups.append((name, sum(h for _, h, _ in leaves), [(path, None, h) for path, h, _ in leaves]))
    return groups, hours, cuts


SPEC = {
    "asr:a": [("a/long", 60.0, 30.0), ("a/mid", 40.0, 10.0)],
    "asr:b": [("b/short", 100.0, 5.0)],
}


def test_cut_weights_deliver_the_intended_hours():
    groups, hours, cuts = _groups(SPEC)
    group_w, leaf_w = bcc.mux_weights(groups, hours, cuts)

    # Equal hours, but b's cuts are short, so b needs most of the draws.
    n_a = 60 * 3600 / 30 + 40 * 3600 / 10
    n_b = 100 * 3600 / 5
    assert group_w == pytest.approx([n_a / (n_a + n_b), n_b / (n_a + n_b)])
    assert sum(group_w) == pytest.approx(1.0)
    assert all(sum(row) == pytest.approx(1.0) for row in leaf_w)

    shares = bcc.implied_hour_shares(groups, group_w, leaf_w, hours, cuts)
    assert shares[0] == pytest.approx([0.3, 0.2])
    assert shares[1] == pytest.approx([0.5])


def test_cut_weights_match_a_simulated_per_cut_draw():
    """Independent of the formula: draw cuts the way lhotse's mux does and add up the audio."""
    groups, hours, cuts = _groups(SPEC)
    group_w, leaf_w = bcc.mux_weights(groups, hours, cuts)
    leaves = [
        (rel, gw * lw) for (_, _, ls), gw, lws in zip(groups, group_w, leaf_w) for (rel, _, _), lw in zip(ls, lws)
    ]
    rng = random.Random(0)
    picks = rng.choices(range(len(leaves)), weights=[p for _, p in leaves], k=200_000)
    seconds = {rel: 0.0 for rel, _ in leaves}
    for i in picks:
        rel = leaves[i][0]
        seconds[rel] += hours[rel] * 3600 / cuts[rel]
    total = sum(seconds.values())
    assert seconds["a/long"] / total == pytest.approx(0.3, abs=0.01)
    assert seconds["a/mid"] / total == pytest.approx(0.2, abs=0.01)
    assert seconds["b/short"] / total == pytest.approx(0.5, abs=0.01)


def test_hour_share_weights_over_draw_long_cuts():
    """The behaviour being replaced: hour-share weights give long-cut sources too much audio."""
    groups, hours, cuts = _groups(SPEC)
    group_w, leaf_w = bcc.mux_weights(groups, hours, None)
    assert group_w == pytest.approx([0.5, 0.5])
    assert leaf_w == [pytest.approx([0.6, 0.4]), [1.0]]
    shares = bcc.implied_hour_shares(groups, group_w, leaf_w, hours, cuts)
    group_a = sum(shares[0])
    assert group_a == pytest.approx(0.5 * (0.6 * 30 + 0.4 * 10) / (0.5 * (0.6 * 30 + 0.4 * 10) + 0.5 * 5))
    assert group_a > 0.8  # a was meant to get 50%


def test_cut_weights_equal_hour_weights_when_every_cut_is_the_same_length():
    spec = {"asr:a": [("a/x", 70.0, 12.0), ("a/y", 30.0, 12.0)], "asr:b": [("b/x", 100.0, 12.0)]}
    groups, hours, cuts = _groups(spec)
    cut_group_w, cut_leaf_w = bcc.mux_weights(groups, hours, cuts)
    hour_group_w, hour_leaf_w = bcc.mux_weights(groups, hours, None)
    assert cut_group_w == pytest.approx(hour_group_w)
    for got, want in zip(cut_leaf_w, hour_leaf_w):
        assert got == pytest.approx(want)


def test_cut_weights_need_a_measured_cut_count():
    groups, hours, cuts = _groups(SPEC)
    del cuts["a/mid"]
    with pytest.raises(SystemExit, match="No cut count measured for a/mid"):
        bcc.mux_weights(groups, hours, cuts)


# ---------------------------------------------------------------------------
# The rendered block
# ---------------------------------------------------------------------------

# Built from fixed inputs with the code as it was before cut weights existed;
# `--weights hours` must keep reproducing these exactly, so older configs can be
# re-rendered unchanged.
LEGACY_TEMPLATE = {"cv22_sidon": 0.3, "mls_sidon": 0.25, "yodas-granary": 0.2, "voxpopuli": 0.1, "yodas3": 0.15}
LEGACY_DIGEST = {
    "asr": "cd5edcfef62bd3613315d7dcb1af8b743932598e0464f1e4cefe259b84646fd5",
    "both": "5dc7a84ed2415f066be16ae36bf95b7e1c9c1ede494372f31b8ce5111d4cd687",
}


def _legacy_hours() -> dict[str, float]:
    hours = {rel: 1000.0 for corpus in bcc.ASR_SOURCES.values() for rel in corpus.values()}
    for spec in list(bcc.ST_SOURCES.values()) + list(bcc.ST_PROBE.values()):
        hours[spec["path"]] = 5000.0 if "yodas" in spec["path"] else 40.0
    return hours


@pytest.mark.parametrize("tasks, total", [("asr", 500.0), ("both", 940.0)])
def test_hours_mode_block_is_byte_identical_to_the_old_one(tasks, total):
    lines, got_total = bcc.yaml_block(LEGACY_TEMPLATE, _legacy_hours(), 100.0, tasks)
    assert got_total == total
    assert hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest() == LEGACY_DIGEST[tasks]


def _cut_block(tasks: str = "asr"):
    hours = _legacy_hours()
    cuts = {}
    for rel in hours:
        # long cuts everywhere except English's granary set, like the real mix
        d = 8.0 if rel == "yodas-granary/English/asr_only" else 20.0
        cuts[rel] = round(hours[rel] * 3600 / d)
    return bcc.yaml_block(LEGACY_TEMPLATE, hours, 100.0, tasks, cuts), hours, cuts


def test_cut_mode_block_documents_itself_and_keeps_the_hours():
    (lines, total), _, _ = _cut_block()
    text = "\n".join(lines)
    assert total == 500.0
    assert "per-CUT draw probability" in text
    assert "of the cuts, 20.0% of the hours" in text
    assert "mean cut 8.0 s" in text and "mean cut 20.0 s" in text
    # The hours beside each source are unchanged: 30 h of cv22_sidon at a 100 h budget.
    assert "# 30.0 h of cv22_sidon, mean cut 20.0 s" in text
    # English has no yodas3, so its yodas3 slot reads yodas-granary, and the comment says so.
    assert "# 15.0 h of yodas3 (read from yodas-granary), mean cut 8.0 s" in text
    assert "# 15.0 h of yodas3, mean cut 20.0 s" in text  # de/fr/es/it really read yodas3


def test_cut_mode_block_emits_weights_that_sum_to_one_per_level():
    yaml = pytest.importorskip("yaml")
    (lines, _), _, _ = _cut_block()
    text = "data:\n  train_ds:\n" + "\n".join(lines).replace("${oc.env:LOCAL_DATASETS_DIR}", "ROOT")
    groups = yaml.safe_load(text)["data"]["train_ds"]["input_cfg"]
    assert sum(g["weight"] for g in groups) == pytest.approx(1.0, abs=1e-7)
    for g in groups:
        assert sum(s["weight"] for s in g["input_cfg"]) == pytest.approx(1.0, abs=1e-7)
    by_lang = {g["tags"]["lang"]: g["weight"] for g in groups}
    # English draws short cuts, so it needs a bigger share of the cuts for the same hours.
    assert by_lang["en"] > max(w for lang, w in by_lang.items() if lang != "en")


def test_cut_mode_covers_st_groups_too():
    (lines, total), hours, cuts = _cut_block("both")
    assert total == 940.0
    text = "\n".join(lines)
    # One note per group: 5 ASR languages, 4 X->en directions and the en->de probe.
    assert text.count("% of the cuts, ") == 5 + 4 + 1
    groups, group_w, leaf_w = bcc.plan_mux(LEGACY_TEMPLATE, hours, 100.0, "both", cuts)
    shares = bcc.implied_hour_shares(groups, group_w, leaf_w, hours, cuts)
    by_group = {name: sum(row) for (name, _, _), row in zip(groups, shares)}
    assert by_group["asr:en"] == pytest.approx(100.0 / 940.0)
    assert by_group["st:en-de"] == pytest.approx(40.0 / 940.0)


# ---------------------------------------------------------------------------
# Bucket bins on the draw
# ---------------------------------------------------------------------------


def test_draw_bins_of_one_source_are_that_sources_bins(tmp_path):
    pytest.importorskip("numpy")
    import bucket_bins

    hist = {str(k): 1 for k in range(100, 1100)}  # 1-11 s, uniform
    cache = {str(tmp_path / "a/train"): {"hist": hist, "res": 0.01}}
    got = bcc.draw_bins([("a/train", 1.0)], cache, tmp_path, 0.5, 60.0, 5)
    want = bucket_bins.estimate_bins_from_histogram(hist, 0.01, 5)
    # The reference has one cut per key, and the bin search drops the overshoot
    # at each boundary, so it sits a few keys (0.01 s) off the exact quantiles
    # that the finer counts used by draw_bins land on.
    assert got == pytest.approx(want, abs=0.05)


def test_draw_bins_follow_the_weights_not_the_pool(tmp_path):
    pytest.importorskip("numpy")
    import bucket_bins

    short = {str(k): 1 for k in range(100, 1000)}  # 900 cuts, 1-10 s
    long = {str(k): 1 for k in range(2000, 3000)}  # 1000 cuts, 20-30 s
    cache = {str(tmp_path / "s"): {"hist": short, "res": 0.01}, str(tmp_path / "l"): {"hist": long, "res": 0.01}}
    # 70% of draws are short cuts, 30% long. Per key: 0.7/900 vs 0.3/1000 -> 70 : 27 in whole numbers.
    exact = {**{int(k): 70 for k in short}, **{int(k): 27 for k in long}}
    got = bcc.draw_bins([("s", 0.7), ("l", 0.3)], cache, tmp_path, 0.5, 60.0, 8)
    assert got == pytest.approx(bucket_bins.estimate_bins_from_histogram(exact, 0.01, 8), abs=0.05)

    # The pool is 900 short + 1000 long cuts, i.e. ~47% short; bins measured on the pool would differ.
    pool = bucket_bins.estimate_bins_from_histogram(
        {**{int(k): 1 for k in short}, **{int(k): 1 for k in long}}, 0.01, 8
    )
    assert max(abs(a - b) for a, b in zip(got, pool)) > 1.0


def test_draw_bins_apply_the_duration_filter_after_normalising(tmp_path):
    pytest.importorskip("numpy")

    hist = {**{str(k): 1 for k in range(100, 600)}, **{str(k): 1 for k in range(7000, 7500)}}  # 1-6 s, 70-75 s
    cache = {str(tmp_path / "a"): {"hist": hist, "res": 0.01}}
    got = bcc.draw_bins([("a", 1.0)], cache, tmp_path, 0.5, 60.0, 4)
    assert max(got) <= 6.0 and min(got) >= 1.0  # the 70 s cuts are gone, not clipped to 60


def test_draw_bins_name_the_sources_missing_a_histogram(tmp_path):
    pytest.importorskip("numpy")

    with pytest.raises(SystemExit, match="No duration histogram cached for: x/train"):
        bcc.draw_bins([("x/train", 1.0)], {}, tmp_path, 0.5, 60.0, 4)


# ---------------------------------------------------------------------------
# End to end: render over a fake collection whose languages differ in cut length
# ---------------------------------------------------------------------------

TEMPLATE = """\
data:
  apply_chat_template: true
  train_ds:
    total_hours: 1.0
    num_buckets: 5
    bucket_duration_bins: [1.0, 2.0, 3.0, 4.0]
    min_duration: 0.5
    max_duration: 60.0
    input_cfg:
      - placeholder
  validation_ds:
    total_hours: 1.0
    input_cfg:
      - placeholder
"""

CORPORA = ("cv22_sidon", "mls_sidon", "yodas-granary", "voxpopuli")


def _cut_length(corpus: str, lang: str) -> float:
    if corpus == "yodas-granary":
        return 6.0 if lang == "en" else 36.0  # English's long-form source has short cuts
    return {"cv22_sidon": 6.0, "mls_sidon": 12.0, "voxpopuli": 12.0}[corpus]


def _make_collection(root: Path) -> None:
    for corpus in CORPORA:
        for lang in bcc.TRAIN_LANGS:
            rel = bcc.ASR_SOURCES[corpus][lang]
            d = _cut_length(corpus, lang)
            write_manifest(root / rel, [d] * round(3600 / d))  # 1 h in every source
            if corpus in bcc.VALIDATION_SPLIT:
                write_manifest(root / bcc.validation_path(rel, bcc.VALIDATION_SPLIT[corpus]), [d] * 10)


def _render(tmp_path, monkeypatch, *flags: str):
    root = tmp_path / "shar"
    if not root.exists():
        _make_collection(root)
    template = tmp_path / "template.yaml"
    template.write_text(TEMPLATE, encoding="utf-8")
    out = tmp_path / "out.yaml"
    argv = [
        "build_campaign_config.py",
        "--template",
        str(template),
        "--datasets-root",
        str(root),
        "--budget-hours",
        "1.0",
        "--tasks",
        "asr",
        "--exclude-corpus",
        "fleurs",
        "--cache",
        str(tmp_path / "cache.json"),
        "--out",
        str(out),
        *flags,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert bcc.main() == 0
    yaml = pytest.importorskip("yaml")
    text = out.read_text(encoding="utf-8")
    return yaml.safe_load(text.replace("${oc.env:LOCAL_DATASETS_DIR}", "ROOT")), text


def _language_hour_shares(cfg) -> dict[str, float]:
    """Each language's share of the audio the muxer would emit, from the weights and the known cut lengths."""
    mass = {}
    for group in cfg["data"]["train_ds"]["input_cfg"]:
        lang = group["tags"]["lang"]
        mass[lang] = group["weight"] * sum(
            s["weight"] * _cut_length(_corpus_of(s["shar_path"]), lang) for s in group["input_cfg"]
        )
    total = sum(mass.values())
    return {lang: m / total for lang, m in mass.items()}


def _corpus_of(shar_path: str) -> str:
    return shar_path.split("/")[1]


def test_render_gives_every_language_an_equal_share_of_the_audio(tmp_path, monkeypatch):
    cfg, text = _render(tmp_path, monkeypatch)  # --weights cuts is the default
    shares = _language_hour_shares(cfg)
    assert all(s == pytest.approx(0.2, abs=1e-5) for s in shares.values()), shares
    assert cfg["data"]["train_ds"]["total_hours"] == 5.0
    assert "per-CUT draw probability" in text


def test_render_with_hour_share_weights_reproduces_the_skew(tmp_path, monkeypatch):
    cfg, text = _render(tmp_path, monkeypatch, "--weights", "hours")
    groups = cfg["data"]["train_ds"]["input_cfg"]
    assert all(g["weight"] == pytest.approx(0.2) for g in groups)
    assert all(s["weight"] == pytest.approx(0.25) for g in groups for s in g["input_cfg"])
    shares = _language_hour_shares(cfg)
    # The 11.9% vs 20% problem, in miniature: English 12.0%, the others 22.0% each.
    assert shares["en"] == pytest.approx(0.12, abs=1e-6)
    assert all(shares[lang] == pytest.approx(0.22, abs=1e-6) for lang in ("de", "fr", "es", "it"))
    assert "per-CUT draw probability" not in text


def test_render_leaves_the_bins_alone_without_a_histogram_cache(tmp_path, monkeypatch):
    _, text = _render(tmp_path, monkeypatch)
    assert "    bucket_duration_bins: [1.0, 2.0, 3.0, 4.0]" in text.splitlines()
    assert "  apply_chat_template: false" in text.splitlines()  # the builder's long-standing pin


def test_render_replaces_the_bins_with_bins_measured_on_the_draw(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    root = tmp_path / "shar"
    _make_collection(root)
    hist_cache = {}
    for corpus in CORPORA:
        for lang in bcc.TRAIN_LANGS:
            d = _cut_length(corpus, lang)
            n = round(3600 / d)
            hist_cache[str(root / bcc.ASR_SOURCES[corpus][lang])] = {
                "hist": {str(round(d / 0.01)): n},
                "res": 0.01,
                "cuts": n,
                "hours": 1.0,
            }
    cache_path = tmp_path / "hist.json"
    cache_path.write_text(json.dumps(hist_cache), encoding="utf-8")

    cfg, _ = _render(tmp_path, monkeypatch, "--bins-hist-cache", str(cache_path))
    bins = cfg["data"]["train_ds"]["bucket_duration_bins"]
    assert 1 <= len(bins) <= 4 and bins == sorted(bins)
    assert set(bins) <= {6.0, 12.0, 36.0}  # every cut in this collection is one of these lengths
    assert bins != [1.0, 2.0, 3.0, 4.0]
