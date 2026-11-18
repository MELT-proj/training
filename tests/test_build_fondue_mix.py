"""Tests for projects/ablation-campaign/fondue/build_fondue_mix.py.

What would silently train the wrong thing: a language the prompt cannot name (the
loader raises mid-run), a cut-filter tag no rule matches (the leaf trains unfiltered),
an unscored ST source left in the annealing mix (its filter rejects every cut and the
mux keeps reading it), and a schedule that does not move with the batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "fondue"))

import build_fondue_mix as bfm  # noqa: E402


def _train(scored: set[str] = frozenset()) -> list[dict]:
    return [e for lang in bfm.ASR_LANGS for e in bfm.asr_leaves(lang, set(scored))] + bfm.st_leaves(x_to_y=True)


def _all_asr_scored() -> set[str]:
    return {e["rel"] for e in _train() if e["tags"]["task"] == "asr"}


def test_every_language_the_prompts_will_name_is_known_to_the_loader():
    helpers = pytest.importorskip("melt.training.data.audio.lhotse.helpers")
    langs = set()
    for e in _train():
        t = e["tags"]
        langs |= {t.get("lang"), t.get("src_lang"), t.get("tgt_lang")}
    for s in bfm.validation_leaves():
        t = s["tags"]
        langs |= {t.get("lang"), t.get("src_lang"), t.get("tgt_lang")}

    assert langs - {None} <= set(helpers.LANGUAGE_ISO_TO_NAME)


def test_pool_follows_the_pi_language_decisions():
    langs = {e["tags"].get("lang") for e in _train()} - {None}
    assert langs == set(bfm.EU24) | {"uk", "ru"}
    spite = {e["tags"]["tgt_lang"] for e in _train() if e["tags"]["scored"] == "spite"}
    assert spite == {"de", "es", "fr", "it", "nl", "pt", "ru"}


def test_every_leaf_carries_the_tag_its_cut_filter_rule_matches():
    scored = {"cv22_sidon/de/train"}
    for e in _train(scored):
        t = e["tags"]
        if t["task"] == "asr":
            assert t["scored"] == ("wer" if e["rel"] in scored else "none")
            assert ("wer_band" in t) == (e["rel"] in scored)
        elif e["rel"].startswith("spite_"):
            assert t["scored"] == "spite" and t["text_field"] == f"custom.translation_{t['tgt_lang']}"
        else:
            assert t["scored"] == "none"


def test_a_language_band_beats_a_corpus_band_beats_the_default():
    assert bfm.wer_band("cv22_sidon/de/train", "de") == bfm.WER_BY_CORPUS["cv22_sidon"]
    assert bfm.wer_band("cv22_sidon/lv/train", "lv") == bfm.WER_BY_LANG["lv"]
    assert bfm.wer_band("yodas3/de/train", "de") == bfm.WER_DEFAULT
    assert bfm.wer_band("voxpopuli_mosel/lv/train", "lv") == bfm.WER_DEFAULT  # pseudo-labels beat the language


@pytest.mark.parametrize("phase", ["constant", "anneal"])
def test_each_scored_asr_leaf_matches_exactly_one_wer_rule_and_the_rules_validate(phase):
    yaml = pytest.importorskip("yaml")
    cut_filter = pytest.importorskip("melt.training.data.audio.lhotse.cut_filter")
    train = _train(_all_asr_scored())
    bands = [bfm.wer_band(e["rel"], e["tags"]["lang"]) for e in train if "wer_band" in e["tags"]]
    rules = yaml.safe_load("rules:\n" + "\n".join(line[6:] for line in bfm.cut_filter_rules(phase, bands)))["rules"]
    cut_filter.validate_config({"scores_root": "/x", "rules": rules})

    for e in train:
        wer_rules = [n for n, r in rules.items() if "array" in r and cut_filter.rule_matches(r["match"], e["tags"])]
        assert len(wer_rules) == (1 if e["tags"]["task"] == "asr" else 0), e["rel"]


def test_granary_ast_feeds_asr_from_the_supervision_and_st_from_the_translation():
    ast = [e for e in _train() if e["rel"] == "yodas-granary/German/ast"]
    by_task = {e["tags"]["task"]: e["tags"] for e in ast}
    assert "text_field" not in by_task["asr"]
    assert by_task["st"]["text_field"] == "custom.translation_en"


def test_x_to_y_only_on_request():
    pairs = {(e["tags"]["src_lang"], e["tags"]["tgt_lang"]) for e in bfm.st_leaves(x_to_y=False)}
    assert all("en" in p for p in pairs)
    assert ("de", "fr") in {(e["tags"]["src_lang"], e["tags"]["tgt_lang"]) for e in bfm.st_leaves(x_to_y=True)}


def test_unscored_st_leaves_the_mix_when_annealing():
    survival = bfm.Survival(bfm.QUALITY)
    s = {"rel": "covost2/de_en/train", "tags": {"task": "st", "src_lang": "de", "tgt_lang": "en", "scored": "none"}}
    assert survival.share(s, "constant") == 1.0
    assert survival.share(s, "anneal") == 0.0


def test_the_filters_keep_less_when_annealing():
    survival = bfm.Survival(bfm.QUALITY)
    asr = {"rel": "yodas3/cs/train", "tags": {"task": "asr", "lang": "cs", "scored": "wer"}}
    mosel = {"rel": "voxpopuli_mosel/mt/train", "tags": {"task": "asr", "lang": "mt", "scored": "wer"}}
    spite = {"rel": "spite_cv16_euro9b/train", "tags": {"task": "st", "tgt_lang": "de", "scored": "spite"}}
    for s in (asr, mosel, spite):
        assert 0 < survival.share(s, "anneal") < survival.share(s, "constant") < 1
    assert not survival.unknown


def test_tasks_split_the_audio_in_proportion_to_their_surviving_hours():
    sources = [
        {"task": "asr", "lang_key": "de", "hours": 300.0},
        {"task": "asr", "lang_key": "mt", "hours": 10.0},
        {"task": "st", "lang_key": "de-en", "hours": 90.0},
    ]
    held = bfm.phase_weights(sources, "constant")
    assert held == {"asr": 310.0, "st": 90.0}
    st = sum(s["p_cl"] for s in sources if s["task"] == "st")
    assert st == pytest.approx(90 / 400)


def test_schedule_draws_total_hours_at_the_measured_fill():
    # 9600 h * 3600 / (10 * 15 * 16 * 4) = 3600 steps at a fill of 1.0; the decay takes the last 720.
    assert bfm.schedule(9600, 10, 15, nodes=16, gpus_per_node=4, fill=1.0) == (3600, 2880)
    # At a fill of 1.5 each step carries 1.5x the audio, so 2/3 of the steps draw the same hours.
    assert bfm.schedule(9600, 10, 15, nodes=16, gpus_per_node=4, fill=1.5) == (2400, 1920)


def test_only_the_annealing_config_restarts_the_data_stream_and_its_keys_exist():
    yaml = pytest.importorskip("yaml")
    config = pytest.importorskip("melt.training.config")
    default_run = config.get_default_config().run
    constant, anneal = (yaml.safe_load(bfm.RENDERED[p].read_text()) for p in ("constant", "anneal"))

    assert "restart_data_stream" in default_run
    assert not constant["run"].get("restart_data_stream", False)
    assert anneal["run"]["restart_data_stream"] is True
    assert anneal["data"]["train_ds"]["seed"] != constant["data"]["train_ds"]["seed"]


def test_every_mosel_language_has_a_leaf_and_a_histogram():
    """Mosel was scored on MN5 only: a language without its histogram would train unfiltered."""
    survival = bfm.Survival(bfm.QUALITY)
    leaves = [e for e in _train(_all_asr_scored()) if e["rel"].startswith("voxpopuli_mosel/")]
    assert {e["tags"]["lang"] for e in leaves} == set(bfm.MOSEL_LANGS)
    for e in leaves:
        assert "text_field" not in e["tags"], "Mosel trains on its supervision text"
        survival.share(e, "constant")
    assert not survival.unknown


# --- SpeechBlocks-Synth-v1 (build_fondue_mix.py synth) ---------------------------------------


def _synth_text():
    """The committed annealing config, the same with the committed spec added (both as text), the spec and the report."""
    import json

    spec = json.loads(bfm.SYNTH_SPEC.read_text())
    base = bfm.RENDERED["anneal"].read_text()
    text, report = bfm.apply_synth(base, spec)
    return base, text, spec, report


def _synth():
    """The committed annealing config with the committed spec added, parsed."""
    yaml = pytest.importorskip("yaml")
    base, text, spec, report = _synth_text()
    return yaml.safe_load(base)["data"]["train_ds"], yaml.safe_load(text)["data"]["train_ds"], spec, report


def _group_audio(text):
    """Per group, the seconds of audio per cut drawn from the whole mixture (weight x mean cut), from the notes."""
    lines = text.splitlines()
    start, end, _ = bfm.cmw._find_block(lines, "train_ds", "input_cfg")
    return [
        (g["key"], g["w"] * sum(x["w"] * x["mean"] for x in g["leaves"])) for g in bfm.parse_mix(lines, start, end)
    ]


def _leaves(ds):
    for g in ds["input_cfg"]:
        for leaf in g["input_cfg"]:
            yield g, leaf


def test_every_synthetic_leaf_matches_exactly_its_own_rule_and_a_noisy_copy_matches_none():
    cut_filter = pytest.importorskip("melt.training.data.audio.lhotse.cut_filter")
    _, new, spec, _ = _synth()
    rules = cut_filter.validate_config(new["cut_filter"])
    assert len(rules) == 13 + len(spec["rules"])
    st = copies = 0
    for g, leaf in _leaves(new):
        tags = {**leaf["tags"], **g["tags"]}
        matched = list(cut_filter.resolve_leaf_rules(rules, [], tags))
        if tags.get("scored") == "synth":
            st += 1
            want = f"synth_qe_{tags['synth_stream']}_{tags['src_lang']}_{tags['tgt_lang']}"
            assert matched == [want], leaf["shar_path"]
            assert rules[matched[0]]["field"] == f"custom.synth.metricx_xxl.{tags['tgt_lang']}"
            assert tags["text_field"] == f"custom.translation_{tags['tgt_lang']}"
        elif tags.get("synth_stream") == "F":
            copies += 1
            assert matched == [], "no array exists for a copy: a WER rule would drop every cut or fail"
    assert (st, copies) == (len(spec["st_leaves"]), len(spec["asr_leaves"]))


def test_every_stream_the_collection_uses_is_explained_in_the_config():
    _, text, spec, _ = _synth_text()
    letters = {x["stream"] for x in spec["st_leaves"]} | {x["group"] for x in spec["st_leaves"]}
    letters |= {r["stream"] for r in spec["rules"]}
    assert letters <= set(bfm.SYNTH_STREAMS)
    for letter, what in bfm.SYNTH_STREAMS.items():
        assert f"#   {letter}: {what}" in text


def test_every_language_the_synthetic_leaves_name_is_known_to_the_prompts():
    helpers = pytest.importorskip("melt.training.data.audio.lhotse.helpers")
    _, new, _, _ = _synth()
    langs = set()
    for g, leaf in _leaves(new):
        t = {**leaf["tags"], **g["tags"]}
        langs |= {t.get("lang"), t.get("src_lang"), t.get("tgt_lang")}
    assert langs - {None} <= set(helpers.LANGUAGE_ISO_TO_NAME)


def test_every_existing_group_keeps_its_share_of_the_audio_and_every_level_sums_to_one():
    old, new, spec, report = _synth()
    assert sum(g["weight"] for g in new["input_cfg"]) == pytest.approx(1.0, abs=1e-6)
    for g in new["input_cfg"]:
        assert sum(x["weight"] for x in g["input_cfg"]) == pytest.approx(1.0, abs=1e-6)
    tail = {c["lang"] for c in spec["asr_leaves"]}
    for before, after in zip(old["input_cfg"], new["input_cfg"]):
        assert before["tags"] == after["tags"]
        if before["tags"]["task"] == "asr" and before["tags"]["lang"] in tail:
            ratios = [a["weight"] / b["weight"] for a, b in zip(after["input_cfg"], before["input_cfg"])]
            assert max(ratios) == pytest.approx(min(ratios), rel=1e-6)  # the old corpora scale together
        else:
            assert all(
                a["weight"] == pytest.approx(b["weight"], abs=1e-9) and a["tags"] == b["tags"]
                for a, b in zip(after["input_cfg"], before["input_cfg"])
            )
    # Weights count cuts, and a group that takes noisy copies in draws shorter cuts: it only keeps its audio if its
    # weight grows. Every existing group's audio, in the config's own numbers, scales by one common factor.
    base, text, _, _ = _synth_text()
    before = _group_audio(base)
    after = _group_audio(text)[: len(before)]
    assert [k for k, _ in after] == [k for k, _ in before]
    factors = {k: a / b for (k, a), (_, b) in zip(after, before)}
    assert max(factors.values()) == pytest.approx(min(factors.values()), rel=5e-3), factors
    # the rest of the weight is the collection's cuts
    assert sum(g["weight"] for g in new["input_cfg"][len(old["input_cfg"]) :]) == pytest.approx(report["cut_share"])


def test_the_synthetic_share_is_a_share_of_audio_not_of_cuts():
    """Weights are cut probabilities, so the share is converted with the mean cut length of each side."""
    old, new, spec, report = _synth()
    n_old = len(old["input_cfg"])
    mean_of = {(x["name"], x["src"], x["tgt"]): x["kept_s"] / x["kept_pairs"] for x in spec["st_leaves"]}
    audio = cuts = 0.0
    for g in new["input_cfg"][n_old:]:
        for x in g["input_cfg"]:
            name = x["shar_path"].split("/")[-2]
            m = mean_of[(name, g["tags"]["src_lang"], g["tags"]["tgt_lang"])]
            audio += g["weight"] * x["weight"] * m
            cuts += g["weight"] * x["weight"]
    assert cuts == pytest.approx(report["cut_share"], rel=1e-6)
    # the existing groups' audio per cut drawn, read back from the rendered config (not from the base's numbers: the
    # noisy copies shorten the cuts of the groups they join)
    _, text, _, _ = _synth_text()
    existing = sum(a for _, a in _group_audio(text)[:n_old])
    assert audio / (audio + existing) == pytest.approx(report["audio_share"], rel=3e-3)


def test_by_default_the_share_is_the_collections_hours_share_of_all_surviving_hours():
    *_, report = _synth()
    assert report["audio_share"] == pytest.approx(
        report["hours_synth"] / (report["hours_synth"] + report["hours_existing"])
    )
    assert 0.05 < report["audio_share"] < 0.25


def test_noisy_copies_take_their_policy_share_of_a_group_and_the_group_keeps_its_audio():
    group = {"w": 0.001, "leaves": [{"w": 0.7, "mean": 10.0, "hours": 100.0}, {"w": 0.3, "mean": 5.0, "hours": 4.0}]}
    copies = [{"s": 36000.0, "cuts": 3600}, {"s": 7200.0, "cuts": 1000}]  # 10 h at 10 s and 2 h at 7.2 s
    sw = bfm.swap_in_tail(group, copies, mean_exist=8.0)
    # the copies' cuts are not as long as the group's 8.5 s on average, so the mean cut moves and the weight must follow
    # (up when the copies are shorter) for the group's audio to stay what it was
    mean_old = 0.7 * 10 + 0.3 * 5
    assert sw["mean_group"] != pytest.approx(mean_old)
    assert group["w"] * sw["stretch"] * sw["mean_group"] == pytest.approx(group["w"] * mean_old)
    # The largest old corpus holds 7.0 of the group's 8.5 s of audio per cut drawn, and its audio share is
    # hours**0.5 over the sum of hours**0.5 of the old corpora: that fixes the sum.
    old = group["leaves"]
    sum_old = old[0]["hours"] ** 0.5 / (old[0]["w"] * old[0]["mean"] / (0.7 * 10 + 0.3 * 5))
    sum_new = 10**0.5 + 2**0.5
    assert sw["audio_new"] == pytest.approx(sum_new / (sum_old + sum_new))
    assert sum(sw["u"]) == pytest.approx(1.0)
    # back from cut weights to audio shares: the copies get exactly their audio share
    mean_old = 0.7 * 10 + 0.3 * 5
    mean_new = sum(u * m for u, m in zip(sw["u"], (10.0, 7.2)))
    audio = sw["cut_new"] * mean_new / (sw["cut_new"] * mean_new + (1 - sw["cut_new"]) * mean_old)
    assert audio == pytest.approx(sw["audio_new"])


def test_a_config_that_already_has_the_collection_is_refused():
    _, _, spec, _ = _synth()
    text, _ = bfm.apply_synth(bfm.RENDERED["anneal"].read_text(), spec)
    with pytest.raises(ValueError, match="already carries"):
        bfm.apply_synth(text, spec)


def test_the_synthetic_config_changes_nothing_but_the_mixture_the_rules_and_the_comments():
    yaml = pytest.importorskip("yaml")
    base = bfm.RENDERED["anneal"].read_text()
    import json

    text, _ = bfm.apply_synth(base, json.loads(bfm.SYNTH_SPEC.read_text()))
    a, b = yaml.safe_load(base), yaml.safe_load(text)
    a["data"]["train_ds"]["input_cfg"] = b["data"]["train_ds"]["input_cfg"] = None
    a["data"]["train_ds"]["cut_filter"]["rules"] = b["data"]["train_ds"]["cut_filter"]["rules"] = None
    # ~1,000 leaves make a snapshot ~6x dearer: the interval is the one key that moves
    assert a["data"]["train_ds"].pop("snapshot_every_n_steps") == 512
    assert b["data"]["train_ds"].pop("snapshot_every_n_steps") == bfm.SYNTH_SNAPSHOT_EVERY
    assert a == b  # the schedule, the run keys, the model and the batching are untouched
