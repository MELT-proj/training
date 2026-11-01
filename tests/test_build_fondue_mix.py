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
    asr = {"rel": "yodas3/de/train", "tags": {"task": "asr", "lang": "de", "scored": "wer"}}
    spite = {"rel": "spite_cv16_euro9b/train", "tags": {"task": "st", "tgt_lang": "de", "scored": "spite"}}
    for s in (asr, spite):
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
