"""Tests for projects/ablation-campaign/fondue/build_fondue_mix.py.

What would silently train the wrong thing: a language the prompt cannot name (the
loader raises mid-run), a cut-filter tag missing from a source (the rule never
applies), and an anneal step that does not move with the batch.
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
    assert spite == {"de", "es", "fr", "it", "nl", "pt"}


def test_every_leaf_carries_the_tag_its_cut_filter_rule_matches():
    scored = {"cv22_sidon/de/train"}
    for e in _train(scored):
        t = e["tags"]
        if t["task"] == "asr":
            assert t["scored"] == ("wer" if e["rel"] in scored else "none")
        elif e["rel"].startswith("spite_"):
            assert t["scored"] == "spite" and t["text_field"] == f"custom.translation_{t['tgt_lang']}"
        else:
            assert t["scored"] == "none"


def test_granary_ast_feeds_asr_from_the_supervision_and_st_from_the_translation():
    ast = [e for e in _train() if e["rel"] == "yodas-granary/German/ast"]
    by_task = {e["tags"]["task"]: e["tags"] for e in ast}
    assert "text_field" not in by_task["asr"]
    assert by_task["st"]["text_field"] == "custom.translation_en"


def test_x_to_y_only_on_request():
    pairs = {(e["tags"]["src_lang"], e["tags"]["tgt_lang"]) for e in bfm.st_leaves(x_to_y=False)}
    assert all("en" in p for p in pairs)
    assert ("de", "fr") in {(e["tags"]["src_lang"], e["tags"]["tgt_lang"]) for e in bfm.st_leaves(x_to_y=True)}


def test_anneal_step_is_four_fifths_of_max_steps():
    lines = ["    total_hours: 9600.0", "    batch_duration: 10", "  gradient_accumulation_steps: 15"]
    # 9600 h * 3600 / (10 * 15 * 16 * 4) = 3600 steps; the decay takes the last 720.
    assert bfm.anneal_step(lines, nodes=16, gpus_per_node=4) == 2880


def test_st_share_is_the_st_part_of_the_audio_whatever_the_entry_count():
    def src(task, key, hours):
        return {"task": task, "lang_key": key, "hours": hours, "path": key}

    sources = [src("asr", "en", 900.0), src("asr", "mt", 4.0)]
    sources += [src("st", f"en-{t}", 100.0) for t in ("de", "es", "fr", "it", "nl", "pt")] + [src("st", "de-en", 50.0)]
    bfm.split_weights(sources, 0.25)

    st = sum(s["p_cl"] for s in sources if s["task"] == "st")
    assert st == pytest.approx(0.25)
    assert sum(s["p_cl"] for s in sources) == pytest.approx(1.0)
    # beta inside ASR only: en against mt, sqrt(900) : sqrt(4) = 30 : 2.
    en, mt = sources[0]["p_l"], sources[1]["p_l"]
    assert en / mt == pytest.approx(30 / 2)
