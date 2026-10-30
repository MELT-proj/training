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


def test_schedule_draws_total_hours_at_the_measured_fill():
    lines = ["    total_hours: 9600.0", "    batch_duration: 10", "  gradient_accumulation_steps: 15"]
    # 9600 h * 3600 / (10 * 15 * 16 * 4) = 3600 steps at a fill of 1.0; the decay takes the last 720.
    assert bfm.schedule(lines, nodes=16, gpus_per_node=4, fill=1.0) == (3600, 2880)
    # At a fill of 1.5 each step carries 1.5x the audio, so 2/3 of the steps draw the same hours.
    assert bfm.schedule(lines, nodes=16, gpus_per_node=4, fill=1.5) == (2400, 1920)
