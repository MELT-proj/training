"""Tests for :class:`melt.training.duration_tracker.DurationTracker`.

Pure-CPU and no distributed process group.  Most tests exercise the
key-building and accumulation logic in isolation; the two at the bottom run a
real batch through ``SpeechToTextDataset`` to pin which batches the duration
metadata is allowed to appear on, and skip if the tokenizer is unavailable.
"""

import pytest

from melt.training.duration_tracker import DurationTracker


def test_make_key_asr():
    assert DurationTracker.make_key("asr", "de", "de", "") == "asr/de"


def test_make_key_st():
    assert DurationTracker.make_key("st", "en", "de", "en") == "st/de-en"


def test_make_key_st_empty_src_lang_falls_back_to_lang():
    # `lang` is the ST target-language collapse from get_tags_from_cut; an
    # empty src_lang should fall back to it, not to "unknown".
    assert DurationTracker.make_key("st", "de", "", "de") == "st/de-de"


def test_make_key_translate_alias_behaves_like_st():
    assert DurationTracker.make_key("translate", "en", "de", "en") == "translate/de-en"


def test_make_key_empty_task():
    assert DurationTracker.make_key("", "de", "de", "") == "unknown/de"


def test_make_key_none_task():
    assert DurationTracker.make_key(None, "de", "de", "") == "unknown/de"


def test_make_key_empty_lang():
    assert DurationTracker.make_key("asr", "", "", "") == "asr/unknown"


def test_make_key_whitespace_only_is_unknown():
    assert DurationTracker.make_key("  ", "  ", "", "") == "unknown/unknown"


def test_update_accumulates_across_calls():
    tracker = DurationTracker()
    tracker.update([1.0, 2.0], ["asr", "asr"], ["de", "de"], ["de", "de"], ["", ""])
    tracker.update([3.0], ["asr"], ["de"], ["de"], [""])
    assert tracker.state_dict() == {"asr/de": 6.0}


def test_update_handles_tasks_none():
    tracker = DurationTracker()
    tracker.update([1.5], None, ["de"], ["de"], [""])
    assert tracker.state_dict() == {"unknown/de": 1.5}


def test_update_handles_short_metadata_lists():
    # Two durations, but the metadata lists only cover the first sample --
    # the second must fall back to "unknown" fields rather than raising or
    # silently dropping its duration.
    tracker = DurationTracker()
    tracker.update([1.0, 2.0], ["asr"], ["de"], ["de"], [""])
    assert tracker.state_dict() == {"asr/de": 1.0, "unknown/unknown": 2.0}


def test_update_handles_durations_none():
    tracker = DurationTracker()
    tracker.update(None, ["asr"], ["de"], ["de"], [""])
    assert tracker.state_dict() == {}


def test_update_handles_empty_durations_list():
    tracker = DurationTracker()
    tracker.update([], ["asr"], ["de"], ["de"], [""])
    assert tracker.state_dict() == {}


def test_reduced_hours_converts_seconds_and_totals():
    tracker = DurationTracker()
    tracker.update([3600.0, 1800.0], ["asr", "asr"], ["de", "en"], ["de", "en"], ["", ""])
    hours = tracker.reduced_hours()
    assert hours["train_hours/asr/de"] == 1.0
    assert hours["train_hours/asr/en"] == 0.5
    assert hours["train_hours/total"] == 1.5


def test_reduced_hours_total_equals_sum_of_parts():
    tracker = DurationTracker()
    tracker.update(
        [100.0, 200.0, 300.0],
        ["asr", "st", "asr"],
        ["de", "en", "fr"],
        ["de", "de", "fr"],
        ["", "en", ""],
    )
    hours = tracker.reduced_hours()
    total = hours.pop("train_hours/total")
    assert total == sum(hours.values())


def test_reduced_hours_custom_prefix():
    tracker = DurationTracker()
    tracker.update([3600.0], ["asr"], ["de"], ["de"], [""])
    hours = tracker.reduced_hours(prefix="foo")
    assert hours == {"foo/asr/de": 1.0, "foo/total": 1.0}


def test_state_dict_round_trip():
    tracker = DurationTracker()
    tracker.update([10.0, 20.0], ["asr", "st"], ["de", "en"], ["de", "de"], ["", "en"])
    state = tracker.state_dict()

    restored = DurationTracker()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_load_state_dict_replaces_not_accumulates():
    tracker = DurationTracker()
    tracker.update([10.0], ["asr"], ["de"], ["de"], [""])
    tracker.load_state_dict({"asr/en": 5.0})
    assert tracker.state_dict() == {"asr/en": 5.0}


def test_genericity_unknown_task_and_language_get_their_own_key():
    """Regression guard: a (task, lang) combination that appears nowhere in
    the codebase or config must still surface as its own key, with no code
    change required. This is the whole point of discovering keys at runtime
    instead of enumerating a fixed universe.
    """
    tracker = DurationTracker()
    tracker.update([42.0], ["sqa"], ["sw"], ["sw"], [""])
    state = tracker.state_dict()
    assert state == {"sqa/sw": 42.0}

    hours = tracker.reduced_hours()
    assert hours["train_hours/sqa/sw"] == 42.0 / 3600.0


# ---------------------------------------------------------------------------
# The hours instrumentation must stay off the evaluation batch.
#
# Today only the training dataloader builds a SpeechToTextDataset (evaluation
# goes through MELTMapDataset + MELTDataCollator), so these keys never reach
# an eval batch.  The gate is on ``is_train`` rather than on that fact, and
# these tests pin it there so a future eval use of this class cannot start
# carrying duration metadata into model(**inputs).
# ---------------------------------------------------------------------------

_HOURS_KEYS = ("durations", "tasks", "src_langs", "tgt_langs")


def _speech_to_text_batch(is_train: bool):
    """Build a one-cut batch through the real dataset, or skip if unavailable."""
    pytest.importorskip("lhotse")
    from lhotse import CutSet
    from omegaconf import OmegaConf

    from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

    try:
        from tests.test_prompt_template_labels import _build_processor, _make_cut
    except ImportError:  # pragma: no cover - layout-dependent import
        from test_prompt_template_labels import _build_processor, _make_cut

    try:
        processor = _build_processor()
    except Exception as exc:  # pragma: no cover - no network / no hub cache
        pytest.skip(f"tokenizer unavailable: {exc}")

    dataset = SpeechToTextDataset(
        processor=processor,
        config=OmegaConf.create({
            "apply_chat_template": False,
            "prompt_template": "{audio_token}{lang}: {t}",
            "sample_rate": 16000,
        }),
        is_train=is_train,
        return_labels=True,
        return_langs=True,
    )
    return dataset[CutSet.from_cuts([_make_cut("en-1", "hello world", "en")])]


def test_eval_batch_carries_no_duration_metadata():
    batch = _speech_to_text_batch(is_train=False)
    assert batch is not None
    for key in _HOURS_KEYS:
        assert key not in batch, f"{key!r} must not reach an eval batch"
    # The pre-existing per-language eval breakdown is untouched.
    assert batch["langs"] == ["en"]


def test_train_batch_carries_duration_metadata():
    batch = _speech_to_text_batch(is_train=True)
    assert batch is not None
    for key in _HOURS_KEYS:
        assert key in batch, f"{key!r} missing from the train batch"
    assert batch["durations"] == [pytest.approx(1.0)]
    assert batch["tasks"] == ["asr"]
    assert batch["langs"] == ["en"]
