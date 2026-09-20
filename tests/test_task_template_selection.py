"""Tests for TASK_TEMPLATES contents and prompt-template selection modes.

The ``asr`` family deliberately holds ``{lang}`` and no-``{lang}`` templates in
one list and every arm so far drew from it with ``"random"``. These tests pin
that ``"random"`` still draws from the whole list, and that the language-naming
variants are reachable through ``"with_language"`` / ``"without_language"``
without changing anything an earlier run trained on.
"""

import random

import pytest

from melt.training.data.audio.lhotse.helpers import (
    LANGUAGE_ISO_TO_NAME,
    TASK_TEMPLATES,
    apply_chat_template_to_texts,
    apply_qe_chat_template_to_texts,
    select_prompt_template,
)

AUDIO = "<|audio|>"


class _StubTokenizer:
    """Renders the user turn (and assistant turn) as plain text, no hub access."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        out = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return out + "[assistant]" if add_generation_prompt else out


def _has_lang(t: str) -> bool:
    return "{lang}" in t


# The pre-change verbatim list, frozen. New entries may only be appended.
_ORIGINAL_VERBATIM_HEADS = [
    "Your task is to repeat verbatim",
    "Repeat back, word for word",
    "Whatever appears between the markers",
    "I'm testing a simple copy task",
    "Your job is verbatim repetition",
    "Below are a couple of examples",
]


class TestVerbatimVariants:
    def test_verbatim_has_six_originals_then_six_language_variants(self):
        verbatim = TASK_TEMPLATES["verbatim"]
        assert len(verbatim) == 12
        assert not any(_has_lang(t) for t in verbatim[:6])
        assert all(_has_lang(t) for t in verbatim[6:])

    def test_originals_keep_their_order_and_position(self):
        for head, template in zip(_ORIGINAL_VERBATIM_HEADS, TASK_TEMPLATES["verbatim"][:6]):
            assert template.startswith(head)

    def test_variants_carry_no_in_context_examples(self):
        """Examples would be English whatever {lang} is."""
        for variant in TASK_TEMPLATES["verbatim"][6:]:
            assert "Example" not in variant
            assert "Input:" not in variant and "Output:" not in variant

    def test_originals_keep_their_examples(self):
        for original in TASK_TEMPLATES["verbatim"][:6]:
            assert "Input:" in original and "Output:" in original

    def test_each_variant_has_one_lang_slot_and_ends_on_the_audio(self):
        for variant in TASK_TEMPLATES["verbatim"][6:]:
            assert variant.count("{lang}") == 1
            assert variant.endswith("{audio_token}")

    def test_verbatim_names_the_real_audio_boundary_tokens(self):
        """The templates once said <|audio__bos|>, a token that does not exist."""
        for template in TASK_TEMPLATES["verbatim"]:
            assert "audio__bos" not in template
            assert "<|audio_bos|>" in template and "<|audio_eos|>" in template

    def test_variants_are_not_copies_of_one_sentence(self):
        sentences = set()
        for variant in TASK_TEMPLATES["verbatim"][6:]:
            first_paragraph = variant.split("\n\n")[0]
            sentences.add(first_paragraph[first_paragraph.index("{lang}") - 30 :])
        assert len(sentences) == 6

    def test_first_variant_uses_the_pi_wording(self):
        assert TASK_TEMPLATES["verbatim"][6].startswith(
            "Your task is to repeat verbatim whatever I write in this chat surrounded by "
            "<|audio_bos|> and <|audio_eos|>. Everything between those tags will be in {lang}.\n\n"
        )

    @pytest.mark.parametrize("family", sorted(TASK_TEMPLATES))
    def test_every_template_has_the_audio_slot(self, family):
        assert all("{audio_token}" in t for t in TASK_TEMPLATES[family])

    def test_asr_list_is_untouched(self):
        asr = TASK_TEMPLATES["asr"]
        assert len(asr) == 12
        assert all(_has_lang(t) for t in asr[:6])
        assert not any(_has_lang(t) for t in asr[6:])
        assert asr[0] == "{audio_token} Transcribe this audio in {lang}."
        assert asr[6] == "{audio_token} Transcribe this audio."


class TestSelectionModes:
    @pytest.mark.parametrize("family", ["asr", "verbatim"])
    def test_without_language_returns_only_templates_lacking_lang(self, family):
        templates = TASK_TEMPLATES[family]
        rng = random.Random(0)
        drawn = {select_prompt_template(templates, "without_language", rng) for _ in range(300)}
        assert drawn == {t for t in templates if not _has_lang(t)}

    @pytest.mark.parametrize("family", ["asr", "verbatim"])
    def test_with_language_returns_only_templates_containing_lang(self, family):
        templates = TASK_TEMPLATES[family]
        rng = random.Random(0)
        drawn = {select_prompt_template(templates, "with_language", rng) for _ in range(300)}
        assert drawn == {t for t in templates if _has_lang(t)}

    def test_with_language_on_verbatim_now_returns_the_new_variants(self):
        rng = random.Random(0)
        drawn = {select_prompt_template(TASK_TEMPLATES["verbatim"], "with_language", rng) for _ in range(300)}
        assert drawn == set(TASK_TEMPLATES["verbatim"][6:])

    @pytest.mark.parametrize("family", sorted(TASK_TEMPLATES))
    def test_random_still_draws_from_the_full_list(self, family):
        templates = TASK_TEMPLATES[family]
        rng = random.Random(0)
        drawn = {select_prompt_template(templates, "random", rng) for _ in range(2000)}
        assert drawn == set(templates)

    def test_random_consumes_the_global_rng_exactly_as_before(self):
        """Seeded runs must keep drawing the same templates: "random" is one
        ``random.choice`` over the unfiltered list, on the global generator."""
        templates = TASK_TEMPLATES["asr"]
        random.seed(1234)
        expected = [random.choice(templates) for _ in range(20)]
        random.seed(1234)
        assert [select_prompt_template(templates, "random") for _ in range(20)] == expected

    def test_without_language_raises_when_every_template_names_the_language(self):
        with pytest.raises(ValueError, match="without_language"):
            select_prompt_template(TASK_TEMPLATES["st"], "without_language")

    def test_with_language_raises_when_no_template_names_the_language(self):
        with pytest.raises(ValueError, match="with_language"):
            select_prompt_template(["{audio_token} Go."], "with_language")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt_template_selection"):
            select_prompt_template(TASK_TEMPLATES["asr"], "best")


class TestBothCallersShareTheModes:
    """The selection branch used to be duplicated; both entry points must now
    accept and reject exactly the same modes."""

    def _asr(self, selection, lang="de", override=None):
        return apply_chat_template_to_texts(
            ["hallo"], ["asr"], [lang], _StubTokenizer(), AUDIO,
            prompt_template_selection=selection, template_task_override=override,
        )[0]

    def _qe(self, selection):
        return apply_qe_chat_template_to_texts(
            ["hello"], ["de"], _StubTokenizer(), AUDIO, prompt_template_selection=selection,
        )[0]

    def test_asr_without_language_renders_no_language_name(self):
        for _ in range(50):
            assert "German" not in self._asr("without_language")

    def test_asr_with_language_renders_the_language_name(self):
        for _ in range(50):
            assert "German" in self._asr("with_language")

    def test_verbatim_override_with_language_reaches_the_prompt(self):
        for _ in range(50):
            rendered = self._asr("with_language", override="verbatim")
            assert "German" in rendered
            assert "{lang}" not in rendered

    def test_language_name_not_iso_code_reaches_the_prompt(self):
        for _ in range(50):
            rendered = self._asr("with_language", lang="it", override="verbatim")
            assert "Italian" in rendered
            assert "{lang}" not in rendered
        assert LANGUAGE_ISO_TO_NAME["it"] == "Italian"

    def test_verbatim_without_language_is_the_original_six(self):
        rendered = {self._asr("without_language", override="verbatim") for _ in range(300)}
        assert len(rendered) == 6
        assert all("German" not in r for r in rendered)

    def test_qe_accepts_with_language_and_rejects_without_symmetrically(self):
        assert "German" in self._qe("with_language")
        with pytest.raises(ValueError, match="without_language"):
            self._qe("without_language")

    def test_both_callers_reject_an_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown prompt_template_selection"):
            self._asr("best")
        with pytest.raises(ValueError, match="Unknown prompt_template_selection"):
            self._qe("best")

    def test_st_without_language_raises_through_the_caller(self):
        with pytest.raises(ValueError, match="without_language"):
            apply_chat_template_to_texts(
                ["x"], ["st"], ["de"], _StubTokenizer(), AUDIO,
                prompt_template_selection="without_language",
            )
