"""Tests for ``model.decoder.chat_template_from``.

A base checkpoint ships no chat template, so training one under
``data.apply_chat_template: true`` cannot render anything.  Borrowing the
template from the matching instruction-tuned checkpoint is what lets a
base arm and an instruct arm be compared on byte-identical rendered text.
"""

import logging

import pytest

from melt.training.setup import borrow_chat_template


INSTRUCT_TEMPLATE = "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"


class _FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


@pytest.fixture
def fake_hub(monkeypatch):
    """Route ``AutoTokenizer.from_pretrained`` to an in-memory registry."""
    registry: dict[str, _FakeTokenizer] = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            if name not in registry:
                raise AssertionError(f"unexpected checkpoint requested: {name!r}")
            return registry[name]

    monkeypatch.setattr("melt.training.setup.AutoTokenizer", _FakeAutoTokenizer)
    return registry


class TestBorrowChatTemplate:
    def test_base_tokenizer_gets_the_instruct_template(self, fake_hub):
        fake_hub["instruct"] = _FakeTokenizer(INSTRUCT_TEMPLATE)
        base = _FakeTokenizer(None)

        borrow_chat_template(base, "instruct", "base")

        assert base.chat_template == INSTRUCT_TEMPLATE

    def test_the_source_tokenizer_is_not_adopted_wholesale(self, fake_hub):
        # Only the template crosses over. Taking the whole tokenizer would pair
        # one vocabulary with another checkpoint's weights.
        source = _FakeTokenizer(INSTRUCT_TEMPLATE)
        source.vocab_marker = "instruct-vocab"
        fake_hub["instruct"] = source
        base = _FakeTokenizer(None)
        base.vocab_marker = "base-vocab"

        borrow_chat_template(base, "instruct", "base")

        assert base.vocab_marker == "base-vocab"

    def test_source_without_a_template_raises(self, fake_hub):
        fake_hub["another-base"] = _FakeTokenizer(None)
        base = _FakeTokenizer(None)

        with pytest.raises(ValueError, match="no chat template either"):
            borrow_chat_template(base, "another-base", "base")

    def test_borrowing_onto_an_instruct_tokenizer_warns(self, fake_hub, caplog):
        # Not an error -- an explicit config wins -- but silently replacing a
        # working template is how two arms stop being comparable.
        fake_hub["instruct"] = _FakeTokenizer(INSTRUCT_TEMPLATE)
        already = _FakeTokenizer("{{ 'some other template' }}")

        with caplog.at_level(logging.WARNING):
            borrow_chat_template(already, "instruct", "also-instruct")

        assert already.chat_template == INSTRUCT_TEMPLATE
        assert "overwriting" in caplog.text.lower()

    def test_borrowing_onto_a_bare_tokenizer_does_not_warn(self, fake_hub, caplog):
        fake_hub["instruct"] = _FakeTokenizer(INSTRUCT_TEMPLATE)
        base = _FakeTokenizer(None)

        with caplog.at_level(logging.WARNING):
            borrow_chat_template(base, "instruct", "base")

        assert "overwriting" not in caplog.text.lower()

    def test_an_empty_source_template_is_treated_as_missing(self, fake_hub):
        # `chat_template: ""` in a tokenizer_config renders nothing; copying it
        # would push the failure back out to the first batch.
        fake_hub["empty"] = _FakeTokenizer("")
        base = _FakeTokenizer(None)

        with pytest.raises(ValueError, match="no chat template either"):
            borrow_chat_template(base, "empty", "base")
