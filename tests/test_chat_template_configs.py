"""Chat-template boundaries must match what the tokenizer actually renders.

Label masking finds ``assistant_start`` / ``assistant_end`` as literal strings.
When they are absent the span is simply never located — nothing raises, and the
run trains on the wrong tokens while reporting a plausible loss. These tests
pin the three shapes the ablation campaign needs and, where the real tokenizer
is cached, check the boundaries against it rather than against a hand-written
fixture. The Qwen 3 `<think>` case in particular is invisible to any test that
does not render with the genuine template.
"""

import os

import pytest

from melt.training.data.chat_templates import (
    CHAT_TEMPLATE_CONFIGS,
    get_chat_template_config,
    validate_chat_template_config,
)


PROBE = [
    {"role": "user", "content": "__melt_probe_user__"},
    {"role": "assistant", "content": "__melt_probe_assistant__"},
]


class _FakeTokenizer:
    """Renders a fixed string, standing in for a real chat template."""

    def __init__(self, rendered, chat_template="present"):
        self._rendered = rendered
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self._rendered


class TestRegisteredConfigs:
    def test_the_campaign_backbones_are_all_covered(self):
        # Qwen 2.5 and EuroLLM are plain ChatML; Qwen 3/3.5 and Llama 3 are not.
        assert {"chatml", "qwen3", "llama3"} <= set(CHAT_TEMPLATE_CONFIGS)

    def test_qwen3_boundary_absorbs_the_empty_think_block(self):
        cfg = get_chat_template_config("qwen3")

        assert cfg.assistant_start.endswith("<think>\n\n</think>\n\n")

    def test_llama3_uses_header_boundaries_not_chatml(self):
        cfg = get_chat_template_config("llama3")

        assert cfg.assistant_start == "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assert cfg.assistant_end == "<|eot_id|>"


class TestValidation:
    def test_missing_boundary_raises(self):
        # A Llama-shaped render under the chatml config: the classic silent case.
        tok = _FakeTokenizer(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "__melt_probe_assistant__<|eot_id|>"
        )

        with pytest.raises(ValueError, match="does not match this tokenizer"):
            validate_chat_template_config(tok, get_chat_template_config("chatml"), "chatml")

    def test_text_between_boundary_and_content_raises(self):
        # Qwen 3 under plain chatml: both boundaries are present, so a
        # substring check passes, yet the think block would be trained on.
        tok = _FakeTokenizer(
            "<|im_start|>user\n__melt_probe_user__<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "__melt_probe_assistant__<|im_end|>\n"
        )

        with pytest.raises(ValueError, match="between the assistant boundary"):
            validate_chat_template_config(tok, get_chat_template_config("chatml"), "chatml")

    def test_matching_config_passes(self):
        tok = _FakeTokenizer(
            "<|im_start|>user\n__melt_probe_user__<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "__melt_probe_assistant__<|im_end|>\n"
        )

        validate_chat_template_config(tok, get_chat_template_config("qwen3"), "qwen3")

    def test_tokenizer_without_a_template_is_skipped(self):
        tok = _FakeTokenizer("irrelevant", chat_template=None)

        validate_chat_template_config(tok, get_chat_template_config("chatml"), "chatml")


# The mock tests above prove the validator's logic. Only a real tokenizer proves
# the boundary strings themselves are right, so run against the cache when it is
# populated and skip cleanly when it is not.
REAL_CASES = [
    ("Qwen/Qwen3.5-9B", "qwen3"),
    ("Qwen/Qwen2.5-1.5B", "chatml"),
    ("utter-project/EuroLLM-9B-Instruct", "chatml"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
]


@pytest.mark.parametrize("model_id,config_name", REAL_CASES)
def test_boundaries_match_the_real_tokenizer(model_id, config_name):
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_id} not in the local HF cache ({type(exc).__name__})")

    validate_chat_template_config(tok, get_chat_template_config(config_name), config_name)
