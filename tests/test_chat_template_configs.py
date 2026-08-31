"""Chat-template boundaries must match what the tokenizer actually renders.

Label masking finds ``assistant_start`` / ``assistant_end`` as literal strings.
When they are absent the span is simply never located — nothing raises, and the
run trains on the wrong tokens while reporting a plausible loss. These tests
pin the three shapes the ablation campaign needs and, where the real tokenizer
is cached, check the boundaries against it rather than against a hand-written
fixture. The Qwen 3 `<think>` case in particular is invisible to any test that
does not render with the genuine template.
"""


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
        # Qwen 2.5, Qwen 3/3.5 and EuroLLM all render ChatML boundaries;
        # Llama 3 does not.
        assert {"chatml", "llama3"} <= set(CHAT_TEMPLATE_CONFIGS)

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

    def test_injected_text_warns_but_does_not_raise(self, caplog):
        # Qwen 3 under chatml. The think block does land inside the loss, but no
        # boundary string can exclude it, so raising here would reject a config
        # that behaves identically to every alternative.
        tok = _FakeTokenizer(
            "<|im_start|>user\n__melt_probe_user__<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "__melt_probe_assistant__<|im_end|>\n"
        )

        with caplog.at_level("WARNING"):
            validate_chat_template_config(
                tok, get_chat_template_config("chatml"), "chatml"
            )

        assert "think" in caplog.text

    def test_tokenizer_without_a_template_raises(self):
        # The base-checkpoint case. This used to return quietly and let
        # apply_chat_template() raise on the first batch instead -- past model
        # construction and, on a cluster, past the queue wait.
        tok = _FakeTokenizer("irrelevant", chat_template=None)

        with pytest.raises(ValueError, match="has no chat template"):
            validate_chat_template_config(tok, get_chat_template_config("chatml"), "chatml")

    def test_the_no_template_error_names_the_way_out(self):
        tok = _FakeTokenizer("irrelevant", chat_template=None)

        with pytest.raises(ValueError) as excinfo:
            validate_chat_template_config(tok, get_chat_template_config("llama3"), "llama3")

        # Both escapes have to be reachable from the message alone.
        assert "chat_template_from" in str(excinfo.value)
        assert "apply_chat_template: false" in str(excinfo.value)


# The mock tests above prove the validator's logic. Only a real tokenizer proves
# the boundary strings themselves are right, so run against the cache when it is
# populated and skip cleanly when it is not.
REAL_CASES = [
    ("Qwen/Qwen3.5-9B", "chatml"),
    ("Qwen/Qwen2.5-1.5B", "chatml"),
    ("utter-project/EuroLLM-9B-Instruct", "chatml"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
]


def test_masking_is_inclusive_of_the_boundaries():
    """Pin the semantics, because they are easy to describe wrongly.

    `mask_non_assistant_tokens` keeps the boundary tokens themselves, so the
    training target includes the assistant header — and, for Qwen 3, the empty
    `<think>` block the template injects after it. No choice of boundary string
    changes that: a longer `assistant_start` still begins the kept span at the
    same index. Excluding the header would require changing the masking, not the
    config, so this test exists to stop a future reader assuming otherwise.
    """
    torch = pytest.importorskip("torch")

    from melt.training.data.audio.lhotse.helpers import mask_non_assistant_tokens

    # start=[1,2], content=[3], end=[4]
    labels = torch.tensor([[9, 1, 2, 3, 4, 9]])
    masked = mask_non_assistant_tokens(labels.clone(), [1, 2], [4])

    kept = [t for t in masked[0].tolist() if t != -100]

    assert kept == [1, 2, 3, 4], (
        "masking is expected to keep the boundaries inclusively; if this "
        "changed, the chat-template configs and their documentation need "
        "revisiting"
    )


@pytest.mark.hub
@pytest.mark.parametrize("model_id,config_name", REAL_CASES)
def test_boundaries_match_the_real_tokenizer(model_id, config_name):
    transformers = pytest.importorskip("transformers")

    # NB: this used to `os.environ.setdefault("HF_HUB_OFFLINE", "1")` here to
    # keep the test cache-only. That never worked -- huggingface_hub reads the
    # variable once, at import time, and transformers is already imported by
    # then -- so the test downloaded anyway, and the stray variable leaked
    # offline mode into every later test in the process. The `hub` marker is
    # what keeps this off the network in CI now.
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_id} not in the local HF cache ({type(exc).__name__})")

    validate_chat_template_config(tok, get_chat_template_config(config_name), config_name)
