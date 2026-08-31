"""Chat template configurations for label masking.

Each configuration defines the string boundaries that delimit assistant
responses inside a fully-rendered chat template.  The masking logic in
:class:`SpeechToTextDataset` uses these boundaries to decide which tokens
in ``input_ids`` should be trained on (the assistant turns) and which
should be masked (everything else).

Example – ChatML format::

    <|im_start|>user
    Transcribe this audio.<|im_end|>
    <|im_start|>assistant
    Hello world.<|im_end|>

Here ``assistant_start = "<|im_start|>assistant\n"`` marks where an
assistant turn begins and ``assistant_end = "<|im_end|>\n"`` marks where
it ends.
"""

from dataclasses import dataclass

from ...logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatTemplateConfig:
    """Boundary strings that delimit assistant responses in a chat template.

    Attributes:
        assistant_start: Token sequence that opens an assistant turn
            (e.g. ``"<|im_start|>assistant\\n"``).
        assistant_end: Token sequence that closes an assistant turn
            (e.g. ``"<|im_end|>\\n"``).
    """

    assistant_start: str
    assistant_end: str


CHAT_TEMPLATE_CONFIGS: dict[str, ChatTemplateConfig] = {
    # Qwen 2.5, EuroLLM, and anything else emitting plain ChatML.
    "chatml": ChatTemplateConfig(
        assistant_start="<|im_start|>assistant\n",
        assistant_end="<|im_end|>\n",
    ),
    # NOTE on Qwen 3 / 3.5: they open the assistant turn with an empty reasoning
    # block, `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and
    # `enable_thinking=False` does not remove it. There is deliberately no
    # separate entry for them, because one would not help:
    # `mask_non_assistant_tokens` keeps the boundaries *inclusive*, so a longer
    # `assistant_start` still begins the kept span at the same index and yields
    # byte-identical labels. The block is inside the loss either way. Fixing that
    # means changing the masking semantics, not the boundary strings — see
    # `docs/` for the write-up.
    # Llama 3.x header format — not ChatML, and silently unfindable under it.
    "llama3": ChatTemplateConfig(
        assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_end="<|eot_id|>",
    ),
}


def get_chat_template_config(name: str) -> ChatTemplateConfig:
    """Return the :class:`ChatTemplateConfig` registered under *name*.

    Args:
        name: Key into :data:`CHAT_TEMPLATE_CONFIGS`.

    Raises:
        ValueError: If *name* is not a registered configuration.
    """
    if name not in CHAT_TEMPLATE_CONFIGS:
        raise ValueError(
            f"Unknown chat template config '{name}'. "
            f"Available: {list(CHAT_TEMPLATE_CONFIGS.keys())}"
        )
    return CHAT_TEMPLATE_CONFIGS[name]


def validate_chat_template_config(tokenizer, config: ChatTemplateConfig, name: str) -> None:
    """Check that *config*'s boundaries appear in what *tokenizer* actually renders.

    Label masking locates ``assistant_start`` / ``assistant_end`` as literal
    strings. When they are absent — a Llama tokenizer under the ``chatml``
    config, say — nothing raises: the span is simply never found, and the run
    trains on the wrong tokens while reporting a perfectly plausible loss. A
    boundary mismatch has to fail the run, not degrade it.

    Args:
        tokenizer: Tokenizer that will render the training samples.
        config: The boundary configuration being used.
        name: Its registry key, for the error message.

    Raises:
        ValueError: If the tokenizer has no chat template at all, or if either
            boundary is missing from the rendered probe.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return
    if not getattr(tokenizer, "chat_template", None):
        # Reached only when the caller has already decided to apply the chat
        # template, so this is always a misconfiguration rather than a tokenizer
        # this function has no opinion about. It used to return quietly, which
        # deferred the failure to `apply_chat_template()` on the first batch --
        # past model construction, past the first shard read, and on MN5 past
        # the queue wait. Base checkpoints are the usual way in: they ship no
        # template while their Instruct siblings do.
        raise ValueError(
            f"chat_template_config is '{name}' and apply_chat_template is on, but this "
            "tokenizer has no chat template, so nothing can render. Base checkpoints "
            "normally ship none.\n"
            "Either set model.decoder.chat_template_from to an instruction-tuned "
            "checkpoint sharing this vocabulary (usually the Instruct sibling of the "
            "decoder), or set data.apply_chat_template: false."
        )

    probe = [
        {"role": "user", "content": "__melt_probe_user__"},
        {"role": "assistant", "content": "__melt_probe_assistant__"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    except Exception:  # noqa: BLE001 - a tokenizer that cannot render is not our problem here
        return

    missing = [
        label
        for label, boundary in (
            ("assistant_start", config.assistant_start),
            ("assistant_end", config.assistant_end),
        )
        if boundary not in rendered
    ]
    if missing:
        raise ValueError(
            f"chat_template_config '{name}' does not match this tokenizer: "
            f"{', '.join(missing)} not found in the rendered chat. Label "
            f"masking would silently train on the wrong tokens.\n"
            f"Rendered probe: {rendered!r}\n"
            f"Available configs: {list(CHAT_TEMPLATE_CONFIGS.keys())}"
        )

    # Anything the template injects between the boundary and the content -- Qwen
    # 3's empty `<think>` block, say -- lands inside the loss. That is worth
    # knowing, but it is not a *mismatch*: masking keeps the boundaries
    # inclusively, so no choice of boundary string excludes it. Warn rather than
    # raise, or a config that behaves identically to the default would be
    # rejected for no gain.
    start = rendered.find(config.assistant_start)
    content = rendered.find("__melt_probe_assistant__")
    if start >= 0 and content >= 0:
        between = rendered[start + len(config.assistant_start) : content]
        if between.strip():
            logger.warning(
                "Chat template for config %r injects %r before the assistant "
                "content. Because label masking is inclusive of the boundaries, "
                "those tokens are part of the training target, and the model "
                "will learn to emit them before every response. Strip them at "
                "evaluation time or the hypothesis will contain them.",
                name,
                between,
            )
