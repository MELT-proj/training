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
    # Qwen 3 / 3.5 open the assistant turn with an empty reasoning block:
    #   <|im_start|>assistant\n<think>\n\n</think>\n\n
    # `enable_thinking=False` does not remove it on these checkpoints, so the
    # block has to be part of the boundary. Masking on plain "chatml" instead
    # starts the trainable span *before* `<think>`, which trains the model to
    # emit an empty reasoning block ahead of every transcript.
    "qwen3": ChatTemplateConfig(
        assistant_start="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        assistant_end="<|im_end|>\n",
    ),
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
        ValueError: If either boundary is missing from the rendered probe.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return
    if not getattr(tokenizer, "chat_template", None):
        return

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

    # The trainable span must be the assistant content alone. If the rendered
    # text puts anything between the boundary and the content -- Qwen 3's
    # `<think>` block is the live example -- the mask would swallow it.
    start = rendered.find(config.assistant_start)
    content = rendered.find("__melt_probe_assistant__")
    if start >= 0 and content >= 0:
        between = rendered[start + len(config.assistant_start) : content]
        if between.strip():
            raise ValueError(
                f"chat_template_config '{name}' leaves {between!r} between the "
                "assistant boundary and the assistant content. That text would "
                "be trained on as part of the target. Extend assistant_start to "
                "cover it, or pick a config that does."
            )
