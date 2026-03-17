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
    "chatml": ChatTemplateConfig(
        assistant_start="<|im_start|>assistant\n",
        assistant_end="<|im_end|>\n",
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
