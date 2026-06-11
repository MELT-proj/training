"""
Shared constants, utilities, and helper functions for Lhotse-based datasets.

These are used by both the batch-oriented :class:`SpeechToTextDataset`
and the map-style :class:`MELTMapDataset`.
"""

from lhotse.cut import Cut

from .....logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Config helpers
# =============================================================================


def _get_config_value(config, key: str, default=None):
    """Get a value from config, supporting both dataclass and dict access."""
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _get_nested_value(obj, path: str, default=None):
    """Get a nested value from an object using dot notation.

    Supports both attribute access and dict key access at each level.

    Args:
        obj: Object to traverse (can have nested dicts/objects).
        path: Dot-separated path like ``"custom.metadata.sentence"``.
        default: Default value if path not found.

    Returns:
        The value at the path, or *default* if not found.
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if current is None:
            return default
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current if current is not None else default


# =============================================================================
# Language & task constants
# =============================================================================

# ISO 639-1 language code to spelled-out language name.
# Regional variants use the format "Language (REGION)".
LANGUAGE_ISO_TO_NAME: dict[str, str] = {
    # Base language codes
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mn": "Mongolian",
    "mt": "Maltese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sv": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
    # Regional variants (keys are lowercase; _get_tags lowercases before lookup)
    "de_de": "German",
    "en_us": "English (US)",
    "es_419": "Spanish (Latin America)",
    "es_es": "Spanish (Spain)",
    "fr_fr": "French (France)",
    "it_it": "Italian (Italy)",
    "pt_br": "Portuguese (BR)",
    "pt_pt": "Portuguese (PT)",
    "sv-se": "Swedish (SE)",
    "zh-cn": "Chinese (CN)",
    "zh-hk": "Chinese (HK)",
    "zh-tw": "Chinese (TW)",
}

# Task-specific prompt templates for chat-template mode.
# Each template must contain {audio_token} and {lang} placeholders.
TASK_TEMPLATES: dict[str, list[str]] = {
    "asr": [
        "{audio_token} Transcribe this audio in {lang}.",
        "Transcribe the following {lang} audio: {audio_token}",
        "{audio_token} Write down what is said in this {lang} recording.",
        "Listen to this {lang} audio and transcribe it: {audio_token}",
        "{audio_token} Provide a transcription of the {lang} speech.",
        "What is being said in this {lang} audio? {audio_token}",
        # Same as above but with no LID
        "{audio_token} Transcribe this audio.",
        "Transcribe the following audio: {audio_token}",
        "{audio_token} Write down what is said in this recording.",
        "Listen to this audio and transcribe it: {audio_token}",
        "{audio_token} Provide a transcription of the speech.",
        "What is being said in this audio? {audio_token}",
    ],
    "st": [
        "{audio_token} Translate this audio to {lang}.",
        "Translate the following audio into {lang}: {audio_token}",
        "{audio_token} Provide a translation of this speech to {lang}.",
        "Listen to this audio and translate it to {lang}: {audio_token}",
        "{audio_token} What is being said? Translate into {lang}.",
        "Convert the speech in this audio to {lang}: {audio_token}",
    ],
    "speechqe": [
        "{audio_token} Score how well this {lang} translation matches the audio. Return a float between 0 and 1.",
        "Given this audio and its {lang} translation, provide a quality score from 0 to 1: {audio_token}",
        "{audio_token} Evaluate the quality of the following {lang} translation for this speech and answer with a single float in [0, 1].",
        "Listen to this audio, assess the provided {lang} translation, and output only a float between 0 and 1: {audio_token}",
    ],
}


# =============================================================================
# Per-cut helpers
# =============================================================================


def load_audio_from_cut(cut: Cut) -> "np.ndarray | None":  # noqa: F821
    """Load audio from a Lhotse cut and return it as a numpy array.

    Args:
        cut: Lhotse Cut object.

    Returns:
        1-D float32 numpy array of audio samples, or ``None`` on failure.
    """
    import numpy as np
    import torch

    try:
        audio = cut.load_audio()
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if audio.ndim > 1:
            audio = audio[0]  # mono
        return audio.astype(np.float32)
    except Exception:
        logger.warning("Failed to load audio for cut %s.", cut.id)
        return None


def get_text_from_cut(cut: Cut, text_field: str) -> str | None:
    """Extract the text transcript from a cut.

    Args:
        cut: Lhotse Cut object.
        text_field: Dot-separated path to the text field (e.g.
            ``"custom.metadata.sentence"``) or ``"text"`` to use
            supervision text directly.

    Returns:
        Text transcript, or ``None`` if no text is available.
    """
    text: str | None = None

    if text_field and text_field != "text":
        text = _get_nested_value(cut, text_field)
        if text is None and hasattr(cut, "custom") and cut.custom:
            text = cut.custom.get(text_field)

    # Fall back to supervision text
    if text is None and cut.supervisions:
        texts = [sup.text for sup in cut.supervisions if sup.text]
        if texts:
            text = " ".join(texts)

    # Final fallback: ``custom.text``
    if text is None and hasattr(cut, "custom") and cut.custom:
        text = cut.custom.get("text")

    if text is not None:
        text = text.strip()

    return text or None


def get_tags_from_cut(cut: Cut) -> tuple[str, str]:
    """Extract task and language tags from a cut.

    For ASR tasks the returned language is the ``lang`` tag.
    For ST tasks the returned language is ``tgt_lang`` (the target
    language), falling back to ``lang``.

    Args:
        cut: Lhotse Cut object.

    Returns:
        Tuple of ``(task, language)`` strings.  Both may be ``""`` if
        the cut carries no tags.
    """
    task = ""
    lang = ""

    # Supervision-level language
    if cut.supervisions:
        for sup in cut.supervisions:
            if sup.language:
                lang = sup.language
                break

    # Per-cut tags (set during dataset construction) take precedence
    if hasattr(cut, "tags") and cut.tags:
        task = cut.tags.get("task", task)
        if task in ("st", "translate"):
            lang = cut.tags.get("tgt_lang", lang)
        else:
            lang = cut.tags.get("lang", lang)

    return task, lang


# =============================================================================
# Chat-template helpers
# =============================================================================


def apply_chat_template_to_texts(
    texts: list[str],
    tasks: list[str],
    langs: list[str],
    tokenizer: "PreTrainedTokenizerBase",  # noqa: F821
    audio_token: str,
) -> list[str]:
    """Format each sample with a random task-specific prompt wrapped in the
    tokenizer's chat template.

    Args:
        texts: Transcripts (assistant responses).
        tasks: Task identifiers (``"asr"``, ``"st"``, …).
        langs: Language ISO codes.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        audio_token: The string representation of the audio placeholder token.

    Returns:
        List of fully formatted chat strings ready for the processor.
    """
    import random

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "apply_chat_template=True but the tokenizer does not support apply_chat_template()."
        )

    formatted: list[str] = []
    for text, task, lang in zip(texts, tasks, langs):
        templates = TASK_TEMPLATES.get(task)
        if templates is None:
            raise ValueError(
                f"Unknown task '{task}'. Expected one of: {', '.join(sorted(TASK_TEMPLATES.keys()))}"
            )
        template = random.choice(templates)

        lang_key = (lang or "").lower()
        language_name = LANGUAGE_ISO_TO_NAME.get(lang_key)
        if language_name is None:
            supported = ", ".join(sorted(LANGUAGE_ISO_TO_NAME.keys()))
            raise ValueError(
                f"Unsupported language ISO code '{lang}'. Expected one of: {supported}"
            )
        prompt = template.format(audio_token=audio_token, lang=language_name)

        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        formatted.append(full_text)

    return formatted


def apply_qe_chat_template_to_texts(
    texts: list[str],
    langs: list[str],
    tokenizer: "PreTrainedTokenizerBase",  # noqa: F821
    audio_token: str,
) -> list[str]:
    """Like :func:`apply_chat_template_to_texts` but for quality-estimation tasks.

    The assistant turn contains the candidate translation text whose quality
    is to be scored.
    """
    import random

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "apply_chat_template=True but the tokenizer does not support apply_chat_template()."
        )

    formatted: list[str] = []
    qe_templates = TASK_TEMPLATES["speechqe"]
    for text, lang in zip(texts, langs):
        template = random.choice(qe_templates)
        lang_key = (lang or "").lower()
        language_name = LANGUAGE_ISO_TO_NAME.get(lang_key)
        if language_name is None:
            supported = ", ".join(sorted(LANGUAGE_ISO_TO_NAME.keys()))
            raise ValueError(
                f"Unsupported language ISO code '{lang}'. Expected one of: {supported}"
            )
        prompt = template.format(audio_token=audio_token, lang=language_name)
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        formatted.append(full_text)

    return formatted


def mask_non_assistant_tokens(
    labels: "torch.Tensor",  # noqa: F821
    assistant_start_ids: list[int],
    assistant_end_ids: list[int],
) -> "torch.Tensor":  # noqa: F821
    """Mask all tokens not belonging to an assistant turn.

    Keeps only tokens between ``assistant_start_ids`` and
    ``assistant_end_ids`` spans (inclusive).  Everything else is set to
    ``-100``.

    Args:
        labels: Integer tensor of shape ``[B, S]``.
        assistant_start_ids: Token ID sequence that opens an assistant turn.
        assistant_end_ids: Token ID sequence that closes an assistant turn.

    Returns:
        Same tensor (mutated in-place).
    """
    start_len = len(assistant_start_ids)
    end_len = len(assistant_end_ids)

    for i in range(labels.size(0)):
        ids = labels[i].tolist()
        keep = [False] * len(ids)

        pos = 0
        while True:
            s = _find_subsequence(ids, assistant_start_ids, pos)
            if s == -1:
                break
            e = _find_subsequence(ids, assistant_end_ids, s + start_len)
            if e == -1:
                # Open-ended assistant turn — keep to end
                for j in range(s, len(ids)):
                    keep[j] = True
                break
            for j in range(s, e + end_len):
                keep[j] = True
            pos = e + end_len

        for j in range(len(ids)):
            if not keep[j]:
                labels[i, j] = -100

    return labels


def _find_subsequence(seq: list[int], subseq: list[int], start: int = 0) -> int:
    """Return the index of *subseq* in *seq* starting from *start*, or -1."""
    sub_len = len(subseq)
    for i in range(start, len(seq) - sub_len + 1):
        if seq[i : i + sub_len] == subseq:
            return i
    return -1
