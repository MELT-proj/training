"""
Speech-to-Text Dataset for Lhotse-based data loading.

This module provides a PyTorch Dataset that processes Lhotse CutSets
for speech-to-text tasks like ASR and speech translation.

The dataset does not hold actual data; instead, it acts as a processing
recipe that transforms CutSets into model inputs via the MELTProcessor.
"""

import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import Cut

from ....config import DataConfig
from .....logging_utils import get_logger
from .....modeling import MELTProcessor

logger = get_logger(__name__)


def _get_config_value(config, key: str, default=None):
    """Get a value from config, supporting both dataclass and dict access."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


class SpeechToTextDataset(torch.utils.data.Dataset):
    """A dataset for Speech-to-Text tasks that processes Lhotse CutSets.

    This dataset follows the Lhotse convention where __getitem__ receives
    a CutSet (batch of cuts) rather than an index. It loads audio and text
    from the cuts and processes them using MELTProcessor.

    Similar to NeMo's SALMDataset, this dataset:
    - Does not hold actual data
    - Acts as a processing function mapping CutSet -> model inputs
    - Supports task tags (asr, st) and language tags from cut metadata

    Args:
        processor: MELTProcessor instance for audio/text processing.
        config: DataConfig containing processing settings.
        is_train: Whether this is for training (affects augmentation, etc.).

    Example:
        >>> processor = MELTProcessor(feature_extractor, tokenizer)
        >>> config = DataConfig(apply_chat_template=False)
        >>> dataset = SpeechToTextDataset(processor, config)
        >>> # Called by Lhotse sampler/dataloader:
        >>> batch = dataset[cuts]  # cuts is a CutSet
    """

    def __init__(
        self,
        processor: MELTProcessor,
        config: DataConfig,
        is_train: bool = True,
    ) -> None:
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.apply_chat_template = bool(_get_config_value(config, "apply_chat_template", False))
        self.sample_rate = int(_get_config_value(config, "sample_rate", 16000))
        self.min_chars = int(_get_config_value(config, "min_chars", 0))

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor] | None:
        """Process a batch of cuts into model inputs.

        Args:
            cuts: A CutSet containing the cuts to process.

        Returns:
            Dictionary with model inputs:
                - input_features: Audio features [B, T, F]
                - feature_attention_mask: Audio attention mask [B, T]
                - input_ids: Text token IDs [B, S]
                - attention_mask: Text attention mask [B, S]
                - labels: Target labels for training [B, S]
                - audio_lengths: List of audio frame lengths per sample
            Returns None if all cuts fail to load (for fault tolerance).
        """
        if cuts is None or len(cuts) == 0:
            return None

        # Load audio and text from cuts
        audios = []
        texts = []
        tasks = []
        langs = []
        failed_indices = []

        for idx, cut in enumerate(cuts):
            try:
                # Load audio
                audio = self._load_audio(cut)
                if audio is None:
                    failed_indices.append(idx)
                    continue

                # Get text transcript
                text = self._get_text(cut)
                if text is None or len(text.strip()) < self.min_chars:
                    failed_indices.append(idx)
                    continue

                # Get task and language tags
                task, lang = self._get_tags(cut)

                audios.append(audio)
                texts.append(text)
                tasks.append(task)
                langs.append(lang)

            except Exception as e:
                logger.warning(f"Failed to process cut {cut.id}: {e}")
                failed_indices.append(idx)
                continue

        if len(audios) == 0:
            logger.warning("All cuts in batch failed to load")
            return None

        if failed_indices:
            logger.debug(f"Skipped {len(failed_indices)} cuts due to loading errors")

        # Format texts with audio token for the processor
        # This adds <|AUDIO|> token to indicate where audio embeddings go
        if self.apply_chat_template:
            formatted_texts = self._apply_chat_template(texts, tasks, langs)
        else:
            # Simple format: audio token + transcription
            formatted_texts = [self._format_text_with_audio_token(t, task, lang) for t, task, lang in zip(texts, tasks, langs)]

        # Process through MELTProcessor
        try:
            # Wrap each audio in a list since processor expects list of lists for batched input
            audio_inputs = [[a] for a in audios]

            inputs = self.processor(
                text=formatted_texts,
                audio=audio_inputs,
                sampling_rate=self.sample_rate,
                padding=True,
                return_tensors="pt",
                return_labels=True,
            )

            # Add the labels for loss computation. The labels for this class or exclusively the text token IDs. 
            # The modeling code will handle  logit slicing based on this tensor's shape and loss masking as needed
            # (Crucial: the training code must set loss_ignore_index to match the padding token ID of the tokenizer).
            inputs["labels"] = self.processor.tokenizer.pad(
                texts,
                padding_side="left",
                padding="longest",
                return_tensors="pt",
            )["input_ids"]

            return inputs

        except Exception as e:
            logger.error(f"Failed to process batch through processor: {e}")
            return None

    def _load_audio(self, cut: Cut) -> torch.Tensor | None:
        """Load audio from a cut.

        Args:
            cut: Lhotse Cut object.

        Returns:
            Audio tensor or None if loading fails.
        """
        try:
            # Load audio at target sample rate
            audio = cut.load_audio()

            # Convert to tensor if needed
            if not isinstance(audio, torch.Tensor):
                audio = torch.from_numpy(audio)

            # Ensure 1D (mono) - take first channel if stereo
            if audio.ndim > 1:
                audio = audio[0]

            return audio

        except Exception as e:
            logger.warning(f"Failed to load audio for cut {cut.id}: {e}")
            return None

    def _get_text(self, cut: Cut) -> str | None:
        """Extract text transcript from a cut.

        Args:
            cut: Lhotse Cut object.

        Returns:
            Text transcript or None if not available.
        """
        text = None

        # Try to get text from supervisions
        if cut.supervisions:
            texts = []
            for sup in cut.supervisions:
                if sup.text:
                    texts.append(sup.text)
            if texts:
                text = " ".join(texts)

        # Try custom field
        if text is None and hasattr(cut, "custom") and cut.custom:
            train_ds = self.config.get("train_ds") or {}
            text_field = train_ds.get("text_field", "text") if hasattr(train_ds, "get") else "text"
            if text_field in cut.custom:
                text = cut.custom[text_field]

        return text

    def _format_text_with_audio_token(self, text: str, task: str, lang: str) -> str:
        """Format text with audio token for the processor.

        The processor expects text to contain <|AUDIO|> token(s) indicating where
        audio embeddings should be inserted. This formats the text appropriately
        based on the task.

        Args:
            text: Raw transcript text.
            task: Task identifier (e.g., "transcribe", "asr", "translate", "st").
            lang: Language code.

        Returns:
            Formatted text with audio token.
        """
        audio_token = self.processor.audio_token

        # TODO: update this logic by supporting a config-specified template or prefix
        # E.g., "Transcribe the following audio: {audio_token}{text}" for ASR
        if task in ("transcribe", "asr"):
            # For ASR: audio followed by transcription
            return f"{audio_token}{text}"
        elif task in ("translate", "st"):
            # For translation: audio followed by translation
            return f"{audio_token}{text}"
        else:
            # Default: audio followed by text
            return f"{audio_token}{text}"

    def _get_tags(self, cut: Cut) -> tuple[str, str]:
        """Extract task and language tags from a cut.

        Args:
            cut: Lhotse Cut object.

        Returns:
            Tuple of (task, language) strings.
        """
        task = "transcribe"  # Default task
        lang = "en"  # Default language

        # Try to get from cut custom metadata
        if hasattr(cut, "custom") and cut.custom:
            task = cut.custom.get("task", task)
            lang = cut.custom.get("lang", lang)

        # Try to get language from supervision
        if cut.supervisions:
            for sup in cut.supervisions:
                if sup.language:
                    lang = sup.language
                    break

        # Check for tags added during dataset loading
        if hasattr(cut, "tags") and cut.tags:
            task = cut.tags.get("task", task)
            lang = cut.tags.get("lang", lang)

        return task, lang

    def _apply_chat_template(self, texts: list[str], tasks: list[str], langs: list[str]) -> list[str]:
        """Apply chat template formatting to texts.

        Args:
            texts: List of transcripts.
            tasks: List of task identifiers.
            langs: List of language codes.

        Returns:
            List of formatted texts with chat template and audio token.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "apply_chat_template=True but the processor tokenizer does not support apply_chat_template()."
            )

        audio_token = self.processor.audio_token
        formatted: list[str] = []

        for text, task, lang in zip(texts, tasks, langs):
            if task in ("transcribe", "asr"):
                prompt = f"Transcribe the following audio in {lang}: {audio_token}"
            elif task in ("translate", "st"):
                prompt = f"Translate the following audio to {lang}: {audio_token}"
            else:
                prompt = f"Process the following audio: {audio_token}"

            # Use tokenizer chat template (as done in tests) to build the final prompt.
            # For training, we include the assistant message with the ground-truth text.
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append(formatted_text)

        return formatted


class FallbackDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns previous batch on failure.

    This is useful for fault-tolerant training where some cuts may fail
    to load. Instead of crashing, it returns the previous successful batch.
    """

    def __init__(self, dataset: SpeechToTextDataset):
        self.dataset = dataset
        self._last_good_batch: dict[str, torch.Tensor] | None = None

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor]:
        result = self.dataset[cuts]

        if result is not None:
            self._last_good_batch = result
            return result

        if self._last_good_batch is not None:
            logger.warning("Using fallback batch due to loading failure")
            return self._last_good_batch

        raise RuntimeError("No fallback batch available and current batch failed to load")


__all__ = [
    "SpeechToTextDataset",
    "FallbackDataset",
]
