"""
Speech-to-Text Dataset for Lhotse-based data loading.

This module provides a PyTorch Dataset that processes Lhotse CutSets
for speech-to-text tasks like ASR and speech translation.

The dataset does not hold actual data; instead, it acts as a processing
recipe that transforms CutSets into model inputs via the MELTProcessor.
"""

import json
import os
import time

import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import Cut
from omegaconf import DictConfig

from .....logging_utils import get_logger
from .....modeling import MELTProcessor
from ....data.chat_templates import ChatTemplateConfig, get_chat_template_config
from .helpers import (
    LANGUAGE_ISO_TO_NAME,
    TASK_TEMPLATES,
    _get_config_value,
    _get_nested_value,
    apply_chat_template_to_texts,
    apply_qe_chat_template_to_texts,
    get_tags_from_cut,
    get_text_from_cut,
    load_audio_from_cut,
    mask_non_assistant_tokens,
)


logger = get_logger(__name__)


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
        config: DictConfig containing data processing settings.
        is_train: Whether this is for training (affects augmentation, etc.).

    Example:
        >>> processor = MELTProcessor(feature_extractor, tokenizer)
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({"apply_chat_template": False})
        >>> dataset = SpeechToTextDataset(processor, config)
        >>> # Called by Lhotse sampler/dataloader:
        >>> batch = dataset[cuts]  # cuts is a CutSet
    """

    def __init__(
        self,
        processor: MELTProcessor,
        config: DictConfig,
        is_train: bool = True,
        return_labels: bool = True,
        return_langs: bool = False,
    ) -> None:
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.return_labels = return_labels
        self.return_langs = return_langs
        self.apply_chat_template = bool(_get_config_value(config, "apply_chat_template", False))
        self.sample_rate = int(_get_config_value(config, "sample_rate", 16000))
        self.min_chars = int(_get_config_value(config, "min_chars", 0))

        # Template selection strategy when apply_chat_template is True.
        # "random"        – randomly pick from all templates for the task (current default)
        # "with_language" – only pick templates that include the {lang} placeholder
        # "custom"        – use the exact template string from self.prompt_template
        self.prompt_template_selection = str(_get_config_value(config, "prompt_template_selection", "random"))
        if self.prompt_template_selection not in ("random", "with_language", "custom"):
            raise ValueError(
                f"Invalid prompt_template_selection '{self.prompt_template_selection}'. "
                "Must be one of: 'random', 'with_language', 'custom'."
            )

        # Optional custom prompt template. Used when:
        # - apply_chat_template=True and prompt_template_selection="custom"
        # - apply_chat_template=False, to override the default "{audio_token}{t}" format
        self.prompt_template = _get_config_value(config, "prompt_template", None)
        if self.prompt_template_selection == "custom" and not self.prompt_template:
            raise ValueError("prompt_template_selection='custom' requires prompt_template to be set in config.")

        # Pre-compute boundary token IDs for chat-template label masking.
        if self.apply_chat_template:
            ct_name = str(_get_config_value(config, "chat_template_config", "chatml"))
            ct_cfg: ChatTemplateConfig = get_chat_template_config(ct_name)
            self._assistant_start_ids: list[int] = processor.tokenizer.encode(
                ct_cfg.assistant_start, add_special_tokens=False
            )
            self._assistant_end_ids: list[int] = processor.tokenizer.encode(
                ct_cfg.assistant_end, add_special_tokens=False
            )

        self._debug_cut_ids_dir = os.environ.get("MELT_DEBUG_CUT_IDS_DIR")
        self._debug_cut_ids_max_batches = int(os.environ.get("MELT_DEBUG_CUT_IDS_MAX_BATCHES", "0") or "0")
        self._debug_cut_ids_every = int(os.environ.get("MELT_DEBUG_CUT_IDS_EVERY", "1") or "1")
        self._debug_cut_ids_batch_idx = 0
        self._debug_cut_ids_fh = None

    def _maybe_log_cut_ids(self, cuts: CutSet) -> None:
        if not self._debug_cut_ids_dir:
            return
        if self._debug_cut_ids_max_batches <= 0:
            return
        if self._debug_cut_ids_batch_idx >= self._debug_cut_ids_max_batches:
            return
        if self._debug_cut_ids_every > 1 and (self._debug_cut_ids_batch_idx % self._debug_cut_ids_every) != 0:
            self._debug_cut_ids_batch_idx += 1
            return

        # Note: in DataLoader worker processes, torch.distributed is typically not initialized.
        # We rely on env vars; Lhotse's make_worker_init_fn sets RANK/WORLD_SIZE for WebDataset.
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")) or "0")
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")) or "1")
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        os.makedirs(self._debug_cut_ids_dir, exist_ok=True)
        if self._debug_cut_ids_fh is None:
            pid = os.getpid()
            path = os.path.join(
                self._debug_cut_ids_dir,
                f"cut_ids.rank{rank:05d}-ws{world_size:05d}.worker{worker_id:02d}.pid{pid}.jsonl",
            )
            # Line-buffered for "tail -f" usability.
            self._debug_cut_ids_fh = open(path, "a", encoding="utf-8", buffering=1)

        cut_ids = [c.id for c in cuts]
        record = {
            "time": time.time(),
            "rank": rank,
            "world_size": world_size,
            "worker_id": worker_id,
            "batch_idx": self._debug_cut_ids_batch_idx,
            "num_cuts": len(cut_ids),
            "cut_ids": cut_ids,
        }
        self._debug_cut_ids_fh.write(json.dumps(record) + "\n")
        self._debug_cut_ids_batch_idx += 1

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor] | None:
        """Process a batch of cuts into model inputs.

        Args:
            cuts: A CutSet containing the cuts to process.

        Returns:
            Dictionary with model inputs:
                - input_features: Audio features [B, T, F]
                - features_attention_mask: Audio attention mask [B, T]
                - input_ids: Text token IDs [B, S]
                - attention_mask: Text attention mask [B, S]
                - labels: Target labels for training [B, S]
                - audio_lengths: List of audio frame lengths per sample
            Returns None if all cuts fail to load (for fault tolerance).
        """
        if cuts is None or len(cuts) == 0:
            return None

        self._maybe_log_cut_ids(cuts)

        # Load audio and text from cuts
        audios = []
        texts = []
        tasks = []
        langs = []
        for idx, cut in enumerate(cuts):
            # Load audio
            audio = self._load_audio(cut)
            if audio is None:
                logger.warning("Skipping cut %s: audio failed to load.", cut.id)
                continue

            # Get text transcript
            text = self._get_text(cut)
            if not text or not text.strip():
                logger.warning(
                    "Skipping cut %s: empty or missing text (text_field=%r, supervisions=%s).",
                    cut.id,
                    getattr(getattr(cut, "custom", None) or {}, "get", lambda *a: None)("tags", {}).get("text_field")
                    if hasattr(cut, "custom") and cut.custom
                    else None,
                    len(cut.supervisions) if cut.supervisions else 0,
                )
                continue

            # Strip leading/trailing whitespace and make it lowercase
            text = text.strip().lower()

            # Get task and language tags
            task, lang = self._get_tags(cut)
            audios.append(audio)
            texts.append(text)
            tasks.append(task)
            langs.append(lang)

        if not audios:
            logger.warning("All %d cuts in this batch were skipped.", len(cuts))
            return None

        # Format texts with audio token for the processor
        # This adds <|audio|> token to indicate where audio embeddings go
        if self.apply_chat_template:
            formatted_texts = self._apply_chat_template(texts, tasks, langs)
        else:
            _prompt_texts = None  # only populated when self.prompt_template is set
            if self.prompt_template:
                # Custom template: supports {audio_token}, {t}, and {lang} placeholders.
                formatted_texts = []
                _prompt_texts = []  # template with {t} stripped — used for label masking
                for t, lang in zip(texts, langs):
                    language_name = self._resolve_language_name(lang)
                    full_text = self.prompt_template.format(
                        audio_token=self.processor.audio_token,
                        t=t,
                        lang=language_name,
                    )
                    prompt_text = self.prompt_template.format(
                        audio_token=self.processor.audio_token,
                        t="",
                        lang=language_name,
                    )
                    formatted_texts.append(full_text)
                    _prompt_texts.append(prompt_text)
            else:
                # Simple format: audio token + transcription
                formatted_texts = [f"{self.processor.audio_token}{t}" for t in texts]

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
            )

            if self.return_labels:
                labels = inputs["input_ids"].clone()
                if self.apply_chat_template:
                    labels = self._mask_non_assistant_tokens(labels)
                else:
                    # Non-chat-template mode: mask prompt tokens (if using a custom
                    # template) so that only the target text {t} contributes to the loss.
                    if self.prompt_template and _prompt_texts:
                        for i in range(labels.size(0)):
                            # Apply the same BOS/EOS wrapping the processor does
                            # internally (see MELTProcessor.__call__).
                            wrapped_prompt = self.processor._surround_bos_eos_mm_tokens(_prompt_texts[i])
                            wrapped_full = self.processor._surround_bos_eos_mm_tokens(formatted_texts[i])
                            prompt_char_len = len(wrapped_prompt)

                            # Use character-offset mapping to find the true
                            # token boundary.  BPE-based tokenizers often merge
                            # the last prompt character (e.g. a space) with the
                            # first target word into a single token, so we
                            # cannot simply measure the prompt token count in
                            # isolation — we must check where each token's
                            # character span lies.
                            encoding = self.processor.tokenizer(
                                wrapped_full,
                                add_special_tokens=True,
                                return_offsets_mapping=True,
                                return_tensors=None,
                            )
                            offsets = encoding["offset_mapping"]

                            for j, (start, end) in enumerate(offsets):
                                # Skip special tokens (offset 0,0) — handled
                                # by the element-wise mask below.
                                if start == 0 and end == 0:
                                    continue
                                # Mask tokens whose character span is entirely
                                # within the prompt portion.  Tokens that
                                # straddle the boundary (start < prompt_char_len
                                # < end) belong to the target and are kept.
                                if end <= prompt_char_len:
                                    labels[i, j] = -100

                    # Mask individual special tokens (overlapping masks are idempotent).
                    mask = (
                        (labels == self.processor.audio_token_id)
                        | (labels == self.processor.audio_bos_token_id)
                        | (labels == self.processor.audio_eos_token_id)
                        | (labels == self.processor.tokenizer.pad_token_id)
                        | (labels == self.processor.tokenizer.bos_token_id)
                    )
                    labels[mask] = -100
                inputs["labels"] = labels

            # Attach per-sample language codes (evaluation only) so metrics
            # can compute a per-language WER/CER breakdown.
            if self.return_langs:
                inputs["langs"] = langs

            return inputs

        except Exception as e:
            logger.error(f"Failed to process batch through processor: {e}")
            return None

    def _load_audio(self, cut: Cut):
        """Load audio from a cut (delegates to shared helper)."""
        return load_audio_from_cut(cut)

    def _get_text(self, cut: Cut) -> str | None:
        """Extract text transcript from a cut (delegates to shared helper)."""
        ds_config = _get_config_value(self.config, "train_ds" if self.is_train else "validation_ds", None)
        text_field = _get_config_value(ds_config, "text_field", "text") if ds_config else "text"
        # Per-cut text_field override from tags
        if hasattr(cut, "custom") and cut.custom:
            tags = cut.custom.get("tags", {})
            if isinstance(tags, dict) and tags.get("text_field"):
                text_field = tags["text_field"]
        return get_text_from_cut(cut, text_field)

    def _get_tags(self, cut: Cut) -> tuple[str, str]:
        """Extract task and language tags from a cut (delegates to shared helper)."""
        return get_tags_from_cut(cut)

    @staticmethod
    def _resolve_language_name(lang: str | None) -> str:
        """Resolve an ISO 639-1 language code to a human-readable name.

        Args:
            lang: Lowercase ISO code (e.g. ``"en"``, ``"fr"``) or ``None``.

        Returns:
            Spelled-out language name (e.g. ``"English"``, ``"French"``).

        Raises:
            ValueError: If *lang* is ``None`` or an unsupported code.
        """
        lang_key = (lang or "").lower()
        language_name = LANGUAGE_ISO_TO_NAME.get(lang_key)
        if language_name is None:
            supported = ", ".join(sorted(LANGUAGE_ISO_TO_NAME.keys()))
            raise ValueError(f"Unsupported language ISO code '{lang}'. Expected one of: {supported}")
        return language_name

    def _select_template(self, templates: list[str]) -> str:
        """Select a template from *templates* based on the configured strategy.

        Args:
            templates: List of template strings to choose from.

        Returns:
            A single template string.

        Raises:
            ValueError: If ``prompt_template_selection`` is unrecognised or
                if no suitable templates are found for the chosen strategy.
        """
        if self.prompt_template_selection == "random":
            return random.choice(templates)
        elif self.prompt_template_selection == "with_language":
            # Only pick templates that explicitly mention the source language.
            lang_templates = [t for t in templates if "{lang}" in t]
            if not lang_templates:
                raise ValueError(
                    "prompt_template_selection='with_language' but no templates "
                    f"with {{lang}} placeholder found among: {templates}"
                )
            return random.choice(lang_templates)
        elif self.prompt_template_selection == "custom":
            if self.prompt_template is None:
                raise ValueError("prompt_template_selection='custom' but prompt_template has not been set.")
            return self.prompt_template
        else:
            raise ValueError(f"Unknown prompt_template_selection: {self.prompt_template_selection}")

    def _apply_chat_template(self, texts: list[str], tasks: list[str], langs: list[str]) -> list[str]:
        """Apply chat template formatting to texts.

        For each sample a task-specific prompt is randomly sampled from
        ``TASK_TEMPLATES`` and wrapped with the tokenizer's chat template.

        Args:
            texts: List of transcripts (ground-truth assistant responses).
            tasks: List of task identifiers ("asr", "transcribe", "st", …).
            langs: List of language codes.

        Returns:
            List of fully formatted chat strings ready for the processor.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "apply_chat_template=True but the processor tokenizer does not support apply_chat_template()."
            )

        audio_token = self.processor.audio_token
        formatted: list[str] = []

        for text, task, lang in zip(texts, tasks, langs):
            # Pick a prompt template for the task according to the selection strategy
            templates = TASK_TEMPLATES.get(task)
            if not templates:
                raise ValueError(
                    f"No templates defined for task '{task}'. Available tasks: {list(TASK_TEMPLATES.keys())}"
                )
            template = self._select_template(templates)
            language_name = self._resolve_language_name(lang)
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

    def _apply_qe_chat_template(self, texts: list[str], langs: list[str]) -> list[str]:
        """Apply chat-template formatting for quality estimation inputs.

        Each sample is formatted as a user prompt containing the audio token and
        a request to score a translation, followed by an assistant turn that
        contains the candidate translation text.

        Args:
            texts: Candidate translations to be evaluated.
            langs: List of target language ISO codes.

        Returns:
            List of fully formatted chat strings ready for the processor.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "apply_chat_template=True but the processor tokenizer does not support apply_chat_template()."
            )

        audio_token = self.processor.audio_token
        formatted: list[str] = []

        qe_task_templates = TASK_TEMPLATES["speechqe"]

        for text, lang in zip(texts, langs):
            template = self._select_template(qe_task_templates)
            language_name = self._resolve_language_name(lang)
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

    # ------------------------------------------------------------------
    # Token-level label masking for chat-template mode
    # ------------------------------------------------------------------

    @staticmethod
    def _find_subsequence(seq: list[int], subseq: list[int], start: int = 0) -> int:
        """Return index of *subseq* in *seq* starting from *start*, or -1."""
        sub_len = len(subseq)
        for i in range(start, len(seq) - sub_len + 1):
            if seq[i : i + sub_len] == subseq:
                return i
        return -1

    def _mask_non_assistant_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """Mask non-assistant tokens (delegates to shared helper)."""
        return mask_non_assistant_tokens(
            labels,
            self._assistant_start_ids,
            self._assistant_end_ids,
        )


class SpeechTextQEDataset(SpeechToTextDataset):
    """A dataset for Speech Quality Estimation tasks that processes Lhotse CutSets.

    Identical to SpeechToTextDataset in audio packing, but replaces the
    text-based labels with a scalar floating-point quality score read from
    the cut's custom tags.

    The name of the tag that holds the score is taken from
    ``config.train_ds.target_field`` (or ``validation_ds.target_field``).
    The raw score is expected to be in [0, 100] and is divided by 100 so
    that the returned label is in [0, 1].

    Args:
        processor: MELTProcessor instance for audio processing.
        config: DictConfig containing data processing settings.
        is_train: Whether this is the training split.
        return_labels: Whether to include the ``labels`` key in the output.

    Raises:
        ValueError: If ``target_field`` is not set in the dataset config.
        RuntimeError: If the score cannot be cast to a float for a cut.
    """

    def __init__(
        self,
        processor: MELTProcessor,
        config: DictConfig,
        is_train: bool = True,
        return_labels: bool = True,
        return_langs: bool = False,
    ) -> None:
        super().__init__(
            processor=processor,
            config=config,
            is_train=is_train,
            return_labels=return_labels,
            return_langs=return_langs,
        )

    def _get_score(self, cut: Cut) -> float:
        """Extract and validate the quality score from a cut's custom tags.

        The field path is read per-cut from ``cut.custom["tags"]["target_field"]``
        (set via the ``tags`` block of the corresponding ``input_cfg`` entry in the
        YAML config, e.g. ``target_field: custom.score``).  The value at that path
        is then resolved with dot-notation via ``_get_nested_value``.

        Args:
            cut: Lhotse Cut object.

        Returns:
            Score in [0, 1] (raw value divided by 100).

        Raises:
            RuntimeError: If ``target_field`` is absent from the cut's tags, if the
                resolved value is missing, or if it cannot be cast to float.
        """
        # Step 1: resolve the field path from the per-cut tags.
        target_field: str | None = None
        normalize_factor: float = 1.0

        if hasattr(cut, "custom") and cut.custom:
            tags = cut.custom.get("tags", {})
            if isinstance(tags, dict):
                target_field = tags.get("target_field")
                normalize_factor = tags.get("normalize_factor", normalize_factor)

        if not target_field:
            raise RuntimeError(
                f"Cut '{cut.id}' has no 'target_field' entry in its tags. "
                "Ensure each input_cfg entry in the YAML has a tags.target_field value."
            )

        # Step 2: resolve the actual score value using dot-notation.
        raw = _get_nested_value(cut, target_field)

        if raw is None:
            raise RuntimeError(
                f"Cut '{cut.id}': could not find value at path '{target_field}'. "
                "Ensure the CutSet was prepared with this field."
            )

        try:
            score = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cut '{cut.id}': cannot convert '{target_field}' value {raw!r} to float.") from exc

        return score / normalize_factor

    def _get_translation_text(self, cut: Cut) -> str:
        """Extract the candidate translation from supervision text.

        Args:
            cut: Lhotse Cut object.

        Returns:
            Translation text from the cut supervisions.

        Raises:
            RuntimeError: If no non-empty supervision text is available.
        """
        if cut.supervisions:
            texts = [sup.text for sup in cut.supervisions if sup.text]
            if texts:
                return " ".join(texts).strip()

        raise RuntimeError(f"Empty or missing supervision text for cut {cut.id}. Cut: {cut}")

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor] | None:
        """Process a batch of cuts into model inputs with scalar QE labels.

        Audio features are packed exactly as in SpeechToTextDataset; the
        ``labels`` tensor is replaced with a 1-D float tensor of shape [B]
        containing the normalised quality scores.

        Args:
            cuts: A CutSet containing the cuts to process.

        Returns:
            Dictionary with model inputs (same keys as SpeechToTextDataset
            except ``labels`` is a float tensor of shape [B]).
            Returns None if all cuts fail to load.
        """
        if cuts is None or len(cuts) == 0:
            return None

        self._maybe_log_cut_ids(cuts)

        audios: list[torch.Tensor] = []
        scores: list[float] = []
        texts: list[str] = []
        langs: list[str] = []

        for cut in cuts:
            audio = self._load_audio(cut)
            if audio is None:
                logger.warning("Skipping cut %s: audio failed to load.", cut.id)
                continue

            try:
                score = self._get_score(cut)
            except RuntimeError as exc:
                logger.warning("Skipping cut %s: score extraction failed: %s", cut.id, exc)
                continue

            try:
                text = self._get_translation_text(cut)
            except RuntimeError as exc:
                logger.warning("Skipping cut %s: translation text missing: %s", cut.id, exc)
                continue

            _, lang = self._get_tags(cut)
            audios.append(audio)
            scores.append(score)
            texts.append(text)
            langs.append(lang)

        if not audios:
            logger.warning("All %d cuts in this batch were skipped.", len(cuts))
            return None

        try:
            audio_inputs = [[a] for a in audios]
            if self.apply_chat_template:
                formatted_texts = self._apply_qe_chat_template(texts, langs)
            else:
                if self.prompt_template:
                    # Custom template: supports {audio_token}, {t}, and {lang} placeholders.
                    formatted_texts = [
                        self.prompt_template.format(
                            audio_token=self.processor.audio_token,
                            t=t,
                            lang=self._resolve_language_name(lang),
                        )
                        for t, lang in zip(texts, langs)
                    ]
                else:
                    formatted_texts = [f"{self.processor.audio_token}{t}" for t in texts]

            inputs = self.processor(
                text=formatted_texts,
                audio=audio_inputs,
                sampling_rate=self.sample_rate,
                padding=True,
                return_tensors="pt",
            )

            if self.return_labels:
                # Shape [B, 1] to match the (batch_size, num_labels) convention
                # expected by AutoModelForSequenceClassification with num_labels=1.
                inputs["labels"] = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

            if self.return_langs:
                inputs["langs"] = langs

            return inputs

        except Exception as e:
            logger.error(f"Failed to process batch through processor: {e}")
            return None


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
    "SpeechTextQEDataset",
    "FallbackDataset",
]
