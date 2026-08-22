"""
Data-collator for MELT map-style datasets.

Takes a list of raw item dicts (each with a numpy audio array and text
string) and produces a model-ready batch through :class:`MELTProcessor`.
"""

import torch

from .....logging_utils import get_logger
from .....modeling import MELTProcessor
from ....data.chat_templates import (
    ChatTemplateConfig,
    get_chat_template_config,
    validate_chat_template_config,
)
from .helpers import (
    LANGUAGE_ISO_TO_NAME,
    _get_config_value,
    _normalize_prompt_template,
    _resolve_language_name_safe,
    apply_chat_template_to_texts,
    mask_non_assistant_tokens,
    resolve_custom_template,
)

logger = get_logger(__name__)


class MELTDataCollator:
    """Collates individual items from :class:`MELTMapDataset` into a batch.

    Args:
        processor: :class:`MELTProcessor` for audio featurisation and tokenisation.
        config: DictConfig with data processing settings.
        is_train: Whether this is for training (affects chat-template behaviour).
    """

    def __init__(
        self,
        processor: MELTProcessor,
        config,
        is_train: bool = False,
    ) -> None:
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.apply_chat_template = bool(
            _get_config_value(config, "apply_chat_template", False)
        )
        self.sample_rate = int(_get_config_value(config, "sample_rate", 16000))

        # Template selection strategy when apply_chat_template is True.
        raw_prompt_template = _get_config_value(config, "prompt_template", None)
        self.prompt_template = _normalize_prompt_template(raw_prompt_template)

        self.prompt_template_selection = str(
            _get_config_value(config, "prompt_template_selection", "random")
        )
        if self.prompt_template_selection not in ("random", "with_language", "custom"):
            raise ValueError(
                f"Invalid prompt_template_selection '{self.prompt_template_selection}'. "
                "Must be one of: 'random', 'with_language', 'custom'."
            )

        # Pre-compute boundary token IDs for chat-template label masking
        if self.apply_chat_template:
            ct_name = str(
                _get_config_value(config, "chat_template_config", "chatml")
            )
            ct_cfg: ChatTemplateConfig = get_chat_template_config(ct_name)
            # Validated here as well as on the training path, because the two
            # can disagree: resolve_eval_data_config lets `validation_ds` win
            # over the parent `data` block, so `validation_ds.chat_template_config`
            # may name a different template than training uses -- and the
            # training-side check would pass while eval's boundaries matched
            # nothing. Masking would then blank every label, references would
            # decode to the empty string, and the reported WER would be noise.
            validate_chat_template_config(processor.tokenizer, ct_cfg, ct_name)
            self._assistant_start_ids: list[int] = processor.tokenizer.encode(
                ct_cfg.assistant_start, add_special_tokens=False
            )
            self._assistant_end_ids: list[int] = processor.tokenizer.encode(
                ct_cfg.assistant_end, add_special_tokens=False
            )

    def __call__(self, items: list[dict]) -> dict:
        # 1. Filter out invalid sentinels
        valid = [it for it in items if not it.get("__invalid__", False)]
        if not valid:
            raise RuntimeError(
                "MELTDataCollator: all items in the batch were invalid."
            )

        # 2. Gather per-item fields
        audios: list = [it["audio"] for it in valid]
        texts: list[str] = [it["text"] for it in valid]
        tasks: list[str] = [it.get("task", "asr") for it in valid]
        langs: list[str] = [it.get("lang", "") for it in valid]
        src_langs: list[str] = [it.get("src_lang", "") for it in valid]
        tgt_langs: list[str] = [it.get("tgt_lang", "") for it in valid]

        # 3. Format texts.  Evaluation additionally needs the *prompt* alone --
        # everything up to where the target transcript starts -- because
        # `MELTTrainer.prediction_step` decodes it with `generate()` and would
        # otherwise be handed the ground truth it is meant to predict.
        want_prompts = not self.is_train
        prompts: list[str] | None = None

        if self.apply_chat_template:
            formatted = apply_chat_template_to_texts(
                texts,
                tasks,
                langs,
                tokenizer=self.processor.tokenizer,
                audio_token=self.processor.audio_token,
                prompt_template=self.prompt_template,
                prompt_template_selection=self.prompt_template_selection,
                src_langs=src_langs,
                tgt_langs=tgt_langs,
                return_prompts=want_prompts,
            )
            if want_prompts:
                formatted, prompts = formatted
        else:
            if self.prompt_template:
                # Custom template: supports {audio_token}, {t}, {lang},
                # {src_lang}, and {tgt_lang}.
                # May be a single string or a dict mapping task → template.
                formatted = []
                prompts = [] if want_prompts else None
                for t, lang, task, src_lang, tgt_lang in zip(
                    texts, langs, tasks, src_langs, tgt_langs
                ):
                    template = (
                        resolve_custom_template(self.prompt_template, task)
                        if isinstance(self.prompt_template, dict)
                        else self.prompt_template
                    )
                    lang_key = (lang or "").lower()
                    language_name = LANGUAGE_ISO_TO_NAME.get(lang_key, "")
                    fmt_kwargs = dict(
                        audio_token=self.processor.audio_token,
                        t=t,
                        lang=language_name,
                        src_lang=_resolve_language_name_safe(src_lang),
                        tgt_lang=_resolve_language_name_safe(tgt_lang),
                    )
                    formatted.append(template.format(**fmt_kwargs))
                    if want_prompts:
                        # Substituting an empty transcript leaves the template's
                        # own lead-in intact -- the same trick
                        # SpeechToTextDataset uses to locate the prompt/target
                        # boundary for label masking (dataset.py).
                        prompts.append(template.format(**{**fmt_kwargs, "t": ""}))
            else:
                formatted = [f"{self.processor.audio_token}{t}" for t in texts]
                if want_prompts:
                    prompts = [self.processor.audio_token] * len(texts)

        # 4. Batch-process through MELTProcessor
        audio_inputs = [[a] for a in audios]  # list of lists
        batch = self.processor(
            text=formatted,
            audio=audio_inputs,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        # 5. Build labels
        labels = batch["input_ids"].clone()
        if self.apply_chat_template:
            labels = mask_non_assistant_tokens(
                labels, self._assistant_start_ids, self._assistant_end_ids
            )
        else:
            mask = (
                (labels == self.processor.audio_token_id)
                | (labels == self.processor.audio_bos_token_id)
                | (labels == self.processor.audio_eos_token_id)
                | (labels == self.processor.tokenizer.pad_token_id)
                | (labels == self.processor.tokenizer.bos_token_id)
            )
            labels[mask] = -100
        batch["labels"] = labels

        # 6. Attach the generation prompt (evaluation only).
        #
        # Tokenised separately from `formatted` but with the same audio bos/eos
        # wrapping the processor applies internally, and left-padded, which is
        # what batched generation needs so that every sequence's last real token
        # sits at the same index.  The audio features are *not* recomputed: the
        # prompt reuses the ones already in `batch`.
        if prompts is not None:
            prompt_inputs = self.processor.tokenizer(
                [self.processor._surround_bos_eos_mm_tokens(p) for p in prompts],
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            batch["prompt_input_ids"] = prompt_inputs["input_ids"]
            batch["prompt_attention_mask"] = prompt_inputs["attention_mask"]

        # 7. Attach langs/tasks for per-language and per-task WER/CER
        batch["langs"] = langs
        batch["tasks"] = tasks

        return batch


__all__ = ["MELTDataCollator"]
