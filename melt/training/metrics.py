from collections import defaultdict

import jiwer
import torch
from transformers import EvalPrediction

from ..evaluation import BasicTextNormalizer
from ..logging_utils import get_logger

logger = get_logger(__name__)


def pull_final_logits(logits: torch.Tensor, labels: torch.Tensor):
    """Pull the final logits corresponding to the label sequence length."""
    return logits[:, -labels.shape[1]:, :]


class TrainingEvaluator:
    """Computes WER and CER during evaluation, with optional per-language breakdown.

    Args:
        config: Evaluation configuration (enable_whisper_normalization, etc.).
        processor: MELTProcessor whose tokenizer is used for decoding.
    """

    def __init__(self, config, processor):
        self.config = config
        self.processor = processor

        self._predictions: list[str] = []
        self._references: list[str] = []
        self._langs: list[str] = []

        if self.config.enable_whisper_normalization:
            self.normalizer = BasicTextNormalizer()

    def _normalize_text(self, text: str) -> str:
        return self.normalizer(text).strip()

    def __call__(self, predictions: EvalPrediction, compute_result: bool) -> dict[str, float]:
        logits, labels = predictions

        # Access per-sample language codes if available.  Under
        # `batch_eval_metrics` this is called once per batch and `langs` holds
        # every code seen so far in this evaluation, while `logits`/`labels`
        # hold only the current batch — so the codes for this batch start where
        # the previous call stopped, not at 0.
        langs = getattr(predictions, "langs", None)
        offset = len(self._langs)

        # Logits and label_ids contain n_batch elements of shape
        # (seq_len, vocab_size) and (seq_len,) respectively.
        # 1. iterate over each row and extract predicted / reference tokens
        #    only when the label is not -100 (HF ignore index).
        # 2. decode using the processor's tokenizer.
        for i, (logit, label_id) in enumerate(zip(logits, labels)):
            ref_tokens = label_id.cpu().tolist()
            pred_tokens = logit.argmax(dim=-1).cpu().tolist()

            # Filter out tokens where the label is -100
            pred_tokens = [p for p, l in zip(pred_tokens, ref_tokens) if l != -100]
            ref_tokens = [l for l in ref_tokens if l != -100]

            self._predictions.append(self.processor.decode(pred_tokens, skip_special_tokens=True))
            self._references.append(self.processor.decode(ref_tokens, skip_special_tokens=True))

            lang = langs[offset + i] if langs is not None and offset + i < len(langs) else ""
            # A cut with no language tag reaches us as an empty string; bucket
            # it with the genuinely missing ones rather than emitting a `wer_`
            # key with nothing after the underscore.
            self._langs.append(lang or "unknown")

        if compute_result:
            preds = list(self._predictions)
            refs = list(self._references)
            run_langs = list(self._langs)

            if self.config.enable_whisper_normalization:
                preds = [self._normalize_text(p) for p in preds]
                refs = [self._normalize_text(r) for r in refs]

            # --- Overall metrics ---
            r: dict[str, float] = {
                "wer": jiwer.wer(refs, preds),
                "cer": jiwer.cer(refs, preds),
            }

            # --- Per-language metrics ---
            lang_to_preds: dict[str, list[str]] = defaultdict(list)
            lang_to_refs: dict[str, list[str]] = defaultdict(list)
            for pred, ref, lang in zip(preds, refs, run_langs):
                lang_to_preds[lang].append(pred)
                lang_to_refs[lang].append(ref)

            # An "unknown" bucket means predictions arrived without a language
            # code to pair them with, which makes the whole breakdown suspect —
            # say so rather than reporting a `_unknown` metric and leaving the
            # reader to work out where it came from.
            n_unknown = len(lang_to_preds.get("unknown", ()))
            if n_unknown:
                logger.warning(
                    "%d/%d evaluated samples had no language code; their WER/CER "
                    "is reported under `unknown` and the per-language split may "
                    "be misaligned.",
                    n_unknown,
                    len(preds),
                )

            # With no codes at all the single bucket just restates the overall
            # numbers, so skip it.
            if list(lang_to_preds) != ["unknown"]:
                # Log the split's sizes, not just its scores: they are what you
                # check against the manifest to know the breakdown is counting
                # the samples you think it is.
                logger.info(
                    "Per-language eval samples: %s",
                    ", ".join(
                        f"{lang}={len(lang_to_preds[lang])}"
                        for lang in sorted(lang_to_preds)
                    ),
                )
                for lang in sorted(lang_to_preds):
                    lp = lang_to_preds[lang]
                    lr = lang_to_refs[lang]
                    r[f"wer_{lang}"] = jiwer.wer(lr, lp)
                    r[f"cer_{lang}"] = jiwer.cer(lr, lp)

            # We finished, hence we clean the buffers for the next evaluation phase
            self._predictions = []
            self._references = []
            self._langs = []

            return r
