from collections import defaultdict

import jiwer
import torch
from transformers import EvalPrediction

from ..evaluation import BasicTextNormalizer


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

        # Access per-sample language codes if available
        langs = getattr(predictions, "langs", None)

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

            if langs is not None and i < len(langs):
                self._langs.append(langs[i])
            else:
                self._langs.append("unknown")

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
