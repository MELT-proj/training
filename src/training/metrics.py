from transformers import EvalPrediction
from ..evaluation import BasicTextNormalizer
import jiwer
import torch


def pull_final_logits(logits: torch.Tensor, labels: torch.Tensor):
    """Pull the final logits corresponding to the label sequence length."""
    return logits[:, -labels.shape[1]:, :]

class TrainingEvaluator:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor

        self._predictions = []
        self._references = []

        if self.config.enable_whisper_normalization:
            self.normalizer = BasicTextNormalizer()

    def _normalize_text(self, text: str) -> str:
        return self.normalizer(text).strip()

    def __call__(self, predictions: list[EvalPrediction], compute_result: bool) -> dict[str, float]:
        logits, label_ids = predictions

        # Logits and label_ids contain n_batch elements of shape (seq_len, vocab_size) and (seq_len,) respectively
        # 1. iterate over each row and extract the predicted tokens and reference token only when the label is not -100 (the default ignore index for labels in Hugging Face)
        # 2. decode using the processor's tokenizer to get the predicted and reference transcripts
        for logit, label_id in zip(logits, label_ids):
            ref_tokens = label_id.cpu().tolist()
            pred_tokens = logit.argmax(dim=-1).cpu().tolist()

            # Filter out tokens where the label is -100
            pred_tokens = [p for p, l in zip(pred_tokens, ref_tokens) if l != -100]
            ref_tokens = [l for l in ref_tokens if l != -100]

            self._predictions.append(self.processor.decode(pred_tokens, skip_special_tokens=True))
            self._references.append(self.processor.decode(ref_tokens, skip_special_tokens=True))

        if compute_result:

            if self.config.enable_whisper_normalization:
                # TODO: technically, there another normalizer for English but we are not using it for now
                self._predictions = [self._normalize_text(pred) for pred in self._predictions]
                self._references = [self._normalize_text(ref) for ref in self._references]

            r = {
                "wer": jiwer.wer(self._references, self._predictions),
                "cer": jiwer.cer(self._references, self._predictions)
            }
            # We finished, hence we clean the buffers for the next evaluation phase
            self._predictions = []
            self._references = []
            return r