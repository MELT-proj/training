from collections import defaultdict

import jiwer
import torch
from transformers import EvalPrediction

from ..evaluation import BasicTextNormalizer
from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Number of decoded reference/hypothesis pairs stashed on the evaluator for
#: the trainer to log, when the config does not say otherwise.
DEFAULT_LOG_NUM_SAMPLES = 10


def _config_value(config, key: str, default):
    """Read *key* off an OmegaConf node, a dataclass, or a plain namespace."""
    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


class TrainingEvaluator:
    """Computes WER and CER during evaluation, with optional per-language and
    per-task breakdowns.

    Predictions arrive as *generated token ids* -- ``MELTTrainer.prediction_step``
    decodes each batch with ``generate()`` rather than taking the argmax of a
    teacher-forced forward pass, so what is scored here is what the model would
    actually emit at inference.

    Args:
        config: Evaluation configuration (``enable_whisper_normalization``,
            ``log_num_samples``).
        processor: MELTProcessor whose tokenizer is used for decoding.

    Attributes:
        last_samples: The first ``log_num_samples`` decoded pairs of the most
            recent evaluation, as ``{lang, task, reference, prediction,
            reference_raw, prediction_raw}`` dicts.  Populated on the final
            (``compute_result=True``) call and consumed by the trainer, which
            is where the global step and the W&B run live.
    """

    def __init__(self, config, processor):
        self.config = config
        self.processor = processor

        self._predictions: list[str] = []
        self._references: list[str] = []
        self._langs: list[str] = []
        self._tasks: list[str] = []

        self.log_num_samples = int(
            _config_value(config, "log_num_samples", DEFAULT_LOG_NUM_SAMPLES)
        )
        self.last_samples: list[dict[str, str]] = []

        if self.config.enable_whisper_normalization:
            self.normalizer = BasicTextNormalizer()

    def _normalize_text(self, text: str) -> str:
        return self.normalizer(text).strip()

    def _decode(self, token_ids) -> list[str]:
        """Decode a batch of token ids to strings.

        ``-100`` appears in two places and means "nothing" in both: HF's ignore
        index in the labels, and the value ``evaluation_loop`` pads with when it
        squares up ragged tensors across ranks.  The tokenizer would reject it,
        so swap it for the pad id -- which ``skip_special_tokens`` then drops.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu()
        else:
            token_ids = torch.as_tensor(token_ids)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        token_ids = torch.where(
            token_ids == -100,
            torch.full_like(token_ids, pad_token_id),
            token_ids,
        )
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)

    def __call__(self, predictions: EvalPrediction, compute_result: bool) -> dict[str, float]:
        pred_ids, label_ids = predictions

        # Access per-sample language codes if available.  Under
        # `batch_eval_metrics` this is called once per batch and `langs` holds
        # every code seen so far in this evaluation, while `predictions`/
        # `labels` hold only the current batch — so the codes for this batch
        # start where the previous call stopped, not at 0.
        langs = getattr(predictions, "langs", None)
        tasks = getattr(predictions, "tasks", None)
        offset = len(self._langs)

        batch_predictions = self._decode(pred_ids)
        batch_references = self._decode(label_ids)

        self._predictions.extend(batch_predictions)
        self._references.extend(batch_references)

        for i in range(len(batch_predictions)):
            lang = langs[offset + i] if langs is not None and offset + i < len(langs) else ""
            # A cut with no language tag reaches us as an empty string; bucket
            # it with the genuinely missing ones rather than emitting a `wer_`
            # key with nothing after the underscore.
            self._langs.append(lang or "unknown")

            task = tasks[offset + i] if tasks is not None and offset + i < len(tasks) else ""
            self._tasks.append(task or "unknown")

        if compute_result:
            raw_preds = list(self._predictions)
            raw_refs = list(self._references)
            preds = raw_preds
            refs = raw_refs
            run_langs = list(self._langs)
            run_tasks = list(self._tasks)

            if self.config.enable_whisper_normalization:
                preds = [self._normalize_text(p) for p in preds]
                refs = [self._normalize_text(r) for r in refs]

            # Stash a few decoded pairs for the trainer to log.  Both forms are
            # kept: the normaliser rewrites the text (case, punctuation), and
            # what you want to read when a run looks wrong is what the model
            # actually emitted.
            self.last_samples = [
                {
                    "lang": run_langs[i],
                    "task": run_tasks[i],
                    "reference": refs[i],
                    "prediction": preds[i],
                    "reference_raw": raw_refs[i],
                    "prediction_raw": raw_preds[i],
                }
                for i in range(min(self.log_num_samples, len(preds)))
            ]

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

            # --- Per-task metrics ---
            task_to_preds: dict[str, list[str]] = defaultdict(list)
            task_to_refs: dict[str, list[str]] = defaultdict(list)
            for pred, ref, task in zip(preds, refs, run_tasks):
                task_to_preds[task].append(pred)
                task_to_refs[task].append(ref)

            # Unlike the language split, an "unknown" task bucket is not
            # emitted as a metric: every dataset item carries a task (the
            # collator defaults to "asr"), so a missing code means the
            # plumbing broke rather than the data being untagged — and the
            # same samples already land in the per-language `unknown` bucket.
            n_unknown_task = len(task_to_preds.get("unknown", ()))
            if n_unknown_task:
                logger.warning(
                    "%d/%d evaluated samples had no task code; they are "
                    "excluded from the per-task WER/CER split.",
                    n_unknown_task,
                    len(preds),
                )

            if list(task_to_preds) != ["unknown"]:
                logger.info(
                    "Per-task eval samples: %s",
                    ", ".join(
                        f"{task}={len(task_to_preds[task])}"
                        for task in sorted(task_to_preds)
                    ),
                )
                for task in sorted(task_to_preds):
                    if task == "unknown":
                        continue
                    tp = task_to_preds[task]
                    tr = task_to_refs[task]
                    r[f"wer_{task}"] = jiwer.wer(tr, tp)
                    r[f"cer_{task}"] = jiwer.cer(tr, tp)

            # We finished, hence we clean the buffers for the next evaluation phase
            self._predictions = []
            self._references = []
            self._langs = []
            self._tasks = []

            return r
