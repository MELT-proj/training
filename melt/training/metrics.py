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

    A breakdown is only worth reporting when it distinguishes something the
    overall `wer`/`cer` does not -- i.e. when the evaluated samples span more
    than one language (or task) bucket, counting `unknown` as a bucket of its
    own. A campaign config that names a validation source (e.g. `asr_es`)
    hands the evaluator a homogeneous set -- one language, one task -- so its
    per-language and per-task splits would just restate `wer`/`cer` under a
    different key; the legacy unnamed path, which mixes every language and
    task into one set, is unaffected since it always has more than one
    bucket. See `__call__` for where this is decided.

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
        self._scaffold_ids: list[int] | None = None
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

    def _assistant_scaffold_ids(self) -> list[int]:
        """Token ids the chat template inserts between the prompt and the answer.

        With `apply_chat_template`, references and hypotheses do not start at
        the same place.  The generation prompt is built with
        `add_generation_prompt=True`, so it already contains the assistant
        header -- and, for Qwen3, the empty reasoning block it emits even under
        `enable_thinking=False`.  `generate()` returns only what comes after,
        so the hypothesis is the bare answer.  The reference, though, is the
        label span, and `mask_non_assistant_tokens` keeps its boundaries
        *inclusive*, so it still carries that scaffolding: every REF read
        `assistant <think> </think> <the actual target>` while no HYP could.
        Scoring them against each other charges the model insertions it had no
        opportunity to produce -- on a ten-word target those three tokens alone
        move WER by tens of points.

        Derived from the tokenizer rather than from a ChatTemplateConfig so it
        stays correct for any template: the difference between templating the
        user turn with and without a generation prompt *is* the scaffolding,
        whatever it happens to contain.

        Returns an empty list when the tokenizer has no chat template, which is
        also the right answer for runs with `apply_chat_template: false` --
        their references never carried the scaffolding in the first place.
        """
        if self._scaffold_ids is not None:
            return self._scaffold_ids

        tokenizer = self.processor.tokenizer
        probe = [{"role": "user", "content": ""}]
        try:
            without = tokenizer.apply_chat_template(
                probe,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            with_prompt = tokenizer.apply_chat_template(
                probe,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:  # no chat template, or one that rejects the probe
            self._scaffold_ids = []
            return self._scaffold_ids

        if not with_prompt.startswith(without):
            # The derivation assumes the generation prompt is the non-generation
            # form plus a suffix.  Every template checked holds to that (Qwen3,
            # Qwen3.5, Llama 3.x, EuroLLM), but a template that rewrote earlier
            # turns would break it -- and the failure is invisible in the
            # metrics, which would just quietly go back to scoring the
            # scaffolding as part of the target.  Say so.
            logger.warning(
                "Could not derive the assistant scaffolding from the chat "
                "template: the generation prompt does not extend the plain "
                "form. Evaluation references may still contain the assistant "
                "header, which inflates WER/CER."
            )
            self._scaffold_ids = []
            return self._scaffold_ids

        scaffold = with_prompt[len(without):]
        self._scaffold_ids = (
            tokenizer(scaffold, add_special_tokens=False)["input_ids"]
            if scaffold
            else []
        )
        return self._scaffold_ids

    def _strip_assistant_scaffold(self, token_ids):
        """Blank the chat scaffolding leading each reference row.

        Rewrites it to ``-100`` rather than slicing so the tensor keeps its
        shape and `_decode` -- which already maps ``-100`` to the pad id and
        lets `skip_special_tokens` drop it -- needs no special case.

        Only strips at the start of the *kept* span (labels open with a run of
        ``-100`` over the masked prompt), and only when the ids match exactly,
        so a reference that genuinely begins some other way is left alone.
        """
        scaffold = self._assistant_scaffold_ids()
        if not scaffold:
            return token_ids

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().clone()
        else:
            token_ids = torch.as_tensor(token_ids).clone()

        width = len(scaffold)
        flat = token_ids.reshape(-1, token_ids.shape[-1]) if token_ids.dim() > 1 else token_ids.reshape(1, -1)
        scaffold_tensor = torch.tensor(scaffold, dtype=flat.dtype)
        for row in flat:
            kept = (row != -100).nonzero()
            if kept.numel() == 0:
                continue
            start = int(kept[0])
            if start + width > row.shape[0]:
                continue
            if torch.equal(row[start : start + width], scaffold_tensor):
                row[start : start + width] = -100
        return token_ids

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
        batch_references = self._decode(self._strip_assistant_scaffold(label_ids))

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

            # Log the split's sizes whenever at least one sample carried a
            # language code, even if the breakdown below ends up skipped: for
            # a named eval set this count is the check against the manifest
            # that the set holds the number of samples it is supposed to.
            if list(lang_to_preds) != ["unknown"]:
                logger.info(
                    "Per-language eval samples: %s",
                    ", ".join(
                        f"{lang}={len(lang_to_preds[lang])}"
                        for lang in sorted(lang_to_preds)
                    ),
                )

            # A single bucket -- one language, or nothing but `unknown` --
            # means the split can't tell you anything the overall wer/cer
            # doesn't, so skip emitting it.
            if len(lang_to_preds) > 1:
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

            # Same rule as the language split: a single task bucket (or an
            # `unknown`-only one) restates the overall metric, so skip it.
            # `{"asr", "unknown"}` is still two buckets and does get a split,
            # even though the `unknown` half is never itself emitted below.
            if len(task_to_preds) > 1:
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
