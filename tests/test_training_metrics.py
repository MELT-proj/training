"""Tests for :class:`melt.training.metrics.TrainingEvaluator`.

The per-language and per-task breakdowns are only meaningful if each decoded
prediction is paired with the language/task code of the sample it came from.
Under ``batch_eval_metrics`` the evaluator is called once per batch while the
buffers it reads from hold every code seen so far in the evaluation, so the
pairing depends on an offset that these tests pin down.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import EvalPrediction

from melt.training.metrics import TrainingEvaluator

VOCAB_SIZE = 16


class _StubProcessor:
    """Decodes token ids to `w<id>` words — enough for jiwer to score."""

    def decode(self, tokens, skip_special_tokens=True):  # noqa: ARG002
        return " ".join(f"w{t}" for t in tokens)


def _batch(samples: list[tuple[int, bool]]):
    """Build (logits, labels) for a batch.

    Args:
        samples: one ``(token_id, correct)`` pair per sample.  ``correct=False``
            makes the argmax land on a different token, i.e. WER 1.0 for that
            sample.
    """
    logits = torch.zeros(len(samples), 1, VOCAB_SIZE)
    labels = torch.zeros(len(samples), 1, dtype=torch.long)
    for i, (token, correct) in enumerate(samples):
        labels[i, 0] = token
        predicted = token if correct else (token + 1) % VOCAB_SIZE
        logits[i, 0, predicted] = 1.0
    return logits, labels


@pytest.fixture
def evaluator():
    return TrainingEvaluator(
        config=SimpleNamespace(enable_whisper_normalization=False),
        processor=_StubProcessor(),
    )


def _feed(evaluator, batches, langs_per_batch, tasks_per_batch=None):
    """Drive the evaluator the way Trainer.evaluation_loop does.

    ``langs`` (and ``tasks``, when given) accumulate across batches — the
    trainer buffers every code it has seen — while ``predictions``/
    ``label_ids`` carry only the current batch.
    """
    seen: list[str] = []
    seen_tasks: list[str] = []
    result = None
    for i, (batch, langs) in enumerate(zip(batches, langs_per_batch)):
        seen.extend(langs)
        logits, labels = _batch(batch)
        prediction = EvalPrediction(predictions=logits, label_ids=labels)
        prediction.langs = list(seen)
        if tasks_per_batch is not None:
            seen_tasks.extend(tasks_per_batch[i])
            prediction.tasks = list(seen_tasks)
        result = evaluator(prediction, compute_result=(i == len(batches) - 1))
    return result


def test_per_language_metrics_track_the_right_samples(evaluator):
    """Codes from batch 2 must not be read with batch 1's indices.

    German is transcribed perfectly and French never is, but the languages sit
    at different positions in each batch — so an evaluator that indexed the
    cumulative buffer from 0 every call would keep re-reading batch 1's ordering
    and both languages would land somewhere in between.
    """
    batches = [
        [(1, True), (2, True), (3, False), (4, False)],
        [(5, False), (6, False), (7, True), (8, True)],
        [(9, True), (10, False), (11, True), (12, False)],
    ]
    langs_per_batch = [
        ["de", "de", "fr", "fr"],
        ["fr", "fr", "de", "de"],
        ["de", "fr", "de", "fr"],
    ]

    result = _feed(evaluator, batches, langs_per_batch)

    assert result["wer_de"] == 0.0
    assert result["wer_fr"] == 1.0
    assert result["wer"] == 0.5
    assert "wer_unknown" not in result


def test_no_unknown_bucket_when_every_sample_is_tagged(evaluator):
    """A full set of codes must produce no `unknown` metrics at all.

    This is the shape of the original defect: the buffer was shorter than the
    batch (it held one rank's codes while the logits were gathered from all of
    them), so the tail of every early batch fell through to `unknown`.
    """
    batches = [[(1, True), (2, False)], [(3, True), (4, False)]]
    langs_per_batch = [["de", "es"], ["fr", "de"]]

    result = _feed(evaluator, batches, langs_per_batch)

    assert not [k for k in result if k.endswith("_unknown")]
    assert sorted(k for k in result if k.startswith("wer_")) == [
        "wer_de",
        "wer_es",
        "wer_fr",
    ]


def test_short_buffer_falls_back_to_unknown(evaluator):
    """A buffer that runs out still scores; the gap is named, not guessed."""
    batches = [[(1, True), (2, False), (3, True), (4, False)]]
    langs_per_batch = [["de", "es"]]

    result = _feed(evaluator, batches, langs_per_batch)

    assert result["wer_unknown"] == 0.5
    assert result["wer_de"] == 0.0
    assert result["wer_es"] == 1.0


def test_empty_language_code_is_bucketed_as_unknown(evaluator):
    """Untagged cuts arrive as `''`; they must not create a bare `wer_` key."""
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["de", ""]]

    result = _feed(evaluator, batches, langs_per_batch)

    assert "wer_" not in result
    assert result["wer_unknown"] == 1.0


def test_breakdown_is_skipped_when_no_codes_are_available(evaluator):
    """With nothing to split by, `wer_unknown` would just restate `wer`."""
    batches = [[(1, True), (2, False)]]
    logits, labels = _batch(batches[0])
    result = evaluator(
        EvalPrediction(predictions=logits, label_ids=labels), compute_result=True
    )

    assert result == {"wer": 0.5, "cer": pytest.approx(result["cer"])}
    assert not [k for k in result if "unknown" in k]


def test_buffers_are_cleared_between_evaluations(evaluator):
    """A second evaluation must not inherit the first one's offset."""
    first = _feed(evaluator, [[(1, True), (2, False)]], [["de", "fr"]])
    second = _feed(evaluator, [[(3, True), (4, True)]], [["de", "fr"]])

    assert first["wer_de"] == 0.0 and first["wer_fr"] == 1.0
    assert second["wer_de"] == 0.0 and second["wer_fr"] == 0.0


def test_per_task_metrics_split_asr_from_st(evaluator):
    """`wer_asr`/`wer_st` each count only their own samples.

    The language split still reports the mixed bucket (the samples all share
    one language), so the task keys are what separates the two groups.
    """
    batches = [[(1, True), (2, False)], [(3, True), (4, False)]]
    langs_per_batch = [["de", "de"], ["de", "de"]]
    tasks_per_batch = [["asr", "st"], ["asr", "st"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert result["wer_asr"] == 0.0
    assert result["wer_st"] == 1.0
    assert result["wer"] == 0.5
    assert result["wer_de"] == 0.5


def test_task_codes_pair_with_the_right_batch(evaluator):
    """Task codes use the same cumulative offset as language codes.

    The batches interleave the tasks in opposite order, so an evaluator that
    read batch 2's codes from index 0 would swap them and report both tasks at
    0.5 instead of 0.0 / 1.0.
    """
    batches = [[(1, True), (2, False)], [(3, False), (4, True)]]
    langs_per_batch = [["de", "de"], ["de", "de"]]
    tasks_per_batch = [["asr", "st"], ["st", "asr"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert result["wer_asr"] == 0.0  # samples 1 and 4, both correct
    assert result["wer_st"] == 1.0  # samples 2 and 3, both wrong


def test_short_task_buffer_is_excluded_not_guessed(evaluator):
    """Samples whose task code is missing stay out of the per-task split.

    They must not collide with the language split's `wer_unknown` key either.
    """
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["de", "de"]]
    tasks_per_batch = [["asr"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert result["wer_asr"] == 0.0
    assert "wer_st" not in result
    assert "wer_unknown" not in result  # the language split is unaffected


def test_no_task_keys_when_no_task_codes_are_available(evaluator):
    """Old-style calls without `tasks` keep their exact previous output."""
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["de", "fr"]]

    result = _feed(evaluator, batches, langs_per_batch)

    assert not [k for k in result if k in ("wer_asr", "wer_st", "cer_asr", "cer_st")]
    assert result["wer_de"] == 0.0
    assert result["wer_fr"] == 1.0
