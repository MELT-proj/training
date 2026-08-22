"""Tests for :class:`melt.training.metrics.TrainingEvaluator`.

The evaluator scores *generated token ids* against label ids: both sides are
decoded to text and handed to jiwer, so what is measured is what the model
would emit at inference.

Two properties are pinned down here.  First, the per-language and per-task
breakdowns are only meaningful if each decoded prediction is paired with the
language/task code of the sample it came from; under ``batch_eval_metrics`` the
evaluator is called once per batch while the buffers it reads from hold every
code seen so far, so the pairing depends on an offset.  Second, sequences are
built with ``seq_len > 1`` throughout: the previous fixture used ``seq_len=1``,
where a one-position misalignment between predictions and references is
invisible, and that is exactly how an off-by-one survived in the old
teacher-forced path.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import EvalPrediction

from melt.training.metrics import TrainingEvaluator

PAD_TOKEN_ID = 0
#: Added to every token of a wrong prediction, so no word of it survives.
WRONG_OFFSET = 500
#: Words per sample.  Anything above 1 exposes a prediction/reference shift.
SEQ_LEN = 3


class _StubProcessor:
    """Decodes token ids to `w<id>` words — enough for jiwer to score."""

    tokenizer = SimpleNamespace(pad_token_id=PAD_TOKEN_ID, eos_token_id=PAD_TOKEN_ID)

    def batch_decode(self, sequences, skip_special_tokens=True):  # noqa: ARG002
        return [
            " ".join(f"w{int(t)}" for t in row if int(t) != PAD_TOKEN_ID)
            for row in sequences
        ]


def _words(token: int) -> list[int]:
    """The `SEQ_LEN` distinct token ids that spell out sample *token*."""
    return [token * 10 + k for k in range(1, SEQ_LEN + 1)]


def _batch(samples: list[tuple[int, bool]]):
    """Build (prediction_ids, label_ids) for a batch.

    Args:
        samples: one ``(token_id, correct)`` pair per sample.  ``correct=False``
            shifts every word of the prediction out of range of its reference,
            i.e. WER 1.0 for that sample.
    """
    labels = torch.tensor([_words(token) for token, _ in samples], dtype=torch.long)
    predictions = torch.tensor(
        [
            _words(token) if correct else [w + WRONG_OFFSET for w in _words(token)]
            for token, correct in samples
        ],
        dtype=torch.long,
    )
    return predictions, labels


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
        predictions, labels = _batch(batch)
        prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        prediction.langs = list(seen)
        if tasks_per_batch is not None:
            seen_tasks.extend(tasks_per_batch[i])
            prediction.tasks = list(seen_tasks)
        result = evaluator(prediction, compute_result=(i == len(batches) - 1))
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_an_exact_multi_token_match_scores_zero(evaluator):
    """The regression the old `seq_len=1` fixture could not catch.

    Teacher forcing compared `argmax(logit[j])` with `label[j]` while
    `logit[j]` predicts token `j+1`, so a *perfect* model scored WER 0.4 on a
    five-token reference.  With one token per sample the shift has nothing to
    slide against and the bug is unobservable; with three it is immediate.
    """
    predictions, labels = _batch([(1, True), (2, True), (3, True)])

    result = evaluator(
        EvalPrediction(predictions=predictions, label_ids=labels),
        compute_result=True,
    )

    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_a_one_position_shift_is_not_free(evaluator):
    """Predictions offset by one position must cost real errors."""
    _, labels = _batch([(1, True), (2, True)])
    # Drop each reference's first word and pad the tail: exactly the shape of
    # the old off-by-one.
    shifted = torch.cat(
        [labels[:, 1:], torch.full((labels.shape[0], 1), PAD_TOKEN_ID)], dim=1
    )

    result = evaluator(
        EvalPrediction(predictions=shifted, label_ids=labels), compute_result=True
    )

    assert result["wer"] > 0.0


def test_ignore_index_is_decoded_as_nothing(evaluator):
    """`-100` reaches the evaluator from two directions and means "nothing".

    HF writes it into the labels for masked positions, and `evaluation_loop`
    pads ragged tensors with it before gathering.  Feeding it to the tokenizer
    raises, so it has to be mapped to the pad id first.
    """
    predictions, labels = _batch([(1, True), (2, True)])
    padded_labels = torch.cat(
        [labels, torch.full((labels.shape[0], 2), -100)], dim=1
    )
    padded_predictions = torch.cat(
        [predictions, torch.full((predictions.shape[0], 2), -100)], dim=1
    )

    result = evaluator(
        EvalPrediction(predictions=padded_predictions, label_ids=padded_labels),
        compute_result=True,
    )

    assert result["wer"] == 0.0


def test_numpy_predictions_are_accepted(evaluator):
    """Without `batch_eval_metrics` the loop hands over numpy, not tensors."""
    predictions, labels = _batch([(1, True), (2, False)])

    result = evaluator(
        EvalPrediction(
            predictions=predictions.numpy(), label_ids=labels.numpy()
        ),
        compute_result=True,
    )

    assert result["wer"] == 0.5


# ---------------------------------------------------------------------------
# Sample logging
# ---------------------------------------------------------------------------


def test_sample_texts_are_stashed_for_the_trainer(evaluator):
    """`last_samples` is what the trainer prints and sends to W&B."""
    evaluator.log_num_samples = 2
    batches = [[(1, True), (2, False), (3, True)]]

    _feed(evaluator, batches, [["de", "fr", "it"]], [["asr", "st", "asr"]])

    assert len(evaluator.last_samples) == 2
    first = evaluator.last_samples[0]
    assert first["lang"] == "de" and first["task"] == "asr"
    assert first["reference"] == first["prediction"] == "w11 w12 w13"
    # The raw text is kept alongside the normalised one; with normalisation off
    # they coincide, but both fields must be present for the trainer to read.
    assert first["reference_raw"] == first["reference"]
    second = evaluator.last_samples[1]
    assert second["reference"] != second["prediction"]


def test_sample_logging_can_be_switched_off(evaluator):
    evaluator.log_num_samples = 0

    _feed(evaluator, [[(1, True), (2, False)]], [["de", "fr"]])

    assert evaluator.last_samples == []


def test_log_num_samples_is_read_from_the_config():
    evaluator = TrainingEvaluator(
        config=SimpleNamespace(
            enable_whisper_normalization=False, log_num_samples=3
        ),
        processor=_StubProcessor(),
    )

    assert evaluator.log_num_samples == 3


# ---------------------------------------------------------------------------
# Per-language / per-task breakdowns
# ---------------------------------------------------------------------------


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
    batch (it held one rank's codes while the predictions were gathered from all
    of them), so the tail of every early batch fell through to `unknown`.
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
    predictions, labels = _batch(batches[0])
    result = evaluator(
        EvalPrediction(predictions=predictions, label_ids=labels),
        compute_result=True,
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

    All samples share one language, so the language split has a single
    bucket and would only restate `wer` under a `wer_de` key -- exactly the
    redundancy a named, homogeneous eval set (e.g. `asr_es`) would produce.
    It must be skipped; the task keys are what separates the two groups.
    """
    batches = [[(1, True), (2, False)], [(3, True), (4, False)]]
    langs_per_batch = [["de", "de"], ["de", "de"]]
    tasks_per_batch = [["asr", "st"], ["asr", "st"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert result["wer_asr"] == 0.0
    assert result["wer_st"] == 1.0
    assert result["wer"] == 0.5
    assert "wer_de" not in result


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


def test_single_language_breakdown_is_skipped_as_redundant(evaluator):
    """A homogeneous eval set gets only `wer`/`cer`, nothing per-bucket.

    This is the shape a campaign config's named validation source produces
    (e.g. `asr_es`, all samples lang `es` and task `asr`): every cut shares
    the same language and task, so `wer_es`/`wer_asr` would just restate
    `wer` under a different key. W&B should see one series, not four.
    """
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["es", "es"]]
    tasks_per_batch = [["asr", "asr"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert set(result) == {"wer", "cer"}


def test_two_languages_one_task_gets_language_split_not_task_split(evaluator):
    """Only the split that actually distinguishes samples is reported.

    Two languages but a single shared task is the mirror image of
    `test_per_task_metrics_split_asr_from_st`: the language keys carry
    information the task key would not.
    """
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["de", "fr"]]
    tasks_per_batch = [["asr", "asr"]]

    result = _feed(evaluator, batches, langs_per_batch, tasks_per_batch)

    assert result["wer_de"] == 0.0
    assert result["wer_fr"] == 1.0
    assert "wer_asr" not in result


def test_single_language_plus_unknown_still_emits_both(evaluator):
    """An `unknown` bucket alongside one real language is not collapsed away.

    Two buckets is two buckets even when one of them is `unknown`: that
    bucket is a misalignment signal, not padding, so silencing it here would
    hide the exact kind of plumbing break the `unknown` handling exists to
    surface.
    """
    batches = [[(1, True), (2, False)]]
    langs_per_batch = [["es", ""]]

    result = _feed(evaluator, batches, langs_per_batch)

    assert result["wer_es"] == 0.0
    assert result["wer_unknown"] == 1.0


# ---------------------------------------------------------------------------
# Chat-template scaffolding in the reference
# ---------------------------------------------------------------------------

#: What the stub's chat template puts between the prompt and the answer, and
#: the ids it tokenises to — standing in for
#: ``<|im_start|>assistant\n<think>\n\n</think>\n\n``.
SCAFFOLD_TEXT = "<assistant-header><think></think>"
SCAFFOLD_IDS = [901, 902]


class _ChatTokenizer:
    """Minimal tokenizer with a chat template shaped like Qwen3's.

    ``add_generation_prompt=True`` appends the assistant header and an empty
    reasoning block: the scaffolding the generation prompt already covers, so
    the model never emits it and the reference must not carry it either.
    """

    pad_token_id = PAD_TOKEN_ID
    eos_token_id = PAD_TOKEN_ID

    def apply_chat_template(
        self,
        messages,  # noqa: ARG002
        tokenize=False,  # noqa: ARG002
        add_generation_prompt=False,
        enable_thinking=False,  # noqa: ARG002
    ):
        return "<user>" + (SCAFFOLD_TEXT if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        assert text == SCAFFOLD_TEXT
        return {"input_ids": list(SCAFFOLD_IDS)}


class _ChatStubProcessor(_StubProcessor):
    tokenizer = _ChatTokenizer()


def _chat_evaluator():
    return TrainingEvaluator(
        config=SimpleNamespace(enable_whisper_normalization=False),
        processor=_ChatStubProcessor(),
    )


def _score(evaluator, labels, predictions):
    prediction = EvalPrediction(
        predictions=torch.tensor(predictions, dtype=torch.long),
        label_ids=torch.tensor(labels, dtype=torch.long),
    )
    return evaluator(prediction, compute_result=True)


def test_reference_drops_the_chat_scaffolding():
    """A hypothesis that matches the target exactly must score WER 0.

    The label span keeps the assistant header and empty reasoning block
    (`mask_non_assistant_tokens` is inclusive), while `generate()` returns only
    what follows the generation prompt.  Left in, those tokens are counted as
    deletions the model could not have avoided — here 2 of 5 reference words,
    i.e. WER 0.4 for a perfect transcription.
    """
    result = _score(
        _chat_evaluator(),
        labels=[SCAFFOLD_IDS + [11, 12, 13]],
        predictions=[[11, 12, 13, PAD_TOKEN_ID, PAD_TOKEN_ID]],
    )

    assert result["wer"] == 0.0


def test_scaffolding_is_stripped_after_the_masked_prompt_region():
    """Labels open with a run of -100 over the masked prompt.

    The scaffolding starts where the kept span starts, not at index 0.
    """
    result = _score(
        _chat_evaluator(),
        labels=[[-100, -100] + SCAFFOLD_IDS + [11, 12, 13]],
        predictions=[[11, 12, 13, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]],
    )

    assert result["wer"] == 0.0


def test_a_reference_without_the_scaffolding_is_left_alone():
    """Only an exact leading match is stripped.

    A target that genuinely begins with other tokens must survive intact, or
    the strip would eat real words.
    """
    result = _score(
        _chat_evaluator(),
        labels=[[11, 12, 13]],
        predictions=[[11, 12, 13]],
    )
    assert result["wer"] == 0.0

    # And the words are really still there: a wrong hypothesis still scores 1.0
    # rather than being compared against an emptied reference.
    result = _score(
        _chat_evaluator(),
        labels=[[11, 12, 13]],
        predictions=[[21, 22, 23]],
    )
    assert result["wer"] == 1.0


def test_a_tokenizer_without_a_chat_template_is_a_no_op():
    """Runs with `apply_chat_template: false` never had the scaffolding."""
    evaluator = TrainingEvaluator(
        config=SimpleNamespace(enable_whisper_normalization=False),
        processor=_StubProcessor(),
    )

    result = _score(
        evaluator,
        labels=[[11, 12, 13]],
        predictions=[[11, 12, 13]],
    )

    assert result["wer"] == 0.0
