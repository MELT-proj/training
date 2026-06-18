"""Tests verifying that only {t} (target text) tokens contribute to the loss
when using a custom ``prompt_template`` in non-chat-template mode.

These tests use the exact configuration from ``runs/debug_VP-only.yaml``:

    apply_chat_template: false
    prompt_template: "{audio_token}{lang}: {t}"

and verify that prompt-token positions are set to -100 in ``labels`` while
target-text positions retain their token IDs.
"""

import io
import wave

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# All tests in this module require lhotse.
pytest.importorskip("lhotse")

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKENIZER_NAME = "Qwen/Qwen3-1.7B"


def _build_processor():
    """Build a minimal MELTProcessor around the Qwen3-1.7B tokenizer.

    The tokenizer is augmented with the MELT-required special tokens so that
    the processor validates successfully.  A dummy feature extractor is wired
    in so that neither real audio nor model downloads are needed (only the
    tokenizer is loaded from HuggingFace).
    """
    from transformers import AutoTokenizer
    from transformers.feature_extraction_utils import FeatureExtractionMixin

    from melt.modeling.processing_melt import MELTProcessor

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

    # Add MELT-required special tokens.
    specials = {
        "audio_token": "<|audio|>",
        "audio_bos_token": "<|audio_bos|>",
        "audio_eos_token": "<|audio_eos|>",
    }
    tokenizer.add_special_tokens({"additional_special_tokens": list(specials.values())})
    for attr, token_str in specials.items():
        setattr(tokenizer, attr, token_str)

    class _DummyFeatureExtractor(FeatureExtractionMixin):
        """Returns fixed-size random features — no real audio processing."""

        def __call__(self, audio, sampling_rate=16000, return_attention_mask=True,
                     pad_to_multiple_of=8, **kwargs):
            if isinstance(audio, list):
                batch_size = len(audio)
            else:
                batch_size = 1
            seq_len = 80  # arbitrary > 0
            return {
                "input_features": torch.randn(batch_size, seq_len, 80),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            }

    return MELTProcessor(feature_extractor=_DummyFeatureExtractor(), tokenizer=tokenizer)


def _make_wav_bytes(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate mono 16-bit PCM WAV bytes for *duration* seconds of silence."""
    num_samples = int(duration * sample_rate)
    # Low-amplitude noise so the audio isn't pure silence.
    samples = (np.random.randn(num_samples) * 50).clip(-32767, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_cut(cut_id: str, text: str, lang: str, task: str = "asr") -> MonoCut:
    """Create a minimal Lhotse MonoCut with *text* supervision and language tag."""
    duration = 1.0
    sr = 16000
    wav_bytes = _make_wav_bytes(duration, sr)

    recording = Recording.from_bytes(wav_bytes, recording_id=cut_id)

    return MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=[
            SupervisionSegment(
                id=f"{cut_id}-sup",
                recording_id=cut_id,
                start=0.0,
                duration=duration,
                text=text,
                language=lang,
            )
        ],
        custom={"tags": {"task": task, "lang": lang}},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPromptTemplateLabels:
    """Verify label masking when ``apply_chat_template=False`` and a custom
    ``prompt_template`` is provided."""

    @pytest.fixture(scope="class")
    def processor(self):
        return _build_processor()

    @pytest.fixture(scope="class")
    def dataset(self, processor):
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": False,
            "prompt_template": "{audio_token}{lang}: {t}",
            "sample_rate": 16000,
        })
        return SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

    # -- single cut ----------------------------------------------------------

    def test_only_target_tokens_survive(self, dataset, processor):
        """A single English cut — only "hello world" tokens should remain."""
        cut = _make_cut("en-1", "hello world", "en")
        cuts = CutSet.from_cuts([cut])

        batch = dataset[cuts]
        assert batch is not None, "Batch should not be None"
        labels = batch["labels"]  # [1, seq_len]

        self._assert_labels_match_target(
            labels, processor.tokenizer, target_text="hello world",
        )

    def test_german_target_only(self, dataset, processor):
        """German cut — only the German transcription tokens survive."""
        cut = _make_cut("de-1", "guten tag", "de")
        cuts = CutSet.from_cuts([cut])
        batch = dataset[cuts]
        assert batch is not None
        labels = batch["labels"]

        self._assert_labels_match_target(
            labels, processor.tokenizer, target_text="guten tag",
        )

    # -- batch of mixed-language cuts ---------------------------------------

    def test_batch_mixed_languages(self, dataset, processor):
        """Each cut in a batch should only have its own target tokens unmasked."""
        cut_en = _make_cut("en-1", "hello world", "en")
        cut_de = _make_cut("de-1", "guten tag", "de")
        cuts = CutSet.from_cuts([cut_en, cut_de])

        batch = dataset[cuts]
        assert batch is not None
        labels = batch["labels"]  # [2, seq_len]

        self._assert_labels_match_target(
            labels[0:1], processor.tokenizer, target_text="hello world",
        )
        self._assert_labels_match_target(
            labels[1:2], processor.tokenizer, target_text="guten tag",
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _assert_labels_match_target(
        labels: torch.Tensor,
        tokenizer,
        target_text: str,
    ):
        """Assert that *labels* only contain valid token IDs at positions that
        decode to *target_text*, with everything else set to -100."""
        for i in range(labels.size(0)):
            row = labels[i]
            # Gather positions that are NOT masked.
            active_mask = row != -100
            active_ids = row[active_mask].tolist()

            assert len(active_ids) > 0, (
                f"Sample {i}: all labels are -100 — nothing contributes to loss"
            )

            decoded = tokenizer.decode(active_ids, skip_special_tokens=True).strip()
            # Normalise whitespace for comparison.
            expected = target_text.strip()
            actual = " ".join(decoded.split())

            assert actual == expected, (
                f"Sample {i}: expected target {expected!r}, got {actual!r}\n"
                f"Full label row: {row.tolist()}"
            )

            # Sanity: the active count should approximately match the target
            # token count (allow off-by-two for subword variance and EOS).
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            assert abs(len(active_ids) - len(target_ids)) <= 3, (
                f"Sample {i}: active token count {len(active_ids)} deviates too much "
                f"from expected {len(target_ids)} for target {target_text!r}"
            )


# ---------------------------------------------------------------------------
# Per-task prompt_template tests (non-chat-template mode)
# ---------------------------------------------------------------------------


class TestPerTaskPromptTemplateLabels:
    """Verify per-task custom ``prompt_template`` dict in non-chat-template mode."""

    @pytest.fixture(scope="class")
    def processor(self):
        return _build_processor()

    @pytest.fixture(scope="class")
    def per_task_dataset(self, processor):
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": False,
            "prompt_template": {
                "asr": "{audio_token}ASR: {t}",
                "st": "{audio_token}ST: {t}",
            },
            "sample_rate": 16000,
        })
        return SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

    def test_asr_cut_uses_asr_template(self, per_task_dataset, processor):
        """An ASR cut should use the 'asr' template from the dict."""
        cut = _make_cut("en-1", "hello world", "en", task="asr")
        cuts = CutSet.from_cuts([cut])
        batch = per_task_dataset[cuts]
        assert batch is not None

        # Decode the full input to verify the prompt prefix is ASR-specific.
        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "ASR" in decoded, f"Expected ASR prompt in decoded text, got: {decoded!r}"
        assert "ST" not in decoded, f"ST prompt should not appear for ASR task"

        # Verify the target text is present in the unmasked labels.
        labels = batch["labels"]
        self._assert_target_in_labels(labels, processor.tokenizer, target_text="hello world")

    def test_st_cut_uses_st_template(self, per_task_dataset, processor):
        """An ST cut should use the 'st' template from the dict."""
        cut = _make_cut("en-1", "bonjour le monde", "en", task="st")
        cuts = CutSet.from_cuts([cut])
        batch = per_task_dataset[cuts]
        assert batch is not None

        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "ST" in decoded, f"Expected ST prompt in decoded text, got: {decoded!r}"
        assert "ASR" not in decoded, f"ASR prompt should not appear for ST task"

        labels = batch["labels"]
        self._assert_target_in_labels(labels, processor.tokenizer, target_text="bonjour le monde")

    def test_mixed_asr_st_batch(self, per_task_dataset, processor):
        """A mixed ASR+ST batch should use the correct template per sample."""
        cut_asr = _make_cut("en-1", "hello world", "en", task="asr")
        cut_st = _make_cut("fr-1", "bonjour le monde", "fr", task="st")
        cuts = CutSet.from_cuts([cut_asr, cut_st])
        batch = per_task_dataset[cuts]
        assert batch is not None
        assert batch["input_ids"].size(0) == 2

        # Sample 0 = ASR
        decoded_0 = processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        assert "ASR" in decoded_0
        assert "ST" not in decoded_0

        # Sample 1 = ST
        decoded_1 = processor.tokenizer.decode(batch["input_ids"][1], skip_special_tokens=False)
        assert "ST" in decoded_1
        assert "ASR" not in decoded_1

        # Both labels should contain their respective targets
        labels = batch["labels"]
        self._assert_target_in_labels(labels[0:1], processor.tokenizer, target_text="hello world")
        self._assert_target_in_labels(labels[1:2], processor.tokenizer, target_text="bonjour le monde")

    @staticmethod
    def _assert_target_in_labels(labels, tokenizer, target_text: str):
        """Assert that *labels* contain *target_text* among the non-masked tokens."""
        for i in range(labels.size(0)):
            row = labels[i]
            active_mask = row != -100
            active_ids = row[active_mask].tolist()
            assert len(active_ids) > 0, (
                f"Sample {i}: all labels are -100 — nothing contributes to loss"
            )
            decoded = tokenizer.decode(active_ids, skip_special_tokens=True).strip()
            decoded_normalized = " ".join(decoded.split())
            target_normalized = " ".join(target_text.split())
            assert target_normalized in decoded_normalized, (
                f"Sample {i}: expected target {target_text!r} not found in decoded {decoded!r}"
            )

    def test_missing_task_in_dict_raises(self, per_task_dataset):
        """A task not in the dict should raise ValueError."""
        cut = _make_cut("en-1", "hello world", "en", task="transcribe")
        cuts = CutSet.from_cuts([cut])
        with pytest.raises(ValueError, match="Task 'transcribe' not found"):
            per_task_dataset[cuts]


# ---------------------------------------------------------------------------
# Per-task prompt_template tests (chat-template mode)
# ---------------------------------------------------------------------------


class TestPerTaskChatTemplateLabels:
    """Verify per-task custom ``prompt_template`` dict in chat-template mode."""

    @pytest.fixture(scope="class")
    def processor(self):
        return _build_processor()

    @pytest.fixture(scope="class")
    def per_task_chat_dataset(self, processor):
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": True,
            "prompt_template_selection": "custom",
            "prompt_template": {
                "asr": "{audio_token} Transcribe this {lang} audio.",
                "st": "{audio_token} Translate this audio to {lang}.",
            },
            "sample_rate": 16000,
        })
        return SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

    def test_asr_cut_uses_custom_chat_template(self, per_task_chat_dataset, processor):
        """ASR cut with chat-template + custom per-task prompt."""
        cut = _make_cut("en-1", "hello world", "en", task="asr")
        cuts = CutSet.from_cuts([cut])
        batch = per_task_chat_dataset[cuts]
        assert batch is not None

        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "Transcribe" in decoded
        assert "Translate" not in decoded

        # Labels: only the assistant (target) tokens survive
        labels = batch["labels"][0]
        active = labels[labels != -100]
        assert len(active) > 0
        target_decoded = processor.tokenizer.decode(
            active.tolist(), skip_special_tokens=True
        ).strip()
        assert "hello world" in target_decoded

    def test_st_cut_uses_custom_chat_template(self, per_task_chat_dataset, processor):
        """ST cut with chat-template + custom per-task prompt."""
        cut = _make_cut("fr-1", "bonjour le monde", "fr", task="st")
        cuts = CutSet.from_cuts([cut])
        batch = per_task_chat_dataset[cuts]
        assert batch is not None

        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "Translate" in decoded
        assert "Transcribe" not in decoded

        labels = batch["labels"][0]
        active = labels[labels != -100]
        target_decoded = processor.tokenizer.decode(
            active.tolist(), skip_special_tokens=True
        ).strip()
        assert "bonjour le monde" in target_decoded

    def test_missing_task_in_chat_dict_raises(self, per_task_chat_dataset):
        """A task not in the prompt_template dict should raise ValueError.

        Uses ``"speechqe"`` which is a valid task in TASK_TEMPLATES but not
        present in the per-task prompt_template dict configured above.
        """
        cut = _make_cut("en-1", "hello", "en", task="speechqe")
        cuts = CutSet.from_cuts([cut])
        with pytest.raises(ValueError, match="Task 'speechqe' not found"):
            per_task_chat_dataset[cuts]


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestPromptTemplateBackwardCompat:
    """Verify single-string prompt_template still works identically."""

    @pytest.fixture(scope="class")
    def processor(self):
        return _build_processor()

    def test_string_template_still_works(self, processor):
        """Single-string prompt_template is backward compatible."""
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": False,
            "prompt_template": "{audio_token}{lang}: {t}",
            "sample_rate": 16000,
        })
        dataset = SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

        cut = _make_cut("en-1", "hello world", "en")
        cuts = CutSet.from_cuts([cut])
        batch = dataset[cuts]
        assert batch is not None

        labels = batch["labels"]
        TestPromptTemplateLabels._assert_labels_match_target(
            labels, processor.tokenizer, target_text="hello world",
        )

    def test_string_template_chat_mode_still_works(self, processor):
        """Single-string prompt_template with chat-template mode is backward compatible."""
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": True,
            "prompt_template_selection": "custom",
            "prompt_template": "{audio_token} Listen and transcribe in {lang}.",
            "sample_rate": 16000,
        })
        dataset = SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

        cut = _make_cut("en-1", "hello world", "en")
        cuts = CutSet.from_cuts([cut])
        batch = dataset[cuts]
        assert batch is not None

        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "Listen and transcribe" in decoded


# ---------------------------------------------------------------------------
# OmegaConf list-of-dicts normalization test
# ---------------------------------------------------------------------------


class TestPromptTemplateNormalization:
    """Verify prompt_template normalization handles various YAML shapes."""

    def test_list_of_dicts_normalized(self):
        """OmegaConf list-of-dicts format normalizes to a plain dict."""
        from melt.training.data.audio.lhotse.helpers import _normalize_prompt_template

        config = OmegaConf.create({
            "prompt_template": [
                {"asr": "ASR template {t}"},
                {"st": "ST template {t}"},
            ],
        })
        result = _normalize_prompt_template(config.prompt_template)
        assert result == {"asr": "ASR template {t}", "st": "ST template {t}"}

    def test_plain_dict_preserved(self):
        """Plain dict is preserved as-is."""
        from melt.training.data.audio.lhotse.helpers import _normalize_prompt_template

        result = _normalize_prompt_template({"asr": "template {t}"})
        assert result == {"asr": "template {t}"}

    def test_string_preserved(self):
        """String is returned as-is."""
        from melt.training.data.audio.lhotse.helpers import _normalize_prompt_template

        result = _normalize_prompt_template("template {t}")
        assert result == "template {t}"

    def test_none_returns_none(self):
        """None returns None."""
        from melt.training.data.audio.lhotse.helpers import _normalize_prompt_template

        result = _normalize_prompt_template(None)
        assert result is None


# ---------------------------------------------------------------------------
# ST templates with src_lang / tgt_lang placeholders
# ---------------------------------------------------------------------------


class TestSTLanguagePlaceholders:
    """Verify that {src_lang} and {tgt_lang} work in ST prompt templates."""

    @pytest.fixture(scope="class")
    def processor(self):
        return _build_processor()

    @pytest.fixture(scope="class")
    def st_dataset(self, processor):
        from melt.training.data.audio.lhotse.dataset import SpeechToTextDataset

        config = OmegaConf.create({
            "apply_chat_template": False,
            "prompt_template": (
                "{audio_token} Translate this {src_lang} source audio to {tgt_lang}: {t}"
            ),
            "sample_rate": 16000,
        })
        return SpeechToTextDataset(
            processor=processor,
            config=config,
            is_train=False,
            return_labels=True,
            return_langs=False,
        )

    def _make_st_cut(self, cut_id: str, text: str, src_lang: str, tgt_lang: str):
        """Create a cut with ST task and both src_lang and tgt_lang tags."""
        cut = _make_cut(cut_id, text, src_lang, task="st")
        # _make_cut sets tags = {"task": "st", "lang": src_lang}
        # but for ST we also need tgt_lang in the tags.
        if hasattr(cut, "custom") and cut.custom:
            cut.custom["tags"]["tgt_lang"] = tgt_lang
            cut.custom["tags"]["src_lang"] = src_lang
        return cut

    def test_src_lang_and_tgt_lang_in_prompt(self, st_dataset, processor):
        """Verify src_lang and tgt_lang appear resolved in the formatted text."""
        cut = self._make_st_cut("st-1", "bonjour", "en", "fr")
        cuts = CutSet.from_cuts([cut])
        batch = st_dataset[cuts]
        assert batch is not None

        input_ids = batch["input_ids"][0]
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "English" in decoded, f"Expected 'English' (src_lang) in: {decoded!r}"
        assert "French" in decoded, f"Expected 'French' (tgt_lang) in: {decoded!r}"

    def test_labels_only_contain_target_text(self, st_dataset, processor):
        """Verify only the target text survives in labels."""
        cut = self._make_st_cut("st-1", "bonjour", "en", "fr")
        cuts = CutSet.from_cuts([cut])
        batch = st_dataset[cuts]
        assert batch is not None

        labels = batch["labels"][0]
        active = labels[labels != -100]
        assert len(active) > 0
        decoded = processor.tokenizer.decode(active.tolist(), skip_special_tokens=True).strip()
        assert "bonjour" in decoded
        assert "English" not in decoded  # src_lang should be masked
        assert "French" not in decoded   # tgt_lang should be masked
