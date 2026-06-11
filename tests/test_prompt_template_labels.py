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

def _build_processor():
    """Build a minimal MELTProcessor around a GPT-2 tokenizer.

    The GPT-2 tokenizer is augmented with the MELT-required special tokens
    so that the processor validates successfully.  A dummy feature extractor
    is wired in so that neither real audio nor model downloads are needed.
    """
    from transformers import AutoTokenizer
    from transformers.feature_extraction_utils import FeatureExtractionMixin

    from melt.modeling.processing_melt import MELTProcessor

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add MELT-required special tokens.
    specials = {
        "audio_token": "<|audio|>",
        "audio_bos_token": "<|audio_bos|>",
        "audio_eos_token": "<|audio_eos|>",
    }
    tokenizer.add_special_tokens({"additional_special_tokens": list(specials.values())})
    for attr, token_str in specials.items():
        setattr(tokenizer, attr, token_str)
    # GPT-2 uses EOS as pad; ensure pad_token is set.
    tokenizer.pad_token = tokenizer.eos_token

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
