"""Tests for the generation prompt :class:`MELTDataCollator` emits at eval time.

Generation-based evaluation stands or falls on this split.  The eval
``input_ids`` hold the audio placeholder *and* the target transcript — feeding
them to ``generate()`` would hand the model the answer and produce a WER that
looks like a very good model rather than a bug.  So the collator emits a second,
prompt-only pair, and these tests assert that the target text is not in it, in
every formatting mode.

The failure mode is quiet: a prompt with a stray leading space, or one missing
its generation prefix, degrades WER in a way indistinguishable from a modelling
problem, and the "eval_loss is unchanged" check does not cover it.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import FeatureExtractionMixin

# Every test here builds a real tokenizer/processor, so the whole module
# needs the HuggingFace Hub (or a warm cache). GitHub CI deselects it.
pytestmark = pytest.mark.hub

_TOKENIZER_NAME = "Qwen/Qwen3-1.7B"

AUDIO_TOKEN = "<|audio|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"


class _CountingFeatureExtractor(FeatureExtractionMixin):
    """Fixed-size random features, and a tally of how often it ran."""

    def __init__(self):
        self.calls = 0

    def __call__(self, audio, sampling_rate=16000, return_attention_mask=True,
                 pad_to_multiple_of=8, **kwargs):
        self.calls += 1
        batch_size = len(audio) if isinstance(audio, list) else 1
        seq_len = 80
        return {
            "input_features": torch.randn(batch_size, seq_len, 80),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


def _build_processor():
    """A MELTProcessor around the real Qwen3 tokenizer and a dummy extractor.

    Only the tokenizer is fetched from HuggingFace; no model weights and no real
    audio are involved.  Mirrors ``tests/test_prompt_template_labels.py``.
    """
    from transformers import AutoTokenizer

    from melt.modeling.processing_melt import MELTProcessor

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    specials = {
        "audio_token": AUDIO_TOKEN,
        "audio_bos_token": AUDIO_BOS_TOKEN,
        "audio_eos_token": AUDIO_EOS_TOKEN,
    }
    tokenizer.add_special_tokens({"additional_special_tokens": list(specials.values())})
    for attr, token_str in specials.items():
        setattr(tokenizer, attr, token_str)

    extractor = _CountingFeatureExtractor()
    processor = MELTProcessor(feature_extractor=extractor, tokenizer=tokenizer)
    return processor, extractor


def _items():
    """Two eval samples, in the dict shape MELTMapDataset yields."""
    return [
        {
            "audio": np.zeros(16000, dtype=np.float32),
            "text": "hello world",
            "task": "asr",
            "lang": "en",
        },
        {
            "audio": np.zeros(16000, dtype=np.float32),
            "text": "guten tag",
            "task": "asr",
            "lang": "de",
        },
    ]


def _collator(processor, overrides: dict, is_train: bool = False):
    from melt.training.data.audio.lhotse.collator import MELTDataCollator

    config = OmegaConf.create({"sample_rate": 16000, **overrides})
    return MELTDataCollator(processor=processor, config=config, is_train=is_train)


def _decode_prompts(processor, batch):
    """Decode `prompt_input_ids`, keeping the special tokens visible."""
    return processor.batch_decode(
        batch["prompt_input_ids"], skip_special_tokens=False
    )


@pytest.fixture(scope="module")
def processor_and_extractor():
    return _build_processor()


@pytest.fixture
def processor(processor_and_extractor):
    return processor_and_extractor[0]


class TestPlainMode:
    """`apply_chat_template: false` with no custom template."""

    def test_prompt_is_the_audio_placeholder_alone(self, processor):
        batch = _collator(processor, {"apply_chat_template": False})(_items())

        prompts = _decode_prompts(processor, batch)
        for prompt in prompts:
            assert AUDIO_TOKEN in prompt
            assert AUDIO_BOS_TOKEN in prompt and AUDIO_EOS_TOKEN in prompt

        assert "hello world" not in prompts[0]
        assert "guten tag" not in prompts[1]

    def test_the_full_inputs_still_carry_the_target(self, processor):
        """The prompt is an addition, not a replacement: the loss forward and
        the labels must go on seeing the transcript."""
        batch = _collator(processor, {"apply_chat_template": False})(_items())

        full = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
        assert "hello world" in full[0]
        assert "guten tag" in full[1]

    def test_prompts_are_left_padded(self, processor):
        """Batched generation needs every sequence's last real token at the same
        index; that is what left-padding buys."""
        batch = _collator(processor, {"apply_chat_template": False})(_items())

        mask = batch["prompt_attention_mask"]
        # Any padding sits at the front: the mask never goes 1 -> 0.
        assert torch.all(mask[:, -1] == 1)
        for row in mask:
            ones = row.nonzero().flatten()
            assert ones[-1].item() == mask.shape[1] - 1
            assert torch.all(row[ones[0]:] == 1)


class TestCustomTemplateMode:
    """A custom `prompt_template` whose `{t}` marks where the target starts."""

    CONFIG = {
        "apply_chat_template": False,
        "prompt_template": "{audio_token}{lang}: {t}",
    }

    def test_prompt_keeps_the_lead_in_and_drops_the_target(self, processor):
        batch = _collator(processor, self.CONFIG)(_items())

        prompts = _decode_prompts(processor, batch)
        assert "English:" in prompts[0]
        assert "German:" in prompts[1]
        assert "hello world" not in prompts[0]
        assert "guten tag" not in prompts[1]

    def test_prompt_is_a_prefix_of_the_full_text(self, processor):
        """Anything else means the model is decoding from a context it was never
        trained to continue."""
        batch = _collator(processor, self.CONFIG)(_items())

        prompts = _decode_prompts(processor, batch)
        full = processor.batch_decode(batch["input_ids"], skip_special_tokens=False)
        for prompt, text in zip(prompts, full):
            stripped_prompt = prompt.replace(
                processor.tokenizer.pad_token, ""
            )
            stripped_full = text.replace(processor.tokenizer.pad_token, "")
            assert stripped_full.startswith(stripped_prompt)


class TestChatTemplateMode:
    """`apply_chat_template: true` — the prompt ends at the assistant turn."""

    CONFIG = {"apply_chat_template": True, "prompt_template_selection": "random"}

    def test_prompt_stops_at_the_assistant_boundary(self, processor):
        batch = _collator(processor, self.CONFIG)(_items())

        prompts = _decode_prompts(processor, batch)
        for prompt in prompts:
            assert "<|im_start|>assistant" in prompt
        assert "hello world" not in prompts[0]
        assert "guten tag" not in prompts[1]

    def test_prompt_and_full_text_use_the_same_drawn_template(self, processor):
        """`apply_chat_template_to_texts` draws a task template at random.

        Rendering the prompt in a second call would re-draw it, and the prompt
        would stop being a prefix of what the labels were built from — a
        mismatch no shape check would catch.
        """
        batch = _collator(processor, self.CONFIG)(_items())

        prompts = _decode_prompts(processor, batch)
        full = processor.batch_decode(batch["input_ids"], skip_special_tokens=False)
        pad = processor.tokenizer.pad_token
        for prompt, text in zip(prompts, full):
            assert text.replace(pad, "").startswith(prompt.replace(pad, ""))


class TestGating:
    def test_training_batches_have_no_prompt_fields(self, processor):
        """Nothing generates during training; the extra tokenisation would be
        pure cost."""
        batch = _collator(processor, {"apply_chat_template": False}, is_train=True)(
            _items()
        )

        assert "prompt_input_ids" not in batch
        assert "prompt_attention_mask" not in batch

    def test_audio_is_featurised_once(self, processor_and_extractor):
        """The prompt shares the batch's audio features; re-running the feature
        extractor would double the collator's cost for nothing."""
        processor, extractor = processor_and_extractor
        before = extractor.calls

        batch = _collator(processor, {"apply_chat_template": False})(_items())

        assert extractor.calls == before + len(_items())
        assert batch["prompt_input_ids"].shape[0] == batch["input_ids"].shape[0]


class TestChatTemplateConfigPairing:
    """The eval collator validates its own pairing, not training's.

    `resolve_eval_data_config` lets `validation_ds` win over the parent `data`
    block for the chat-template keys, so eval can be configured with a
    different `chat_template_config` than training uses. The training-side
    check in MELTMapDataset would pass on training's pairing and never see
    eval's — and a mismatched one blanks every eval label, so references decode
    to the empty string and the reported WER is noise rather than an error.
    """

    def test_a_mismatched_config_raises(self, processor):
        # The processor is built on the real Qwen3 tokenizer, which renders
        # ChatML; `llama3` boundaries appear nowhere in it.
        with pytest.raises(ValueError, match="does not match this tokenizer"):
            _collator(
                processor,
                {
                    "apply_chat_template": True,
                    "prompt_template_selection": "random",
                    "chat_template_config": "llama3",
                },
            )

    def test_the_matching_config_is_accepted(self, processor):
        collator = _collator(
            processor,
            {
                "apply_chat_template": True,
                "prompt_template_selection": "random",
                "chat_template_config": "chatml",
            },
        )

        # Built, and with boundaries that masking can actually locate.
        assert collator._assistant_start_ids
        assert collator._assistant_end_ids

    def test_the_error_names_the_alternatives(self, processor):
        """Whoever hits this needs to know what to set instead."""
        with pytest.raises(ValueError, match="chatml"):
            _collator(
                processor,
                {
                    "apply_chat_template": True,
                    "prompt_template_selection": "random",
                    "chat_template_config": "llama3",
                },
            )
