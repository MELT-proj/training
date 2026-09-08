"""Tests for the YAML -> :class:`MELTConfig` hand-off in ``melt.training.setup``.

The thing being pinned is narrow but was broken for a long time: ``MELTConfig``
accepts ``encoder_kwargs`` and ``decoder_kwargs`` symmetrically, but
``prepare_melt_config`` only ever built the decoder half, so
``model.encoder.attn_implementation`` had nowhere to go. It stayed invisible while
facebook/w2v-bert-2.0 was the only encoder -- it applies a relative-position bias
inside self-attention and transformers offers it nothing but sdpa -- and stops being
invisible with facebook/mms-1b, which supports flash_attention_2.
"""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from melt.training.setup import prepare_melt_config


# Small, and already in every cache the suite uses.
AUDIO_ENCODER = "facebook/wav2vec2-base"
TEXT_DECODER = "gpt2"


def _cfg(**encoder_overrides):
    encoder = {"name": AUDIO_ENCODER, "freeze": True, "max_audio_seq_len": 960_000}
    encoder.update(encoder_overrides)
    return OmegaConf.create(
        {
            "model": {
                "encoder": encoder,
                "decoder": {"name": TEXT_DECODER, "attn_implementation": "sdpa"},
                "adapter": {"_type": "mlp"},
            }
        }
    )


def _processor():
    """A processor stub that only has to answer token-id lookups."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = [0]
    tokenizer.pad_token = "<pad>"
    processor = MagicMock()
    processor.tokenizer = tokenizer
    processor.audio_token = "<|audio|>"
    processor.audio_bos_token = "<|audio_bos|>"
    processor.audio_eos_token = "<|audio_eos|>"
    return processor


@pytest.mark.hub
class TestEncoderAttnImplementation:
    def test_the_yaml_value_reaches_the_encoder_sub_config(self):
        config = prepare_melt_config(
            _cfg(attn_implementation="flash_attention_2"), _processor()
        )
        assert config.audio_encoder_config._attn_implementation == "flash_attention_2"

    def test_it_does_not_leak_into_the_decoder(self):
        """Each side keeps its own; MELTConfig does not broadcast across sub-configs."""
        config = prepare_melt_config(
            _cfg(attn_implementation="flash_attention_2"), _processor()
        )
        assert config.text_decoder_config._attn_implementation == "sdpa"

    def test_omitting_it_preserves_the_previous_behaviour(self):
        """No key in the YAML must still mean sdpa, or the w2v-BERT arm changes."""
        config = prepare_melt_config(_cfg(), _processor())
        assert config.audio_encoder_config._attn_implementation == "sdpa"
