"""Tests for the YAML -> :class:`MELTConfig` hand-off in ``melt.training.setup``.

The thing being pinned is narrow but was broken for a long time: ``MELTConfig``
accepts ``encoder_kwargs`` and ``decoder_kwargs`` symmetrically, but
``prepare_melt_config`` only ever built the decoder half, so
``model.encoder.attn_implementation`` had nowhere to go. It stayed invisible while
facebook/w2v-bert-2.0 was the only encoder -- it applies a relative-position bias
inside self-attention and transformers offers it nothing but sdpa -- and stops being
invisible with facebook/mms-1b, which supports flash_attention_2.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from melt.training.setup import _merge_chat_template_eos_token_id, prepare_melt_config


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


# ============================================================================
# _merge_chat_template_eos_token_id (issue #124)
#
# A backbone's own config.json carries its raw pretraining EOS (Qwen's
# <|endoftext|>), not necessarily the token that closes a rendered chat turn
# (<|im_end|>). MELTForCausalLM builds its decoder via
# AutoModelForCausalLM.from_config, which never reads a generation_config.json,
# so text_decoder_config.eos_token_id is the only place a stop condition can
# come from -- left as the raw EOS alone, generation on a freshly-built ChatML
# decoder never stops at the end of a turn. These tests isolate the merge
# logic with a fake tokenizer/config, no real model or hub access needed.
# ============================================================================


def _fake_tokenizer(vocab: dict[str, int]):
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = vocab
    tokenizer.convert_tokens_to_ids.side_effect = lambda toks: [vocab[t] for t in toks]
    return tokenizer


class TestMergeChatTemplateEosTokenId:
    def test_noop_when_apply_chat_template_is_off(self):
        """Without chat formatting the model never learns to emit a turn-end
        token, so there is nothing to add -- and no vocab lookup either."""
        cfg = OmegaConf.create({"data": {"apply_chat_template": False}})
        tokenizer = _fake_tokenizer({})
        text_decoder_config = SimpleNamespace(eos_token_id=248044)

        _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)

        assert text_decoder_config.eos_token_id == 248044
        tokenizer.get_vocab.assert_not_called()

    def test_noop_when_data_block_is_absent(self):
        """prepare_melt_config's other tests build a cfg with no `data` key
        at all; this must not raise on the missing node."""
        cfg = OmegaConf.create({})
        tokenizer = _fake_tokenizer({})
        text_decoder_config = SimpleNamespace(eos_token_id=248044)

        _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)

        assert text_decoder_config.eos_token_id == 248044

    def test_appends_the_turn_end_token_to_a_qwen_style_scalar_eos(self):
        """The issue #124 scenario: config.json's eos_token_id is the raw
        <|endoftext|> alone; <|im_end|> (chatml's turn_end_token) is missing
        and must be appended, as a list."""
        cfg = OmegaConf.create(
            {"data": {"apply_chat_template": True, "chat_template_config": "chatml"}}
        )
        tokenizer = _fake_tokenizer({"<|im_end|>": 248046})
        text_decoder_config = SimpleNamespace(eos_token_id=248044)

        _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)

        assert text_decoder_config.eos_token_id == [248044, 248046]

    def test_is_a_noop_when_the_turn_end_token_is_already_covered(self):
        """The Llama pattern: upstream config.json already lists the chat
        stop tokens (eos/eom/eot) alongside the raw EOS, so nothing changes."""
        cfg = OmegaConf.create(
            {"data": {"apply_chat_template": True, "chat_template_config": "llama3"}}
        )
        tokenizer = _fake_tokenizer({"<|eot_id|>": 128009})
        text_decoder_config = SimpleNamespace(eos_token_id=[128001, 128008, 128009])

        _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)

        assert text_decoder_config.eos_token_id == [128001, 128008, 128009]

    def test_handles_a_missing_existing_eos_token_id(self):
        cfg = OmegaConf.create(
            {"data": {"apply_chat_template": True, "chat_template_config": "chatml"}}
        )
        tokenizer = _fake_tokenizer({"<|im_end|>": 248046})
        text_decoder_config = SimpleNamespace(eos_token_id=None)

        _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)

        assert text_decoder_config.eos_token_id == [248046]

    def test_raises_when_the_turn_end_token_is_missing_from_the_vocab(self):
        """Silently converting a missing token would map it to <unk> (or
        whatever id 0 happens to be) and add *that* as a stop condition --
        worse than the bug being fixed. Must fail loudly instead."""
        cfg = OmegaConf.create(
            {"data": {"apply_chat_template": True, "chat_template_config": "chatml"}}
        )
        tokenizer = _fake_tokenizer({})
        text_decoder_config = SimpleNamespace(eos_token_id=248044)

        with pytest.raises(RuntimeError, match="not in this decoder's tokenizer vocabulary"):
            _merge_chat_template_eos_token_id(cfg, tokenizer, text_decoder_config)
