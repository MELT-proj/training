import json

import pytest

from melt.modeling.configuration_melt import MELTAdapterConfig, MELTConfig
from transformers import AutoConfig


# ---------------------------------------------------------------------------
# Default model identifiers used across tests (small models for speed).
# ---------------------------------------------------------------------------
AUDIO_ENCODER = "facebook/wav2vec2-base"
TEXT_DECODER = "gpt2"


class TestMELTAdapterConfig:
    def test_default_initialization(self):
        config = MELTAdapterConfig(_type="mlp")

        assert config._type == "mlp"
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 2
        assert config.intermediate_size == 4096
        assert config.hidden_act == "gelu"
        assert config.dropout == 0.1
        assert config.model_type == "melt_adapter"

    def test_custom_initialization(self):
        config = MELTAdapterConfig(
            _type="mlp",
            hidden_size=512,
            num_hidden_layers=4,
            intermediate_size=2048,
            hidden_act="relu",
            dropout=0.2,
        )

        assert config._type == "mlp"
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 4
        assert config.intermediate_size == 2048
        assert config.hidden_act == "relu"
        assert config.dropout == 0.2

    def test_qformer_params(self):
        config = MELTAdapterConfig(_type="qformer", downsample_rate=7, window_size=21)
        assert config._type == "qformer"
        assert config.downsample_rate == 7
        assert config.window_size == 21


@pytest.mark.hub
class TestMELTConfig:
    def test_initialization_with_model_names(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )

        assert config.model_type == "melt"
        assert config.audio_encoder == AUDIO_ENCODER
        assert config.text_decoder == TEXT_DECODER
        assert config.audio_encoder_config is not None
        assert config.text_decoder_config is not None
        assert isinstance(config.adapter_config, MELTAdapterConfig)

    def test_adapter_config_custom(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp", "hidden_size": 768, "num_hidden_layers": 3},
        )

        assert config.adapter_config.hidden_size == 768
        assert config.adapter_config.num_hidden_layers == 3

    def test_default_parameters(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )

        assert config.initializer_range == 0.02
        assert config.loss_type == "ForCausalLMLoss"

    def test_custom_initializer_range(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
            initializer_range=0.01,
        )

        assert config.initializer_range == 0.01

    def test_vocab_size_property(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )

        expected_vocab_size = AutoConfig.from_pretrained(TEXT_DECODER).vocab_size
        assert config.vocab_size == expected_vocab_size

    def test_is_composition(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )

        assert config.is_composition is True

    def test_adapter_type_from_adapter_config(self):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "qformer", "downsample_rate": 7},
        )

        assert config.adapter_config._type == "qformer"
        assert config.adapter_config.downsample_rate == 7

    def test_saved_config_does_not_carry_the_attn_implementation(self, tmp_path):
        """A round-tripped config loses the decoder's attention implementation.

        This is why train.py re-applies `model.decoder.attn_implementation`
        from the YAML on the `model.ckpt` path: without it the reloaded config
        falls back to transformers' sdpa default, which on an H100 dispatches
        to the cuDNN backend and makes generation ~2 s per token (artemis job
        328287). If transformers ever starts persisting the value this test
        fails, and that re-application can go.
        """
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
            decoder_kwargs={"attn_implementation": "flash_attention_2"},
        )
        assert config.text_decoder_config._attn_implementation == "flash_attention_2"

        config.save_pretrained(tmp_path)
        reloaded = MELTConfig.from_pretrained(tmp_path)

        assert reloaded.text_decoder_config._attn_implementation != "flash_attention_2"

    def test_attn_implementation_can_be_set_on_the_decoder_sub_config(self, tmp_path):
        """The re-application train.py performs has to actually stick.

        MELTConfig overrides the `_attn_implementation` setter so the parent
        class cannot broadcast one value over every sub-config, so the decoder
        has to be addressed directly rather than through the parent.
        """
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )
        config.save_pretrained(tmp_path)
        reloaded = MELTConfig.from_pretrained(tmp_path)

        reloaded.text_decoder_config._attn_implementation = "flash_attention_2"

        assert reloaded.text_decoder_config._attn_implementation == "flash_attention_2"

    def test_save_pretrained_writes_config_json(self, tmp_path):
        config = MELTConfig(
            audio_encoder=AUDIO_ENCODER,
            text_decoder=TEXT_DECODER,
            adapter_config={"_type": "mlp"},
        )

        config_path = tmp_path / "config.json"
        config.to_json_file(str(config_path), use_diff=False)
        assert config_path.exists()

        saved = json.loads(config_path.read_text())
        assert saved["model_type"] == "melt"
        assert "audio_encoder_config" in saved
        assert "text_decoder_config" in saved
