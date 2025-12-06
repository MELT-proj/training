import pytest

from src.melt.configuration_melt import MELTConfig, MELTProjectorConfig
from transformers import AutoConfig


class TestMELTProjectorConfig:
    def test_default_initialization(self):
        config = MELTProjectorConfig()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 2
        assert config.intermediate_size == 4096
        assert config.hidden_act == "gelu"
        assert config.dropout == 0.1
        assert config.model_type == "melt_projector"

    def test_custom_initialization(self):
        config = MELTProjectorConfig(
            hidden_size=512,
            num_hidden_layers=4,
            intermediate_size=2048,
            hidden_act="relu",
            dropout=0.2,
        )

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 4
        assert config.intermediate_size == 2048
        assert config.hidden_act == "relu"
        assert config.dropout == 0.2


class TestMELTConfig:
    @pytest.fixture
    def audio_encoder_config(self):
        return AutoConfig.from_pretrained("facebook/w2v-bert-2.0")

    @pytest.fixture
    def text_decoder_config(self):
        return AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")

    def test_initialization_with_config_objects(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
        )

        assert config.model_type == "melt"
        assert config.audio_encoder_config.model_type == "wav2vec2-bert"
        assert config.text_decoder_config.model_type == "qwen2"
        assert isinstance(config.projector_config, MELTProjectorConfig)

    def test_initialization_with_dicts(self):
        audio_encoder_dict = AutoConfig.from_pretrained("facebook/w2v-bert-2.0").to_dict()
        text_decoder_dict = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B").to_dict()

        config = MELTConfig(
            audio_encoder_config=audio_encoder_dict,
            text_decoder_config=text_decoder_dict,
        )

        assert config.model_type == "melt"
        assert config.audio_encoder_config.model_type == "wav2vec2-bert"
        assert config.text_decoder_config.model_type == "qwen2"

    def test_initialization_with_custom_projector(self, audio_encoder_config, text_decoder_config):
        projector_config = MELTProjectorConfig(hidden_size=768, num_hidden_layers=3)

        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            projector_config=projector_config,
        )

        assert config.projector_config.hidden_size == 768
        assert config.projector_config.num_hidden_layers == 3

    def test_initialization_with_projector_dict(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            projector_config={"hidden_size": 512, "dropout": 0.3},
        )

        assert config.projector_config.hidden_size == 512
        assert config.projector_config.dropout == 0.3

    def test_default_parameters(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
        )

        assert config.audio_token_index == 32000
        assert config.initializer_range == 0.02
        assert config.has_lora_adapter is False
        assert config.adapter_type == "mlp"
        assert config.num_latents == 64
        assert config.loss_type == "ForCausalLMLoss"

    def test_custom_parameters(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            audio_token_index=50000,
            initializer_range=0.01,
            has_lora_adapter=True,
            adapter_type="perceiver",
            num_latents=128,
        )

        assert config.audio_token_index == 50000
        assert config.initializer_range == 0.01
        assert config.has_lora_adapter is True
        assert config.adapter_type == "perceiver"
        assert config.num_latents == 128

    def test_vocab_size_property(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
        )

        assert config.vocab_size == text_decoder_config.vocab_size

    def test_missing_audio_encoder_config_raises(self, text_decoder_config):
        with pytest.raises(ValueError, match="audio_encoder_config must be provided"):
            MELTConfig(text_decoder_config=text_decoder_config)

    def test_missing_text_decoder_config_raises(self, audio_encoder_config):
        with pytest.raises(ValueError, match="text_decoder_config must be provided"):
            MELTConfig(audio_encoder_config=audio_encoder_config)

    def test_dict_without_model_type_raises(self, text_decoder_config):
        with pytest.raises(ValueError, match="audio_encoder_config dict must contain 'model_type'"):
            MELTConfig(
                audio_encoder_config={"hidden_size": 1024},
                text_decoder_config=text_decoder_config,
            )

    def test_is_composition(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
        )

        assert config.is_composition is True
