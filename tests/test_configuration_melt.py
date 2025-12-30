import pytest
import json

from src.melt.configuration_melt import MELTConfig, MELTAdapterConfig
from transformers import AutoConfig


class TestMELTAdapterConfig:
    def test_default_initialization(self):
        config = MELTAdapterConfig()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 2
        assert config.intermediate_size == 4096
        assert config.hidden_act == "gelu"
        assert config.dropout == 0.1
        assert config.model_type == "melt_adapter"

    def test_custom_initialization(self):
        config = MELTAdapterConfig(
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

    def test_qformer_params(self):
        config = MELTAdapterConfig(downsample_rate=7, window_size=21)
        assert config.downsample_rate == 7
        assert config.window_size == 21


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
        assert isinstance(config.adapter_config, MELTAdapterConfig)

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
        adapter_config = MELTAdapterConfig(hidden_size=768, num_hidden_layers=3)

        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            adapter_config=adapter_config,
        )

        assert config.adapter_config.hidden_size == 768
        assert config.adapter_config.num_hidden_layers == 3

    def test_initialization_with_projector_dict(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            adapter_config={"hidden_size": 512, "dropout": 0.3},
        )

        assert config.adapter_config.hidden_size == 512
        assert config.adapter_config.dropout == 0.3

    def test_default_parameters(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
        )

        # audio_token_index removed from config; ensure defaults still set elsewhere
        assert hasattr(config, "initializer_range") and config.initializer_range == 0.02
        assert config.initializer_range == 0.02
        assert config.has_lora_adapter is False
        assert config.adapter_type == "mlp"
        assert config.num_latents == 64
        assert config.loss_type == "ForCausalLMLoss"

    def test_custom_parameters(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            initializer_range=0.01,
            has_lora_adapter=True,
            adapter_type="perceiver",
            num_latents=128,
        )

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

    def test_adapter_type_from_adapter_config(self, audio_encoder_config, text_decoder_config):
        config = MELTConfig(
            audio_encoder_config=audio_encoder_config,
            text_decoder_config=text_decoder_config,
            adapter_config={"type": "qformer", "downsample_rate": 7},
        )

        assert config.adapter_type == "qformer"
        assert config.adapter_config.downsample_rate == 7

    def test_save_pretrained_writes_config_json(self, tmp_path):
        # Use local config objects (no network) but still exercise HF save_pretrained
        # which triggers Transformers' diff-serialization logic.
        config = MELTConfig(
            audio_encoder_config=AutoConfig.for_model("wav2vec2"),
            text_decoder_config=AutoConfig.for_model("gpt2"),
        )

        config.save_pretrained(tmp_path)

        config_path = tmp_path / "config.json"
        assert config_path.exists()

        saved = json.loads(config_path.read_text())
        assert saved["model_type"] == "melt"
        assert "audio_encoder_config" in saved
        assert "text_decoder_config" in saved
