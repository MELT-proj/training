import json

from src.melt.configuration_melt import MELTConfig
from transformers import AutoConfig


def test_melt_config_save_pretrained_roundtrip(tmp_path):
    # Use `for_model` to avoid network access.
    audio_encoder_config = AutoConfig.for_model("wav2vec2")
    text_decoder_config = AutoConfig.for_model("gpt2")

    config = MELTConfig(
        audio_encoder_config=audio_encoder_config,
        text_decoder_config=text_decoder_config,
    )

    config.save_pretrained(tmp_path)

    config_path = tmp_path / "config.json"
    assert config_path.exists()

    # Sanity-check the JSON includes the nested configs.
    payload = json.loads(config_path.read_text())
    assert payload["model_type"] == "melt"
    assert payload["audio_encoder_config"]["model_type"] == "wav2vec2"
    assert payload["text_decoder_config"]["model_type"] == "gpt2"

    loaded = MELTConfig.from_pretrained(tmp_path)
    assert loaded.model_type == "melt"
    assert loaded.audio_encoder_config.model_type == "wav2vec2"
    assert loaded.text_decoder_config.model_type == "gpt2"
