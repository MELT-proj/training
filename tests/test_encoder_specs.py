"""Tests for the per-encoder spec table and the encoder half of the shape contract.

The shape contract used to live only on adapters, which made
``_get_audio_embeddings`` assume every encoder was frame-synchronous. That holds for
w2v-BERT and is wrong for anything with a strided frontend, so these tests pin both
halves: that the default spec still reproduces w2v-BERT exactly, and that a
downsampling encoder's lengths are actually halved by the time the adapter sees them.
"""

import types

import pytest
import torch

from melt.modeling.encoder_specs import (
    DEFAULT_ENCODER_SPEC,
    EncoderSpec,
    get_encoder_spec,
    get_encoder_spec_for_config,
    get_encoder_spec_for_feature_extractor,
)
from melt.modeling.modeling_melt import (
    MELTAudioStack,
    MELTMLPAdapter,
    _get_encoder_hidden_size,
)
from unittest.mock import MagicMock

from melt.modeling.configuration_melt import MELTConfig


# ============================================================================
# The spec table
# ============================================================================


class TestEncoderSpecTable:
    def test_default_is_length_preserving_and_time_major(self):
        """The default must reproduce w2v-BERT, or the existing arm silently changes."""
        spec = DEFAULT_ENCODER_SPEC
        assert spec.feature_layout == "tf"
        assert spec.is_channel_major is False
        assert spec.window_frames is None
        assert spec.window_seconds() is None
        assert spec.frame_seconds == 0.02
        for n in (1, 7, 150, 1500):
            assert spec.output_lengths(n) == n

    def test_wav2vec2_bert_has_no_entry_and_gets_the_default(self):
        assert get_encoder_spec("wav2vec2-bert") is DEFAULT_ENCODER_SPEC
        assert get_encoder_spec(None) is DEFAULT_ENCODER_SPEC
        assert get_encoder_spec("some-encoder-we-have-not-met") is DEFAULT_ENCODER_SPEC

    def test_whisper_spec_values(self):
        spec = get_encoder_spec("whisper")
        assert spec.is_channel_major is True
        assert spec.window_frames == 3000
        assert spec.frame_seconds == 0.01
        # 3000 mel frames at 10 ms is exactly Whisper's 30 s window.
        assert spec.window_seconds() == pytest.approx(30.0)

    @pytest.mark.parametrize(
        "n_in, n_out",
        [(0, 0), (1, 1), (2, 1), (3, 2), (300, 150), (2999, 1500), (3000, 1500), (6000, 3000)],
    )
    def test_whisper_output_lengths_round_up(self, n_in, n_out):
        assert get_encoder_spec("whisper").output_lengths(n_in) == n_out

    def test_output_lengths_accepts_tensors(self):
        spec = get_encoder_spec("whisper")
        lengths = torch.tensor([0, 1, 300, 3000])
        assert torch.equal(spec.output_lengths(lengths), torch.tensor([0, 1, 150, 1500]))

    def test_lookup_by_feature_extractor_class(self):
        # The processor holds only a feature extractor, so the spec is resolved by its
        # class name rather than by a model config.
        whisper_fe = type("WhisperFeatureExtractor", (), {})()
        assert get_encoder_spec_for_feature_extractor(whisper_fe).window_frames == 3000

        seamless_fe = type("SeamlessM4TFeatureExtractor", (), {})()
        assert get_encoder_spec_for_feature_extractor(seamless_fe) is DEFAULT_ENCODER_SPEC

    def test_lookup_by_config_model_type(self):
        cfg = types.SimpleNamespace(model_type="whisper")
        assert get_encoder_spec_for_config(cfg).window_frames == 3000
        assert get_encoder_spec_for_config(types.SimpleNamespace()) is DEFAULT_ENCODER_SPEC

    def test_unwrap_keeps_the_encoder_of_an_encoder_decoder(self):
        """AutoModel resolves Whisper to the full model; the decoder is dead weight."""
        encoder = object()
        model = types.SimpleNamespace(get_encoder=lambda: encoder)
        assert get_encoder_spec("whisper").unwrap(model) is encoder

    def test_unwrap_is_a_no_op_for_encoder_only_models(self):
        model = types.SimpleNamespace(get_encoder=lambda: "should not be called")
        assert DEFAULT_ENCODER_SPEC.unwrap(model) is model


# ============================================================================
# Adapter input width
# ============================================================================


class TestEncoderHiddenSize:
    def test_prefers_output_hidden_size(self):
        cfg = types.SimpleNamespace(output_hidden_size=1024, hidden_size=768, d_model=512)
        assert _get_encoder_hidden_size(cfg) == 1024

    def test_falls_back_to_hidden_size(self):
        cfg = types.SimpleNamespace(hidden_size=768)
        assert _get_encoder_hidden_size(cfg) == 768

    def test_falls_back_to_d_model(self):
        """WhisperConfig names its width d_model and has neither of the other two."""
        cfg = types.SimpleNamespace(d_model=1280)
        assert _get_encoder_hidden_size(cfg) == 1280

    def test_raises_when_the_width_cannot_be_found(self):
        with pytest.raises(AttributeError, match="_ENCODER_HIDDEN_SIZE_ATTRS"):
            _get_encoder_hidden_size(types.SimpleNamespace())

    def test_mlp_adapter_builds_against_a_whisper_shaped_config(self):
        """Regression: the old nested-getattr lookup raised before the fallback ran."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "mlp"
        config.adapter_config.mlp_hidden_size = None
        config.audio_encoder_config = types.SimpleNamespace(d_model=1280, model_type="whisper")
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 2048

        adapter = MELTMLPAdapter(config)
        assert adapter.fc1.in_features == 1280
        assert adapter.fc2.out_features == 2048


# ============================================================================
# The composed encoder -> adapter shape contract
# ============================================================================


def _stack_with(spec, adapter):
    """A stand-in for MELTAudioStack that skips loading a real encoder.

    ``_get_output_features_shape`` only reads ``self.encoder.output_lengths`` and
    ``self.adapter``, so this exercises the real composition without pulling 635 M
    parameters off the Hub.
    """
    encoder = types.SimpleNamespace(output_lengths=spec.output_lengths)
    stack = types.SimpleNamespace(encoder=encoder, adapter=adapter)
    # _get_output_features_shape delegates the mask mapping to this sibling method.
    stack._encoder_output_mask = types.MethodType(MELTAudioStack._encoder_output_mask, stack)
    return stack


def _mlp_adapter(hidden_size=1280, out=2048):
    config = MagicMock(spec=MELTConfig)
    config.adapter_config = MagicMock()
    config.adapter_config._type = "mlp"
    config.adapter_config.mlp_hidden_size = None
    config.audio_encoder_config = types.SimpleNamespace(hidden_size=hidden_size)
    config.text_decoder_config = MagicMock()
    config.text_decoder_config.hidden_size = out
    return MELTMLPAdapter(config)


class TestStackOutputFeaturesShape:
    def test_length_preserving_encoder_passes_the_mask_through(self):
        stack = _stack_with(DEFAULT_ENCODER_SPEC, _mlp_adapter(hidden_size=160))
        features = torch.randn(2, 600, 160)
        mask = torch.ones(2, 600, dtype=torch.long)
        mask[1, 400:] = 0

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, features, mask)

        assert shape == (2, 600, 2048)
        assert out_mask.tolist() == mask.tolist()

    def test_downsampling_encoder_halves_lengths_and_mask(self):
        spec = get_encoder_spec("whisper")
        stack = _stack_with(spec, _mlp_adapter(hidden_size=128))
        # One 30 s window of mel frames, with 12 s of real audio in it.
        features = torch.randn(2, 3000, 128)
        mask = torch.zeros(2, 3000, dtype=torch.long)
        mask[0, :1200] = 1
        mask[1, :300] = 1

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, features, mask)

        assert shape == (2, 1500, 2048)
        assert out_mask.shape == (2, 1500)
        # 50 audio tokens per second of real audio -- the same rate as w2v-BERT.
        assert out_mask.sum(-1).tolist() == [600, 150]
        # And they are a prefix, not scattered.
        assert out_mask[0, :600].all() and not out_mask[0, 600:].any()

    def test_chunked_input_spans_multiple_windows(self):
        """A 45 s cut folds into two 30 s windows; lengths still map exactly."""
        spec = get_encoder_spec("whisper")
        stack = _stack_with(spec, _mlp_adapter(hidden_size=128))
        features = torch.randn(1, 6000, 128)
        mask = torch.zeros(1, 6000, dtype=torch.long)
        mask[0, :4500] = 1  # 45 s of real audio

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, features, mask)

        assert shape == (1, 3000, 2048)
        assert int(out_mask.sum()) == 2250  # 45 s x 50 Hz

    def test_no_mask_is_passed_straight_through(self):
        stack = _stack_with(get_encoder_spec("whisper"), _mlp_adapter(hidden_size=128))
        features = torch.randn(2, 3000, 128)

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, features, None)

        assert shape == (2, 1500, 2048)
        assert out_mask is None


# ============================================================================
# Against the real checkpoints
# ============================================================================


@pytest.mark.hub
class TestRealConfigs:
    def test_whisper_large_v3_resolves_to_the_whisper_spec(self):
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("openai/whisper-large-v3")
        spec = get_encoder_spec_for_config(cfg)
        assert spec.window_frames == cfg.max_source_positions * 2
        assert spec.output_lengths(spec.window_frames) == cfg.max_source_positions
        assert _get_encoder_hidden_size(cfg) == cfg.d_model

    def test_w2v_bert_still_resolves_to_the_default(self):
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("facebook/w2v-bert-2.0")
        assert get_encoder_spec_for_config(cfg) is DEFAULT_ENCODER_SPEC
