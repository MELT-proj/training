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
    MELTAudioEncoder,
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


def _stack_with(spec, adapter, max_seq_len=None):
    """A stand-in for MELTAudioStack that skips loading a real encoder.

    ``_get_output_features_shape`` only reads ``self.encoder.output_lengths`` and
    ``self.adapter``, so this exercises the real composition without pulling 635 M
    parameters off the Hub.
    """
    if max_seq_len is None:
        encoder = types.SimpleNamespace(output_lengths=spec.output_lengths)
    else:
        # The chunk-aware mapping, which is what the real encoder exposes.
        encoder = types.SimpleNamespace(spec=spec, max_seq_len=max_seq_len)
        encoder.output_lengths = types.MethodType(MELTAudioEncoder.output_lengths, encoder)
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
# Raw-waveform encoders (the wav2vec2 / HuBERT family)
# ============================================================================


def _tiny_hubert_config(**overrides):
    """A HuBERT config small enough to instantiate, with the real conv frontend.

    The frontend is what the length arithmetic is about, so its kernels and strides
    are left at HuBERT's own defaults; only the transformer on top is shrunk.
    """
    from transformers import HubertConfig

    cfg = HubertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(8,) * 7,
        num_feat_extract_layers=7,
        feat_extract_norm="group",
        layerdrop=0.0,
        # Off so a forward is deterministic. Production keeps whatever the checkpoint
        # ships -- mHuBERT-147 ships it on.
        apply_spec_augment=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _audio_encoder(max_audio_seq_len, cfg=None):
    """A real MELTAudioEncoder around a randomly-initialised tiny HuBERT."""
    cfg = cfg if cfg is not None else _tiny_hubert_config()
    cfg.max_audio_seq_len = max_audio_seq_len
    melt_config = types.SimpleNamespace(audio_encoder_config=cfg, audio_encoder="tiny-hubert")
    return MELTAudioEncoder(melt_config, load_pretrained=False)


class TestWaveformSpecTable:
    @pytest.mark.parametrize(
        "model_type", ["hubert", "wav2vec2", "wavlm", "data2vec-audio", "unispeech-sat"]
    )
    def test_the_whole_family_resolves_to_the_waveform_spec(self, model_type):
        spec = get_encoder_spec(model_type)
        assert spec.is_waveform is True
        assert spec.feature_key == "input_values"
        # Group-norm checkpoints declare return_attention_mask: false.
        assert spec.passes_attention_mask is False
        # "Frames" are samples: 1/16 kHz, not 20 ms.
        assert spec.frame_seconds == pytest.approx(1 / 16_000)
        assert spec.window_frames is None

    def test_w2v_bert_is_not_swept_up_by_the_family(self):
        """`wav2vec2-bert` is a different model_type and a frame-based encoder."""
        assert get_encoder_spec("wav2vec2-bert") is DEFAULT_ENCODER_SPEC
        assert get_encoder_spec("wav2vec2-bert").is_waveform is False

    def test_lookup_by_feature_extractor_class(self):
        fe = type("Wav2Vec2FeatureExtractor", (), {})()
        assert get_encoder_spec_for_feature_extractor(fe).is_waveform is True

    def test_for_config_reads_the_conv_frontend_off_the_checkpoint(self):
        cfg = _tiny_hubert_config()
        spec = get_encoder_spec_for_config(cfg)
        assert spec.conv_kernels == tuple(cfg.conv_kernel)
        assert spec.conv_strides == tuple(cfg.conv_stride)
        assert spec.total_stride == 320

    def test_for_config_raises_when_the_frontend_is_undescribed(self):
        cfg = types.SimpleNamespace(model_type="hubert")
        with pytest.raises(ValueError, match="conv_kernel"):
            get_encoder_spec_for_config(cfg)

    def test_for_config_is_a_no_op_for_spectral_encoders(self):
        cfg = types.SimpleNamespace(model_type="whisper")
        assert get_encoder_spec_for_config(cfg) is get_encoder_spec("whisper")

    @pytest.mark.parametrize(
        "n_samples", [0, 100, 399, 400, 401, 8000, 12345, 16000, 480_000, 960_000]
    )
    def test_output_lengths_match_hf_exactly(self, n_samples):
        """The whole point: a ratio is not good enough.

        HuBERT's frontend rounds *down* seven times over, so 960 000 samples give 2999
        frames where ``ceil(960000 / 320)`` says 3000. One frame of disagreement here
        misaligns every audio embedding after it.
        """
        from transformers import HubertModel

        cfg = _tiny_hubert_config()
        model = HubertModel(cfg)
        expected = max(int(model._get_feat_extract_output_lengths(n_samples)), 0)
        assert get_encoder_spec_for_config(cfg).output_lengths(n_samples) == expected

    def test_a_ratio_would_have_been_wrong(self):
        spec = get_encoder_spec_for_config(_tiny_hubert_config())
        assert spec.output_lengths(960_000) == 2999
        assert -(-960_000 // spec.total_stride) == 3000  # what ceil-division would say

    def test_output_lengths_accepts_tensors(self):
        spec = get_encoder_spec_for_config(_tiny_hubert_config())
        lengths = torch.tensor([0, 100, 400, 16_000, 480_000])
        assert spec.output_lengths(lengths).tolist() == [0, 0, 1, 49, 1499]


class TestWaveformAudioEncoder:
    @pytest.mark.parametrize("max_len", [1500, 500, 16_001, 320])
    def test_frame_sized_max_audio_seq_len_is_rejected(self, max_len):
        """Left at the frame-based default this would slice a clip into ~100 chunks."""
        with pytest.raises(ValueError, match="SAMPLES"):
            _audio_encoder(max_len)

    def test_a_sample_sized_window_is_accepted(self):
        encoder = _audio_encoder(16_000)
        assert encoder.max_seq_len == 16_000
        assert encoder.spec.is_waveform

    @pytest.mark.parametrize("n_samples", [8000, 16_000, 16_320, 40_000, 48_000])
    def test_predicted_lengths_match_a_real_chunked_forward(self, n_samples):
        """The contract that matters: what the stack predicts is what the encoder emits.

        ``forward`` runs the encoder once per chunk, so the total is a sum of
        per-chunk lengths. Under floor-rounding that is not the same as one pass over
        the whole sequence -- 2 x out(16000) is 98, out(32000) is 99 -- and the
        prediction has to follow the chunking, not the ideal.
        """
        encoder = _audio_encoder(16_000).eval()
        waveform = torch.randn(2, n_samples, 1)
        mask = torch.ones(2, n_samples, dtype=torch.long)

        with torch.no_grad():
            out = encoder(waveform, features_attention_mask=mask)

        assert out.shape[1] == encoder.output_lengths(n_samples)

    def test_chunking_really_is_the_reason_the_naive_formula_fails(self):
        encoder = _audio_encoder(16_000)
        assert encoder.output_lengths(32_000) == 2 * encoder.spec.output_lengths(16_000)
        # ...which is one frame short of what a single unchunked pass would give.
        assert encoder.spec.output_lengths(32_000) == encoder.output_lengths(32_000) + 1

    def test_the_encoder_never_sees_the_attention_mask(self):
        """mHuBERT declares return_attention_mask: false; MELT keeps its own copy."""
        encoder = _audio_encoder(16_000).eval()
        seen = {}
        inner = encoder.model.forward

        def spy(input_values, attention_mask=None, **kwargs):
            seen["mask"] = attention_mask
            seen["shape"] = tuple(input_values.shape)
            return inner(input_values, attention_mask=attention_mask, **kwargs)

        encoder.model.forward = spy
        mask = torch.ones(2, 8000, dtype=torch.long)
        mask[1, 4000:] = 0
        with torch.no_grad():
            encoder(torch.randn(2, 8000, 1), features_attention_mask=mask)

        assert seen["mask"] is None
        # And the trailing feature axis MELT carries a waveform on is gone by then.
        assert seen["shape"] == (2, 8000)

    def test_a_masked_encoder_still_gets_its_mask(self):
        """The flag is per-spec, not a blanket "drop the mask"."""
        assert DEFAULT_ENCODER_SPEC.passes_attention_mask is True
        assert get_encoder_spec("whisper").passes_attention_mask is True


class TestWaveformStackShape:
    def test_sample_lengths_become_frame_lengths(self):
        spec = get_encoder_spec_for_config(_tiny_hubert_config())
        stack = _stack_with(spec, _mlp_adapter(hidden_size=32), max_seq_len=16_000)
        # 1 s of waveform, the second item holding 0.5 s of real audio.
        waveform = torch.randn(2, 16_000, 1)
        mask = torch.zeros(2, 16_000, dtype=torch.long)
        mask[0, :] = 1
        mask[1, :8000] = 1

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, waveform, mask)

        assert shape == (2, 49, 2048)
        # ~50 audio tokens per second of real audio, the same rate as w2v-BERT.
        assert out_mask.sum(-1).tolist() == [49, 24]
        assert out_mask[1, :24].all() and not out_mask[1, 24:].any()

    def test_a_chunked_waveform_agrees_with_the_chunked_forward(self):
        spec = get_encoder_spec_for_config(_tiny_hubert_config())
        stack = _stack_with(spec, _mlp_adapter(hidden_size=32), max_seq_len=16_000)
        waveform = torch.randn(1, 32_000, 1)
        mask = torch.ones(1, 32_000, dtype=torch.long)

        shape, out_mask = MELTAudioStack._get_output_features_shape(stack, waveform, mask)

        # 98, not the 99 a single unchunked pass would emit.
        assert shape == (1, 98, 2048)
        assert int(out_mask.sum()) == 98


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

    def test_mhubert_147_resolves_to_the_waveform_spec(self):
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("utter-project/mHuBERT-147")
        spec = get_encoder_spec_for_config(cfg)
        assert spec.is_waveform is True
        assert spec.conv_strides == tuple(cfg.conv_stride)
        assert spec.total_stride == 320  # 16 kHz in, 50 Hz out
        assert _get_encoder_hidden_size(cfg) == 768
        # 60 s of audio, the campaign's max_duration, at the launcher's window.
        assert spec.output_lengths(60 * 16_000) == 2999
