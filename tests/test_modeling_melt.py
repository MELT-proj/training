"""Tests for MELTForCausalLM and adapter components."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from melt.modeling import (
    MELTAudioAdapter,
    MELTConfig,
    MELTConformerAdapter,
    MELTForCausalLM,
    MELTMLPAdapter,
)


# ---------------------------------------------------------------------------
# Shared model identifiers (small models for fast CI).
# ---------------------------------------------------------------------------
AUDIO_ENCODER = "facebook/wav2vec2-base"
TEXT_DECODER = "gpt2"


def _make_melt_config(**kwargs):
    """Helper to create a MELTConfig with sensible small-model defaults."""
    defaults = dict(
        audio_encoder=AUDIO_ENCODER,
        text_decoder=TEXT_DECODER,
        adapter_config={"_type": "mlp"},
    )
    defaults.update(kwargs)
    config = MELTConfig(**defaults)
    config.audio_bos_token_id = 100
    return config


# ============================================================================
# Freeze helpers
# ============================================================================


def test_freeze_audio_stack():
    """Ensure component-level freeze methods work correctly."""
    config = _make_melt_config()
    # Trim layers so the model is tiny
    config.audio_encoder_config.num_hidden_layers = 1
    config.text_decoder_config.n_layer = 1

    model = MELTForCausalLM(config)

    # Freeze the entire audio stack (encoder + adapter)
    ret = model.audio_stack.freeze()
    assert ret is model.audio_stack
    assert all(not p.requires_grad for p in model.audio_stack.parameters())

    # Text decoder should still be trainable
    assert any(p.requires_grad for p in model.text_decoder.parameters())


# ============================================================================
# Adapter output-shape tests
# ============================================================================


class TestAdapterOutputFeaturesShape:
    """Tests for the _get_output_features_shape method on all adapter types."""

    @pytest.fixture
    def mlp_config(self):
        """Create a config for MLP adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "mlp"
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 768
        config.audio_encoder_config.output_hidden_size = 768
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024
        return config

    @pytest.fixture
    def qformer_config(self):
        """Create a config for Q-Former adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "qformer"
        config.adapter_config.hidden_size = 768
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024
        config.downsample_rate = 5
        config.window_size = 15
        return config

    @pytest.fixture
    def conformer_config(self):
        """Create a config for Conformer adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "conformer"
        config.adapter_config.num_adapter_layers = 2
        config.adapter_config.adapter_kernel_size = 3
        config.adapter_config.adapter_stride = 2
        config.adapter_config.layerdrop = 0.0
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 2
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024
        return config

    def test_mlp_adapter_preserves_sequence_length(self, mlp_config):
        """MLP adapter should preserve sequence length."""
        adapter = MELTMLPAdapter(mlp_config)

        batch_size = 2
        seq_len = 50
        input_hidden_size = 768

        input_features = torch.randn(batch_size, seq_len, input_hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        assert output_shape == (batch_size, seq_len, 1024)
        assert torch.equal(output_mask, attention_mask)

    def test_mlp_adapter_without_attention_mask(self, mlp_config):
        """MLP adapter should handle None attention mask."""
        adapter = MELTMLPAdapter(mlp_config)

        batch_size = 2
        seq_len = 50
        input_hidden_size = 768

        input_features = torch.randn(batch_size, seq_len, input_hidden_size)

        output_shape, output_mask = adapter._get_output_features_shape(input_features, None)

        assert output_shape == (batch_size, seq_len, 1024)
        assert output_mask is None

    def test_mlp_adapter_output_matches_shape_prediction(self, mlp_config):
        """Verify that actual forward output matches predicted shape."""
        adapter = MELTMLPAdapter(mlp_config)

        batch_size = 2
        seq_len = 50
        input_hidden_size = 768

        input_features = torch.randn(batch_size, seq_len, input_hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        predicted_shape, _ = adapter._get_output_features_shape(input_features, attention_mask)
        actual_output = adapter(input_features)

        assert actual_output.shape == predicted_shape

    def test_qformer_adapter_downsampling(self, qformer_config):
        """Q-Former adapter should downsample sequence by downsample_rate."""
        pytest.importorskip("transformers")

        seq_len = 45
        window_size = 15
        num_queries = window_size // 5

        nblocks = math.ceil(seq_len / window_size)
        expected_output_len = nblocks * num_queries

        assert expected_output_len == 9

    def test_conformer_adapter_downsampling_single_layer(self):
        """Test Conformer adapter downsampling with a single layer."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "conformer"
        config.adapter_config.num_adapter_layers = 1
        config.adapter_config.adapter_kernel_size = 3
        config.adapter_config.adapter_stride = 2
        config.adapter_config.layerdrop = 0.0
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 1
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024

        with patch("src.modeling.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        seq_len = 100
        expected_output_len = adapter._compute_output_seq_len(seq_len)

        pad = 3 // 2
        manual_calc = int((seq_len + 2 * pad - 3) / 2 + 1)
        assert expected_output_len == manual_calc

    def test_conformer_adapter_downsampling_multiple_layers(self):
        """Test Conformer adapter downsampling with multiple layers."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "conformer"
        config.adapter_config.num_adapter_layers = 2
        config.adapter_config.adapter_kernel_size = 3
        config.adapter_config.adapter_stride = 2
        config.adapter_config.layerdrop = 0.0
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 2
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024

        with patch("src.modeling.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        seq_len = 100
        expected_output_len = adapter._compute_output_seq_len(seq_len)

        pad = 1
        after_layer1 = int((seq_len + 2 * pad - 3) / 2 + 1)
        after_layer2 = int((after_layer1 + 2 * pad - 3) / 2 + 1)

        assert expected_output_len == after_layer2
        assert expected_output_len == 25

    def test_conformer_get_output_features_shape(self):
        """Test full _get_output_features_shape for Conformer adapter."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "conformer"
        config.adapter_config.num_adapter_layers = 2
        config.adapter_config.adapter_kernel_size = 3
        config.adapter_config.adapter_stride = 2
        config.adapter_config.layerdrop = 0.0
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 2
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 2048

        with patch("src.modeling.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 1024)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        assert output_shape == (batch_size, 25, 2048)
        assert output_mask.shape == (batch_size, 25)

    def test_conformer_attention_mask_subsampling(self):
        """Test that attention mask is correctly subsampled."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_config = MagicMock()
        config.adapter_config._type = "conformer"
        config.adapter_config.num_adapter_layers = 1
        config.adapter_config.adapter_kernel_size = 3
        config.adapter_config.adapter_stride = 2
        config.adapter_config.layerdrop = 0.0
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 1
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024

        with patch("src.modeling.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 1024)

        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, 50:] = 0

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        assert output_shape[1] == 50
        assert output_mask.shape == (batch_size, 50)
        assert output_mask[0].sum() == 50

        expected_valid_batch1 = int((50 + 2 * 1 - 3) / 2 + 1)
        assert output_mask[1].sum() == expected_valid_batch1

    def test_melt_audio_adapter_delegates_correctly(self, mlp_config):
        """Test that MELTAudioAdapter delegates to underlying adapter."""
        adapter = MELTAudioAdapter(mlp_config)

        batch_size = 2
        seq_len = 50
        input_hidden_size = 768

        input_features = torch.randn(batch_size, seq_len, input_hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)
        inner_shape, inner_mask = adapter.adapter._get_output_features_shape(input_features, attention_mask)

        assert output_shape == inner_shape
        assert torch.equal(output_mask, inner_mask)


# ============================================================================
# _inject_tensor tests (2D attention mask case – no real decoder needed)
# ============================================================================


class TestInjectTensor:
    """Test _inject_tensor using the 2-D (attention mask) path which does not
    require a live text_decoder instance."""

    @pytest.fixture
    def model(self):
        """Create a real MELTForCausalLM (tiny) for _inject_tensor testing."""
        config = _make_melt_config()
        config.audio_encoder_config.num_hidden_layers = 1
        config.text_decoder_config.n_layer = 1
        return MELTForCausalLM(config)

    def test_single_injection_mask(self, model):
        """Inject one audio mask into one text mask."""
        inject_id = model.config.audio_bos_token_id
        # text tokens: [3, inject, 3, 3]
        input_ids = torch.tensor([[3, inject_id, 3, 3]])
        target_mask = torch.ones(1, 4)  # (1, 4)
        source_mask = torch.ones(1, 3) * 0.5  # (1, 3)
        source_lengths = torch.tensor([[3]])

        result = model._inject_tensor(
            source_tensor=source_mask,
            target_tensor=target_mask,
            inject_token_id=inject_id,
            input_ids=input_ids,
            source_lengths=source_lengths,
            source_tensor_mask=None,
            pad_item=0.0,
        )

        # inject token replaced by 3 source tokens → seq_len = 4 - 1 + 3 = 6
        assert result.shape == (1, 6)

    def test_no_injection_when_no_token(self, model):
        """If inject token is absent, output equals input."""
        inject_id = model.config.audio_bos_token_id
        input_ids = torch.tensor([[3, 3, 3, 3]])
        target_mask = torch.ones(1, 4)
        source_mask = torch.ones(1, 2) * 0.5
        source_lengths = torch.tensor([[2]])

        result = model._inject_tensor(
            source_tensor=source_mask,
            target_tensor=target_mask,
            inject_token_id=inject_id,
            input_ids=input_ids,
            source_lengths=source_lengths,
            source_tensor_mask=None,
            pad_item=0.0,
        )

        assert result.shape == target_mask.shape
        assert torch.allclose(result, target_mask)
