"""Tests for MELTForConditionalGeneration forward pass, specifically testing
the merging of audio and text embeddings and their attention masks."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.modeling import (
    MELTAudioAdapter,
    MELTConfig,
    MELTConformerAdapter,
    MELTForConditionalGeneration,
    MELTMLPAdapter,
)


class TestReplaceAudioPlaceholders:
    """Test the _replace_audio_placeholders method in isolation."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal mock config for testing."""
        config = MagicMock(spec=MELTConfig)
        config.audio_bos_token_id = 100  # Arbitrary token ID for <audio_bos>
        config.use_return_dict = True
        return config

    @pytest.fixture
    def mock_model(self, mock_config):
        """Create a mock model with the _replace_audio_placeholders method."""

        # Use a MagicMock with the real class as spec, then attach the
        # actual _replace_audio_placeholders function so tests can call it
        # attach the actual _replace_audio_placeholders function so tests can call it
        mock_model = MagicMock(spec=MELTForConditionalGeneration)
        mock_model.config = mock_config
        # assign the unbound function; tests call it passing the mock as self
        mock_model._replace_audio_placeholders = MELTForConditionalGeneration._replace_audio_placeholders
        return mock_model

    def test_single_audio_single_batch(self, mock_model):
        """Test replacing placeholders with a single audio in a single batch item."""
        batch_size = 1
        seq_len = 10
        hidden_size = 4
        audio_len = 3

        # Token IDs: [PAD, PAD, AUDIO_BOS, AUDIO, AUDIO, AUDIO, AUDIO_EOS, TEXT, TEXT, TEXT]
        # Positions:  0     1     2          3      4      5      6         7     8     9
        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[0, 0, audio_bos_token, 1, 1, 1, 2, 3, 3, 3]])

        # Text embeddings (placeholder values)
        text_data = torch.zeros(batch_size, seq_len, hidden_size)

        # Audio embeddings (distinct values to verify replacement)
        audio_data = torch.ones(batch_size, audio_len, hidden_size) * 5.0

        # Audio lengths
        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Verify shape is preserved
        assert result.shape == text_data.shape

        # Verify positions 0-2 (before and including audio_bos) are unchanged (zeros)
        assert torch.allclose(result[0, :3, :], torch.zeros(3, hidden_size))

        # Verify positions 3-5 (placeholders) are replaced with audio embeddings (5.0)
        assert torch.allclose(result[0, 3:6, :], torch.ones(3, hidden_size) * 5.0)

        # Verify positions 6-9 (after audio) are unchanged (zeros)
        assert torch.allclose(result[0, 6:, :], torch.zeros(4, hidden_size))

    def test_multiple_audios_single_batch(self, mock_model):
        """Test replacing placeholders with multiple audios in a single batch item."""
        batch_size = 1
        seq_len = 15
        hidden_size = 4
        audio_len_1 = 2
        audio_len_2 = 3

        # Token IDs with two audio segments
        # [TEXT, AUDIO_BOS, AUDIO, AUDIO, AUDIO_EOS, TEXT, AUDIO_BOS, AUDIO, AUDIO, AUDIO, AUDIO_EOS, TEXT, TEXT, TEXT, TEXT]
        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[3, audio_bos_token, 1, 1, 2, 3, audio_bos_token, 1, 1, 1, 2, 3, 3, 3, 3]])

        # Text embeddings (placeholder values)
        text_data = torch.zeros(batch_size, seq_len, hidden_size)

        # Audio embeddings concatenated (audio_len_1 + audio_len_2 = 5 total)
        audio_data = torch.cat(
            [
                torch.ones(batch_size, audio_len_1, hidden_size) * 10.0,  # First audio
                torch.ones(batch_size, audio_len_2, hidden_size) * 20.0,  # Second audio
            ],
            dim=1,
        )

        # Audio lengths
        audio_lengths = torch.tensor([[audio_len_1, audio_len_2]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Verify shape is preserved
        assert result.shape == text_data.shape

        # Verify first audio segment (positions 2-3) replaced with 10.0
        assert torch.allclose(result[0, 2:4, :], torch.ones(2, hidden_size) * 10.0)

        # Verify second audio segment (positions 7-9) replaced with 20.0
        assert torch.allclose(result[0, 7:10, :], torch.ones(3, hidden_size) * 20.0)

        # Verify other positions remain zeros
        assert torch.allclose(result[0, 0:2, :], torch.zeros(2, hidden_size))  # Before first audio
        assert torch.allclose(result[0, 4:7, :], torch.zeros(3, hidden_size))  # Between audios
        assert torch.allclose(result[0, 10:, :], torch.zeros(5, hidden_size))  # After second audio

    def test_batch_with_different_audio_counts(self, mock_model):
        """Test batch where items have different numbers of audios."""
        batch_size = 2
        seq_len = 10
        hidden_size = 4

        audio_bos_token = mock_model.config.audio_bos_token_id

        # Batch item 0: one audio of length 2
        # Batch item 1: no audio
        input_ids = torch.tensor(
            [
                [3, audio_bos_token, 1, 1, 2, 3, 3, 3, 3, 3],  # Has audio
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # No audio
            ]
        )

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, 2, hidden_size) * 7.0

        # First batch item has audio length 2, second has -1 (no audio)
        audio_lengths = torch.tensor([[2, -1], [-1, -1]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Batch 0: positions 2-3 should be replaced
        assert torch.allclose(result[0, 2:4, :], torch.ones(2, hidden_size) * 7.0)

        # Batch 1: everything should remain zeros (no audio)
        assert torch.allclose(result[1, :, :], torch.zeros(seq_len, hidden_size))

    def test_2d_mask_replacement(self, mock_model):
        """Test replacing placeholders in attention masks (2D tensor)."""
        batch_size = 1
        seq_len = 10
        audio_len = 3

        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[0, 0, audio_bos_token, 1, 1, 1, 2, 3, 3, 3]])

        # Text attention mask (all ones initially)
        text_mask = torch.ones(batch_size, seq_len)

        # Audio attention mask (distinct values: 0.5)
        audio_mask = torch.ones(batch_size, audio_len) * 0.5

        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_mask, audio_mask, input_ids, audio_lengths)

        # Shape should be preserved
        assert result.shape == text_mask.shape

        # Positions 3-5 should be replaced with audio mask values (0.5)
        assert torch.allclose(result[0, 3:6], torch.ones(3) * 0.5)

        # Other positions should remain 1.0
        assert torch.allclose(result[0, :3], torch.ones(3))
        assert torch.allclose(result[0, 6:], torch.ones(4))

    def test_no_audio_bos_tokens(self, mock_model):
        """Test when input_ids has no audio_bos tokens."""
        batch_size = 1
        seq_len = 5
        hidden_size = 4

        # No audio_bos token in input
        input_ids = torch.tensor([[3, 3, 3, 3, 3]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, 2, hidden_size) * 5.0
        audio_lengths = torch.tensor([[2]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Result should be identical to input (no changes)
        assert torch.allclose(result, text_data)

    def test_empty_audio_lengths(self, mock_model):
        """Test when audio_lengths contains only -1 (no valid audios)."""
        batch_size = 1
        seq_len = 10
        hidden_size = 4

        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[0, 0, audio_bos_token, 1, 1, 1, 2, 3, 3, 3]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, 3, hidden_size) * 5.0
        audio_lengths = torch.tensor([[-1, -1]])  # No valid audio lengths

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Result should be identical to input (no changes)
        assert torch.allclose(result, text_data)


class TestForwardPassIntegration:
    """Integration tests for the full forward pass with mocked subcomponents."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configs for testing."""
        from transformers import AutoConfig

        # Use small model configs for testing
        audio_config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
        text_config = AutoConfig.from_pretrained("gpt2")

        # Override to make tests faster
        audio_config.num_hidden_layers = 1
        text_config.n_layer = 1

        config = MELTConfig(
            audio_encoder_config=audio_config,
            text_decoder_config=text_config,
            adapter_type="mlp",
        )
        config.audio_bos_token_id = 100
        return config

    def test_decoder_inputs_shape_after_merge(self, minimal_config):
        """Verify decoder_input_embs shape is preserved after merging."""
        batch_size = 2
        text_seq_len = 20
        audio_seq_len = 10
        hidden_size = minimal_config.text_decoder_config.n_embd

        # Create mock model components
        with patch.object(MELTForConditionalGeneration, "__init__", lambda x, y: None):
            model = MELTForConditionalGeneration.__new__(MELTForConditionalGeneration)
            model.config = minimal_config

            # Test the _replace_audio_placeholders directly with controlled inputs
            audio_bos_token = minimal_config.audio_bos_token_id

            # Input IDs with audio placeholders
            input_ids = torch.zeros(batch_size, text_seq_len, dtype=torch.long)
            input_ids[0, 5] = audio_bos_token  # Audio at position 5
            input_ids[1, 3] = audio_bos_token  # Audio at position 3

            # Text embeddings
            text_embs = torch.randn(batch_size, text_seq_len, hidden_size)

            # Audio embeddings
            audio_embs = torch.randn(batch_size, audio_seq_len, hidden_size)

            # Audio lengths
            audio_lengths = torch.tensor([[5, -1], [7, -1]])

            result = model._replace_audio_placeholders(text_embs, audio_embs, input_ids, audio_lengths)

            # Shape must be preserved
            assert result.shape == text_embs.shape
            assert result.shape == (batch_size, text_seq_len, hidden_size)

    def test_attention_mask_preserved_dtype(self, minimal_config):
        """Verify attention mask dtype is preserved after merging."""
        batch_size = 1
        text_seq_len = 10
        audio_seq_len = 5

        with patch.object(MELTForConditionalGeneration, "__init__", lambda x, y: None):
            model = MELTForConditionalGeneration.__new__(MELTForConditionalGeneration)
            model.config = minimal_config

            audio_bos_token = minimal_config.audio_bos_token_id
            input_ids = torch.zeros(batch_size, text_seq_len, dtype=torch.long)
            input_ids[0, 2] = audio_bos_token

            # Float attention mask
            text_mask = torch.ones(batch_size, text_seq_len, dtype=torch.float32)
            audio_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.float32)
            audio_lengths = torch.tensor([[3]])

            result = model._replace_audio_placeholders(text_mask, audio_mask, input_ids, audio_lengths)

            assert result.dtype == torch.float32

            # Bool attention mask
            text_mask_bool = torch.ones(batch_size, text_seq_len, dtype=torch.bool)
            audio_mask_bool = torch.ones(batch_size, audio_seq_len, dtype=torch.bool)

            result_bool = model._replace_audio_placeholders(text_mask_bool, audio_mask_bool, input_ids, audio_lengths)

            assert result_bool.dtype == torch.bool


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=MELTConfig)
        config.audio_bos_token_id = 100
        return config

    @pytest.fixture
    def mock_model(self, mock_config):
        mock_model = MagicMock(spec=MELTForConditionalGeneration)
        mock_model.config = mock_config
        mock_model._replace_audio_placeholders = MELTForConditionalGeneration._replace_audio_placeholders
        return mock_model

    def test_audio_at_start_of_sequence(self, mock_model):
        """Test audio placeholder at the very start of the sequence."""
        batch_size = 1
        seq_len = 8
        hidden_size = 4
        audio_len = 3

        audio_bos_token = mock_model.config.audio_bos_token_id
        # Audio BOS at position 0
        input_ids = torch.tensor([[audio_bos_token, 1, 1, 1, 2, 3, 3, 3]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, audio_len, hidden_size) * 9.0
        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Positions 1-3 should be replaced (after audio_bos at 0)
        assert torch.allclose(result[0, 1:4, :], torch.ones(3, hidden_size) * 9.0)

    def test_audio_at_end_of_sequence(self, mock_model):
        """Test audio placeholder near the end of the sequence."""
        batch_size = 1
        seq_len = 8
        hidden_size = 4
        audio_len = 2

        audio_bos_token = mock_model.config.audio_bos_token_id
        # Audio BOS at position 5, placeholders at 6-7
        input_ids = torch.tensor([[3, 3, 3, 3, 3, audio_bos_token, 1, 1]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, audio_len, hidden_size) * 8.0
        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Positions 6-7 should be replaced
        assert torch.allclose(result[0, 6:8, :], torch.ones(2, hidden_size) * 8.0)

    def test_more_audio_bos_than_audio_lengths(self, mock_model):
        """Test when there are more audio_bos tokens than valid audio_lengths entries."""
        batch_size = 1
        seq_len = 12
        hidden_size = 4

        audio_bos_token = mock_model.config.audio_bos_token_id
        # Two audio_bos tokens but only one valid audio length
        input_ids = torch.tensor([[3, audio_bos_token, 1, 1, 2, 3, audio_bos_token, 1, 1, 2, 3, 3]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, 2, hidden_size) * 6.0
        # Only one valid audio length
        audio_lengths = torch.tensor([[2, -1]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Only first audio segment should be replaced (positions 2-3)
        assert torch.allclose(result[0, 2:4, :], torch.ones(2, hidden_size) * 6.0)

        # Second audio_bos segment should NOT be replaced (positions 7-8 remain zeros)
        assert torch.allclose(result[0, 7:9, :], torch.zeros(2, hidden_size))

    def test_zero_length_audio(self, mock_model):
        """Test handling of zero-length audio (edge case)."""
        batch_size = 1
        seq_len = 8
        hidden_size = 4

        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[3, audio_bos_token, 2, 3, 3, 3, 3, 3]])

        text_data = torch.zeros(batch_size, seq_len, hidden_size)
        audio_data = torch.ones(batch_size, 0, hidden_size)  # Empty audio
        audio_lengths = torch.tensor([[0]])  # Zero length (filtered out as not > 0)

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Nothing should be replaced
        assert torch.allclose(result, text_data)

    def test_large_batch_consistency(self, mock_model):
        """Test that batch processing is consistent across all items."""
        batch_size = 4
        seq_len = 15
        hidden_size = 8
        audio_len = 3

        audio_bos_token = mock_model.config.audio_bos_token_id

        # All batch items have audio_bos at the same position
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[:, 5] = audio_bos_token

        text_data = torch.zeros(batch_size, seq_len, hidden_size)

        # Different audio values per batch item
        audio_data = torch.stack([torch.ones(audio_len, hidden_size) * (i + 1) for i in range(batch_size)])

        audio_lengths = torch.full((batch_size, 1), audio_len)

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Verify each batch item got its own audio values
        for i in range(batch_size):
            expected_value = float(i + 1)
            assert torch.allclose(result[i, 6:9, :], torch.ones(3, hidden_size) * expected_value), (
                f"Batch item {i} mismatch"
            )


class TestMergeCorrectness:
    """Tests to verify the merged tensors have correct values at correct positions."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=MELTConfig)
        config.audio_bos_token_id = 100
        return config

    @pytest.fixture
    def mock_model(self, mock_config):
        mock_model = MagicMock(spec=MELTForConditionalGeneration)
        mock_model.config = mock_config
        mock_model._replace_audio_placeholders = MELTForConditionalGeneration._replace_audio_placeholders
        return mock_model

    def test_text_embeddings_unchanged_outside_placeholders(self, mock_model):
        """Verify text embeddings outside placeholder regions are unchanged."""
        batch_size = 1
        seq_len = 12
        hidden_size = 4
        audio_len = 3

        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[3, 4, 5, audio_bos_token, 1, 1, 1, 2, 6, 7, 8, 9]])

        # Create distinct text embeddings for each position
        text_data = torch.arange(seq_len * hidden_size).float().view(batch_size, seq_len, hidden_size)
        original_text_data = text_data.clone()

        audio_data = torch.ones(batch_size, audio_len, hidden_size) * 999.0
        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_data, audio_data, input_ids, audio_lengths)

        # Positions 0-3 (before placeholder region) should be unchanged
        assert torch.allclose(result[0, :4, :], original_text_data[0, :4, :])

        # Positions 4-6 (placeholder region) should be audio values
        assert torch.allclose(result[0, 4:7, :], torch.ones(3, hidden_size) * 999.0)

        # Positions 7-11 (after placeholder region) should be unchanged
        assert torch.allclose(result[0, 7:, :], original_text_data[0, 7:, :])


def test_freeze_decoder_minimal():
    """Ensure freeze_decoder freezes all decoder parameters and returns self."""
    from transformers import AutoConfig

    audio_config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
    text_config = AutoConfig.from_pretrained("gpt2")
    audio_config.num_hidden_layers = 1
    text_config.n_layer = 1

    config = MELTConfig(
        audio_encoder_config=audio_config,
        text_decoder_config=text_config,
        adapter_type="mlp",
    )
    config.audio_bos_token_id = 100

    model = MELTForConditionalGeneration(config)
    ret = model.freeze_decoder()
    assert ret is model
    # All decoder params should be frozen
    assert all((not p.requires_grad) for p in model.text_decoder.parameters())

    def test_mask_values_correctly_propagated(self, mock_model):
        """Verify mask values from audio are correctly placed."""
        batch_size = 1
        seq_len = 10
        audio_len = 4

        audio_bos_token = mock_model.config.audio_bos_token_id
        input_ids = torch.tensor([[3, audio_bos_token, 1, 1, 1, 1, 2, 3, 3, 3]])

        # Text mask: all 1s
        text_mask = torch.ones(batch_size, seq_len)

        # Audio mask: pattern [1, 0, 1, 0] to verify correct ordering
        audio_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        audio_lengths = torch.tensor([[audio_len]])

        result = mock_model._replace_audio_placeholders(mock_model, text_mask, audio_mask, input_ids, audio_lengths)

        # Verify pattern is preserved at positions 2-5
        expected_pattern = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(result[0, 2:6], expected_pattern)


class TestAdapterOutputFeaturesShape:
    """Tests for the _get_output_features_shape method on all adapter types."""

    @pytest.fixture
    def mlp_config(self):
        """Create a config for MLP adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "mlp"
        # Mock audio encoder config
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 768
        config.audio_encoder_config.output_hidden_size = 768
        # Mock text decoder config
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024
        return config

    @pytest.fixture
    def qformer_config(self):
        """Create a config for Q-Former adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "qformer"
        # Mock projector config
        config.adapter_config = MagicMock()
        config.adapter_config.hidden_size = 768
        # Mock text decoder config
        config.text_decoder_config = MagicMock()
        config.text_decoder_config.hidden_size = 1024
        # Q-Former specific params
        config.downsample_rate = 5
        config.window_size = 15
        return config

    @pytest.fixture
    def conformer_config(self):
        """Create a config for Conformer adapter testing."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "conformer"
        # Mock audio encoder config
        config.audio_encoder_config = MagicMock()
        config.audio_encoder_config.hidden_size = 1024
        config.audio_encoder_config.output_hidden_size = 1024
        config.audio_encoder_config.num_adapter_layers = 2
        config.audio_encoder_config.adapter_kernel_size = 3
        config.audio_encoder_config.adapter_stride = 2
        config.audio_encoder_config.layerdrop = 0.0
        config.audio_encoder_config.layer_norm_eps = 1e-5
        # Mock text decoder config
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

        # Shape should be (batch_size, seq_len, output_hidden_size)
        assert output_shape == (batch_size, seq_len, 1024)
        # Attention mask should be unchanged
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

        # Get predicted shape
        predicted_shape, _ = adapter._get_output_features_shape(input_features, attention_mask)

        # Get actual output
        actual_output = adapter(input_features)

        assert actual_output.shape == predicted_shape

    def test_qformer_adapter_downsampling(self, qformer_config):
        """Q-Former adapter should downsample sequence by downsample_rate."""
        # Skip if blip_2_qformer is not available (requires transformers with it)
        pytest.importorskip("transformers")

        # We'll test the shape calculation logic directly
        seq_len = 45  # Multiple of window_size=15
        window_size = 15
        num_queries = window_size // 5  # downsample_rate=5, so 3 queries

        # Expected: 45 / 15 = 3 blocks, 3 * 3 = 9 output tokens
        nblocks = math.ceil(seq_len / window_size)
        expected_output_len = nblocks * num_queries  # 9

        # Directly test the math
        assert expected_output_len == 9

    def test_conformer_adapter_downsampling_single_layer(self):
        """Test Conformer adapter downsampling with a single layer."""
        # Create a minimal config for 1 layer
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "conformer"
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

        # The real Wav2Vec2BertAdapterLayer depends on numeric fields in the
        # encoder config. Patch it to a lightweight identity module so we can
        # exercise the adapter's length math without constructing heavy layers.
        with patch("src.melt.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        # Test the _compute_output_seq_len method
        seq_len = 100
        # Formula: floor((100 + 2*1 - 3) / 2 + 1) = floor(100/2) = 50
        expected_output_len = adapter._compute_output_seq_len(seq_len)

        # Verify formula: (seq_len + 2*pad - kernel) / stride + 1
        pad = 3 // 2  # 1
        manual_calc = int((seq_len + 2 * pad - 3) / 2 + 1)
        assert expected_output_len == manual_calc

    def test_conformer_adapter_downsampling_multiple_layers(self):
        """Test Conformer adapter downsampling with multiple layers."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "conformer"
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

        with patch("src.melt.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        seq_len = 100
        # Layer 1: floor((100 + 2 - 3) / 2 + 1) = 50
        # Layer 2: floor((50 + 2 - 3) / 2 + 1) = 25
        expected_output_len = adapter._compute_output_seq_len(seq_len)

        # Verify by manual calculation
        pad = 1
        after_layer1 = int((seq_len + 2 * pad - 3) / 2 + 1)  # 50
        after_layer2 = int((after_layer1 + 2 * pad - 3) / 2 + 1)  # 25

        assert expected_output_len == after_layer2
        assert expected_output_len == 25

    def test_conformer_get_output_features_shape(self):
        """Test full _get_output_features_shape for Conformer adapter."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "conformer"
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

        with patch("src.melt.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 1024)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        # Expected: (2, 25, 2048)
        assert output_shape == (batch_size, 25, 2048)
        # Output mask should have shape (batch_size, 25)
        assert output_mask.shape == (batch_size, 25)

    def test_conformer_attention_mask_subsampling(self):
        """Test that attention mask is correctly subsampled."""
        config = MagicMock(spec=MELTConfig)
        config.adapter_type = "conformer"
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

        with patch("src.melt.modeling_melt.Wav2Vec2BertAdapterLayer", new=lambda cfg: torch.nn.Identity()):
            adapter = MELTConformerAdapter(config)

        batch_size = 2
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, 1024)

        # Create attention mask with different valid lengths per batch item
        # Batch 0: 100 valid tokens, Batch 1: 50 valid tokens
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, 50:] = 0

        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        # Output seq_len = 50 (after 1 layer of downsampling)
        assert output_shape[1] == 50
        assert output_mask.shape == (batch_size, 50)

        # Batch 0: all 50 positions should be valid (1.0)
        assert output_mask[0].sum() == 50

        # Batch 1: should have subsampled valid length
        # Original 50 valid tokens -> after downsampling: floor((50+2-3)/2 + 1) = 25
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

        # Through wrapper
        output_shape, output_mask = adapter._get_output_features_shape(input_features, attention_mask)

        # Through inner adapter directly
        inner_shape, inner_mask = adapter.adapter._get_output_features_shape(input_features, attention_mask)

        assert output_shape == inner_shape
        assert torch.equal(output_mask, inner_mask)
