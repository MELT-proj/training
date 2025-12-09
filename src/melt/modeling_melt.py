"""MELT (Multimodal Encoder Language Transformer) architecture"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertAdapterLayer
from transformers.utils import logging

from .configuration_melt import MELTConfig


logger = logging.get_logger(__name__)


# =============================================================================
# Audio Adapter Architectures
# =============================================================================


class MELTMLPAdapter(nn.Module):
    """
    Simple MLP-based audio adapter (similar to Qwen2AudioMultiModalProjector).
    Projects audio encoder hidden states to the text decoder hidden size.
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        audio_hidden_size = getattr(
            config.audio_encoder_config,
            "output_hidden_size",
            getattr(config.audio_encoder_config, "hidden_size", config.audio_encoder_config.output_hidden_size),
        )
        self.output_hidden_size = config.text_decoder_config.hidden_size
        self.linear = nn.Linear(audio_hidden_size, self.output_hidden_size, bias=True)

    def _get_output_features_shape(
        self,
        input_features: torch.Tensor,
        features_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int, int], torch.Tensor | None]:
        """
        Compute the expected output shape after passing through this adapter.

        Args:
            input_features: Input tensor of shape (batch_size, seq_len, hidden_size)
            features_attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (output_shape, output_attention_mask):
                - output_shape: (batch_size, output_seq_len, output_hidden_size)
                - output_attention_mask: Same as input (MLP preserves sequence length)
        """
        batch_size, seq_len, _ = input_features.shape
        output_shape = (batch_size, seq_len, self.output_hidden_size)
        return output_shape, features_attention_mask

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(audio_features)
        return hidden_states


class MELTQFormerAdapter(nn.Module):
    """
    Q-Former based audio adapter (similar to GraniteSpeechEncoderProjector).
    Uses learnable queries and cross-attention to downsample and project audio features.
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = getattr(config, "downsample_rate", 5)
        self.window_size = getattr(config, "window_size", 15)
        self.num_queries = self.window_size // self.downsample_rate
        self.output_hidden_size = config.text_decoder_config.hidden_size

        self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.projector_config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)

        # Q-Former model from config (typically blip_2_qformer)
        self.qformer = AutoModel.from_config(config.projector_config)

        # Final projection to text decoder hidden size
        self.linear = nn.Linear(config.projector_config.hidden_size, self.output_hidden_size)

    def _get_output_features_shape(
        self,
        input_features: torch.Tensor,
        features_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int, int], torch.Tensor | None]:
        """
        Compute the expected output shape after passing through this adapter.

        Args:
            input_features: Input tensor of shape (batch_size, seq_len, hidden_size)
            features_attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (output_shape, output_attention_mask):
                - output_shape: (batch_size, output_seq_len, output_hidden_size)
                - output_attention_mask: All ones since Q-Former doesn't use input mask
        """
        batch_size, seq_len, _ = input_features.shape
        nblocks = math.ceil(seq_len / self.window_size)
        output_seq_len = nblocks * self.num_queries
        output_shape = (batch_size, output_seq_len, self.output_hidden_size)
        # Q-Former doesn't propagate the attention mask; output is always valid
        output_attention_mask = torch.ones(batch_size, output_seq_len, device=input_features.device)
        return output_shape, output_attention_mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad), "constant", 0)
        hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, dim)

        query_output = self.qformer(
            query_embeds=self.query,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=None,
            return_dict=True,
        )
        query_proj = self.linear(
            query_output.last_hidden_state.view(batch_size, nblocks * self.window_size // self.downsample_rate, -1)
        )
        return query_proj


class MELTConformerAdapter(nn.Module):
    """
    Conformer-based audio adapter (similar to Wav2Vec2BertAdapter).
    Uses conformer layers with optional downsampling via strided convolutions.
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        encoder_config = config.audio_encoder_config

        # Feature projection if output_hidden_size differs from hidden_size
        output_hidden_size = getattr(encoder_config, "output_hidden_size", encoder_config.hidden_size)
        if output_hidden_size != encoder_config.hidden_size:
            self.proj = nn.Linear(encoder_config.hidden_size, output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(output_hidden_size, eps=encoder_config.layer_norm_eps)
        else:
            self.proj = None
            self.proj_layer_norm = None

        num_adapter_layers = getattr(encoder_config, "num_adapter_layers", 1)
        self.num_adapter_layers = num_adapter_layers
        self.layers = nn.ModuleList(Wav2Vec2BertAdapterLayer(encoder_config) for _ in range(num_adapter_layers))
        self.layerdrop = getattr(encoder_config, "layerdrop", 0.0)

        self.kernel_size = getattr(encoder_config, "adapter_kernel_size", 3)
        self.stride = getattr(encoder_config, "adapter_stride", 2)

        # Final projection to text decoder hidden size
        adapter_output_size = output_hidden_size
        self.output_hidden_size = config.text_decoder_config.hidden_size
        if adapter_output_size != self.output_hidden_size:
            self.out_proj = nn.Linear(adapter_output_size, self.output_hidden_size)
        else:
            self.out_proj = None

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    def _compute_output_seq_len(self, seq_len: int) -> int:
        """
        Compute the output sequence length after all conformer adapter layers.

        Each layer applies a strided convolution that reduces sequence length.
        Formula: out_len = floor((in_len + 2*pad - kernel_size) / stride + 1)
        """
        output_len = seq_len
        pad = self.kernel_size // 2
        for _ in range(self.num_adapter_layers):
            output_len = int((output_len + 2 * pad - self.kernel_size) / self.stride + 1)
        return output_len

    def _get_output_features_shape(
        self,
        input_features: torch.Tensor,
        features_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int, int], torch.Tensor | None]:
        """
        Compute the expected output shape after passing through this adapter.

        Args:
            input_features: Input tensor of shape (batch_size, seq_len, hidden_size)
            features_attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (output_shape, output_attention_mask):
                - output_shape: (batch_size, output_seq_len, output_hidden_size)
                - output_attention_mask: Subsampled attention mask matching output_seq_len
        """
        batch_size, seq_len, _ = input_features.shape
        output_seq_len = self._compute_output_seq_len(seq_len)
        output_shape = (batch_size, output_seq_len, self.output_hidden_size)

        # Compute subsampled attention mask
        output_attention_mask = None
        if features_attention_mask is not None:
            # Compute effective lengths from attention mask
            seq_lens = (features_attention_mask.size(1) - (1 - features_attention_mask.int()).sum(1)).float()
            # Apply subsampling for each layer
            for _ in range(self.num_adapter_layers):
                seq_lens = self._compute_sub_sample_lengths_from_attention_mask(seq_lens)
            # Create output attention mask from computed lengths
            output_attention_mask = torch.arange(output_seq_len, device=input_features.device).unsqueeze(0)
            output_attention_mask = (output_attention_mask < seq_lens.unsqueeze(1)).float()

        return output_shape, output_attention_mask

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = (attention_mask.size(1) - (1 - attention_mask.int()).sum(1)).to(hidden_states.device)

        for layer in self.layers:
            layerdrop_prob = torch.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(sub_sampled_lengths)
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, sub_sampled_lengths=sub_sampled_lengths
                )

        # Final projection to text decoder hidden size
        if self.out_proj is not None:
            hidden_states = self.out_proj(hidden_states)

        return hidden_states


class MELTAudioAdapter(nn.Module):
    """
    Factory class for audio adapters that projects audio encoder outputs to text decoder space.

    Supports multiple architectures:
    - "mlp": Simple linear projection (Qwen2Audio style)
    - "qformer": Q-Former with learnable queries (GraniteSpeech style)
    - "conformer": Conformer adapter layers (Wav2Vec2Bert style)
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        architecture = config.adapter_type

        if architecture == "mlp":
            self.adapter = MELTMLPAdapter(config)
        elif architecture == "qformer":
            self.adapter = MELTQFormerAdapter(config)
        elif architecture == "conformer":
            self.adapter = MELTConformerAdapter(config)
        else:
            raise ValueError(
                f"Unknown adapter architecture: {architecture}. Supported architectures: 'mlp', 'qformer', 'conformer'"
            )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Some adapters (conformer) need attention_mask, others don't
        if isinstance(self.adapter, MELTConformerAdapter):
            return self.adapter(hidden_states, attention_mask=attention_mask)
        return self.adapter(hidden_states)

    def _get_output_features_shape(
        self,
        input_features: torch.Tensor,
        features_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int, int], torch.Tensor | None]:
        """
        Compute the expected output shape after passing through this adapter.

        Delegates to the underlying adapter implementation.

        Args:
            input_features: Input tensor of shape (batch_size, seq_len, hidden_size)
            features_attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (output_shape, output_attention_mask):
                - output_shape: (batch_size, output_seq_len, output_hidden_size)
                - output_attention_mask: Adapter-specific output attention mask
        """
        return self.adapter._get_output_features_shape(input_features, features_attention_mask)


# =============================================================================
# Audio Encoder with Chunking Support
# =============================================================================


def _unfold_tensor(tensor: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfolds it to a (N*T', max_seq_len, D) where T' is T // max_seq_len.

    Args:
        tensor: Input tensor of shape (N, T, D)
        max_seq_len: Maximum sequence length for each chunk

    Returns:
        Unfolded tensor of shape (N*T', max_seq_len, D)
    """
    _, _, D = tensor.shape
    tensor = tensor.transpose(-1, -2)
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    tensor = F.unfold(tensor[..., None, :], kernel_size=(1, max_seq_len), stride=(1, max_seq_len))

    new_bsz, _, slen = tensor.shape
    tensor = tensor.view(new_bsz, -1, max_seq_len, slen)
    tensor = tensor.permute(0, 3, 2, 1)
    tensor = tensor.view(-1, max_seq_len, D).contiguous()
    return tensor


class MELTAudioEncoder(nn.Module):
    """
    Audio encoder module that encapsulates an audio model and an adapter.

    This module handles:
    - Loading the audio encoder model
    - Projecting encoder outputs through an adapter
    - Chunking long sequences that exceed max_seq_len

    Args:
        config: MELTConfig containing audio_encoder_config and adapter settings
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        self.config = config

        # Maximum sequence length for chunking (default 500 like Phi4)
        self.max_seq_len = getattr(config, "max_audio_seq_len", 500)

        # Initialize the audio encoder
        self.encoder = AutoModel.from_config(config.audio_encoder_config)

        # Validate that the encoder doesn't have an LM head
        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The audio encoder {self.encoder} should not have a LM Head. Please use a model without LM Head."
            )

        # Initialize the audio adapter (projector)
        self.adapter = MELTAudioAdapter(config)

    def forward(
        self,
        input_features: torch.Tensor,
        features_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the audio encoder and adapter.

        Args:
            input_features: Audio features of shape (batch_size, seq_len, feature_dim)
            features_attention_mask: Attention mask of shape (batch_size, seq_len)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple
            **kwargs: Additional arguments passed to the encoder

        Returns:
            Projected audio features ready for the text decoder
        """
        hidden_states = input_features
        bs, seq_len, _ = hidden_states.shape
        unfolded = False
        chunk_pad_size = 0

        # Handle long sequences by chunking
        if seq_len > self.max_seq_len:
            unfolded = True

            # Pad to multiple of max_seq_len for clean unfolding
            if seq_len % self.max_seq_len > 0:
                chunk_pad_size = self.max_seq_len - (seq_len % self.max_seq_len)

            if chunk_pad_size > 0:
                hidden_states = F.pad(hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0)

            # Unfold into chunks
            hidden_states = _unfold_tensor(hidden_states, self.max_seq_len)

            # Handle attention mask for unfolded tensor
            chunk_mask = None
            if features_attention_mask is not None:
                # Pad the attention mask similarly
                padded_mask = F.pad(features_attention_mask, (0, chunk_pad_size), "constant", 0)
                padded_mask = padded_mask.unsqueeze(-1).float()
                chunk_mask = _unfold_tensor(padded_mask, self.max_seq_len)
                chunk_mask = chunk_mask.squeeze(-1).bool()
        else:
            chunk_mask = features_attention_mask

        # Pass through the encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=chunk_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = encoder_outputs[0]

        # Project through adapter
        hidden_states = self.adapter(hidden_states, attention_mask=chunk_mask)

        # Reshape back and remove padding if we unfolded
        if unfolded:
            hidden_states = hidden_states.reshape(bs, -1, hidden_states.shape[-1])
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states

    def freeze_encoder(self):
        """Freeze the audio encoder parameters (not the adapter)."""
        for param in self.encoder.parameters():
            param.requires_grad = False


# =============================================================================
# MELT Model Classes
# =============================================================================


class MELTPreTrainedModel(PreTrainedModel, GenerationMixin):
    """Base class for MELT models."""

    config_class = MELTConfig
    base_model_prefix = "melt"
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = [
        "LlamaDecoderLayer",
        "Qwen2DecoderLayer",
        "MELTAudioAdapter",
        "Wav2Vec2BertAdapterLayer",
        "Wav2Vec2BertEncoderLayer",
    ]
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    _supports_flash_attn_2 = False
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module):
        """Initialize the weights for MELT adapter modules.

        Weight initialization follows the patterns from:
        - Qwen2Audio: Standard normal initialization for Linear layers
        - GraniteSpeech: Query parameter initialization for Q-Former
        - Wav2Vec2Bert: Kaiming initialization for Conv1d, xavier for attention biases
        """
        std = self.config.initializer_range

        # Q-Former adapter query initialization (GraniteSpeech style)
        if isinstance(module, MELTQFormerAdapter):
            module.query.data.normal_(mean=0.0, std=1.0)

        # MLP adapter initialization (Qwen2Audio style)
        elif isinstance(module, MELTMLPAdapter):
            module.linear.weight.data.normal_(mean=0.0, std=std)
            if module.linear.bias is not None:
                module.linear.bias.data.zero_()

        # Conformer adapter initialization (Wav2Vec2Bert style)
        elif isinstance(module, MELTConformerAdapter):
            if module.proj is not None:
                module.proj.weight.data.normal_(mean=0.0, std=std)
                if module.proj.bias is not None:
                    module.proj.bias.data.zero_()
            if module.out_proj is not None:
                module.out_proj.weight.data.normal_(mean=0.0, std=std)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()

        # Generic Linear layers (Qwen2Audio style)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        # Conv1d layers (Wav2Vec2Bert style - Kaiming initialization)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

        # LayerNorm and BatchNorm (common pattern)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        # Embedding layers
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MELTForConditionalGeneration(MELTPreTrainedModel):
    r"""
    MELT model for conditional generation, consisting of an audio encoder, audio adapter, and text decoder.

    The model takes audio features as input, processes them through an encoder, projects them to the
    text decoder's embedding space via an adapter, and generates text autoregressively.
    """

    def __init__(self, config: MELTConfig):
        super().__init__(config)

        # Initialize the text decoder (language model)
        self.text_decoder = AutoModelForCausalLM.from_config(config.text_decoder_config)

        # Propagate tied weights keys if present
        if self.text_decoder._tied_weights_keys is not None:
            self._tied_weights_keys = [f"text_decoder.{k}" for k in self.text_decoder._tied_weights_keys]

        # Initialize the audio encoder (includes encoder model + adapter)
        self.audio_encoder = MELTAudioEncoder(config)

        # Sync attention implementation between config and models
        self.config.audio_encoder_config._attn_implementation = self.audio_encoder.encoder.config._attn_implementation
        self.config.text_decoder_config._attn_implementation = self.text_decoder.config._attn_implementation
        self.audio_encoder.encoder.config = self.config.audio_encoder_config
        self.text_decoder.config = self.config.text_decoder_config

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.audio_encoder.encoder

    def get_decoder(self):
        return self.text_decoder

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.text_decoder.get_output_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        return self.text_decoder.set_output_embeddings(new_embeddings)

    def get_audio_features(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get the audio features projected to the text decoder embedding space."""
        return self.audio_encoder(input_features, features_attention_mask=attention_mask)

    def _replace_audio_placeholders(
        self,
        text_data: torch.Tensor,
        audio_data: torch.Tensor,
        input_ids: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace placeholder <|AUDIO|> tokens with actual audio data (embeddings or masks).

        The text data is expected to contain placeholder tokens after each <audio_bos>:
        <audio_bos> <|AUDIO|> <|AUDIO|> ... <audio_eos>

        This function replaces the <|AUDIO|> placeholders with actual audio data.

        Args:
            text_data: Text data of shape (batch_size, seq_len, ...) - embeddings or masks
            audio_data: Audio data of shape (batch_size, audio_seq_len, ...) - embeddings or masks
            input_ids: Token IDs of shape (batch_size, seq_len)
            audio_lengths: Audio lengths of shape (batch_size, num_audios) where -1 means empty slot

        Returns:
            Data with <|AUDIO|> placeholders replaced by actual audio data
        """
        batch_size = text_data.shape[0]

        # Get audio_bos token ID from tokenizer
        audio_bos_token = self.config.audio_bos_token_id

        # Clone text data to avoid in-place modification
        merged_data = text_data.clone()

        # Track current position in audio_data
        audio_pos = 0

        for batch_idx in range(batch_size):
            input_id_seq = input_ids[batch_idx]  # (seq_len,)
            audio_lens = audio_lengths[batch_idx]  # (num_audios,)

            # Find positions of audio_bos token in this sequence
            audio_bos_positions = torch.where(input_id_seq == audio_bos_token)[0]

            # Filter valid audio lengths (remove -1 padding)
            valid_audio_lens = audio_lens[audio_lens > 0].tolist()

            # If no audio tokens or no audio lengths, keep text data as-is
            if len(audio_bos_positions) == 0 or len(valid_audio_lens) == 0:
                continue

            # Replace placeholder tokens with actual audio data
            for audio_idx, pos in enumerate(audio_bos_positions):
                if audio_idx >= len(valid_audio_lens):
                    break

                audio_len = valid_audio_lens[audio_idx]

                # Get the audio data for this segment
                if audio_data.ndim == 3:  # Embeddings (batch, seq, hidden)
                    audio_slice = audio_data[batch_idx, audio_pos : audio_pos + audio_len, :]
                else:  # Masks (batch, seq)
                    audio_slice = audio_data[batch_idx, audio_pos : audio_pos + audio_len]

                # Replace the placeholder tokens (positions after audio_bos)
                # Placeholders are at positions: pos+1, pos+2, ..., pos+audio_len
                placeholder_start = pos + 1
                placeholder_end = pos + 1 + audio_len

                if audio_data.ndim == 3:
                    merged_data[batch_idx, placeholder_start:placeholder_end, :] = audio_slice
                else:
                    merged_data[batch_idx, placeholder_start:placeholder_end] = audio_slice

                # Update audio position for next iteration
                audio_pos += audio_len

            # Reset audio_pos for next batch item
            audio_pos = 0

        return merged_data

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MELTForConditionalGeneration. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(
        self,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        features_attention_mask: torch.FloatTensor | None = None,
        audio_lengths: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("encoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        # extract input embeds from the decoder
        decoder_input_embs = self.text_decoder.get_input_embeddings()(input_ids)

        # Process audio through encoder (includes chunking and adapter projection)
        # We assume that if we are using cache then we are caching encoder_outputs
        if input_features is not None:
            encoder_hidden_states = None
            if not use_cache or (use_cache and past_key_values is None):
                encoder_hidden_states = self.audio_encoder(
                    input_features,
                    features_attention_mask=features_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs_encoder,
                )

                # Create attention mask for encoder outputs
                if features_attention_mask is not None:
                    encoder_outputs_mask = self.audio_encoder.encoder._get_feature_vector_attention_mask(
                        encoder_hidden_states.shape[1], features_attention_mask
                    )
                else:
                    encoder_outputs_mask = torch.ones(
                        encoder_hidden_states.shape[:2],
                        dtype=attention_mask.dtype if attention_mask is not None else torch.float32,
                        device=encoder_hidden_states.device,
                    )

            # If we are not using the cache, or it's the first pass with the cache on.
            # Hence, we need to build new inputs for the decoder
            if not use_cache or (use_cache and past_key_values is None):
                # Replace placeholder audio tokens with actual audio embeddings
                if audio_lengths is not None and encoder_hidden_states is not None:
                    decoder_input_embs = self._replace_audio_placeholders(
                        decoder_input_embs,
                        encoder_hidden_states,
                        input_ids,
                        audio_lengths,
                    )

                if attention_mask is not None and encoder_outputs_mask is not None:
                    attention_mask = self._replace_audio_placeholders(
                        attention_mask,
                        encoder_outputs_mask,
                        input_ids,
                        audio_lengths,
                    )

        if logits_to_keep == 0:
            logits_to_keep = labels.shape[1] if labels is not None else input_ids.shape[1]

        decoder_outputs = self.text_decoder(
            inputs_embeds=decoder_input_embs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=decoder_outputs.logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                ignore_index=self.config.text_decoder_config.pad_token_id,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via MELTForConditionalGeneration directly is not supported. "
            "Please use the respective methods of the wrapped decoder object "
            "(model.text_decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.text_decoder._reorder_cache(past_key_values, beam_idx)

    def generate(self, *args, **kwargs):
        if hasattr(self, "audio_attention_mask"):
            del self.audio_attention_mask
        return super().generate(*args, **kwargs)

    def can_generate(self):
        return True


__all__ = [
    "MELTPreTrainedModel",
    "MELTForConditionalGeneration",
    "MELTAudioEncoder",
    "MELTAudioAdapter",
    "MELTMLPAdapter",
    "MELTQFormerAdapter",
    "MELTConformerAdapter",
]
