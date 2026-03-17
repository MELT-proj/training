"""MELT (Multimodal Encoder Language Transformer) architecture"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

import transformers.modeling_utils
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
    Wav2Vec2BertAdapterLayer,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ..logging_utils import get_logger
from .configuration_melt import MELTConfig


logger = get_logger(__name__)


# =============================================================================
# Audio Adapter Architectures
# =============================================================================


class MELTMLPAdapter(nn.Module):
    """2-layer MLP projector with normalization for stable LLM injection."""

    def __init__(self, config: MELTConfig):
        super().__init__()
        audio_hidden_size = getattr(
            config.audio_encoder_config,
            "output_hidden_size",
            getattr(
                config.audio_encoder_config,
                "hidden_size",
                config.audio_encoder_config.output_hidden_size,
            ),
        )
        out = config.text_decoder_config.hidden_size
        adapter_cfg = getattr(config, "adapter_config", None)
        mid = (
            getattr(adapter_cfg, "mlp_hidden_size", out)
            if adapter_cfg is not None
            else out
        )
        if mid is None:
            mid = out

        self.fc1 = nn.Linear(audio_hidden_size, mid, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, out, bias=True)

        self.post_norm = nn.LayerNorm(out)
        self.gain = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.output_hidden_size = out

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
        hidden_states = self.fc1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.post_norm(hidden_states) * self.gain.to(
            dtype=hidden_states.dtype
        )
        return hidden_states


class MELTQFormerAdapter(nn.Module):
    """
    Q-Former based audio adapter (similar to GraniteSpeechEncoderProjector).
    Uses learnable queries and cross-attention to downsample and project audio features.
    """

    def __init__(self, config: MELTConfig):
        super().__init__()
        adapter_cfg = getattr(config, "adapter_config", None)
        if adapter_cfg is None:
            raise ValueError(
                "MELTQFormerAdapter requires config.adapter_config to be set"
            )

        self.hidden_size = adapter_cfg.hidden_size
        # Read Q-Former parameters from adapter_config when present
        self.downsample_rate = getattr(
            adapter_cfg, "downsample_rate", getattr(config, "downsample_rate", 5)
        )
        self.window_size = getattr(
            adapter_cfg, "window_size", getattr(config, "window_size", 15)
        )
        self.num_queries = self.window_size // self.downsample_rate
        self.output_hidden_size = config.text_decoder_config.hidden_size

        self.query = nn.Parameter(
            torch.zeros(1, self.num_queries, adapter_cfg.hidden_size)
        )
        self.query.data.normal_(mean=0.0, std=1.0)

        # Q-Former model from config (typically blip_2_qformer)
        self.qformer = AutoModel.from_config(adapter_cfg)

        # Final projection to text decoder hidden size
        self.linear = nn.Linear(adapter_cfg.hidden_size, self.output_hidden_size)

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
        output_attention_mask = torch.ones(
            batch_size, output_seq_len, device=input_features.device
        )
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
            query_output.last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.downsample_rate, -1
            )
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
        adapter_config = getattr(config, "adapter_config", None) or encoder_config

        # Feature projection if output_hidden_size differs from hidden_size
        output_hidden_size = getattr(
            encoder_config, "output_hidden_size", encoder_config.hidden_size
        )
        if output_hidden_size != encoder_config.hidden_size:
            self.proj = nn.Linear(encoder_config.hidden_size, output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(
                output_hidden_size, eps=encoder_config.layer_norm_eps
            )
        else:
            self.proj = None
            self.proj_layer_norm = None

        # Prefer adapter_config values, fall back to encoder_config, then defaults
        num_adapter_layers = getattr(
            adapter_config,
            "num_adapter_layers",
            getattr(encoder_config, "num_adapter_layers", 1),
        )
        self.num_adapter_layers = num_adapter_layers
        self.layers = nn.ModuleList(
            Wav2Vec2BertAdapterLayer(encoder_config) for _ in range(num_adapter_layers)
        )
        self.layerdrop = getattr(
            adapter_config, "layerdrop", getattr(encoder_config, "layerdrop", 0.0)
        )

        self.kernel_size = getattr(
            adapter_config,
            "adapter_kernel_size",
            getattr(encoder_config, "adapter_kernel_size", 3),
        )
        self.stride = getattr(
            adapter_config,
            "adapter_stride",
            getattr(encoder_config, "adapter_stride", 2),
        )

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
            output_len = int(
                (output_len + 2 * pad - self.kernel_size) / self.stride + 1
            )
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

        output_attention_mask = None
        if features_attention_mask is not None:
            # Non-padded lengths (robust, standard)
            non_padded_lengths = (
                features_attention_mask.to(torch.long).sum(dim=-1).to(torch.float32)
            )

            # Apply subsampling for each adapter layer
            out_lengths = non_padded_lengths
            for _ in range(self.num_adapter_layers):
                out_lengths = self._compute_sub_sample_lengths_from_attention_mask(
                    out_lengths
                )

            out_lengths = out_lengths.to(torch.long).clamp(min=0, max=output_seq_len)

            # Build boolean prefix mask of shape (B, output_seq_len)
            output_attention_mask = torch.zeros(
                (batch_size, output_seq_len),
                dtype=torch.bool,
                device=input_features.device,
            )
            valid = out_lengths > 0
            if valid.any():
                output_attention_mask[
                    torch.arange(batch_size, device=input_features.device)[valid],
                    out_lengths[valid] - 1,
                ] = True
                output_attention_mask = (
                    output_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
                )

        return output_shape, output_attention_mask

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = (
                attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
            ).to(hidden_states.device)

        for layer in self.layers:
            layerdrop_prob = torch.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(
                sub_sampled_lengths
            )
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sub_sampled_lengths=sub_sampled_lengths,
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
        architecture = config.adapter_config._type

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

        logger.info("MELT instantiated with adapter architecture: %s", architecture)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        return self.adapter._get_output_features_shape(
            input_features, features_attention_mask
        )

    def freeze(self):
        """Freeze all adapter parameters."""
        for param in self.parameters():
            param.requires_grad = False

        return self


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
    tensor = F.unfold(
        tensor[..., None, :], kernel_size=(1, max_seq_len), stride=(1, max_seq_len)
    )

    new_bsz, _, slen = tensor.shape
    tensor = tensor.reshape(new_bsz, -1, max_seq_len, slen)
    tensor = tensor.permute(0, 3, 2, 1)
    tensor = tensor.reshape(-1, max_seq_len, D).contiguous()
    return tensor


class MELTAudioEncoder(nn.Module):
    """
    Audio encoder module that encapsulates an audio model with chunking support.

    This module handles:
    - Loading the audio encoder model
    - Chunking long sequences that exceed max_seq_len

    Args:
        config: MELTConfig containing audio_encoder_config and chunking settings
    """

    def __init__(self, config: MELTConfig, load_pretrained: bool = True):
        super().__init__()
        self.config = config

        # Maximum sequence length for chunking (default 500 like Phi4)
        self.max_seq_len = getattr(
            config.audio_encoder_config, "max_audio_seq_len", 500
        )

        # Initialize the audio encoder – when loading from a checkpoint the
        # pretrained weights are unnecessary (the checkpoint will overwrite
        # everything), so we use ``from_config`` which is much faster.
        if load_pretrained:
            self.model = AutoModel.from_pretrained(
                config.audio_encoder, config=config.audio_encoder_config
            )
        else:
            self.model = AutoModel.from_config(config.audio_encoder_config)

        # Validate that the encoder doesn't have an LM head
        if self.model.get_output_embeddings() is not None:
            raise ValueError(
                f"The audio encoder {self.model} should not have a LM Head. Please use a model without LM Head."
            )

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
        Forward pass through the audio encoder.

        Args:
            input_features: Audio features of shape (batch_size, seq_len, feature_dim)
            features_attention_mask: Attention mask of shape (batch_size, seq_len)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple
            **kwargs: Additional arguments passed to the encoder

        Returns:
            Audio encoder hidden states
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
                hidden_states = F.pad(
                    hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0
                )

            # Unfold into chunks
            hidden_states = _unfold_tensor(hidden_states, self.max_seq_len)

            # Handle attention mask for unfolded tensor
            chunk_mask = None
            if features_attention_mask is not None:
                # Pad the attention mask similarly
                padded_mask = F.pad(
                    features_attention_mask, (0, chunk_pad_size), "constant", 0
                )
                padded_mask = padded_mask.unsqueeze(-1).float()
                chunk_mask = _unfold_tensor(padded_mask, self.max_seq_len)
                chunk_mask = chunk_mask.squeeze(-1).bool()
        else:
            chunk_mask = features_attention_mask

        # Pass through the encoder
        encoder_outputs = self.model(
            hidden_states,
            attention_mask=chunk_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = encoder_outputs[0]

        # Reshape back and remove padding if we unfolded
        if unfolded:
            hidden_states = hidden_states.reshape(bs, -1, hidden_states.shape[-1])
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

        return self


class MELTAudioStack(nn.Module):
    """Audio stack = encoder + adapter.

    This mirrors the historical behavior of :class:`MELTAudioEncoder` (which used to
    include the adapter), but keeps the responsibilities separated.
    """

    def __init__(self, config: MELTConfig, load_pretrained: bool = True):
        super().__init__()
        self.config = config
        self.encoder = MELTAudioEncoder(config, load_pretrained=load_pretrained)
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
        hidden_states = self.encoder(
            input_features,
            features_attention_mask=features_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return self.adapter(hidden_states, attention_mask=features_attention_mask)

    def freeze(self):
        """Freeze all stack parameters (encoder + adapter)."""
        for param in self.parameters():
            param.requires_grad = False

        return self


# =============================================================================
# MELT Model Classes
# =============================================================================


class MELTPreTrainedModel(PreTrainedModel):
    """Base class for MELT models."""

    config_class = MELTConfig
    base_model_prefix = "melt"
    _skip_keys_device_placement = ["past_key_values"]
    supports_gradient_checkpointing = False
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
            module.fc1.weight.data.normal_(mean=0.0, std=std)
            if module.fc1.bias is not None:
                module.fc1.bias.data.zero_()

            module.fc2.weight.data.normal_(mean=0.0, std=std)
            if module.fc2.bias is not None:
                module.fc2.bias.data.zero_()

            module.post_norm.weight.data.fill_(1.0)
            module.post_norm.bias.data.zero_()

            # Keep the initial injected-audio scale small for stability.
            module.gain.data.fill_(0.1)

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
                k = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
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


class MELTForCausalLM(MELTPreTrainedModel, GenerationMixin):
    r"""
    MELT model for conditional generation, consisting of an audio encoder, audio adapter, and text decoder.

    The model takes audio features as input, processes them through an encoder, projects them to the
    text decoder's embedding space via an adapter, and generates text autoregressively.
    """

    # When loading from a local checkpoint the audio encoder and text decoder
    # weights are supplied by the checkpoint itself.  We mark their keys as
    # *ignorable-on-missing* so that ``from_pretrained`` does not complain if
    # only the adapter weights were saved (partial checkpoint).
    _keys_to_ignore_on_load_missing = ["audio_stack.encoder.*", "text_decoder.*"]

    def __init__(self, config: MELTConfig):
        super().__init__(config)

        # Initialize the text decoder (language model)
        self.text_decoder = self._create_text_stack(config)

        # Propagate tied weights keys if present
        if self.text_decoder._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"text_decoder.{k}" for k in self.text_decoder._tied_weights_keys
            ]

        # Initialize the audio stack (encoder model + adapter)
        self.audio_stack = self._create_audio_stack(config)

        # Sync attention implementation between config and models
        self.config.audio_encoder_config._attn_implementation = (
            self.audio_stack.encoder.model.config._attn_implementation
        )
        self.config.text_decoder_config._attn_implementation = (
            self.text_decoder.config._attn_implementation
        )
        self.audio_stack.encoder.model.config = self.config.audio_encoder_config
        self.text_decoder.config = self.config.text_decoder_config

        # Initialize weights and apply final processing
        self.post_init()

    # -----------------------------------------------------------------
    # Sub-model factory methods
    # -----------------------------------------------------------------

    @staticmethod
    def _create_text_stack(config: MELTConfig):
        """Instantiate the text decoder (language model).

        When ``transformers.modeling_utils._init_weights`` is ``True`` (i.e. we
        are creating a *fresh* model), the decoder is loaded via
        ``from_pretrained`` so that it ships with pretrained weights.

        When the flag is ``False`` (i.e. ``MELTForCausalLM.from_pretrained`` is
        loading a local checkpoint), we use ``from_config`` instead – the
        checkpoint's own state-dict will overwrite all weights, so downloading
        the original pretrained weights would be wasteful.
        """
        if (
            transformers.modeling_utils._init_weights
            and config.text_decoder is not None
        ):
            return AutoModelForCausalLM.from_pretrained(
                config.text_decoder, config=config.text_decoder_config
            )
        return AutoModelForCausalLM.from_config(config.text_decoder_config)

    @staticmethod
    def _create_audio_stack(config: MELTConfig) -> MELTAudioStack:
        """Instantiate the audio stack (encoder + adapter).

        Mirrors :meth:`_create_text_stack` – the audio encoder inside the stack
        is loaded via ``from_pretrained`` only when we are creating a fresh
        model; otherwise ``from_config`` is used.
        """
        load_pretrained = bool(
            transformers.modeling_utils._init_weights
            and config.audio_encoder is not None
        )
        return MELTAudioStack(config, load_pretrained=load_pretrained)

    # -----------------------------------------------------------------
    # Weight initialisation (overrides MELTPreTrainedModel._init_weights)
    # -----------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        """Initialise weights, skipping pre-trained sub-models.

        The text decoder and audio encoder carry their own pretrained weights
        (loaded via ``from_pretrained`` or supplied by the checkpoint).  We
        must **not** re-initialise them with the generic random-init rules
        defined in the base class; only MELT-specific adapter modules should
        be touched.
        """
        # Skip modules that belong to pretrained sub-models.
        if hasattr(self, "text_decoder") and module in self.text_decoder.modules():
            return
        if (
            hasattr(self, "audio_stack")
            and hasattr(self.audio_stack, "encoder")
            and hasattr(self.audio_stack.encoder, "model")
            and module in self.audio_stack.encoder.model.modules()
        ):
            return

        # Delegate adapter / generic initialisation to the base class.
        super()._init_weights(module)

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.text_decoder.get_output_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        return self.text_decoder.set_output_embeddings(new_embeddings)

    def _inject_tensor(
        self,
        source_tensor: torch.Tensor,  # audio embeddings (b, s_a, d) or masks (b, s_a)
        target_tensor: torch.Tensor,  # text embeddings (b, s_t, d) or masks (b, s_t)
        inject_token_id: int,  # token id to inject at
        input_ids: torch.Tensor,  # token ids (b, s_t), with inject_token_id in them
        source_lengths: torch.Tensor,  # lengths of source tensor (b, max_num_injects), where max_num_injects is the max number of inject_token_id in input_ids
        source_tensor_mask: torch.Tensor
        | None = None,  # mask for source tensor (b, s_a), only used for embeddings
        pad_item: torch.Tensor
        | float = 0.0,  # item to pad with for masks (float) or embeddings (tensor of shape (d,))
    ) -> torch.Tensor:
        """
        Inject source_tensor embeddings or masks into target_tensor at positions specified by input_ids.
        For each batch item [i], for each occurrence of inject_token_id in input_ids [i,j] we inject a part of source_tensor
        dictated by source_lengths[i, j].

        This function inserts source slices into the middle of target_tensor (rather than replacing),
        so the resulting sequences may be longer than the original. The output is left-padded to
        the longest sequence.

        For embeddings (ndim=3): uses the decoder's eos_token embedding for padding.
        For attention masks (ndim=2): uses pad_item value for padding.

        Returns:
            Tensor of shape (batch_size, max_new_seq_len, hidden_size) for embeddings,
            or (batch_size, max_new_seq_len) for masks, with left-padding.
        """
        ndim = target_tensor.ndim
        batch_size = target_tensor.shape[0]

        # Determine pad_item based on tensor type
        if ndim == 3:
            # Embeddings: use eos embedding for padding
            hidden_size = target_tensor.shape[-1]
            eos_token_id = self.text_decoder.config.eos_token_id
            pad_item = self.text_decoder.get_input_embeddings()(
                torch.tensor(
                    [eos_token_id], device=target_tensor.device, dtype=torch.long
                )
            ).squeeze(0)  # (hidden_size,)
        else:
            # Attention masks: use 0.0 for padding (masked positions)
            pad_item = torch.tensor(
                pad_item, device=target_tensor.device, dtype=target_tensor.dtype
            )

        merged_sequences = []

        for batch_idx in range(batch_size):
            input_id_seq = input_ids[batch_idx]  # (seq_len,)
            item_lengths = source_lengths[batch_idx]  # (max_num_injects,)

            # Filter valid audio lengths (remove -1 or 0 padding)
            valid_audio_lens = item_lengths[item_lengths > 0].tolist()

            # Filter valid source tensor based on source mask
            if ndim == 3 and source_tensor_mask is not None:
                item_source_mask = source_tensor_mask[batch_idx]  # (s_a,)
                valid_source_tensor = source_tensor[batch_idx][
                    item_source_mask
                ]  # (valid_s_a, d)
            else:
                valid_source_tensor = source_tensor[batch_idx]  # (d)

            # Find positions where injection should happen
            inject_positions = torch.where(input_id_seq == inject_token_id)[0]

            # If no inject tokens or no audio lengths, keep target data as-is
            if len(inject_positions) == 0 or len(valid_audio_lens) == 0:
                merged_sequences.append(target_tensor[batch_idx])
                continue

            # Build the merged sequence by concatenating slices
            slices = []
            prev_pos = 0
            source_pos = 0

            for audio_idx, pos in enumerate(inject_positions):
                if audio_idx >= len(valid_audio_lens):
                    # No more audio to inject, but there may be more inject tokens
                    # Skip remaining inject tokens
                    break

                pos = pos.item()
                audio_len = int(valid_audio_lens[audio_idx])

                # Add target slice before the inject position (excluding the inject token itself)
                if pos > prev_pos:
                    slices.append(target_tensor[batch_idx, prev_pos:pos])

                # Add the source (audio) slice
                audio_slice = valid_source_tensor[source_pos : source_pos + audio_len]
                slices.append(audio_slice)

                # Update positions: skip the inject_token_id in target
                prev_pos = pos + 1
                source_pos += audio_len

            # Add remaining target slice after the last injection
            if prev_pos < target_tensor.shape[1]:
                slices.append(target_tensor[batch_idx, prev_pos:])

            # Concatenate all slices for this batch item
            if slices:
                merged_seq = torch.cat(slices, dim=0)  # (new_seq_len, hidden_size)
            else:
                merged_seq = target_tensor[batch_idx]

            merged_sequences.append(merged_seq)

        # Find max sequence length for padding
        max_seq_len = max(seq.shape[0] for seq in merged_sequences)

        # Left-pad all sequences to max_seq_len
        padded_sequences = []
        for seq in merged_sequences:
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                if ndim == 3:
                    # Embeddings: expand pad_item to (pad_len, hidden_size)
                    padding = pad_item.unsqueeze(0).expand(pad_len, hidden_size)
                else:
                    # Masks: create tensor of zeros with shape (pad_len,)
                    padding = pad_item.expand(pad_len)
                padded_seq = torch.cat([padding, seq], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack into batch tensor
        # Shape: (batch_size, max_seq_len, hidden_size) for embeddings
        # Shape: (batch_size, max_seq_len) for masks
        merged_tensor = torch.stack(padded_sequences, dim=0)

        return merged_tensor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MELTForCausalLM. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def _get_text_embeddings(self, input_ids):
        embedding = self.text_decoder.get_input_embeddings()
        return embedding(input_ids)

    def _get_audio_embeddings(
        self,
        input_features,
        features_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        **kwargs_encoder,
    ):
        encoder_hidden_states = self.audio_stack(
            input_features,
            features_attention_mask=features_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )

        # Create attention mask for encoder outputs
        if features_attention_mask is not None:
            # Tuple of (output_shape, output_attention_mask):
            # - output_shape: (batch_size, output_seq_len, output_hidden_size)
            # - output_attention_mask: Subsampled attention mask matching output_seq_len
            output_shape, encoder_outputs_mask = (
                self.audio_stack.adapter._get_output_features_shape(
                    input_features, features_attention_mask
                )
            )
        else:
            output_shape = encoder_hidden_states.shape
            encoder_outputs_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device,
            )

        # The audio stack might have modified the audio lengths (e.g., subsampling in conformer).
        # We need to compute the new audio lengths after the audio stack.

        # TODO: IMPORTANT -- this implementation only works if the is only one audio per batch item!
        # The reason why it breaks is with more than one is because audio_lengths should be
        # a list of lists of lenghts. Moreover, since we concatenate them and process them together in the audio encoder
        # we will have to compute the new lengths for each audio accordingly.
        audio_lengths = encoder_outputs_mask.sum(dim=1).unsqueeze(-1)

        return encoder_hidden_states, encoder_outputs_mask, audio_lengths

    def _merge_embeddings(
        self,
        decoder_input_embs,
        encoder_hidden_states,
        encoder_outputs_mask,
        audio_lengths,
        input_ids,
        attention_mask,
        labels,
    ):
        """Replace placeholder audio tokens with actual audio embeddings."""
        decoder_input_embs = self._inject_tensor(
            source_tensor=encoder_hidden_states,
            target_tensor=decoder_input_embs,
            inject_token_id=self.config.audio_token_id,
            input_ids=input_ids,
            source_lengths=audio_lengths,
            source_tensor_mask=encoder_outputs_mask,  # not all encoder outputs are valid, this is used to filter them
        )
        if attention_mask is not None and encoder_outputs_mask is not None:
            attention_mask = self._inject_tensor(
                source_tensor=encoder_outputs_mask,
                target_tensor=attention_mask,
                inject_token_id=self.config.audio_token_id,
                input_ids=input_ids,
                source_lengths=audio_lengths,
            )
        if labels is not None:
            labels = self._inject_tensor(
                source_tensor=torch.full(
                    encoder_outputs_mask.shape,
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                ),
                target_tensor=labels,
                inject_token_id=self.config.audio_token_id,
                input_ids=input_ids,
                source_lengths=audio_lengths,
                pad_item=-100,
            )  # type: ignore
        return decoder_input_embs, attention_mask, labels

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
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
        if input_ids is None and input_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }
        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        # HF Trainer passes `num_items_in_batch` (not prefixed); forward it to the decoder if present.
        if "num_items_in_batch" in kwargs:
            kwargs_decoder["num_items_in_batch"] = kwargs["num_items_in_batch"]

        # 1. Extract text embeddings
        decoder_input_embs = self._get_text_embeddings(
            input_ids=input_ids,
        )  # (batch_size, seq_len, hidden_size)

        # 2. If the user provided audio features, extract audio embeddings
        # In this step chunking and adapter projection happens. Moreover, we return the new
        # features mask and audio lenghts since some adapters modify the sequence length.
        if input_features is not None:
            encoder_hidden_states, encoder_outputs_mask, audio_lengths = (
                self._get_audio_embeddings(
                    input_features=input_features,
                    features_attention_mask=features_attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    **kwargs_encoder,
                )
            )

            decoder_input_embs, attention_mask, labels = self._merge_embeddings(
                decoder_input_embs=decoder_input_embs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_outputs_mask=encoder_outputs_mask,
                audio_lengths=audio_lengths,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # TODO: Below this point (logits to keep and loss computation) only works
        # for causal language modeling with a single text response.
        # For training, we keep only the last `logits_to_keep` logits for computing loss on text tokens.
        # For generation, we typically only need the last token's logits.
        # if logits_to_keep == 0:
        #     logits_to_keep = labels.shape[1] if labels is not None else input_ids.shape[1]

        # We do not pass labels to the LLM and compute the loss ourselves
        outputs: CausalLMOutputWithPast = self.text_decoder(
            inputs_embeds=decoder_input_embs,
            attention_mask=attention_mask,
            input_features=input_features,
            features_attention_mask=features_attention_mask,
            audio_lengths=audio_lengths,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = outputs.logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                ignore_index=-100,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.text_decoder._reorder_cache(past_key_values, beam_idx)

    def generate(self, *args, **kwargs):
        # TODO this needs to be improved using cache in output classes
        if hasattr(self, "audio_attention_mask"):
            del self.audio_attention_mask
        return super().generate(*args, **kwargs)


class MELTForSequenceClassification(MELTPreTrainedModel):
    r"""
    MELT model for sequence classification, consisting of an audio encoder, audio adapter, and text decoder.

    The model takes audio features as input, processes them through an encoder, projects them to the
    text decoder's embedding space via an adapter, and predicts sequence labels.
    """

    # Same loading behavior as MELTForCausalLM for shared stacks.
    _keys_to_ignore_on_load_missing = ["audio_stack.encoder.*", "text_decoder.*"]

    def __init__(self, config: MELTConfig):
        super().__init__(config)

        # Initialize the text decoder (sequence classification model)
        self.text_decoder = self._create_text_stack(config)

        # Propagate tied weights keys if present
        if self.text_decoder._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"text_decoder.{k}" for k in self.text_decoder._tied_weights_keys
            ]

        # Initialize the audio stack (encoder model + adapter)
        self.audio_stack = self._create_audio_stack(config)

        # Sync attention implementation between config and models
        self.config.audio_encoder_config._attn_implementation = (
            self.audio_stack.encoder.model.config._attn_implementation
        )
        self.config.text_decoder_config._attn_implementation = (
            self.text_decoder.config._attn_implementation
        )
        self.audio_stack.encoder.model.config = self.config.audio_encoder_config
        self.text_decoder.config = self.config.text_decoder_config

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _create_text_stack(config: MELTConfig):
        """Instantiate the text decoder (sequence classification model)."""
        if (
            transformers.modeling_utils._init_weights
            and config.text_decoder is not None
        ):
            return AutoModelForSequenceClassification.from_pretrained(
                config.text_decoder, config=config.text_decoder_config
            )
        return AutoModelForSequenceClassification.from_config(config.text_decoder_config)

    @staticmethod
    def _create_audio_stack(config: MELTConfig) -> MELTAudioStack:
        """Instantiate the audio stack (encoder + adapter)."""
        load_pretrained = bool(
            transformers.modeling_utils._init_weights
            and config.audio_encoder is not None
        )
        return MELTAudioStack(config, load_pretrained=load_pretrained)

    def _init_weights(self, module: nn.Module):
        """Initialise weights, skipping pre-trained sub-models."""
        if hasattr(self, "text_decoder") and module in self.text_decoder.modules():
            return
        if (
            hasattr(self, "audio_stack")
            and hasattr(self.audio_stack, "encoder")
            and hasattr(self.audio_stack.encoder, "model")
            and module in self.audio_stack.encoder.model.modules()
        ):
            return

        super()._init_weights(module)

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.text_decoder.get_output_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        return self.text_decoder.set_output_embeddings(new_embeddings)

    def _inject_tensor(
        self,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        inject_token_id: int,
        input_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        source_tensor_mask: torch.Tensor | None = None,
        pad_item: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Inject source_tensor embeddings or masks into target_tensor."""
        ndim = target_tensor.ndim
        batch_size = target_tensor.shape[0]

        if ndim == 3:
            hidden_size = target_tensor.shape[-1]
            eos_token_id = self.text_decoder.config.eos_token_id
            pad_item = self.text_decoder.get_input_embeddings()(
                torch.tensor(
                    [eos_token_id], device=target_tensor.device, dtype=torch.long
                )
            ).squeeze(0)
        else:
            pad_item = torch.tensor(
                pad_item, device=target_tensor.device, dtype=target_tensor.dtype
            )

        merged_sequences = []

        for batch_idx in range(batch_size):
            input_id_seq = input_ids[batch_idx]
            item_lengths = source_lengths[batch_idx]
            valid_audio_lens = item_lengths[item_lengths > 0].tolist()

            if ndim == 3 and source_tensor_mask is not None:
                item_source_mask = source_tensor_mask[batch_idx]
                valid_source_tensor = source_tensor[batch_idx][item_source_mask]
            else:
                valid_source_tensor = source_tensor[batch_idx]

            inject_positions = torch.where(input_id_seq == inject_token_id)[0]

            if len(inject_positions) == 0 or len(valid_audio_lens) == 0:
                merged_sequences.append(target_tensor[batch_idx])
                continue

            slices = []
            prev_pos = 0
            source_pos = 0

            for audio_idx, pos in enumerate(inject_positions):
                if audio_idx >= len(valid_audio_lens):
                    break

                pos = pos.item()
                audio_len = int(valid_audio_lens[audio_idx])

                if pos > prev_pos:
                    slices.append(target_tensor[batch_idx, prev_pos:pos])

                audio_slice = valid_source_tensor[source_pos : source_pos + audio_len]
                slices.append(audio_slice)

                prev_pos = pos + 1
                source_pos += audio_len

            if prev_pos < target_tensor.shape[1]:
                slices.append(target_tensor[batch_idx, prev_pos:])

            if slices:
                merged_seq = torch.cat(slices, dim=0)
            else:
                merged_seq = target_tensor[batch_idx]

            merged_sequences.append(merged_seq)

        max_seq_len = max(seq.shape[0] for seq in merged_sequences)

        padded_sequences = []
        for seq in merged_sequences:
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                if ndim == 3:
                    padding = pad_item.unsqueeze(0).expand(pad_len, hidden_size)
                else:
                    padding = pad_item.expand(pad_len)
                padded_seq = torch.cat([padding, seq], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences, dim=0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MELTForSequenceClassification. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        # Pop MELT-specific sub-model kwargs before they reach transformers'
        # from_pretrained machinery (which would not know what to do with them).
        text_decoder_kwargs = kwargs.pop("text_decoder_kwargs", None) or {}
        audio_encoder_kwargs = kwargs.pop("audio_encoder_kwargs", None) or {}

        if text_decoder_kwargs or audio_encoder_kwargs:
            # Load (or reuse) the config and patch it in-place so that
            # _create_text_stack / _create_audio_stack pick the values up
            # naturally (e.g. AutoModelForSequenceClassification.from_pretrained
            # will receive num_labels through config.text_decoder_config).
            config = kwargs.pop("config", None)
            if config is None:
                config = MELTConfig.from_pretrained(pretrained_model_name_or_path)
            for k, v in text_decoder_kwargs.items():
                setattr(config.text_decoder_config, k, v)
            for k, v in audio_encoder_kwargs.items():
                setattr(config.audio_encoder_config, k, v)
            kwargs["config"] = config

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def _get_text_embeddings(self, input_ids):
        embedding = self.text_decoder.get_input_embeddings()
        return embedding(input_ids)

    def _get_audio_embeddings(
        self,
        input_features,
        features_attention_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        **kwargs_encoder,
    ):
        encoder_hidden_states = self.audio_stack(
            input_features,
            features_attention_mask=features_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )

        if features_attention_mask is not None:
            output_shape, encoder_outputs_mask = (
                self.audio_stack.adapter._get_output_features_shape(
                    input_features, features_attention_mask
                )
            )
        else:
            output_shape = encoder_hidden_states.shape
            encoder_outputs_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device,
            )

        audio_lengths = encoder_outputs_mask.sum(dim=1).unsqueeze(-1)

        return encoder_hidden_states, encoder_outputs_mask, audio_lengths

    def _merge_embeddings(
        self,
        decoder_input_embs,
        encoder_hidden_states,
        encoder_outputs_mask,
        audio_lengths,
        input_ids,
        attention_mask,
        labels,
    ):
        """Replace placeholder audio tokens with actual audio embeddings."""
        decoder_input_embs = self._inject_tensor(
            source_tensor=encoder_hidden_states,
            target_tensor=decoder_input_embs,
            inject_token_id=self.config.audio_token_id,
            input_ids=input_ids,
            source_lengths=audio_lengths,
            source_tensor_mask=encoder_outputs_mask,
        )
        if attention_mask is not None and encoder_outputs_mask is not None:
            attention_mask = self._inject_tensor(
                source_tensor=encoder_outputs_mask,
                target_tensor=attention_mask,
                inject_token_id=self.config.audio_token_id,
                input_ids=input_ids,
                source_lengths=audio_lengths,
            )
        # We don't touch labels as they are of shape [batch_size, num_labels]
        return decoder_input_embs, attention_mask, labels

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        features_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        decoder_input_embs = self._get_text_embeddings(input_ids)

        if input_features is not None:
            encoder_hidden_states, encoder_outputs_mask, audio_lengths = (
                self._get_audio_embeddings(
                    input_features=input_features,
                    features_attention_mask=features_attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    **kwargs_encoder,
                )
            )

            decoder_input_embs, attention_mask, labels = self._merge_embeddings(
                decoder_input_embs=decoder_input_embs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_outputs_mask=encoder_outputs_mask,
                audio_lengths=audio_lengths,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Delegate both logits and loss computation to the underlying sequence classifier.
        return self.text_decoder(
            inputs_embeds=decoder_input_embs,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


__all__ = [
    "MELTPreTrainedModel",
    "MELTForCausalLM",
    "MELTForSequenceClassification",
    "MELTAudioEncoder",
    "MELTAudioStack",
    "MELTAudioAdapter",
    "MELTMLPAdapter",
    "MELTQFormerAdapter",
    "MELTConformerAdapter",
]
