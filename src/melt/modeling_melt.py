# coding=utf-8
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
            getattr(config.audio_encoder_config, "hidden_size", config.audio_encoder_config.d_model),
        )
        text_hidden_size = config.text_decoder_config.hidden_size
        self.linear = nn.Linear(audio_hidden_size, text_hidden_size, bias=True)

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

        self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.projector_config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)

        # Q-Former model from config (typically blip_2_qformer)
        self.qformer = AutoModel.from_config(config.projector_config)

        # Final projection to text decoder hidden size
        self.linear = nn.Linear(config.projector_config.hidden_size, config.text_decoder_config.hidden_size)

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
        self.layers = nn.ModuleList(Wav2Vec2BertAdapterLayer(encoder_config) for _ in range(num_adapter_layers))
        self.layerdrop = getattr(encoder_config, "layerdrop", 0.0)

        self.kernel_size = getattr(encoder_config, "adapter_kernel_size", 3)
        self.stride = getattr(encoder_config, "adapter_stride", 2)

        # Final projection to text decoder hidden size
        adapter_output_size = output_hidden_size
        text_hidden_size = config.text_decoder_config.hidden_size
        if adapter_output_size != text_hidden_size:
            self.out_proj = nn.Linear(adapter_output_size, text_hidden_size)
        else:
            self.out_proj = None

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

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

        # Initialize the audio encoder
        self.audio_encoder = AutoModel.from_config(config.audio_encoder_config)

        # Validate that the encoder doesn't have an LM head
        if self.audio_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The audio encoder {self.audio_encoder} should not have a LM Head. "
                "Please use a model without LM Head."
            )

        # Initialize the audio adapter (projector)
        self.audio_adapter = MELTAudioAdapter(config)

        # Sync attention implementation between config and models
        self.config.audio_encoder_config._attn_implementation = self.audio_encoder.config._attn_implementation
        self.config.text_decoder_config._attn_implementation = self.text_decoder.config._attn_implementation
        self.audio_encoder.config = self.config.audio_encoder_config
        self.text_decoder.config = self.config.text_decoder_config

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.audio_encoder

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
        encoder_outputs = self.audio_encoder(input_features, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs[0]
        projected_features = self.audio_adapter(encoder_hidden_states, attention_mask=attention_mask)
        return projected_features

    def freeze_audio_encoder(self):
        """
        Freeze the audio encoder parameters.
        This disables gradient computation for the encoder so its parameters won't be updated during training.
        """
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def freeze_text_decoder(self):
        """
        Freeze the text decoder parameters.
        This disables gradient computation for the decoder so its parameters won't be updated during training.
        """
        for param in self.text_decoder.parameters():
            param.requires_grad = False

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
        audio_input_features: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        audio_attention_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
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

        # we assume that if we are using cache then we are caching encoder_outputs
        if not use_cache or (use_cache and past_key_values is None):
            encoder_outputs = self.audio_encoder(
                audio_input_features,
                attention_mask=audio_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_hidden_states = encoder_outputs[0]

            # Project through audio adapter
            encoder_hidden_states = self.audio_adapter(encoder_hidden_states, attention_mask=audio_attention_mask)

            if audio_attention_mask is not None:
                encoder_outputs_mask = self.audio_encoder._get_feature_vector_attention_mask(
                    encoder_hidden_states.shape[1], audio_attention_mask
                )
            else:
                encoder_outputs_mask = torch.ones(
                    encoder_hidden_states.shape[:2],
                    dtype=attention_mask.dtype if attention_mask is not None else torch.float32,
                    device=encoder_hidden_states.device,
                )

        # extract input embeds from the decoder
        decoder_input_embs = self.text_decoder.get_input_embeddings()(input_ids)

        # If we are not using the cache, or it's the first pass with the cache on.
        # Hence, we need to build new inputs for the decoder
        if not use_cache or (use_cache and past_key_values is None):
            # prepend audio representations to the text input embeddings
            decoder_input_embs = torch.cat([encoder_hidden_states, decoder_input_embs], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat([encoder_outputs_mask, attention_mask], dim=1)

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
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

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
    "MELTAudioAdapter",
    "MELTMLPAdapter",
    "MELTQFormerAdapter",
    "MELTConformerAdapter",
]
