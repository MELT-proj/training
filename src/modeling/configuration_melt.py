from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


# Required special token names (no hard-coded defaults).
# These are the token *names* the model config is expected to provide under
# `model.config.decoder.<name>` or that must already exist in the tokenizer.
MELT_REQUIRED_SPECIAL_TOKENS = [
    "audio_token",
    "audio_bos_token",
    "audio_eos_token",
]


class MELTAdapterConfig(PretrainedConfig):
    r"""
    Configuration class for MELT adapters (MLP, Q-Former, Conformer).

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of adapter hidden layers (used by Q-Former and other adapters).
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers (if applicable).
        intermediate_size (`int`, *optional*, defaults to 4096):
            Intermediate (FFN) size for adapters that need it.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        downsample_rate (`int`, *optional*, defaults to 5):
            Q-Former downsample rate (used by Q-Former adapter).
        window_size (`int`, *optional*, defaults to 15):
            Q-Former window size (used by Q-Former adapter).
    """

    model_type = "melt_adapter"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=2,
        intermediate_size=4096,
        hidden_act="gelu",
        dropout=0.1,
        downsample_rate=5,
        window_size=15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        # Q-Former specific
        self.downsample_rate = downsample_rate
        self.window_size = window_size


class MELTConfig(PretrainedConfig):
    r"""
    Configuration class for MELT (Multimodal Encoder Language Transformer).

    Args:
        audio_encoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder backbone.
        text_decoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text decoder backbone.
        adapter_config (`MELTAdapterConfig`, *optional*):
            The config object or dictionary of the audio adapter.

        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        has_lora_adapter (`bool`, *optional*, defaults to `False`):
            Indicates whether or not the model has a LoRA adapter.
        adapter_type (`str`, *optional*, defaults to `"mlp"`):
            Type of adapter to use for modality projection. One of: "mlp", "qformer", or "conformer".
        num_latents (`int`, *optional*, defaults to 64):
            Number of latent vectors for perceiver-style adapters.
        max_audio_seq_len (`int`, *optional*, defaults to 500):
            Maximum sequence length for audio features before chunking is applied.
            Sequences longer than this will be split into chunks.
    """

    model_type = "melt"
    # Transformers' `PretrainedConfig.to_diff_dict()` will attempt to instantiate
    # `self.__class__()` to compute defaults unless we opt out. Since MELTConfig
    # requires nested sub-configs at init, we must disable that behavior.
    has_no_defaults_at_init = True
    sub_configs = {
        "audio_encoder_config": AutoConfig,
        "text_decoder_config": AutoConfig,
        "adapter_config": MELTAdapterConfig,
    }  # type: ignore
    is_composition = True

    def __init__(
        self,
        audio_encoder_config=None,
        text_decoder_config=None,
        adapter_config=None,
        initializer_range=0.02,
        has_lora_adapter=False,
        adapter_type="mlp",
        num_latents=64,
        max_audio_seq_len=1500,  # for frames of 20ms, this is 30s
        **kwargs,
    ):
        if audio_encoder_config is None:
            raise ValueError("audio_encoder_config must be provided")
        if text_decoder_config is None:
            raise ValueError("text_decoder_config must be provided")

        # Handle audio encoder config
        if isinstance(audio_encoder_config, dict):
            encoder_model_type = audio_encoder_config.get("model_type")
            if encoder_model_type is None:
                raise ValueError("audio_encoder_config dict must contain 'model_type'")
            encoder_cfg = dict(audio_encoder_config)
            encoder_cfg.pop("model_type", None)
            self.audio_encoder_config = AutoConfig.for_model(encoder_model_type, **encoder_cfg)
        else:
            self.audio_encoder_config = audio_encoder_config

        # Handle text decoder config
        if isinstance(text_decoder_config, dict):
            decoder_model_type = text_decoder_config.get("model_type")
            if decoder_model_type is None:
                raise ValueError("text_decoder_config dict must contain 'model_type'")
            decoder_cfg = dict(text_decoder_config)
            decoder_cfg.pop("model_type", None)
            self.text_decoder_config = AutoConfig.for_model(decoder_model_type, **decoder_cfg)
        else:
            self.text_decoder_config = text_decoder_config

        # Handle adapter config
        if not isinstance(adapter_config, MELTAdapterConfig):
            adapter_cfg = {} if adapter_config is None else adapter_config
            # If adapter_cfg is a dict and contains a 'type' entry, prefer that as adapter_type
            if isinstance(adapter_cfg, dict) and "type" in adapter_cfg:
                adapter_type = adapter_cfg.get("type", adapter_type)
            self.adapter_config = MELTAdapterConfig(**adapter_cfg)
        else:
            self.adapter_config = adapter_config

        self.initializer_range = initializer_range
        self.has_lora_adapter = has_lora_adapter
        self.adapter_type = adapter_type
        self.num_latents = num_latents

        # Audio encoder chunking settings
        self.max_audio_seq_len = max_audio_seq_len

        # Set decoder-related attributes
        self.loss_type = "ForCausalLMLoss"
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self.text_decoder_config.vocab_size


__all__ = ["MELTConfig", "MELTAdapterConfig"]
