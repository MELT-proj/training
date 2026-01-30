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
        num_adapter_layers (`int`, *optional*, defaults to 1):
            Number of conformer adapter layers (used by Conformer adapter).
        layerdrop (`float`, *optional*, defaults to 0.0):
            Layer drop probability for conformer layers (used by Conformer adapter).
        adapter_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for conformer convolutions (used by Conformer adapter).
        adapter_stride (`int`, *optional*, defaults to 2):
            Stride for conformer convolutions (used by Conformer adapter).
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
        num_adapter_layers=1,
        layerdrop=0.0,
        adapter_kernel_size=3,
        adapter_stride=2,
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
        # Conformer specific
        self.num_adapter_layers = num_adapter_layers
        self.layerdrop = layerdrop
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride


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
        adapter_type (`str`, *optional*, defaults to `"mlp"`):
            Type of adapter to use for modality projection. One of: "mlp", "qformer", or "conformer".
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
        audio_encoder: str,
        text_decoder: str,
        adapter_config: MELTAdapterConfig,
        initializer_range: float = 0.02,
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        **kwargs,
    ):
        audio_config = AutoConfig.from_pretrained(audio_encoder, **encoder_kwargs)
        text_config = AutoConfig.from_pretrained(text_decoder, **decoder_kwargs)

        self.audio_encoder = audio_encoder
        self.text_decoder = text_decoder
        self.audio_encoder_config = audio_config
        self.text_decoder_config = text_config

        # Handle adapter config
        if not isinstance(adapter_config, MELTAdapterConfig):
            # Accept mapping, dataclass (e.g., Training's AdapterConfig), or namespace-like objects
            if adapter_config is None:
                adapter_cfg = {}
            elif isinstance(adapter_config, dict):
                adapter_cfg = dict(adapter_config)
            else:
                # Defer importing dataclasses utilities lazily to avoid import cycles
                from dataclasses import asdict, is_dataclass

                if is_dataclass(adapter_config):
                    adapter_cfg = asdict(adapter_config)
                elif hasattr(adapter_config, "__dict__"):
                    adapter_cfg = dict(adapter_config.__dict__)
                else:
                    raise TypeError(
                        "adapter_config must be a MELTAdapterConfig, dict, or dataclass/namespace-like object"
                    )

            # Extract user-provided adapter `type` (mlp/qformer/conformer) if present.
            adapter_type = adapter_cfg.pop("type", getattr(self, "adapter_type", "mlp"))
            self.adapter_type = adapter_type

            # Initialize MELTAdapterConfig with remaining adapter kwargs
            self.adapter_config = MELTAdapterConfig(**adapter_cfg)
        else:
            self.adapter_config = adapter_config
            # Ensure adapter_type attribute exists and defaults to 'mlp' if not present
            self.adapter_type = getattr(self, "adapter_type", "mlp")

        self.initializer_range = initializer_range

        # Set decoder-related attributes
        self.loss_type = "ForCausalLMLoss"
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self.text_decoder_config.vocab_size


__all__ = ["MELTConfig", "MELTAdapterConfig"]
