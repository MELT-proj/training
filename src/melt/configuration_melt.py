from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


class MELTProjectorConfig(PretrainedConfig):
    r"""
    Configuration class for the MELT projector module.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the projector hidden layers.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the projector.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the intermediate (feed-forward) layer.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function in the projector.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for projector layers.
    """

    model_type = "melt_projector"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=2,
        intermediate_size=4096,
        hidden_act="gelu",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout


class MELTConfig(PretrainedConfig):
    r"""
    Configuration class for MELT (Multimodal Encoder Language Transformer).

    Args:
        audio_encoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder backbone.
        text_decoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text decoder backbone.
        projector_config (`MELTProjectorConfig`, *optional*):
            The config object or dictionary of the audio projector.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        has_lora_adapter (`bool`, *optional*, defaults to `False`):
            Indicates whether or not the model has a LoRA adapter.
        adapter_type (`str`, *optional*, defaults to `"mlp"`):
            Type of adapter to use for modality projection. One of: "mlp", "qformer", or "conformer".
        num_latents (`int`, *optional*, defaults to 64):
            Number of latent vectors for perceiver-style adapters.
    """

    model_type = "melt"
    sub_configs = {
        "audio_encoder_config": AutoConfig,
        "text_decoder_config": AutoConfig,
        "projector_config": MELTProjectorConfig,
    }
    is_composition = True

    def __init__(
        self,
        audio_encoder_config=None,
        text_decoder_config=None,
        projector_config=None,
        audio_token_index=32000,
        initializer_range=0.02,
        has_lora_adapter=False,
        # add_pre_adapter=False,
        # num_pre_adapter_layers=3,
        adapter_type="mlp",
        num_latents=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if audio_encoder_config is None:
            raise ValueError("audio_encoder_config must be provided")
        if text_decoder_config is None:
            raise ValueError("text_decoder_config must be provided")

        # Handle audio encoder config
        if isinstance(audio_encoder_config, dict):
            encoder_model_type = audio_encoder_config.get("model_type")
            if encoder_model_type is None:
                raise ValueError("audio_encoder_config dict must contain 'model_type'")
            print(audio_encoder_config)
            self.audio_encoder_config = AutoConfig.for_model(**audio_encoder_config)
        else:
            self.audio_encoder_config = audio_encoder_config

        # Handle text decoder config
        if isinstance(text_decoder_config, dict):
            decoder_model_type = text_decoder_config.get("model_type")
            if decoder_model_type is None:
                raise ValueError("text_decoder_config dict must contain 'model_type'")
            self.text_decoder_config = AutoConfig.for_model(**text_decoder_config)
        else:
            self.text_decoder_config = text_decoder_config

        # Handle projector config
        if not isinstance(projector_config, MELTProjectorConfig):
            projector_config = {} if projector_config is None else projector_config
            self.projector_config = MELTProjectorConfig(**projector_config)
        else:
            self.projector_config = projector_config

        self.audio_token_index = audio_token_index
        self.initializer_range = initializer_range
        self.has_lora_adapter = has_lora_adapter
        # self.add_pre_adapter = add_pre_adapter
        # self.num_pre_adapter_layers = num_pre_adapter_layers
        self.adapter_type = adapter_type
        self.num_latents = num_latents

        # Set decoder-related attributes
        self.loss_type = "ForCausalLMLoss"

    @property
    def vocab_size(self):
        return self.text_decoder_config.vocab_size


__all__ = ["MELTConfig", "MELTProjectorConfig"]
