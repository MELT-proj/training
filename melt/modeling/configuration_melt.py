from typing import ClassVar

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


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
    has_no_defaults_at_init = True

    # No sub-configs → the recursive _attn_implementation setter is a no-op.
    # ClassVar, not a plain annotation: transformers 5 turns every
    # PretrainedConfig subclass into a dataclass, which rejects a mutable
    # default ({}) on a real field. ClassVar is what PretrainedConfig itself
    # uses for this same attribute.
    sub_configs: ClassVar[dict] = {}

    def __init__(
        self,
        _type=None,
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

        self._type = _type

        # MLP specific
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
        audio_encoder (`str`, *optional*):
            HuggingFace model identifier for the audio encoder (e.g.
            ``"facebook/wav2vec2-base"``).  Used to download the encoder config
            the *first* time a model is created.  Ignored when a pre-serialised
            ``audio_encoder_config`` is supplied (e.g. when loading from a
            checkpoint).
        text_decoder (`str`, *optional*):
            HuggingFace model identifier for the text decoder (e.g.
            ``"gpt2"``).  Same caching semantics as *audio_encoder*.
        audio_encoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder backbone.
            When provided, ``audio_encoder`` is **not** contacted via
            ``AutoConfig.from_pretrained``.
        text_decoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text decoder backbone.
        adapter_config (`Union[MELTAdapterConfig, dict]`, *optional*):
            The config object or dictionary of the audio adapter.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
    """

    model_type = "melt"
    # Transformers' ``PretrainedConfig.to_diff_dict()`` and
    # ``_get_non_default_generation_parameters()`` may attempt to instantiate
    # ``self.__class__()``.  Since MELTConfig has nested sub-configs that
    # cannot be created without model identifiers, we must allow a no-arg
    # construction.
    has_no_defaults_at_init = True
    sub_configs = {
        "audio_encoder_config": AutoConfig,
        "text_decoder_config": AutoConfig,
        "adapter_config": MELTAdapterConfig,
    }  # type: ignore

    # ------------------------------------------------------------------
    # Override the _attn_implementation property so that the parent class
    # does NOT try to propagate the value recursively into sub-configs.
    # Each sub-config should keep the attn_implementation it was created
    # with (from its pretrained config or config file).  The composite
    # model sets it explicitly after construction in modeling_melt.py.
    # ------------------------------------------------------------------
    @property
    def _attn_implementation(self):  # type: ignore[override]
        return getattr(self, "_attn_implementation_internal", None)

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        if isinstance(value, dict):
            value = value.get("", getattr(self, "_attn_implementation_internal", None))
        self._attn_implementation_internal = value

    def __init__(
        self,
        audio_encoder: str | None = None,
        text_decoder: str | None = None,
        adapter_config: dict | MELTAdapterConfig | None = None,
        audio_encoder_config: PretrainedConfig | dict | None = None,
        text_decoder_config: PretrainedConfig | dict | None = None,
        initializer_range: float = 0.02,
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        **kwargs,
    ):
        # Build sub-configs from model identifiers *only* when they are not
        # already supplied.  This avoids redundant HuggingFace downloads when
        # loading from a local checkpoint (whose ``config.json`` already
        # contains serialised sub-configs).
        if audio_encoder_config is None and audio_encoder is not None:
            audio_encoder_config = AutoConfig.from_pretrained(audio_encoder, **encoder_kwargs)
        elif isinstance(audio_encoder_config, dict):
            # Reconstruct a proper config object from a serialised dict
            # (happens during ``from_dict`` / ``from_pretrained``).
            audio_encoder_config = AutoConfig.for_model(**audio_encoder_config)

        if text_decoder_config is None and text_decoder is not None:
            text_decoder_config = AutoConfig.from_pretrained(text_decoder, **decoder_kwargs)
        elif isinstance(text_decoder_config, dict):
            text_decoder_config = AutoConfig.for_model(**text_decoder_config)

        if text_decoder_config is not None:
            # Some decoders (e.g. Qwen3.5) nest their real text config under a
            # sub-config and don't expose vocab_size/hidden_size/eos_token_id
            # etc. at the top level. get_text_config() is the standard
            # transformers method for resolving that -- and a no-op (returns
            # self) for every decoder that isn't nested this way -- so
            # flattening here once means every downstream read of
            # text_decoder_config.<attr> across the codebase just works.
            text_decoder_config = text_decoder_config.get_text_config(decoder=True)

        self.audio_encoder = audio_encoder
        self.text_decoder = text_decoder
        self.audio_encoder_config = audio_encoder_config
        self.text_decoder_config = text_decoder_config

        if adapter_config is not None and not isinstance(adapter_config, MELTAdapterConfig):
            adapter_config = MELTAdapterConfig(**adapter_config)
        self.adapter_config = adapter_config

        self.initializer_range = initializer_range

        # Set decoder-related attributes. "ForCausalLM" (not "ForCausalLMLoss")
        # is the actual key in transformers' loss registry -- both keys
        # resolve to the same loss function since the old, invalid key fell
        # back to this one with a warning; using the real key just quiets it.
        self.loss_type = "ForCausalLM"
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False):
        """Return the text decoder sub-config for generation-related helpers."""
        if self.text_decoder_config is not None:
            return self.text_decoder_config
        return super().get_text_config(decoder=decoder)

    @property
    def vocab_size(self):
        # get_text_config(), not a direct attribute read: some decoders (e.g.
        # Qwen3.5) nest their real text config under a sub-config and don't
        # expose vocab_size at the top level -- get_text_config() is the
        # standard transformers method for resolving that, and is a no-op
        # (returns self) for decoders that aren't nested this way.
        return self.text_decoder_config.get_text_config().vocab_size

    @property
    def audio_token_id(self):
        return self.text_decoder_config.audio_token_id


__all__ = ["MELTConfig", "MELTAdapterConfig"]
