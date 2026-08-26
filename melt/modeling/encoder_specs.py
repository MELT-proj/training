"""Per-encoder behavioural specs.

The audio stack loads whatever ``AutoModel`` returns for ``model.encoder.name``, but
speech encoders disagree on three things that the rest of the pipeline has to know
about *before* it runs a forward pass:

1. **Feature layout.** ``SeamlessM4TFeatureExtractor`` (w2v-BERT) emits ``(B, T, F)``;
   ``WhisperFeatureExtractor`` emits ``(B, F, T)``. Everything in MELT is written
   against ``(B, T, F)``, so the channel-major ones are normalised at the processor
   boundary and transposed back immediately before the HF encoder is called.
2. **A fixed input window.** Whisper's encoder rejects anything that is not exactly
   ``max_source_positions * 2`` mel frames and ignores the attention mask entirely, so
   its input has to be padded to whole 30 s windows.
3. **Sequence-length change.** w2v-BERT is frame-synchronous, Whisper halves the
   sequence with a strided conv. ``_get_audio_embeddings`` derives the number of audio
   placeholder tokens from this, so getting it wrong silently injects the wrong number
   of embeddings.

The default spec reproduces the pre-existing w2v-BERT behaviour exactly, so encoders
without an entry here keep working unchanged.
"""

from dataclasses import dataclass

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Layout of the tensor the feature extractor emits, and that the HF encoder expects.
LAYOUT_TIME_MAJOR = "tf"  # (batch, time, features) -- MELT's internal convention
LAYOUT_CHANNEL_MAJOR = "ft"  # (batch, features, time)


@dataclass(frozen=True)
class EncoderSpec:
    """How one family of speech encoders differs from MELT's defaults.

    Attributes:
        feature_layout: Layout the feature extractor emits and the encoder expects.
            ``"tf"`` needs no conversion; ``"ft"`` is transposed to time-major at the
            processor boundary and back again before the encoder call.
        window_frames: Number of input frames the encoder demands, or ``None`` when it
            accepts any length. When set, the processor pads audio up to a whole
            multiple of this and ``max_audio_seq_len`` should equal it so the existing
            chunking in :class:`~melt.modeling.modeling_melt.MELTAudioEncoder` folds one
            window per forward pass.
        frame_seconds: Seconds of audio per *input* frame. Used for memory
            preallocation and by ``infra/check_training_config.py``.
        downsample_factor: Input frames per output frame. ``1`` is length-preserving.
    """

    feature_layout: str = LAYOUT_TIME_MAJOR
    window_frames: int | None = None
    frame_seconds: float = 0.02
    downsample_factor: int = 1

    @property
    def is_channel_major(self) -> bool:
        return self.feature_layout == LAYOUT_CHANNEL_MAJOR

    def output_lengths(self, input_lengths):
        """Map a count (or tensor of counts) of input frames to output frames.

        Rounds up, matching a stride-``n`` convolution with ``padding=kernel//2``.
        Works for python ints and for integer tensors alike.
        """
        if self.downsample_factor == 1:
            return input_lengths
        return (input_lengths + self.downsample_factor - 1) // self.downsample_factor

    def window_seconds(self) -> float | None:
        """Duration of one fixed encoder window, or ``None`` when unbounded."""
        if self.window_frames is None:
            return None
        return self.window_frames * self.frame_seconds

    def unwrap(self, model):
        """Reduce whatever ``AutoModel`` returned to the encoder we actually train on.

        Encoder-decoder checkpoints (Whisper) resolve to the full model under
        ``AutoModel``; keeping the decoder would cost parameters and memory for nothing.
        """
        if self.window_frames is not None and hasattr(model, "get_encoder"):
            encoder = model.get_encoder()
            if encoder is not model:
                logger.info(
                    "Discarding the decoder of %s; keeping %s as the audio encoder.",
                    type(model).__name__,
                    type(encoder).__name__,
                )
            return encoder
        return model


DEFAULT_ENCODER_SPEC = EncoderSpec()

# Keyed on ``config.model_type``. `wav2vec2-bert` deliberately has no entry: the default
# *is* its behaviour, which is what keeps the existing arm bit-for-bit unchanged.
ENCODER_SPECS: dict[str, EncoderSpec] = {
    "whisper": EncoderSpec(
        feature_layout=LAYOUT_CHANNEL_MAJOR,
        # max_source_positions (1500) * conv2.stride (2). WhisperEncoder.forward raises
        # a ValueError on anything else and ignores the attention mask.
        window_frames=3000,
        frame_seconds=0.01,  # hop_length 160 / 16 kHz, no stride-stacking
        downsample_factor=2,  # conv2 has stride 2: 3000 mel frames -> 1500 positions
    ),
}

# The processor only holds a feature extractor, never a model config, so it resolves the
# same spec through the extractor class name.
FEATURE_EXTRACTOR_TO_MODEL_TYPE: dict[str, str] = {
    "WhisperFeatureExtractor": "whisper",
}


def get_encoder_spec(model_type: str | None) -> EncoderSpec:
    """Look up the spec for an encoder ``model_type``, falling back to the default."""
    if model_type is None:
        return DEFAULT_ENCODER_SPEC
    return ENCODER_SPECS.get(model_type, DEFAULT_ENCODER_SPEC)


def get_encoder_spec_for_config(encoder_config) -> EncoderSpec:
    """Look up the spec for an encoder's HF config object."""
    return get_encoder_spec(getattr(encoder_config, "model_type", None))


def get_encoder_spec_for_feature_extractor(feature_extractor) -> EncoderSpec:
    """Look up the spec that matches a feature extractor instance."""
    model_type = FEATURE_EXTRACTOR_TO_MODEL_TYPE.get(type(feature_extractor).__name__)
    return get_encoder_spec(model_type)
