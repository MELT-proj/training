"""Per-encoder behavioural specs.

The audio stack loads whatever ``AutoModel`` returns for ``model.encoder.name``, but
speech encoders disagree on four things that the rest of the pipeline has to know
about *before* it runs a forward pass:

1. **What the encoder eats.** w2v-BERT and Whisper take a spectrogram; HuBERT and the
   rest of the wav2vec2 family take a raw waveform and run their own conv frontend
   over it. MELT carries everything as ``(B, T, F)``, so a waveform travels as
   ``(B, n_samples, 1)`` and is squeezed back to ``(B, n_samples)`` immediately before
   the HF encoder is called. Note what this does to ``max_audio_seq_len``: for a
   waveform encoder it is a count of *samples*, not of frames.
2. **Feature layout.** ``SeamlessM4TFeatureExtractor`` (w2v-BERT) emits ``(B, T, F)``;
   ``WhisperFeatureExtractor`` emits ``(B, F, T)``. Everything in MELT is written
   against ``(B, T, F)``, so the channel-major ones are normalised at the processor
   boundary and transposed back immediately before the HF encoder is called.
3. **A fixed input window.** Whisper's encoder rejects anything that is not exactly
   ``max_source_positions * 2`` mel frames and ignores the attention mask entirely, so
   its input has to be padded to whole 30 s windows.
4. **Sequence-length change.** w2v-BERT is frame-synchronous, Whisper halves the
   sequence with a strided conv, and the wav2vec2 family reduces 16 kHz samples to
   50 Hz frames through seven chained convolutions. ``_get_audio_embeddings`` derives
   the number of audio placeholder tokens from this, so getting it wrong silently
   injects the wrong number of embeddings. For the conv frontends the arithmetic is
   floor-based and read from the checkpoint's own config, because a ratio is not
   good enough: 60 s of audio is 2999 frames, not ``ceil(960000 / 320) = 3000``.

A fifth difference splits one family down the middle: whether the attention mask is
handed to the HF encoder at all. This is a property of the *checkpoint*, not of the
family, and it follows ``feat_extract_norm`` -- the same rule the checkpoints' own
``preprocessor_config.json`` files follow. Group-norm checkpoints (wav2vec2-base,
mHuBERT-147) declare ``return_attention_mask: false`` and are documented as degrading
when masked, so MELT computes a mask for its own shape bookkeeping and does not pass it
on. Layer-norm checkpoints (facebook/mms-1b and the XLS-R line) declare it ``true`` and
genuinely need it: measured on the real ``facebook/mms-1b``, a 2 s clip batched with a
3 s one reproduces its unbatched output to 2e-5 when masked and diverges by 2.76 when
not.

The default spec reproduces the pre-existing w2v-BERT behaviour exactly, so encoders
without an entry here keep working unchanged.
"""

from dataclasses import dataclass, replace
from math import prod

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Layout of the tensor the feature extractor emits, and that the HF encoder expects.
LAYOUT_TIME_MAJOR = "tf"  # (batch, time, features) -- MELT's internal convention
LAYOUT_CHANNEL_MAJOR = "ft"  # (batch, features, time)

# What the encoder consumes.
INPUT_FEATURES = "features"  # a spectrogram-like tensor with a real feature axis
INPUT_WAVEFORM = "waveform"  # raw samples, carried through MELT as (B, n_samples, 1)


def _wants_attention_mask(encoder_config) -> bool:
    """Whether a raw-waveform checkpoint should be handed an attention mask.

    ``feat_extract_norm`` is the switch. A ``"group"`` checkpoint normalises the conv
    frontend's output over the whole (zero-padded) sequence, was pretrained on
    single-length batches with no mask, and HF's own model card guidance is to leave the
    mask off; its ``preprocessor_config.json`` says ``return_attention_mask: false`` to
    match. A ``"layer"`` checkpoint -- facebook/mms-1b, and XLS-R generally -- says
    ``true``, and measurably needs it.

    Read from the model config rather than the feature extractor because this is asked
    on the model side, where only the model config is in hand, and because the two are
    two spellings of one fact. Anything that does not say defaults to the safe answer.
    """
    return getattr(encoder_config, "feat_extract_norm", None) == "layer"


def _clamp_min_zero(lengths):
    """``max(lengths, 0)`` for python ints, ``.clamp(min=0)`` for tensors."""
    clamp = getattr(lengths, "clamp", None)
    if clamp is not None:
        return clamp(min=0)
    return max(lengths, 0)


@dataclass(frozen=True)
class EncoderSpec:
    """How one family of speech encoders differs from MELT's defaults.

    Attributes:
        feature_layout: Layout the feature extractor emits and the encoder expects.
            ``"tf"`` needs no conversion; ``"ft"`` is transposed to time-major at the
            processor boundary and back again before the encoder call.
        input_kind: ``"features"`` for a spectral frontend, ``"waveform"`` for an
            encoder that runs its own conv frontend over raw samples. A waveform is
            carried as ``(B, n_samples, 1)`` so that the rest of MELT keeps its
            ``(B, T, F)`` contract, and squeezed just before the encoder call.
        feature_key: Key the HF feature extractor puts its tensor under.
            ``Wav2Vec2FeatureExtractor`` uses ``input_values``, everything else here
            uses ``input_features``. MELT always re-emits it as ``input_features``.
        passes_attention_mask: Whether the attention mask is forwarded to the HF
            encoder. For the raw-waveform family this is not fixed by the table but
            read off the checkpoint by :meth:`for_config`; MELT still computes the mask
            for its own length bookkeeping either way.
        window_frames: Number of input frames the encoder demands, or ``None`` when it
            accepts any length. When set, the processor pads audio up to a whole
            multiple of this and ``max_audio_seq_len`` should equal it so the existing
            chunking in :class:`~melt.modeling.modeling_melt.MELTAudioEncoder` folds one
            window per forward pass.
        frame_seconds: Seconds of audio per *input* frame -- ``1 / sampling_rate`` for a
            waveform encoder. Used for memory preallocation and by
            ``infra/check_training_config.py``.
        downsample_factor: Input frames per output frame, used when the exact conv
            arithmetic is not available. ``1`` is length-preserving.
        conv_kernels / conv_strides: The encoder's conv frontend, filled in from the
            checkpoint's own config by :meth:`for_config`. When set they take
            precedence over ``downsample_factor``, because the true mapping is
            ``floor((L - kernel) / stride) + 1`` per layer and rounds *down*.
    """

    feature_layout: str = LAYOUT_TIME_MAJOR
    input_kind: str = INPUT_FEATURES
    feature_key: str = "input_features"
    passes_attention_mask: bool = True
    window_frames: int | None = None
    frame_seconds: float = 0.02
    downsample_factor: int = 1
    conv_kernels: tuple[int, ...] | None = None
    conv_strides: tuple[int, ...] | None = None

    @property
    def is_channel_major(self) -> bool:
        return self.feature_layout == LAYOUT_CHANNEL_MAJOR

    @property
    def is_waveform(self) -> bool:
        return self.input_kind == INPUT_WAVEFORM

    @property
    def total_stride(self) -> int:
        """Input frames consumed per output frame, once the frontend is running."""
        if self.conv_strides is not None:
            return prod(self.conv_strides)
        return self.downsample_factor

    def output_lengths(self, input_lengths):
        """Map a count (or tensor of counts) of input frames to output frames.

        With a conv frontend this is HF's own
        ``_get_feat_extract_output_lengths``: chained ``floor((L - k) / s) + 1``,
        floored at zero for inputs shorter than the receptive field. Otherwise it
        rounds up, matching a stride-``n`` convolution with ``padding=kernel//2``.
        Works for python ints and for integer tensors alike.
        """
        if self.conv_kernels is not None:
            lengths = input_lengths
            for kernel, stride in zip(self.conv_kernels, self.conv_strides):
                # Python's // and torch's rounding_mode="floor" agree on negatives,
                # which is what makes the clamp below the only guard needed.
                lengths = (lengths - kernel) // stride + 1
            return _clamp_min_zero(lengths)
        if self.downsample_factor == 1:
            return input_lengths
        return (input_lengths + self.downsample_factor - 1) // self.downsample_factor

    def window_seconds(self) -> float | None:
        """Duration of one fixed encoder window, or ``None`` when unbounded."""
        if self.window_frames is None:
            return None
        return self.window_frames * self.frame_seconds

    def for_config(self, encoder_config) -> "EncoderSpec":
        """Specialise this spec with what only the checkpoint's own config knows.

        The table is keyed on ``model_type``, but the conv frontend is a property of
        the individual checkpoint, so it is read here rather than hardcoded per family.

        Two things are read here. The conv frontend, because the length arithmetic
        needs the exact kernels and strides. And whether the encoder is masked, because
        that splits the raw-waveform family rather than following it: it tracks
        ``feat_extract_norm``, the same signal the checkpoints' own
        ``preprocessor_config.json`` uses for ``return_attention_mask``.

        Raises:
            ValueError: if a waveform encoder's config does not describe its conv
                frontend, which would leave the length arithmetic guessing.
        """
        if not self.is_waveform:
            return self
        kernels = getattr(encoder_config, "conv_kernel", None)
        strides = getattr(encoder_config, "conv_stride", None)
        if not kernels or not strides or len(kernels) != len(strides):
            raise ValueError(
                f"{type(encoder_config).__name__} is registered as a raw-waveform "
                "encoder but does not describe its conv frontend "
                "(conv_kernel/conv_stride), so the number of audio embeddings it emits "
                "cannot be computed. Check the encoder entry in "
                "melt/modeling/encoder_specs.py."
            )
        return replace(
            self,
            conv_kernels=tuple(kernels),
            conv_strides=tuple(strides),
            passes_attention_mask=_wants_attention_mask(encoder_config),
        )

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

# Every encoder built on ``Wav2Vec2FeatureExtractor``: a raw waveform in and a conv
# frontend that takes 16 kHz down to 50 Hz. Two fields here are only placeholders that
# `for_config` replaces from the checkpoint itself -- `downsample_factor` with the exact
# conv arithmetic, and `passes_attention_mask` with the checkpoint's `feat_extract_norm`.
# The conservative `False` is what a config too sparse to say would otherwise get.
WAVEFORM_ENCODER_SPEC = EncoderSpec(
    input_kind=INPUT_WAVEFORM,
    feature_key="input_values",
    passes_attention_mask=False,
    frame_seconds=1 / 16_000,
    downsample_factor=320,
)

# Keyed on ``config.model_type``. `wav2vec2-bert` deliberately has no entry: the default
# *is* its behaviour, which is what keeps the existing arm bit-for-bit unchanged. Note
# that it is a different `model_type` string from `wav2vec2`, so the family entries
# below do not capture it.
ENCODER_SPECS: dict[str, EncoderSpec] = {
    "whisper": EncoderSpec(
        feature_layout=LAYOUT_CHANNEL_MAJOR,
        # max_source_positions (1500) * conv2.stride (2). WhisperEncoder.forward raises
        # a ValueError on anything else and ignores the attention mask.
        window_frames=3000,
        frame_seconds=0.01,  # hop_length 160 / 16 kHz, no stride-stacking
        downsample_factor=2,  # conv2 has stride 2: 3000 mel frames -> 1500 positions
    ),
    "hubert": WAVEFORM_ENCODER_SPEC,
    "wav2vec2": WAVEFORM_ENCODER_SPEC,
    "wavlm": WAVEFORM_ENCODER_SPEC,
    "data2vec-audio": WAVEFORM_ENCODER_SPEC,
    "unispeech-sat": WAVEFORM_ENCODER_SPEC,
}

# The processor only holds a feature extractor, never a model config, so it resolves the
# same spec through the extractor class name. `hubert` stands in for the whole wav2vec2
# family here: the processor-side behaviour (waveform in, `input_values` out) is
# identical across it, and the parts that do differ per checkpoint are model-side.
FEATURE_EXTRACTOR_TO_MODEL_TYPE: dict[str, str] = {
    "WhisperFeatureExtractor": "whisper",
    "Wav2Vec2FeatureExtractor": "hubert",
}


def get_encoder_spec(model_type: str | None) -> EncoderSpec:
    """Look up the spec for an encoder ``model_type``, falling back to the default."""
    if model_type is None:
        return DEFAULT_ENCODER_SPEC
    return ENCODER_SPECS.get(model_type, DEFAULT_ENCODER_SPEC)


def get_encoder_spec_for_config(encoder_config) -> EncoderSpec:
    """Look up the spec for an encoder's HF config object, specialised to it."""
    spec = get_encoder_spec(getattr(encoder_config, "model_type", None))
    return spec.for_config(encoder_config)


def get_encoder_spec_for_feature_extractor(feature_extractor) -> EncoderSpec:
    """Look up the spec that matches a feature extractor instance.

    Not specialised: a feature extractor carries no conv frontend, so the returned
    spec's ``output_lengths`` is only as good as ``downsample_factor``. Callers that
    need exact lengths go through :func:`get_encoder_spec_for_config`, or through
    :class:`~melt.modeling.modeling_melt.MELTAudioEncoder`, which is chunk-aware too.
    """
    model_type = FEATURE_EXTRACTOR_TO_MODEL_TYPE.get(type(feature_extractor).__name__)
    return get_encoder_spec(model_type)
