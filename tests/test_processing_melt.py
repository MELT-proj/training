import json
import os
import tempfile
import urllib.request

import librosa
import torch
import pytest

from melt.modeling import MELT_REQUIRED_SPECIAL_TOKENS, MELTProcessor
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer


# Every test here builds a real tokenizer or feature extractor, and the audio
# fixture is fetched over the network too, so the whole module needs the
# HuggingFace Hub (or a warm cache). GitHub CI deselects it.
pytestmark = pytest.mark.hub


# Audio sample URL for testing
AUDIO_SAMPLE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"


@pytest.fixture(scope="module")
def audio_sample():
    """Load audio sample from URL using librosa."""
    # Download the audio file to a temporary file
    with urllib.request.urlopen(AUDIO_SAMPLE_URL) as response:
        audio_bytes = response.read()

    # Write to a temporary file (librosa needs a file path or file-like object)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    # Load with librosa
    audio, sr = librosa.load(tmp_path, sr=16000)

    return audio


@pytest.fixture(scope="module")
def feature_extractor():
    """Load a feature extractor for testing."""
    return AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


# Token strings used throughout the test suite — match the production config.
MELT_SPECIAL_TOKEN_STRINGS: dict[str, str] = {
    "audio_token": "<|audio|>",
    "audio_bos_token": "<|audio_bos|>",
    "audio_eos_token": "<|audio_eos|>",
}


@pytest.fixture(scope="module")
def tokenizer():
    """Load a tokenizer for testing with MELT special tokens pre-registered.

    Mirrors :func:`melt.training.setup.prepare_processor`: ``extra_special_tokens``
    is passed to :meth:`AutoTokenizer.from_pretrained` so that the named attributes
    are registered, and then :meth:`~PreTrainedTokenizerBase.add_special_tokens`
    inserts them into the vocabulary (and sets ``additional_special_tokens``).
    """
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        use_fast=True,
        extra_special_tokens=MELT_SPECIAL_TOKEN_STRINGS,
    )
    tok.add_special_tokens(MELT_SPECIAL_TOKEN_STRINGS)
    return tok


@pytest.fixture(scope="module")
def processor(feature_extractor, tokenizer):
    """Create a MELTProcessor instance with special tokens already on the tokenizer."""
    return MELTProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


class TestMELTProcessorInit:
    def test_init_with_required_components(self, feature_extractor, tokenizer):
        # tokenizer fixture already has MELT special tokens pre-configured
        processor = MELTProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        assert processor.feature_extractor is not None
        assert processor.tokenizer is not None

    def test_init_without_required_special_tokens_raises(self, feature_extractor):
        """MELTProcessor must raise RuntimeError when special tokens are missing."""
        fresh_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        with pytest.raises(RuntimeError):
            MELTProcessor(feature_extractor=feature_extractor, tokenizer=fresh_tokenizer)

    def test_init_without_tokenizer_raises(self, feature_extractor, tokenizer):
        with pytest.raises(Exception):
            MELTProcessor(feature_extractor=feature_extractor, tokenizer=None)

    def test_special_tokens_added(self, processor):
        vocab = processor.tokenizer.get_vocab()
        for token_name in MELT_REQUIRED_SPECIAL_TOKENS:
            token_value = getattr(processor, token_name)
            assert token_value in vocab, f"Token {token_value} not found in vocabulary"

    def test_token_attributes_set(self, processor):
        # The processor guarantees audio-related tokens exist; image/video/vision
        # tokens are optional in this implementation and may not be set.
        assert processor.audio_token == "<|audio|>"
        assert processor.audio_bos_token == "<|audio_bos|>"
        assert processor.audio_eos_token == "<|audio_eos|>"


class TestMELTProcessorTextOnly:
    def test_text_only_processing(self, processor):
        text = "Hello, this is a test."
        result = processor(text=text)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_text_only_batch_processing(self, processor):
        texts = ["Hello, this is a test.", "Another test sentence."]
        result = processor(text=texts)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_text_none_raises(self, processor):
        with pytest.raises(ValueError, match="You need to specify a `text` input"):
            processor(text=None)

    def test_decode(self, processor):
        text = "Hello, world!"
        encoded = processor(text=text)
        decoded = processor.decode(encoded["input_ids"][0])
        assert "Hello" in decoded
        assert "world" in decoded

    def test_batch_decode(self, processor):
        texts = ["Hello!", "World!"]
        encoded = processor(text=texts, padding=True)
        decoded = processor.batch_decode(encoded["input_ids"], skip_special_tokens=True)
        assert len(decoded) == 2


class TestMELTProcessorAudio:
    def test_audio_processing(self, processor, audio_sample):
        text = f"Transcribe the following audio: {processor.audio_token}"
        result = processor(text=text, audio=audio_sample)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result

    def test_audio_batch_processing(self, processor, audio_sample):
        texts = [
            f"Audio 1: {processor.audio_token}",
            f"Audio 2: {processor.audio_token}",
        ]
        audios = [[audio_sample], [audio_sample]]
        result = processor(text=texts, audio=audios)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result

    def test_audio_with_return_tensors(self, processor, audio_sample):
        text = f"Transcribe: {processor.audio_token}"
        result = processor(text=text, audio=audio_sample, return_tensors="pt")

        assert result["input_ids"].dim() == 2
        assert result["attention_mask"].dim() == 2

    def test_audio_token_expansion(self, processor, audio_sample):
        text = f"Before {processor.audio_token} after"
        result = processor(text=text, audio=audio_sample)

        # The audio token should be expanded to multiple tokens
        decoded = processor.decode(result["input_ids"][0])
        # Check that the audio token appears multiple times (expanded)
        assert processor.audio_token in decoded


class TestMELTProcessorSpecialTokens:
    def test_audio_bos_eos_tokens_in_vocab(self, processor):
        vocab = processor.tokenizer.get_vocab()
        assert processor.audio_bos_token in vocab
        assert processor.audio_eos_token in vocab

    def test_text_with_special_tokens(self, processor):
        text = f"{processor.audio_bos_token}Some text{processor.audio_eos_token}"
        result = processor(text=text)

        decoded = processor.decode(result["input_ids"][0])
        assert processor.audio_bos_token in decoded
        assert processor.audio_eos_token in decoded


class TestMELTProcessorModelInputNames:
    def test_model_input_names(self, processor):
        input_names = processor.model_input_names
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "features_attention_mask" in input_names


class TestMELTProcessorChatTemplate:
    """Tests for MELTProcessor with chat templates."""

    def test_chat_template_text_only(self, processor):
        """Test processing text with chat template applied."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
        # Apply chat template
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = processor(text=text)

        assert "input_ids" in result
        assert "attention_mask" in result
        # Verify the text was processed correctly
        decoded = processor.decode(result["input_ids"][0])
        assert "Hello" in decoded or "hello" in decoded.lower()

    def test_chat_template_with_audio(self, processor, audio_sample):
        """Test processing chat messages containing audio tokens."""
        audio_token = processor.audio_token
        messages = [
            {"role": "user", "content": f"Transcribe the following audio: {audio_token}"},
        ]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = processor(text=text, audio=audio_sample)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result

    def test_chat_template_multi_turn_with_audio(self, processor, audio_sample):
        """Test multi-turn conversation with audio."""
        audio_token = processor.audio_token
        messages = [
            {"role": "user", "content": f"Listen to this audio: {audio_token}"},
            {"role": "assistant", "content": "I heard someone speaking."},
            {"role": "user", "content": "What did they say exactly?"},
        ]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = processor(text=text, audio=audio_sample)

        assert "input_ids" in result
        assert "input_features" in result

    def test_without_chat_template(self, processor, audio_sample):
        """Test processing raw text without chat template."""
        text = f"Transcribe: {processor.audio_token}"
        result = processor(text=text, audio=audio_sample)

        assert "input_ids" in result
        assert "input_features" in result
        # Should work without any special formatting
        decoded = processor.decode(result["input_ids"][0])
        assert "Transcribe" in decoded


class TestMELTProcessorMultipleAudios:
    """Tests for MELTProcessor with multiple audio inputs per sample."""

    def test_multiple_audio_tokens_single_sample(self, processor, audio_sample):
        """Test a single sample with multiple audio tokens."""
        audio_token = processor.audio_token
        text = f"{audio_token} What is said here? {audio_token} And in this one? {audio_token} Summarize all."
        audios = [audio_sample, audio_sample, audio_sample]

        result = processor(text=text, audio=audios)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result
        # Single text sample => batch size 1; audios concatenated on time dim
        assert result["input_features"].shape[0] == 1
        assert "audio_lengths" in result
        assert len(result["audio_lengths"]) == 3

    def test_multiple_audio_tokens_with_chat_template(self, processor, audio_sample):
        """Test multiple audio tokens with chat template."""
        audio_token = processor.audio_token
        messages = [
            {
                "role": "user",
                "content": f"{audio_token} What is said here? {audio_token} And in this one? {audio_token} Summarize all.",
            },
        ]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios = [audio_sample, audio_sample, audio_sample]

        result = processor(text=text, audio=audios)

        assert "input_ids" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 1
        assert "audio_lengths" in result
        assert len(result["audio_lengths"]) == 3

    def test_two_audio_tokens(self, processor, audio_sample):
        """Test with exactly two audio tokens."""
        audio_token = processor.audio_token
        text = f"Compare {audio_token} with {audio_token}"
        audios = [audio_sample, audio_sample]

        result = processor(text=text, audio=audios)

        assert "input_ids" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 1
        assert "audio_lengths" in result
        assert len(result["audio_lengths"]) == 2

    def test_audio_token_expansion_multiple(self, processor, audio_sample):
        """Test that multiple audio tokens are each expanded correctly."""
        audio_token = processor.audio_token
        text = f"First: {audio_token} Second: {audio_token}"
        audios = [audio_sample, audio_sample]

        result = processor(text=text, audio=audios)
        decoded = processor.decode(result["input_ids"][0])

        # Both "First:" and "Second:" should appear in the decoded text
        assert "First" in decoded
        assert "Second" in decoded
        # Audio tokens should be expanded
        assert decoded.count(audio_token) >= 2

    def test_batch_with_different_audio_counts(self, processor, audio_sample):
        """Test batch processing where samples have different numbers of audios."""
        audio_token = processor.audio_token
        # Sample 1 has 1 audio, Sample 2 has 2 audios
        texts = [
            f"Single audio: {audio_token}",
            f"Two audios: {audio_token} and {audio_token}",
        ]
        audios = [[audio_sample], [audio_sample, audio_sample]]

        result = processor(text=texts, audio=audios, padding=True)

        assert "input_ids" in result
        assert "input_features" in result
        # Two samples in the batch
        assert result["input_features"].shape[0] == 2
        assert "audio_lengths" in result
        assert len(result["audio_lengths"]) == 2

    def test_multiple_audios_preserves_text_structure(self, processor, audio_sample):
        """Test that text structure is preserved with multiple audios."""
        audio_token = processor.audio_token
        text = f"Before first {audio_token} middle text {audio_token} after second"
        audios = [audio_sample, audio_sample]

        result = processor(text=text, audio=audios)
        decoded = processor.decode(result["input_ids"][0])

        assert "Before first" in decoded
        assert "middle text" in decoded
        assert "after second" in decoded

    def test_three_audios_with_questions(self, processor, audio_sample):
        """Test the exact use case from the user request."""
        audio_token = processor.audio_token
        text = f"{audio_token} What is said here? {audio_token} And in this one? {audio_token} Summarize all."
        audios = [audio_sample, audio_sample, audio_sample]

        result = processor(text=text, audio=audios)

        assert "input_ids" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 1
        assert "audio_lengths" in result
        assert len(result["audio_lengths"]) == 3

        decoded = processor.decode(result["input_ids"][0])
        assert "What is said here" in decoded
        assert "And in this one" in decoded
        assert "Summarize all" in decoded

    def test_batch_multi_and_single_audio(self, processor, audio_sample):
        """Test batch of 2 items: one with multiple audios, one with single audio."""
        audio_token = processor.audio_token
        # Sample 1: 3 audios
        text1 = f"{audio_token} What is said here? {audio_token} And in this one? {audio_token} Summarize all."
        # Sample 2: 1 audio
        text2 = f"Transcribe: {audio_token}"

        texts = [text1, text2]
        audios = [[audio_sample, audio_sample, audio_sample], [audio_sample]]

        result = processor(text=texts, audio=audios, padding=True)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 2
        assert len(result["input_ids"]) == 2

    def test_batch_single_and_multi_audio(self, processor, audio_sample):
        """Test batch of 2 items: first with single audio, second with multiple audios."""
        audio_token = processor.audio_token
        # Sample 1: 1 audio
        text1 = f"Single audio sample: {audio_token}"
        # Sample 2: 2 audios
        text2 = f"Compare {audio_token} with {audio_token}"

        texts = [text1, text2]
        audios = [[audio_sample], [audio_sample, audio_sample]]

        result = processor(text=texts, audio=audios, padding=True)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 2
        assert len(result["input_ids"]) == 2

        # Verify text content is preserved for both samples
        decoded_1 = processor.decode(result["input_ids"][0])
        decoded_2 = processor.decode(result["input_ids"][1])
        assert "Single audio sample" in decoded_1
        assert "Compare" in decoded_2

    def test_batch_multi_audio_with_chat_template(self, processor, audio_sample):
        """Test batch processing with chat template and different audio counts."""
        audio_token = processor.audio_token

        # Sample 1: multi-audio with chat template
        messages1 = [
            {
                "role": "user",
                "content": f"{audio_token} Describe this. {audio_token} And this.",
            },
        ]
        text1 = processor.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)

        # Sample 2: single audio with chat template
        messages2 = [
            {"role": "user", "content": f"What do you hear? {audio_token}"},
        ]
        text2 = processor.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)

        texts = [text1, text2]
        audios = [[audio_sample, audio_sample], [audio_sample]]

        result = processor(text=texts, audio=audios, padding=True)

        assert "input_ids" in result
        assert "input_features" in result
        assert result["input_features"].shape[0] == 2
        assert len(result["input_ids"]) == 2


class TestMELTProcessorSaveLoad:
    """Tests for MELTProcessor save_pretrained / from_pretrained round-trip."""

    def test_special_tokens_top_level_in_tokenizer_config(self, processor):
        """MELT special tokens must appear as top-level keys in tokenizer_config.json."""
        with tempfile.TemporaryDirectory() as d:
            processor.save_pretrained(d)
            tc = json.load(open(os.path.join(d, "tokenizer_config.json")))
            for name in MELT_REQUIRED_SPECIAL_TOKENS:
                assert name in tc, f"'{name}' missing from tokenizer_config.json top-level keys"
                assert tc[name] == getattr(processor, name)

    def test_special_tokens_in_extra_special_tokens(self, processor):
        """MELT special tokens must be in extra_special_tokens after a round-trip."""
        with tempfile.TemporaryDirectory() as d:
            processor.save_pretrained(d)
            tc = json.load(open(os.path.join(d, "tokenizer_config.json")))
            extra = tc.get("extra_special_tokens", {})
            for name in MELT_REQUIRED_SPECIAL_TOKENS:
                token_str = getattr(processor, name)
                assert name in extra, (
                    f"'{name}' missing from extra_special_tokens"
                )
                assert extra[name] == token_str, (
                    f"extra_special_tokens['{name}'] mismatch: "
                    f"{extra[name]!r} != {token_str!r}"
                )

    def test_from_pretrained_restores_token_strings(self, processor):
        """from_pretrained must restore all MELT token string attributes."""
        with tempfile.TemporaryDirectory() as d:
            processor.save_pretrained(d)
            loaded = MELTProcessor.from_pretrained(d)
            for name in MELT_REQUIRED_SPECIAL_TOKENS:
                assert getattr(loaded, name) == getattr(processor, name), (
                    f"Token '{name}' mismatch after round-trip"
                )

    def test_from_pretrained_restores_token_ids(self, processor):
        """from_pretrained must restore all MELT token ID attributes."""
        with tempfile.TemporaryDirectory() as d:
            processor.save_pretrained(d)
            loaded = MELTProcessor.from_pretrained(d)
            for name in MELT_REQUIRED_SPECIAL_TOKENS:
                orig_id = getattr(processor, name + "_id")
                loaded_id = getattr(loaded, name + "_id")
                assert loaded_id == orig_id, (
                    f"Token ID for '{name}' mismatch after round-trip: {orig_id} vs {loaded_id}"
                )

    def test_from_pretrained_no_config_argument_needed(self, processor):
        """from_pretrained must succeed without passing a config argument."""
        with tempfile.TemporaryDirectory() as d:
            processor.save_pretrained(d)
            # Must not raise TypeError about missing 'config' argument
            loaded = MELTProcessor.from_pretrained(d)
            assert loaded is not None

    def test_save_and_load_roundtrip(self, processor):
        """Saved processor must reload without errors and preserve token attributes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            loaded = MELTProcessor.from_pretrained(tmp_dir)

        assert loaded.audio_token == processor.audio_token
        assert loaded.audio_bos_token == processor.audio_bos_token
        assert loaded.audio_eos_token == processor.audio_eos_token

    def test_save_and_load_token_ids(self, processor):
        """Token IDs resolved after from_pretrained() must match the originals."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            loaded = MELTProcessor.from_pretrained(tmp_dir)

        assert loaded.audio_token_id == processor.audio_token_id
        assert loaded.audio_bos_token_id == processor.audio_bos_token_id
        assert loaded.audio_eos_token_id == processor.audio_eos_token_id

    def test_load_does_not_require_config(self, processor):
        """from_pretrained() must succeed without passing a config argument."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            # Must not raise TypeError about missing 'config' argument
            loaded = MELTProcessor.from_pretrained(tmp_dir)

        assert isinstance(loaded, MELTProcessor)

    def test_loaded_processor_can_tokenize(self, processor, audio_sample):
        """A reloaded processor must produce the same token IDs as the original."""
        text = f"{processor.audio_token}hello world"
        audio = [[audio_sample]]

        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            loaded = MELTProcessor.from_pretrained(tmp_dir)

        original_out = processor(text=[text], audio=audio, return_tensors="pt")
        loaded_out = loaded(text=[text], audio=audio, return_tensors="pt")

        assert (original_out["input_ids"] == loaded_out["input_ids"]).all()


# ============================================================================
# Fixed-window encoders (Whisper)
# ============================================================================


@pytest.fixture(scope="module")
def whisper_processor(tokenizer):
    """A MELTProcessor wired to Whisper's feature extractor.

    Whisper differs from w2v-BERT in every way the audio path cares about: it emits
    (B, F, T) rather than (B, T, F), it runs at 100 Hz rather than 50 Hz, and its
    encoder only accepts whole 30 s windows.
    """
    fe = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    return MELTProcessor(feature_extractor=fe, tokenizer=tokenizer)


WHISPER_WINDOW_FRAMES = 3000
WHISPER_MEL_BINS = 128


def _tone(seconds: float, sr: int = 16000):
    import numpy as np

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class TestWhisperFixedWindowFeatures:
    def test_spec_is_resolved_from_the_feature_extractor(self, whisper_processor):
        assert whisper_processor.encoder_spec.window_frames == WHISPER_WINDOW_FRAMES
        assert whisper_processor.encoder_spec.is_channel_major is True

    @pytest.mark.parametrize(
        "seconds, expected_windows",
        [(0.5, 1), (3.0, 1), (29.9, 1), (30.0, 1), (30.1, 2), (45.0, 2), (60.0, 2)],
    )
    def test_features_are_time_major_and_a_whole_number_of_windows(
        self, whisper_processor, seconds, expected_windows
    ):
        """Short clips pad up to one window; long ones fold into several."""
        result = whisper_processor(text="<|audio|>", audio=_tone(seconds), return_tensors="pt")

        features = result["input_features"]
        assert features.shape == (
            1,
            expected_windows * WHISPER_WINDOW_FRAMES,
            WHISPER_MEL_BINS,
        )

    @pytest.mark.parametrize("seconds", [0.5, 3.0, 12.0, 45.0])
    def test_mask_counts_real_frames_at_100_hz(self, whisper_processor, seconds):
        """The mask, not the tensor width, is what audio_lengths is derived from."""
        result = whisper_processor(text="<|audio|>", audio=_tone(seconds), return_tensors="pt")

        mask = result["features_attention_mask"]
        assert mask.shape[:2] == result["input_features"].shape[:2]
        # hop_length 160 at 16 kHz -> one mel frame per 10 ms.
        assert int(mask.sum()) == pytest.approx(seconds * 100, abs=1)

    def test_mask_is_a_prefix(self, whisper_processor):
        """Real audio must occupy a prefix, or mapping lengths through the encoder
        downsampling would not describe where the valid outputs are."""
        result = whisper_processor(text="<|audio|>", audio=_tone(12.0), return_tensors="pt")

        mask = result["features_attention_mask"][0].to(torch.bool)
        n_valid = int(mask.sum())
        assert mask[:n_valid].all()
        assert not mask[n_valid:].any()

    def test_batch_pads_on_the_time_axis(self, whisper_processor):
        """Regression: the old path concatenated and padded on dim 1, which for
        Whisper's channel-major output is the mel-bin axis."""
        result = whisper_processor(
            text=["<|audio|>", "<|audio|>"],
            audio=[[_tone(3.0)], [_tone(40.0)]],
            return_tensors="pt",
        )

        features = result["input_features"]
        assert features.shape[0] == 2
        assert features.shape[1] == 2 * WHISPER_WINDOW_FRAMES  # the longer sample
        assert features.shape[2] == WHISPER_MEL_BINS
        # The short sample keeps its own valid frames and is zero-padded beyond them.
        masks = result["features_attention_mask"]
        assert int(masks[0].sum()) == pytest.approx(300, abs=1)
        assert int(masks[1].sum()) == pytest.approx(4000, abs=1)

    def test_whisper_encoder_accepts_what_the_processor_produces(self, whisper_processor):
        """The end of the contract: one window of processor output, transposed the way
        MELTAudioEncoder transposes it, must be exactly what WhisperEncoder demands."""
        from transformers import AutoConfig, AutoModel

        result = whisper_processor(text="<|audio|>", audio=_tone(3.0), return_tensors="pt")
        features = result["input_features"]

        config = AutoConfig.from_pretrained("openai/whisper-large-v3")
        config.encoder_layers = 1
        config.decoder_layers = 1
        encoder = AutoModel.from_config(config).get_encoder().float()

        out = encoder(features.transpose(1, 2).float())[0]

        assert out.shape == (1, config.max_source_positions, config.d_model)


# ============================================================================
# Raw-waveform encoders (mHuBERT / the wav2vec2 family)
# ============================================================================


@pytest.fixture(scope="module")
def mhubert_processor(tokenizer):
    """A MELTProcessor wired to mHuBERT-147's feature extractor.

    Only the tiny preprocessor_config.json is fetched -- no model weights. This
    extractor differs from every other one MELT uses: it emits a raw waveform under
    `input_values`, has no feature axis at all, and declares
    `return_attention_mask: false`.
    """
    fe = AutoFeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
    return MELTProcessor(feature_extractor=fe, tokenizer=tokenizer)


@pytest.mark.hub
class TestWaveformFeatures:
    def test_spec_is_resolved_from_the_feature_extractor(self, mhubert_processor):
        spec = mhubert_processor.encoder_spec
        assert spec.is_waveform is True
        assert spec.feature_key == "input_values"
        assert spec.window_frames is None

    @pytest.mark.parametrize("seconds", [0.5, 3.0, 12.0])
    def test_features_carry_the_waveform_with_a_trailing_axis(
        self, mhubert_processor, seconds
    ):
        """MELT is written against (B, T, F); a waveform travels as (B, n_samples, 1)."""
        result = mhubert_processor(
            text="<|audio|>", audio=_tone(seconds), return_tensors="pt"
        )

        features = result["input_features"]
        n_samples = int(seconds * 16000)
        assert features.ndim == 3
        assert features.shape[0] == 1
        assert features.shape[2] == 1
        # pad_to_multiple_of 8 can round the sample count up, never down.
        assert n_samples <= features.shape[1] <= n_samples + 8

    def test_the_extractor_key_is_renamed_not_passed_through(self, mhubert_processor):
        """Wav2Vec2FeatureExtractor says `input_values`; MELT always says
        `input_features`, and model_input_names has to agree with that."""
        result = mhubert_processor(text="<|audio|>", audio=_tone(1.0), return_tensors="pt")

        assert "input_features" in result
        assert "input_values" not in result
        assert "input_features" in mhubert_processor.model_input_names
        assert "input_values" not in mhubert_processor.model_input_names

    @pytest.mark.parametrize("seconds", [0.5, 3.0, 12.0])
    def test_mask_counts_real_samples(self, mhubert_processor, seconds):
        """A mask is requested even though the checkpoint says not to return one:
        MELT needs the real length to count audio embeddings."""
        result = mhubert_processor(
            text="<|audio|>", audio=_tone(seconds), return_tensors="pt"
        )

        mask = result["features_attention_mask"]
        assert mask.shape == result["input_features"].shape[:2]
        assert int(mask.sum()) == int(seconds * 16000)

    def test_batch_pads_on_the_sample_axis_and_masks_the_pad(self, mhubert_processor):
        result = mhubert_processor(
            text=["<|audio|>", "<|audio|>"],
            audio=[[_tone(1.0)], [_tone(3.0)]],
            return_tensors="pt",
        )

        features = result["input_features"]
        masks = result["features_attention_mask"]
        assert features.shape[0] == 2
        assert features.shape[2] == 1
        assert masks.sum(-1).tolist() == [16000, 48000]
        # The short item is zero-padded, and the pad is a suffix.
        assert features[0, 16000:].abs().max() == 0

    def test_normalisation_is_per_utterance_not_per_batch(self, mhubert_processor):
        """`do_normalize: true` means zero-mean/unit-variance over whatever the
        extractor is handed. Featurising one utterance at a time -- before the batch
        pad -- is what keeps the padding out of the statistics."""
        result = mhubert_processor(
            text=["<|audio|>", "<|audio|>"],
            audio=[[_tone(1.0)], [_tone(3.0)]],
            return_tensors="pt",
        )

        short = result["input_features"][0, :16000, 0]
        assert float(short.mean()) == pytest.approx(0.0, abs=1e-4)
        assert float(short.std()) == pytest.approx(1.0, abs=1e-3)

    def test_hubert_accepts_what_the_processor_produces(self, mhubert_processor):
        """The end of the contract: squeeze the axis MELT added, and the raw HF
        encoder must take it and emit the number of frames the spec predicts."""
        from transformers import HubertConfig, HubertModel

        from melt.modeling.encoder_specs import get_encoder_spec_for_config

        result = mhubert_processor(text="<|audio|>", audio=_tone(3.0), return_tensors="pt")
        features = result["input_features"]

        config = HubertConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            conv_dim=(8,) * 7,
            num_feat_extract_layers=7,
            feat_extract_norm="group",
            layerdrop=0.0,
            apply_spec_augment=False,
        )
        encoder = HubertModel(config).eval()

        with torch.no_grad():
            out = encoder(features.squeeze(-1))[0]

        expected = get_encoder_spec_for_config(config).output_lengths(features.shape[1])
        assert out.shape == (1, expected, 32)
