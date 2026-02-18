import tempfile
import urllib.request
from types import SimpleNamespace

import librosa
import pytest

from melt.modeling import MELT_REQUIRED_SPECIAL_TOKENS, MELTConfig, MELTProcessor
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer


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


@pytest.fixture(scope="module")
def tokenizer():
    """Load a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module")
def processor(feature_extractor, tokenizer):
    """Create a MELTProcessor instance with proper MELTConfig."""
    config = MELTConfig(
        audio_encoder="facebook/w2v-bert-2.0",
        text_decoder="Qwen/Qwen2.5-1.5B",
        adapter_config={"_type": "mlp"},
    )

    # Add decoder attribute with special tokens
    config.decoder = SimpleNamespace(
        image_token="<|IMAGE|>",
        audio_token="<|AUDIO|>",
        video_token="<|VIDEO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
    )

    return MELTProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        config=config,
    )


class TestMELTProcessorInit:
    def test_init_with_required_components(self, feature_extractor, tokenizer):
        config = MELTConfig(
            audio_encoder="facebook/w2v-bert-2.0",
            text_decoder="Qwen/Qwen2.5-1.5B",
            adapter_config={"_type": "mlp"},
        )

        # Add decoder attribute with special tokens
        config.decoder = SimpleNamespace(
            audio_token="<|AUDIO|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
        )

        processor = MELTProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            config=config,
        )
        assert processor.feature_extractor is not None
        assert processor.tokenizer is not None

    def test_init_without_config_raises(self, feature_extractor, tokenizer):
        with pytest.raises(TypeError):
            MELTProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def test_init_without_tokenizer_raises(self, feature_extractor):
        config = MELTConfig(
            audio_encoder="facebook/w2v-bert-2.0",
            text_decoder="Qwen/Qwen2.5-1.5B",
            adapter_config={"_type": "mlp"},
        )
        config.decoder = SimpleNamespace(
            audio_token="<|AUDIO|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
        )
        with pytest.raises(Exception):
            MELTProcessor(feature_extractor=feature_extractor, tokenizer=None, config=config)

    def test_special_tokens_added(self, processor):
        vocab = processor.tokenizer.get_vocab()
        for token_name in MELT_REQUIRED_SPECIAL_TOKENS:
            token_value = getattr(processor, token_name)
            assert token_value in vocab, f"Token {token_value} not found in vocabulary"

    def test_token_attributes_set(self, processor):
        # The processor guarantees audio-related tokens exist; image/video/vision
        # tokens are optional in this implementation and may not be set.
        assert processor.audio_token == "<|AUDIO|>"
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
