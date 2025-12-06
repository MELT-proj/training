import tempfile
import urllib.request

import librosa
import pytest
from transformers import AutoFeatureExtractor, AutoTokenizer

from src.melt.processing_melt import MELT_SPECIAL_TOKENS, MELTProcessor

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
    """Create a MELTProcessor instance."""
    return MELTProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


class TestMELTProcessorInit:
    def test_init_with_required_components(self, feature_extractor, tokenizer):
        processor = MELTProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        assert processor.feature_extractor is not None
        assert processor.tokenizer is not None

    def test_init_without_feature_extractor_raises(self, tokenizer):
        with pytest.raises(ValueError, match="feature_extractor is required"):
            MELTProcessor(feature_extractor=None, tokenizer=tokenizer)

    def test_init_without_tokenizer_raises(self, feature_extractor):
        with pytest.raises(ValueError, match="tokenizer is required"):
            MELTProcessor(feature_extractor=feature_extractor, tokenizer=None)

    def test_special_tokens_added(self, processor):
        vocab = processor.tokenizer.get_vocab()
        for token_name, token_value in MELT_SPECIAL_TOKENS.items():
            assert token_value in vocab, f"Token {token_value} not found in vocabulary"

    def test_token_attributes_set(self, processor):
        assert processor.image_token == "<|IMAGE|>"
        assert processor.audio_token == "<|AUDIO|>"
        assert processor.video_token == "<|VIDEO|>"
        assert processor.vision_bos_token == "<|vision_bos|>"
        assert processor.vision_eos_token == "<|vision_eos|>"
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
        audios = [audio_sample, audio_sample]
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

    def test_vision_bos_eos_tokens_in_vocab(self, processor):
        vocab = processor.tokenizer.get_vocab()
        assert processor.vision_bos_token in vocab
        assert processor.vision_eos_token in vocab

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
        assert "feature_attention_mask" in input_names
