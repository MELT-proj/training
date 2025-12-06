# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for MELT.
"""

import logging
import re

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

logger = logging.getLogger(__name__)


# Redefine kwargs for videos (optional, for future use)
# class MELTVideosKwargs(VideosKwargs, total=False):
#     min_pixels: int
#     max_pixels: int
#     patch_size: int
#     temporal_patch_size: int
#     merge_size: int
#     min_frames: int
#     max_frames: int
#     use_audio_in_video: bool
#     seconds_per_chunk: float
#     position_id_per_seconds: int | float


class MELTProcessorKwargs(ProcessingKwargs, total=False):
    # videos_kwargs: MELTVideosKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "return_attention_mask": True,
        },
        # "videos_kwargs": {
        #     "seconds_per_chunk": 2.0,
        #     "position_id_per_seconds": 25,
        #     "use_audio_in_video": False,
        # },
    }  # type: ignore


# Special tokens for MELT
MELT_SPECIAL_TOKENS = {
    "image_token": "<|IMAGE|>",
    "audio_token": "<|AUDIO|>",
    "video_token": "<|VIDEO|>",
    "vision_bos_token": "<|vision_bos|>",
    "vision_eos_token": "<|vision_eos|>",
    "audio_bos_token": "<|audio_bos|>",
    "audio_eos_token": "<|audio_eos|>",
}


class MELTProcessor(ProcessorMixin):
    r"""
    Constructs a MELT processor which wraps a feature extractor and a tokenizer into a single processor.

    [`MELTProcessor`] offers all the functionalities of an audio feature extractor and a tokenizer.
    See the [`~MELTProcessor.__call__`] and [`~MELTProcessor.decode`] for more information.

    Args:
        feature_extractor ([`AutoFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`AutoTokenizer`], *optional*):
            The text tokenizer.
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor (optional, for future use).
        video_processor ([`AutoVideoProcessor`], *optional*):
            The video processor (optional, for future use).
    """

    attributes = [
        "feature_extractor",
        "tokenizer",
    ]  # , "image_processor", "video_processor"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    # image_processor_class = "AutoImageProcessor"
    # video_processor_class = "AutoVideoProcessor"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        # image_processor=None,
        # video_processor=None,
    ):
        # Ensure we have required components
        if feature_extractor is None:
            raise ValueError("feature_extractor is required for MELTProcessor")
        if tokenizer is None:
            raise ValueError("tokenizer is required for MELTProcessor")

        super().__init__(feature_extractor, tokenizer)  # , image_processor, video_processor)
        self.image_processor = None

        # Add special tokens if not present
        self._ensure_special_tokens()

        # Set token attributes
        self.image_token = MELT_SPECIAL_TOKENS["image_token"]
        self.audio_token = MELT_SPECIAL_TOKENS["audio_token"]
        self.video_token = MELT_SPECIAL_TOKENS["video_token"]
        self.vision_bos_token = MELT_SPECIAL_TOKENS["vision_bos_token"]
        self.vision_eos_token = MELT_SPECIAL_TOKENS["vision_eos_token"]
        self.audio_bos_token = MELT_SPECIAL_TOKENS["audio_bos_token"]
        self.audio_eos_token = MELT_SPECIAL_TOKENS["audio_eos_token"]

    def _ensure_special_tokens(self):
        """Ensure all special tokens are in the tokenizer vocabulary."""
        tokens_to_add = []
        for token_value in MELT_SPECIAL_TOKENS.values():
            if token_value not in self.tokenizer.get_vocab():
                tokens_to_add.append(token_value)

        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
            logger.info(f"Added {len(tokens_to_add)} special tokens to tokenizer: {tokens_to_add}")

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[MELTProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s).

        This method forwards the `text` and `kwargs` arguments to the tokenizer's `__call__` if `text` is not `None`
        to encode the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        the feature extractor's `__call__` if `audio` is not `None`.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `List[np.ndarray]}`, *optional*):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, *optional*):
                The image or batch of images to be prepared (optional, for future use).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The video or batch of videos to be prepared (optional, for future use).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **input_features** -- Audio features to be fed to the model (if audio is provided).
            - **feature_attention_mask** -- Attention mask for audio features (if audio is provided).
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            MELTProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process audio
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            # Rename to prevent conflicts
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask", None)
            if audio_inputs["feature_attention_mask"] is not None:
                input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
                audio_lengths = iter((input_lengths - 2) // 2 + 1)
            else:
                # Estimate audio lengths from input_features shape
                audio_lengths = iter(
                    [audio_inputs["input_features"].shape[-1] // 4] * len(audio_inputs["input_features"])
                )
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        # Process images (optional)
        if images is not None and getattr(self, "image_processor", None) is not None:
            images_inputs = self.image_processor(images=images, **output_kwargs.get("images_kwargs", {}))
            image_grid_thw = iter(images_inputs.get("image_grid_thw", []))
        else:
            images_inputs = {}
            image_grid_thw = None  # iter([])

        # Process videos (optional)
        if videos is not None and getattr(self, "video_processor", None) is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs.get("videos_kwargs", {}))
            video_grid_thw = iter(videos_inputs.get("video_grid_thw", []))
        else:
            videos_inputs = {}
            video_grid_thw = None  # iter([])

        if not isinstance(text, list):
            text = [text]

        # Replace multimodal special tokens with appropriate number of placeholders
        if audio is not None or images is not None or videos is not None:
            text = self.replace_multimodal_special_tokens(
                text,
                audio_lengths=audio_lengths,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **audio_inputs, **images_inputs, **videos_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text: list[str],
        audio_lengths,
        image_grid_thw=None,
        video_grid_thw=None,
    ) -> list[str]:
        """
        Replace multimodal special tokens with the appropriate number of placeholder tokens.

        Args:
            text: List of text strings containing special tokens.
            audio_lengths: Iterator of audio sequence lengths.
            image_grid_thw: Iterator of image grid dimensions (optional).
            video_grid_thw: Iterator of video grid dimensions (optional).

        Returns:
            List of processed text strings with expanded placeholder tokens.
        """
        if image_grid_thw is None:
            image_grid_thw = iter([])
        if video_grid_thw is None:
            video_grid_thw = iter([])

        # Get merge lengths for image/video if processors are available
        merge_length_image = 1
        merge_length_video = 1
        if getattr(self, "image_processor", None) is not None and hasattr(self.image_processor, "merge_size"):
            merge_length_image = self.image_processor.merge_size**2
        if getattr(self, "video_processor", None) is not None and hasattr(self.video_processor, "merge_size"):
            merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            # Find all special tokens and their positions
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    try:
                        audio_len = next(audio_lengths)
                        sample = sample.replace(
                            self.audio_token,
                            "<|audio_placeholder|>" * int(audio_len),
                            1,
                        )
                    except StopIteration:
                        pass
                elif special_token == self.image_token:
                    try:
                        grid = next(image_grid_thw)
                        image_seq_length = grid.prod() // merge_length_image
                        sample = sample.replace(
                            self.image_token,
                            "<|image_placeholder|>" * image_seq_length,
                            1,
                        )
                    except StopIteration:
                        pass
                elif special_token == self.video_token:
                    try:
                        grid = next(video_grid_thw)
                        video_seq_length = grid.prod() // merge_length_video
                        sample = sample.replace(
                            self.video_token,
                            "<|video_placeholder|>" * video_seq_length,
                            1,
                        )
                    except StopIteration:
                        pass

            # Replace placeholders back to actual tokens
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)

        return processed_text

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        names = list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))

        if getattr(self, "image_processor", None) is not None:
            names.extend(self.image_processor.model_input_names)
        if getattr(self, "video_processor", None) is not None:
            names.extend(self.video_processor.model_input_names)

        return list(dict.fromkeys(names))


__all__ = ["MELTProcessor"]

__all__ = ["MELTProcessor"]

__all__ = ["MELTProcessor"]
__all__ = ["MELTProcessor"]
__all__ = ["MELTProcessor"]
__all__ = ["MELTProcessor"]
