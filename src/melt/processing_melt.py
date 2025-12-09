"""
Processor class for MELT.
"""

import logging
import re

import torch

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
            "return_tensors": "pt",
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

    def _validate_audio_token_count(self, text: list[str], audio, is_batched: bool) -> None:
        """Validate that the number of audio tokens in text matches the number of audio inputs.

        Args:
            text: List of text strings.
            audio: Audio input(s). For batched input, should be a list of lists where audio[i]
                   contains the audio arrays for text[i]. For single input, can be a single array
                   or a list of arrays.
            is_batched: Whether we are processing a batch (text is a list of multiple samples).
        """
        audio_token_counts = [sample.count(self.audio_token) for sample in text]
        total_audio_tokens = sum(audio_token_counts)

        if audio is None:
            if total_audio_tokens > 0:
                logger.warning(
                    f"Found {total_audio_tokens} audio token(s) in text but no audio input was provided. "
                    f"The audio token(s) will be treated as normal text tokens."
                )
            return

        if is_batched:
            # For batched input, audio must be a list of lists
            if not isinstance(audio, list) or len(audio) != len(text):
                raise ValueError(
                    f"For batched input, `audio` must be a list of lists with the same length as `text`. "
                    f"Got {len(audio) if isinstance(audio, list) else 'non-list'} audio entries for {len(text)} text samples. "
                    f"Each audio[i] should be a list of audio arrays corresponding to the audio tokens in text[i]."
                )

            # Validate per-sample
            for i, (sample_text, sample_audio, expected_count) in enumerate(zip(text, audio, audio_token_counts)):
                if not isinstance(sample_audio, list):
                    raise ValueError(
                        f"For batched input, audio[{i}] must be a list of audio arrays. "
                        f"Got {type(sample_audio).__name__} instead."
                    )
                if len(sample_audio) != expected_count:
                    raise ValueError(
                        f"Mismatch at sample {i}: found {expected_count} audio token(s) in text "
                        f"but received {len(sample_audio)} audio array(s) in audio[{i}]."
                    )
        else:
            # For single input, audio can be a single array or a list of arrays
            num_audio_inputs = len(audio) if isinstance(audio, list) else 1

            if total_audio_tokens != num_audio_inputs:
                raise ValueError(
                    f"Mismatch between audio tokens and audio inputs: found {total_audio_tokens} audio tokens "
                    f"in text but received {num_audio_inputs} audio inputs. "
                    f"Pass a list of audio arrays if you have multiple audio tokens."
                )

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

        is_batched = isinstance(text, list)
        if not is_batched:
            text = [text]

        # Validate audio token count matches audio inputs
        self._validate_audio_token_count(text, audio, is_batched)

        output_kwargs = self._merge_kwargs(
            MELTProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process audio
        if audio is not None:
            audio_inputs, audio_lengths_output = self._process_audio(audio, is_batched, output_kwargs["audio_kwargs"])
        else:
            audio_inputs = {}
            audio_lengths_output = None

        # Process images (optional)
        # TODO: to support in future
        # if images is not None and getattr(self, "image_processor", None) is not None:
        #     images_inputs = self.image_processor(images=images, **output_kwargs.get("images_kwargs", {}))
        #     image_grid_thw = iter(images_inputs.get("image_grid_thw", []))
        # else:
        #     images_inputs = {}
        #     image_grid_thw = None  # iter([])

        # Process videos (optional)
        # TODO: to support in future
        # if videos is not None and getattr(self, "video_processor", None) is not None:
        #     videos_inputs = self.video_processor(videos=videos, **output_kwargs.get("videos_kwargs", {}))
        #     video_grid_thw = iter(videos_inputs.get("video_grid_thw", []))
        # else:
        #     videos_inputs = {}
        #     video_grid_thw = None  # iter([])

        # Replace multimodal special tokens with appropriate number of placeholders
        if audio is not None or images is not None or videos is not None:
            # Flatten audio_lengths for token replacement
            if audio_lengths_output is not None:
                if is_batched:
                    flat_lengths: list[int] = []
                    if isinstance(audio_lengths_output, list):
                        for sample_lengths in audio_lengths_output:
                            if isinstance(sample_lengths, list):
                                flat_lengths.extend(sample_lengths)
                            else:
                                flat_lengths.append(int(sample_lengths))
                    audio_lengths_flat = iter(flat_lengths)
                else:
                    flat_lengths_single = (
                        audio_lengths_output if isinstance(audio_lengths_output, list) else [audio_lengths_output]
                    )
                    audio_lengths_flat = iter(flat_lengths_single)
            else:
                audio_lengths_flat = iter([])

            text = self.replace_multimodal_special_tokens(
                text,
                audio_lengths=audio_lengths_flat,
                # image_grid_thw=image_grid_thw,
                # video_grid_thw=video_grid_thw,
            )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        output_data = {**texts_inputs, **audio_inputs}
        if audio_lengths_output is not None:
            output_data["audio_lengths"] = audio_lengths_output
            if (
                isinstance(output_data["audio_lengths"], list)
                and output_data["audio_lengths"]
                and isinstance(output_data["audio_lengths"][0], list)
            ):
                max_len = max(len(lengths) for lengths in output_data["audio_lengths"])
                output_data["audio_lengths"] = [
                    lengths + [-1] * (max_len - len(lengths)) for lengths in output_data["audio_lengths"]
                ]

        return BatchFeature(
            data=output_data,  # , **images_inputs, **videos_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def _process_audio(self, audio, is_batched: bool, audio_kwargs: dict) -> tuple[dict, list | int]:
        """
        Process audio inputs and return features with audio lengths.

        Args:
            audio: Audio input(s). For batched input, a list of lists. For single input,
                   a single array or list of arrays.
            is_batched: Whether we are processing a batch.
            audio_kwargs: Keyword arguments for the feature extractor.

        Returns:
            Tuple of (audio_inputs dict, audio_lengths).
            - audio_inputs contains 'input_features' and 'feature_attention_mask' with shape (batch_size, *).
            - audio_lengths is a list of lists for batched input, or a single int/list for single input.
        """

        def _get_features_from_sample(audio):
            """Extract input features, attention mask, and lengths for a single sample."""

            if not isinstance(audio, list):
                audio = [audio]

            # Multiple audios for single text sample - concatenate on time dimension
            all_features = []
            all_masks = []
            audio_lengths_output = []

            # TODO: the length estimation does not consider any reduction due to adapters after the speech encoder.
            # Hence, we are now adding potentially more token placeholders than needed.
            # A better solution would be to have the feature extractor return the exact length after all processing
            audio_kwargs["return_attention_mask"] = True
            audio_kwargs["pad_to_multiple_of"] = 8
            for audio_array in audio:
                audio_out = self.feature_extractor([audio_array], **audio_kwargs)
                all_features.append(audio_out["input_features"])
                mask = audio_out.get("attention_mask")
                if mask is not None:
                    all_masks.append(mask)
                    audio_lengths_output.append(int(mask.sum(-1).item()))
                else:
                    audio_lengths_output.append(audio_out["input_features"].shape[-1])

            input_features = torch.cat(all_features, dim=1)
            attention_mask = torch.cat(all_masks, dim=-1) if all_masks else None

            assert input_features.shape[0] == 1, "Batch size should be 1 for single input"
            assert attention_mask.sum(-1).item() == sum(audio_lengths_output), (
                f"Total sequence length mismatch: {input_features.shape[1]} vs {sum(audio_lengths_output)}"
            )
            return input_features, attention_mask, audio_lengths_output

        if is_batched:
            # audio is a list of lists: audio[i] = list of audio arrays for sample i
            all_features = []
            all_masks = []
            audio_lengths_output = []

            for sample_audios in audio:
                input_features, attention_mask, sample_audio_lengths = _get_features_from_sample(sample_audios)
                all_features.append(input_features)
                all_masks.append(attention_mask)
                audio_lengths_output.append(sample_audio_lengths)

            # Stack all samples to create batch tensors
            # Pad to max length in batch
            max_len = max(f.shape[1] for f in all_features)
            padded_features = []
            padded_masks = []

            for features, mask in zip(all_features, all_masks):
                pad_len = max_len - features.shape[1]
                if pad_len > 0:
                    # Pad features with zeros on seq_len dim
                    padded_features.append(torch.nn.functional.pad(features, (0, 0, 0, pad_len), value=0))
                    # Pad mask with zeros (no attention) on seq_len dim
                    if mask is not None:
                        padded_masks.append(torch.nn.functional.pad(mask, (0, pad_len), value=0))
                else:
                    padded_features.append(features)
                    if mask is not None:
                        padded_masks.append(mask)

            audio_inputs = {
                "input_features": torch.cat(padded_features, dim=0),
                "feature_attention_mask": torch.cat(padded_masks, dim=0) if padded_masks else None,
            }
        else:
            # Single item: here audio is a single array or a list of arrays
            input_features, attention_mask, audio_lengths_output = _get_features_from_sample(audio)

            audio_inputs = {
                "input_features": input_features,
                "feature_attention_mask": attention_mask,
            }

        return audio_inputs, audio_lengths_output

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
        # if image_grid_thw is None:
        #     image_grid_thw = iter([])
        # if video_grid_thw is None:
        #     video_grid_thw = iter([])
        # Get merge lengths for image/video if processors are available
        # merge_length_image = 1
        # merge_length_video = 1
        # if getattr(self, "image_processor", None) is not None and hasattr(self.image_processor, "merge_size"):
        #     merge_length_image = self.image_processor.merge_size**2
        # if getattr(self, "video_processor", None) is not None and hasattr(self.video_processor, "merge_size"):
        #     merge_length_video = self.video_processor.merge_size**2

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
                        replacement = (
                            "<|audio_bos_placeholder|>"
                            + "<|audio_placeholder|>" * int(audio_len)
                            + "<|audio_eos_placeholder|>"
                        )
                        sample = sample.replace(
                            self.audio_token,
                            replacement,
                            1,
                        )
                    except StopIteration:
                        pass
                # elif special_token == self.image_token:
                #     try:
                #         grid = next(image_grid_thw)
                #         image_seq_length = grid.prod() // merge_length_image
                #         sample = sample.replace(
                #             self.image_token,
                #             "<|image_placeholder|>" * image_seq_length,
                #             1,
                #         )
                #     except StopIteration:
                #         pass
                # elif special_token == self.video_token:
                #     try:
                #         grid = next(video_grid_thw)
                #         video_seq_length = grid.prod() // merge_length_video
                #         sample = sample.replace(
                #             self.video_token,
                #             "<|video_placeholder|>" * video_seq_length,
                #             1,
                #         )
                #     except StopIteration:
                #         pass

            # Replace placeholders back to actual tokens
            sample = sample.replace("<|audio_bos_placeholder|>", self.audio_bos_token)
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|audio_eos_placeholder|>", self.audio_eos_token)

            # TODO: Uncomment when image/video processing is supported
            # sample = sample.replace("<|image_placeholder|>", self.image_token)
            # sample = sample.replace("<|video_placeholder|>", self.video_token)
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
