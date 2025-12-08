import os
import random

import torch
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import Trainer


logger = get_logger(__name__)


def current_cpumem_usage():
    import psutil

    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024**2:.2f}"


class MELTTrainer(Trainer):
    @staticmethod
    def num_tokens(train_dl: DataLoader, max_steps: None | int = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            dataset = train_dl.dataset
            words_by_row = [
                len(t.split(" ")) for t in tqdm(dataset["text"], total=len(dataset), desc="Counting tokens")
            ]
            train_tokens = sum(words_by_row)  # it's not tokens, but it's a good approximation
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")

        return train_tokens

    def create_optimizer(self):
        adapter_params = []
        non_adapter_params = []
        for name, param in self.model.encoder.named_parameters():
            if "adapter" in name:
                adapter_params.append(param)
            else:
                non_adapter_params.append(param)

        groups = []
        if not self.args.freeze_adapter:
            groups.append({"params": adapter_params, "lr": self.args.adapter_lr})
        if not self.args.freeze_encoder:
            groups.append({"params": non_adapter_params, "lr": self.args.encoder_lr})
        if not self.args.freeze_decoder:
            (groups.append({"params": self.model.decoder.parameters(), "lr": self.args.decoder_lr}),)

        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )


class AudioTextDataCollator:
    gigaspeech_punctuation = {
        " <COMMA>": ",",
        " <PERIOD>": ".",
        " <QUESTIONMARK>": "?",
        " <EXCLAMATIONPOINT>": "!",
    }

    def __init__(
        self,
        processor,
        add_text: bool = True,
        add_length_probability: float = 0.9,
        add_eos_token: bool = True,
        preamble_col: str | None = None,
        return_labels: bool = True,
    ):
        self.processor = processor
        self.add_length_probability = add_length_probability
        self.add_eos_token = add_eos_token
        self.add_text = add_text
        self.return_labels = return_labels
        self.preamble_col = preamble_col

    def __call__(self, batch: list[dict]):
        audios = []
        langs = []
        tasks = []

        if self.add_text:
            texts = []
            preambles = []
        else:
            texts = None
            preambles = None

        for item in batch:
            audios.append(item["audio"]["array"])
            langs.append(item["lang"])
            tasks.append(item["task"])

            if self.add_text:
                t = item["text"]
                # for k, v in self.gigaspeech_punctuation.items():
                #     t = t.replace(k, v)
                # t = t.lower().capitalize()

            preamble_text = item.get(self.preamble_col, None)
            if preamble_text is None:
                preamble_text = str(len(t)) if random.random() <= self.add_length_probability else ""

            preambles.append(preamble_text)
            texts.append(t)

        inputs = self.processor(
            audio=audios,
            sampling_rate=16000,  # TODO: so far we support only w2v and 16kHz audios
            task=tasks,
            target_lang=langs,
            padding="longest",
            text=texts,
            text_preamble=preambles,
            return_tensors="pt",
            return_labels=self.return_labels,
        )

        return inputs


def sanitize_model_name(x: str) -> str:
    return x.replace("/", "--")


def filter_data(dataset, max_duration: int, min_chars: int):
    logger.info("Filtering the dataset")
    ssize = len(dataset)
    dataset = dataset.filter(
        lambda text, length: (
            length <= max_duration and length > 0 and isinstance(text, str) and len(text) > min_chars
        ),
        input_columns=["text", "length"],
    )
    fsize = len(dataset)
    logger.info(f"Removed {ssize - fsize} ({100 * (ssize - fsize) / ssize:.2f}%) samples from dataset.")
    return dataset


def _format_param_count(count: int, precision: int = 2) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.{precision}f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.{precision}f}M"
    if count >= 1_000:
        return f"{count / 1_000:.{precision}f}K"
    return str(count)


def count_trainable_parameters(model: torch.nn.Module, precision: int = 2, return_int: bool = False):
    """Return the number of trainable parameters, respecting any frozen modules.

    Args:
        model: A torch.nn.Module to inspect.
        precision: Decimal precision for the formatted string.
        return_int: If True, also return the raw integer count.

    Returns:
        str | tuple[int, str]: Formatted count (and optionally the raw integer).
    """

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted = _format_param_count(trainable, precision)
    if return_int:
        return trainable, formatted
    return formatted
