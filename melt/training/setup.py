"""Shared initialization utilities for MELT training and artifact saving.

These functions encapsulate processor and model-config construction so that
``train.py`` and ``infra/save_training_artifacts.py`` share the same logic.
"""

from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, AutoTokenizer

from ..logging_utils import get_logger
from ..modeling import MELTConfig, MELTProcessor
from ..modeling.configuration_melt import MELT_REQUIRED_SPECIAL_TOKENS


logger = get_logger(__name__)


def prepare_processor(cfg: DictConfig) -> MELTProcessor:
    """Build a MELTProcessor from a training config.

    Loads the feature extractor and tokenizer, registers all MELT-required
    special tokens via ``extra_special_tokens``, adds them (along with
    ``pad_token`` / ``eos_token``) to the vocabulary via
    ``add_special_tokens``, and returns a fully configured processor.

    Args:
        cfg: Full training configuration. Must contain ``model.encoder.name``,
            ``model.decoder.name``, and token definitions (``audio_token``,
            ``audio_bos_token``, ``audio_eos_token``) under ``model.decoder``.

    Returns:
        A configured :class:`MELTProcessor` instance.
    """
    encoder_name = cfg.model.encoder.name
    decoder_cfg = cfg.model.decoder

    logger.info("Loading processor for encoder=%s, decoder=%s", encoder_name, decoder_cfg.name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)

    # Build extra_special_tokens dict from config values.
    # This registers the attribute *names* (e.g. ``audio_token``) on the
    # tokenizer class so that ``tokenizer.audio_token`` works.
    extra_special_tokens = {name: getattr(decoder_cfg, name) for name in MELT_REQUIRED_SPECIAL_TOKENS}

    tokenizer = AutoTokenizer.from_pretrained(
        decoder_cfg.name,
        use_fast=True,
        extra_special_tokens=extra_special_tokens,
    )

    # Add all special tokens to the vocabulary in a single call.
    # ``extra_special_tokens`` only registers attribute names; calling
    # ``add_special_tokens`` actually inserts them into the vocab.
    special_tokens_to_add: dict[str, str] = dict(extra_special_tokens)
    eos_token = decoder_cfg.get("eos_token", None)
    pad_token = decoder_cfg.get("pad_token", None)
    if eos_token is not None:
        special_tokens_to_add["eos_token"] = eos_token
    if pad_token is not None:
        special_tokens_to_add["pad_token"] = pad_token
    tokenizer.add_special_tokens(special_tokens_to_add)

    return MELTProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def prepare_melt_config(cfg: DictConfig, processor: MELTProcessor) -> MELTConfig:
    """Build a :class:`MELTConfig` from a training config and processor.

    Args:
        cfg: Full training configuration.
        processor: An already-configured :class:`MELTProcessor` whose tokenizer
            carries the required special-token IDs.

    Returns:
        A configured :class:`MELTConfig` instance.
    """
    encoder_cfg = cfg.model.encoder
    decoder_cfg = cfg.model.decoder
    adapter_cfg = cfg.model.adapter

    max_audio_seq_len = encoder_cfg.get("max_audio_seq_len", 1500)

    config = MELTConfig(
        audio_encoder=encoder_cfg.name,
        text_decoder=decoder_cfg.name,
        adapter_config=adapter_cfg,
        decoder_kwargs={"attn_implementation": decoder_cfg.get("attn_implementation", "sdpa")},
        max_audio_seq_len=max_audio_seq_len,
    )

    config.audio_encoder_config.max_audio_seq_len = max_audio_seq_len

    # Set special token IDs on the text decoder sub-config
    tokenizer = processor.tokenizer
    config.text_decoder_config.audio_token_id = tokenizer.convert_tokens_to_ids([processor.audio_token])[0]
    config.text_decoder_config.audio_bos_token_id = tokenizer.convert_tokens_to_ids([processor.audio_bos_token])[0]
    config.text_decoder_config.audio_eos_token_id = tokenizer.convert_tokens_to_ids([processor.audio_eos_token])[0]

    # pad_token_id becomes nn.Embedding's `padding_idx`, which HF bounds-checks
    # against vocab_size *at model construction time* -- unlike the other ids
    # above, which are just read later. If add_special_tokens (prepare_processor)
    # had to append a brand-new pad token, its id is >= the pretrained decoder's
    # original vocab_size and construction would assert. Leave it off the config
    # here; train.py's prepare_model sets it after resize_token_embeddings has
    # actually grown the embedding table to fit.
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    if pad_token_id < config.text_decoder_config.vocab_size:
        config.text_decoder_config.pad_token_id = pad_token_id

    return config
