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


def borrow_chat_template(tokenizer, source_name: str, decoder_name: str) -> None:
    """Copy the chat template of *source_name* onto *tokenizer*, in place.

    A *base* checkpoint ships no chat template of its own -- verified:
    ``meta-llama/Llama-3.2-1B`` has none, while its Instruct sibling carries
    3,827 bytes of Jinja -- so ``data.apply_chat_template: true`` against one
    has nothing to render with, and
    :func:`~melt.training.data.chat_templates.validate_chat_template_config`
    rejects it when the dataset is built.

    Naming a checkpoint to copy the template from lets a base and an instruct
    arm render byte-identical text, which is what makes the pair a controlled
    comparison of the *backbone* rather than of the input format.

    Only the template string is copied, never the tokenizer itself: borrowing a
    whole tokenizer would pair one vocabulary with another checkpoint's weights,
    and nothing downstream would notice.

    Args:
        tokenizer: The decoder's tokenizer, mutated in place.
        source_name: Checkpoint (hub id or local path) to copy the template
            from. Must share *tokenizer*'s vocabulary; nothing here can check
            that, because the special-token ids are only resolved later.
        decoder_name: The decoder's own name, for the log lines.

    Raises:
        ValueError: If *source_name* has no chat template either.
    """
    borrowed = getattr(
        AutoTokenizer.from_pretrained(source_name, use_fast=True),
        "chat_template",
        None,
    )
    if not borrowed:
        raise ValueError(
            f"model.decoder.chat_template_from is {source_name!r}, but that checkpoint "
            "has no chat template either, so there is nothing to copy. Point it at an "
            "instruction-tuned checkpoint that shares this vocabulary -- typically the "
            "Instruct sibling of the decoder."
        )
    if getattr(tokenizer, "chat_template", None):
        logger.warning(
            "%s already has a chat template; overwriting it with the one from %s. "
            "Drop model.decoder.chat_template_from if that was not intended.",
            decoder_name,
            source_name,
        )
    else:
        logger.info(
            "%s ships no chat template; borrowing the one from %s.",
            decoder_name,
            source_name,
        )
    tokenizer.chat_template = borrowed


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
            ``model.decoder.chat_template_from`` is optional; see
            :func:`borrow_chat_template` for when a base backbone needs it.

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

    chat_template_from = decoder_cfg.get("chat_template_from", None)
    if chat_template_from:
        borrow_chat_template(tokenizer, chat_template_from, decoder_cfg.name)

    # Add all special tokens to the vocabulary in a single call.
    # transformers 5 validates add_special_tokens' keys against the canonical
    # SpecialTokensMixin attributes plus "extra_special_tokens" -- passing
    # "audio_token" etc. as top-level keys now raises ValueError. The MELT
    # tokens go under "extra_special_tokens" as a list instead, matching how
    # they were already registered by name via the extra_special_tokens=
    # kwarg to from_pretrained() above.
    special_tokens_to_add: dict[str, list[str] | str] = {
        "extra_special_tokens": list(extra_special_tokens.values()),
    }
    eos_token = decoder_cfg.get("eos_token", None)
    pad_token = decoder_cfg.get("pad_token", None)
    if eos_token is not None:
        special_tokens_to_add["eos_token"] = eos_token
    if pad_token is not None:
        special_tokens_to_add["pad_token"] = pad_token
    tokenizer.add_special_tokens(special_tokens_to_add)

    # Assertion, not a hope: confirm each MELT special token actually landed
    # in the vocabulary as its own token rather than silently mapping to
    # <unk>, and that tokenizer.<name> still resolves post-add_special_tokens.
    vocab = tokenizer.get_vocab()
    for name, token_str in extra_special_tokens.items():
        if not hasattr(tokenizer, name):
            raise RuntimeError(f"prepare_processor: tokenizer.{name} did not resolve after add_special_tokens.")
        if token_str not in vocab:
            raise RuntimeError(
                f"prepare_processor: special token {token_str!r} ({name}) is not in the tokenizer vocabulary "
                "after add_special_tokens -- it would silently collapse to <unk> at encode time."
            )

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
    if pad_token_id < config.vocab_size:
        config.text_decoder_config.pad_token_id = pad_token_id

    return config
