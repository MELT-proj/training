"""MELT training entrypoint.

This script uses Lhotse-based data loading and dataclass-based configs with tyro.

Usage:
    # Use defaults
    python src/train.py

    # Load from YAML config file
    python src/train.py --config-file config/train/LS_asr.yaml

    # Override specific parameters
    python src/train.py --trainer.max-steps 1000 --trainer.learning-rate 2e-5

    # Combine YAML with CLI overrides
    python src/train.py --config-file config/train/LS_asr.yaml --trainer.max-steps 500
"""

import os
from pathlib import Path

import torch
import tyro

from .. import ddp
from .config import (
    TrainingConfig,
    expand_env_vars_in_config,
    load_config_from_yaml,
    merge_configs,
    save_config,
    trainer_args_dict,
)
from .trainer import MELTTrainer, count_trainable_parameters
from ..logging_utils import configure_logging, get_logger
from ..modeling import MELTConfig, MELTForConditionalGeneration, MELTProcessor
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

logger = get_logger(__name__)

# Optimize matmul precision
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: TrainingConfig,
    targs: TrainingArguments,
    processor: MELTProcessor,
) -> tuple[MELTForConditionalGeneration, str | None]:
    """Prepare the model for training.

    Args:
        cfg: Training configuration.
        targs: HuggingFace TrainingArguments.
        processor: MELTProcessor instance.

    Returns:
        Tuple of (model, last_checkpoint_path).
    """
    # Prepare model configs
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    decoder_cfg = model_cfg.decoder
    adapter_cfg = model_cfg.adapter

    # Two HF ID model names
    encoder_name = encoder_cfg.name
    decoder_name = decoder_cfg.name

    audio_config = AutoConfig.from_pretrained(encoder_name)
    text_config = AutoConfig.from_pretrained(decoder_name, attn_implementation=decoder_cfg.attn_implementation)
    adapter_config = cfg.model.adapter # Used in MELTConfig init

    # Beyond this length in frames, the encoder will unfold the input in chunks
    max_audio_seq_len = getattr(encoder_cfg, "max_audio_seq_len", 1500)

    config = MELTConfig(
        audio_encoder_config=audio_config, 
        text_decoder_config=text_config,
        adapter_config=adapter_config,
    )

    # Set special tokens
    config.audio_bos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_bos_token])[0]
    config.audio_eos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_eos_token])[0]
    config.audio_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_token])[0]

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(targs.output_dir) and targs.do_train and not targs.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(targs.output_dir)
        if last_checkpoint is None and len(os.listdir(targs.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({targs.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and targs.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load or create model
    logger.info("Loading model to CPU (before device placement)...")
    if model_cfg.ckpt is not None:
        logger.info(f"Loading model from checkpoint: {model_cfg.ckpt}")
        model = MELTForConditionalGeneration.from_pretrained(model_cfg.ckpt)
    else:
        model = MELTForConditionalGeneration(config)

    # Ensure decoder embeddings match tokenizer size.
    # If special tokens were added to the tokenizer (e.g., audio tokens), failing to resize
    # will cause CUDA device-side asserts from embedding lookup (out-of-range indices).
    target_vocab_size = len(processor.tokenizer)
    current_vocab_size = model.text_decoder.get_input_embeddings().num_embeddings
    if current_vocab_size < target_vocab_size:
        logger.warning(
            "Resizing text decoder token embeddings to match tokenizer: "
            f"{current_vocab_size} -> {target_vocab_size}"
        )
        model.text_decoder.resize_token_embeddings(target_vocab_size, mean_resizing=False, pad_to_multiple_of=8)
    elif current_vocab_size > target_vocab_size:
        logger.warning(
            "Text decoder embedding table is larger than tokenizer vocab: "
            f"{current_vocab_size} > {target_vocab_size}. Keeping existing embeddings."
        )

    # Apply freezing
    if adapter_cfg.freeze:
        logger.info("Freezing the adapter")
        model.freeze_adapter()
    if encoder_cfg.freeze:
        logger.info("Freezing the encoder")
        model.freeze_encoder()
    if decoder_cfg.freeze:
        logger.info("Freezing the decoder")
        model.freeze_decoder()

    return model, last_checkpoint


def main(cfg: TrainingConfig) -> None:
    """Run training from a loaded config."""
    configure_logging()

    if ddp.is_distributed():
        rank = ddp.get_global_rank()
        world_size = ddp.get_world_size()
        local_rank = ddp.get_local_rank()
        is_local_master = ddp.is_local_master()
        is_global_master = ddp.is_global_master()

        logger.info(f"Distributed setup: rank {rank} out of {world_size}")
        logger.info(
            f"world_size: {world_size}, local_world_size: {ddp.get_local_world_size()}"
            f" local_rank: {local_rank}, group_rank: {ddp.get_group_rank()}"
            f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
        )
    else:
        logger.info("Not in a distributed setup")

    # Create training arguments
    targs = TrainingArguments(**trainer_args_dict(cfg))

    # Set seed
    set_seed(cfg.trainer.seed)

    ##########################
    ## PROCESSOR SETUP
    ##########################
    logger.info(f"Loading processor for encoder={cfg.model.encoder.name}, decoder={cfg.model.decoder.name}")

    processor = MELTProcessor(
        feature_extractor=AutoFeatureExtractor.from_pretrained(cfg.model.encoder.name),
        tokenizer=AutoTokenizer.from_pretrained(cfg.model.decoder.name, use_fast=True),
        config=cfg.model,
    )

    ##########################
    ## MODEL PREPARATION
    ##########################
    model, last_checkpoint = prepare_model(cfg, targs, processor)
    logger.info("Model prepared!")

    trainable_params, trainable_str = count_trainable_parameters(model, return_int=True)
    logger.info(f"Total number of learnable parameters: {trainable_params} ({trainable_str})")

    ##########################
    ## TRAINING
    ##########################
    logger.info("Creating trainer with Lhotse data loading")

    trainer = MELTTrainer(
        model=model,
        args=targs,
        config=cfg,
        processor=processor,
        # No train_dataset/eval_dataset - they are handled by Lhotse
    )

    # Determine checkpoint to resume from
    checkpoint = None
    if targs.resume_from_checkpoint is not None:
        checkpoint = targs.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    logger.info(f"Train_result: {train_result}")

    ##########################
    ## SAVING
    ##########################
    # From: https://huggingface.co/blog/ram-efficient-pytorch-fsdp
    if trainer.is_fsdp_enabled:
        logger.info("Setting FSDP state dict type to FULL_STATE_DICT for saving...")
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    logger.info("Saving model and processor...")
    trainer.save_model()
    processor.save_pretrained(targs.output_dir)

    # Save config for reproducibility
    config_path = str(Path(targs.output_dir) / "training_config.yaml")
    save_config(cfg, config_path)
    logger.info(f"Saved training config to {config_path}")


if __name__ == "__main__":
    configure_logging()
    # Parse CLI arguments using tyro
    cli_cfg = tyro.cli(TrainingConfig)

    # If a config file is specified, load it and merge with CLI overrides
    if cli_cfg.config_file is not None:
        base_cfg = load_config_from_yaml(cli_cfg.config_file)
        cfg = merge_configs(base_cfg, cli_cfg)
    else:
        cfg = cli_cfg

    # Expand environment variables in paths
    cfg = expand_env_vars_in_config(cfg)

    if cfg.dry_run:
        configure_logging()
        logger.info("Dry run mode - config parsed successfully")
        logger.info(f"Config: {cfg}")
    else:
        main(cfg)
