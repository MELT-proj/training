"""MELT training entrypoint.

This script uses Lhotse-based data loading and OmegaConf for configuration.

Usage:
    # Use defaults
    python src/train.py

    # Load from YAML config file
    python src/train.py --config config/train/asr.yaml

    # Override specific parameters
    python src/train.py --config config/train/asr.yaml --trainer.max_steps 1000

    # Override run settings
    python src/train.py --config config/train/asr.yaml --run.exp_name my_experiment
"""

import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    TrainingArguments,
)
from transformers.modeling_utils import find_tied_parameters
from transformers.trainer_utils import get_last_checkpoint

from .. import ddp
from ..logging_utils import configure_logging, get_logger
from ..modeling import MELTConfig, MELTForCausalLM, MELTProcessor
from .config import (
    config_to_dict,
    expand_env_vars_in_config,
    parse_args_and_load_config,
    save_config,
    trainer_args_dict,
)
from .trainer import MELTTrainer, count_trainable_parameters


logger = get_logger(__name__)

# Optimize matmul precision
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: DictConfig,
    targs: TrainingArguments,
    processor: MELTProcessor,
) -> tuple[MELTForCausalLM, str | None]:
    """Prepare the model for training.

    Args:
        cfg: Training configuration (OmegaConf DictConfig).
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

    # Beyond this length in frames, the encoder will unfold the input in chunks
    max_audio_seq_len = encoder_cfg.get("max_audio_seq_len", 1500)

    config = MELTConfig(
        audio_encoder=encoder_cfg.name,
        text_decoder=decoder_cfg.name,
        adapter_config=adapter_cfg,
        decoder_kwargs={"attn_implementation": decoder_cfg.get("attn_implementation", "sdpa")},
        max_audio_seq_len=max_audio_seq_len,
    )

    # Set special tokens
    config.audio_bos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_bos_token])[0]
    config.audio_eos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_eos_token])[0]
    config.pad_token_id = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.pad_token])[0]
    config.audio_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_token])[0]
    config.audio_encoder_config.max_audio_seq_len = max_audio_seq_len

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
    logger.info("Loading model...")
    if model_cfg.ckpt is not None:
        logger.info(f"Loading model from checkpoint: {model_cfg.ckpt}")
        model = MELTForCausalLM.from_pretrained(model_cfg.ckpt)
    else:
        model = MELTForCausalLM(config)

    logger.info("Tied model weights:")
    for tied_pair in find_tied_parameters(model):
        logger.info(f"  {tied_pair[0]} <--> {tied_pair[1]}")

    # If we added new tokens and the model did not have spare embedding entries,
    # we need to resize the token embeddings
    if len(processor.tokenizer) > config.text_decoder_config.vocab_size:
        logger.info(
            f"Resizing token embeddings from {config.text_decoder_config.vocab_size} to {len(processor.tokenizer)}"
        )
        model.text_decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    def _freeze(module: torch.nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    if adapter_cfg.freeze:
        logger.info("Freezing the adapter")
        _freeze(model.audio_stack.adapter)
    if encoder_cfg.freeze:
        logger.info("Freezing the encoder")
        _freeze(model.audio_stack.encoder)
    if decoder_cfg.freeze:
        logger.info("Freezing the decoder")
        _freeze(model.text_decoder)

    return model, last_checkpoint, config


def main(cfg: DictConfig) -> None:
    """Run training from a loaded config.

    Args:
        cfg: Training configuration (OmegaConf DictConfig).
    """
    configure_logging()

    rank = ddp.get_global_rank()
    world_size = ddp.get_world_size()
    local_world_size = ddp.get_local_world_size()
    local_rank = ddp.get_local_rank()
    is_local_master = ddp.is_local_master()
    is_global_master = ddp.is_global_master()
    is_distributed = ddp.is_distributed()

    logger.info(f"Distributed setup: rank {rank} out of {world_size}")
    logger.info(
        f"world_size: {world_size}, local_world_size: {ddp.get_local_world_size()}"
        f" local_rank: {local_rank}, group_rank: {ddp.get_group_rank()}"
        f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
        f" is_distributed: {is_distributed}"
    )

    if is_distributed:
        # It seems gloo breaks on multi-node
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl" if local_world_size == world_size else "nccl"
        )

    # Convert config to dict for wandb logging
    dict_cfg = config_to_dict(cfg)
    dict_cfg |= {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "is_distributed": is_distributed,
    }

    # Starting W&B. HF Trainer can also do this, but this way we can include the config.
    # Initializing sooner also means more of the stdout logs are captured by W&B.
    # Approach taken from: https://github.com/fixie-ai/ultravox/blob/main/ultravox/training/train.py
    if "wandb" in cfg.trainer.report_to and is_global_master:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "melt"),
            config=dict_cfg,
            name=cfg.run.get("exp_name", None),
            # dir="runs",
            # tags=cfg.run_tags,
            # save_code=True,
        )

    # Create training arguments
    targs = TrainingArguments(**trainer_args_dict(cfg))

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
    model, last_checkpoint, config = prepare_model(cfg, targs, processor)
    logger.info("Model prepared!")

    ##########################
    ## TRAINING
    ##########################
    # No train_dataset/eval_dataset - they are handled by Lhotse
    trainer = MELTTrainer(
        model=model,
        args=targs,
        config=cfg,
        processor=processor,
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
    # if trainer.is_fsdp_enabled:
    #     logger.info("Setting FSDP state dict type to FULL_STATE_DICT for saving...")
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    logger.info("Saving model, processor, and config...")
    trainer.save_model()

    if is_global_master:
        processor.save_pretrained(targs.output_dir)
        config.save_pretrained(targs.output_dir)

        # Save config for reproducibility
        config_path = str(Path(targs.output_dir) / "training_config.yaml")
        save_config(cfg, config_path)
        logger.info(f"Saved training config to {config_path}")


if __name__ == "__main__":
    configure_logging()

    # Parse CLI arguments and load config using OmegaConf
    cfg = parse_args_and_load_config()

    # Expand environment variables in paths
    cfg = expand_env_vars_in_config(cfg)

    if cfg.run.get("dry_run", False):
        logger.info("Dry run mode - config parsed successfully")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    else:
        main(cfg)
