"""MELT training entrypoint.

This script uses Lhotse-based data loading and hierarchical YAML configs.

Usage:
    python src/train.py --config-file config/train/LS_asr.yaml
    python src/train.py --config-file config/train/LS_asr.yaml trainer.max_steps=1000
"""

import logging
import os
from pathlib import Path

import torch
import tyro
# Use standard logging module to avoid requiring accelerate state at import time

from omegaconf import DictConfig

import ddp
import transformers
from src.config import TrainingArgs, load_config, save_config, trainer_args_dict
from src.melt import MELTConfig, MELTForConditionalGeneration, MELTProcessor
from src.trainer import MELTTrainer, count_trainable_parameters
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint


# Setup logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

# Optimize matmul precision
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: DictConfig,
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

    encoder_name = encoder_cfg.pop("name")
    decoder_name = decoder_cfg.pop("name")

    audio_config = AutoConfig.from_pretrained(encoder_name, **encoder_cfg)
    text_config = AutoConfig.from_pretrained(decoder_name, **decoder_cfg)
    config = MELTConfig(audio_encoder_config=audio_config, text_decoder_config=text_config)

    # Set special tokens
    config.audio_bos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_bos_token])[0]
    config.text_decoder_config.pad_token_id = processor.tokenizer.pad_token_id

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
    if model_cfg.get("ckpt") is not None:
        logger.info(f"Loading model from checkpoint: {model_cfg.ckpt}")
        model = MELTForConditionalGeneration.from_pretrained(model_cfg.ckpt)
    else:
        model = MELTForConditionalGeneration(config)
        model.text_decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    # Print model layers for inspection/debugging and write to file
    def _print_model_layers(m, out_dir: str | None = None):
        """Log leaf-level model modules (layers) by name and type and optionally write to file."""
        logger.info("Listing model layers (leaf modules):")
        lines = []
        for name, module in m.named_modules():
            if not name:
                continue
            # Consider leaf modules only (no child modules)
            if len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters())
                line = f"{name}: {module.__class__.__name__} | params={params:,}"
                logger.info("  %s", line)
                lines.append(line)
        if out_dir is not None:
            try:
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                file = out_path / "model_layers.txt"
                file.write_text("\n".join(lines) + "\n")
                logger.info("Wrote model layers list to %s", str(file))
            except Exception as e:
                logger.warning("Could not write model layers to %s: %s", out_dir, e)

    # Pass through targs.output_dir if available so layers are saved alongside outputs
    _print_model_layers(model, getattr(targs, "output_dir", None))

    # Apply freezing
    if bool(adapter_cfg.get("freeze", False)):
        logger.info("Freezing the adapter")
        model.freeze_adapter()
    if bool(model_cfg.encoder.get("freeze", False)):
        logger.info("Freezing the encoder")
        model.freeze_encoder()
    if bool(model_cfg.decoder.get("freeze", False)):
        logger.info("Freezing the decoder")
        model.freeze_decoder()

    return model, last_checkpoint


def main(cfg: DictConfig, dry_run: bool = False) -> None:
    """Run training from a loaded config."""
    if ddp.is_distributed():
        rank = ddp.get_global_rank()
        world_size = ddp.get_world_size()
        local_rank = ddp.get_local_rank()
        is_local_master = ddp.is_local_master()
        is_global_master = ddp.is_global_master()

        logger.info(f"Distributed setup: rank {rank} out of {world_size}")
        logging.info(
            f"world_size: {world_size}, local_world_size: {ddp.get_local_world_size()}"
            f" local_rank: {local_rank}, group_rank: {ddp.get_group_rank()}"
            f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
        )

        # Reduce logging noise on non-master processes
        logging.basicConfig(level=logging.INFO if is_local_master else logging.ERROR)
        transformers.logging.set_verbosity(logging.WARNING if is_local_master else logging.ERROR)
    else:
        logger.info("Not in a distributed setup")

    # Create training arguments
    targs = TrainingArguments(**trainer_args_dict(cfg))

    # Set seed
    set_seed(int(cfg.trainer.seed))

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
        # No train_dataset/eval_dataset - handled by Lhotse
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
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model()
    processor.save_pretrained(targs.output_dir)

    # Save config for reproducibility
    config_path = str(Path(targs.output_dir) / "training_config.yaml")
    save_config(cfg, config_path)
    logger.info(f"Saved training config to {config_path}")


if __name__ == "__main__":
    cli_args, unknown = tyro.cli(TrainingArgs, return_unknown_args=True)
    cfg = load_config(cli_args.config_file, dotlist_overrides=unknown)
    main(cfg, dry_run=cli_args.dry_run)
