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

import json
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model

import wandb
from accelerate.utils import find_tied_parameters
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from melt import ddp
from melt.logging_utils import configure_logging, get_logger
from melt.modeling import MELTConfig, MELTForSequenceClassification, MELTProcessor
from melt.training.config import (
    config_to_dict,
    expand_env_vars_in_config,
    parse_args_and_load_config,
    save_config,
    trainer_args_dict,
)
from melt.training.metrics import TrainingEvaluator
from melt.training.trainer import count_trainable_parameters
from .trainer import MELTTrainerForRegression


logger = get_logger(__name__)

# Optimize matmul precision
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: DictConfig,
    targs: Seq2SeqTrainingArguments,
    processor: MELTProcessor,
) -> tuple[MELTForSequenceClassification, str | None]:
    """Prepare the model for training.

    Args:
        cfg: Training configuration (OmegaConf DictConfig).
        targs: HuggingFace Seq2SeqTrainingArguments.
        processor: MELTProcessor instance.

    Returns:
        Tuple of (model, last_checkpoint_path).
    """
    # Prepare model configs
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    decoder_cfg = model_cfg.decoder
    adapter_cfg = model_cfg.adapter

    # Detect last checkpoint
    # transformers 5 removed `overwrite_output_dir` from TrainingArguments
    # entirely, so it can no longer be read off `targs` -- read the raw YAML
    # instead, with the same default (False) TrainingArguments used to apply.
    overwrite_output_dir = bool(cfg.trainer.get("overwrite_output_dir", False))
    last_checkpoint = None
    if os.path.isdir(targs.output_dir) and targs.do_train and not overwrite_output_dir:
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
    if model_cfg.ckpt is None:
        raise ValueError("model.ckpt must be specified to train a metric.")

    logger.info(f"Loading model from checkpoint: {model_cfg.ckpt}")
    model = MELTForSequenceClassification.from_pretrained(
        model_cfg.ckpt,
        text_decoder_kwargs={
            "num_labels": 1,
            "attn_implementation": decoder_cfg.attn_implementation,
        },
    )

    # Sync pad_token_id from the processor tokenizer in case it was not persisted
    # correctly in the checkpoint config (needed for batch sizes > 1).
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "config"):
        model.text_decoder.config.pad_token_id = processor.tokenizer.pad_token_id

    logger.info("Tied model weights:")
    for tied_pair in find_tied_parameters(model):
        logger.info(f"  {tied_pair[0]} <--> {tied_pair[1]}")

    # If we added new tokens and the model did not have spare embedding entries,
    # we need to resize the token embeddings
    if len(processor.tokenizer) > model.config.vocab_size:
        logger.info(
            f"Resizing token embeddings from {model.config.vocab_size} to {len(processor.tokenizer)}"
        )
        model.text_decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    lora_cfg = model_cfg.get("lora", None)
    lora_enabled = lora_cfg is not None and lora_cfg.get("enabled", False)
    if lora_enabled:
        logger.info("Applying LoRA adapters to the model...")
        target_modules = list(lora_cfg.target_modules) if lora_cfg.target_modules is not None else None
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=target_modules,
            bias=lora_cfg.bias,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, peft_config)
        # PEFT freezes all base params; re-enable the regression head so it keeps training
        for name, param in model.named_parameters():
            if "score" in name:
                param.requires_grad = True
        model.print_trainable_parameters()

        logger.info("Parameters containing 'score' after LoRA wrapping:")
        for name, param in model.named_parameters():
            if "score" in name:
                logger.info(f"  {name} | shape={list(param.shape)} | requires_grad={param.requires_grad}")

    def _freeze(module: torch.nn.Module, exclude_names: list[str] | None = None):
        """Freeze parameters in a module, optionally excluding by name.
        
        Args:
            module: The module to freeze.
            exclude_names: List of parameter name substrings to exclude from freezing.
        """
        if exclude_names is None:
            exclude_names = []
        
        for name, param in module.named_parameters():
            # log when a is excluded
            is_excluded = any(exclude in name for exclude in exclude_names)
            if is_excluded:
                logger.info(f"Excluding parameter from freezing: {name}")

            if not any(exclude in name for exclude in exclude_names):
                param.requires_grad = False

    if adapter_cfg.freeze:
        logger.info("Freezing the adapter")
        _freeze(model.audio_stack.adapter)
    if encoder_cfg.freeze:
        logger.info("Freezing the encoder")
        _freeze(model.audio_stack.encoder)
    if decoder_cfg.freeze and not lora_enabled:
        logger.info("Freezing the decoder")
        _freeze(model.text_decoder, exclude_names=["score"])

    return model, last_checkpoint, model.config


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
        )

        # Log SLURM job ID so it is visible in the wandb UI.
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "NOSLURM")
        wandb.config.update({"slurm_job_id": slurm_job_id}, allow_val_change=True)

        # Upload the fully-resolved config (after env-var expansion and CLI
        # overrides) as a wandb artifact for reproducibility.  Write the
        # file into the output directory so it is easy to inspect locally.
        output_dir = cfg.trainer.output_dir
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "resolved_config.json")
        with open(config_path, "w") as f:
            json.dump(dict_cfg, f, indent=2, default=str)
        config_artifact = wandb.Artifact(
            name=f"config-{wandb.run.id}",
            type="config",
            description="Resolved training configuration after env var expansion and CLI overrides",
        )
        config_artifact.add_file(config_path)
        wandb.log_artifact(config_artifact)

    # Create training arguments
    targs = Seq2SeqTrainingArguments(**trainer_args_dict(cfg))

    ##########################
    ## PROCESSOR SETUP
    ##########################
    logger.info(f"Loading processor from checkpoint: {cfg.model.ckpt}")
    processor = MELTProcessor.from_pretrained(cfg.model.ckpt)

    ##########################
    ## MODEL PREPARATION
    ##########################
    model, last_checkpoint, config = prepare_model(cfg, targs, processor)
    logger.info("Model prepared!")

    ##########################
    ## TRAINING
    ##########################
    compute_metrics = None
    if hasattr(cfg, "evaluation") and cfg.evaluation is not None:
        # TrainingEvaluator now scores decoded generations, not logits, so it
        # needs a Seq2SeqTrainingArguments with predict_with_generate and a
        # collator that emits prompt-only inputs. This project's config has no
        # `evaluation` section, so nothing here is exercised today — wiring it
        # up means moving this trainer onto that path first.
        train_evaluator = TrainingEvaluator(cfg.evaluation, processor)
        compute_metrics = train_evaluator

    # No explicit train_dataset/eval_dataset - they are handled by Lhotse
    trainer = MELTTrainerForRegression(
        model=model,
        args=targs,
        config=cfg,
        processor=processor,
        compute_metrics=compute_metrics
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
