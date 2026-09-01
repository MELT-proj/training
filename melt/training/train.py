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
from transformers import Seq2SeqTrainingArguments, set_seed
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
from .metrics import TrainingEvaluator
from .save_checks import verify_saved_weights
from .setup import prepare_melt_config, prepare_processor
from .trainer import MELTTrainer, count_trainable_parameters


logger = get_logger(__name__)

# Optimize matmul precision
torch.set_float32_matmul_precision("high")


def prepare_model(
    cfg: DictConfig,
    targs: Seq2SeqTrainingArguments,
    processor: MELTProcessor,
) -> tuple[MELTForCausalLM, str | None, MELTConfig, MELTProcessor]:  # type: ignore[override]
    """Prepare the model for training.

    When ``model.ckpt`` is set in the config, the model config and processor
    are loaded from the checkpoint directory (which must contain
    ``config.json`` and ``preprocessor_config.json``).  Otherwise they are
    built from scratch using the training config.

    Args:
        cfg: Training configuration (OmegaConf DictConfig).
        targs: HuggingFace Seq2SeqTrainingArguments.
        processor: MELTProcessor instance (used only when no checkpoint is given).

    Returns:
        Tuple of (model, last_checkpoint_path, config, processor).
    """
    # Prepare model configs
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    decoder_cfg = model_cfg.decoder
    adapter_cfg = model_cfg.adapter

    # Detect last checkpoint
    last_checkpoint = None
    resume = targs.resume_from_checkpoint
    if isinstance(resume, str):
        # resume_from_checkpoint is a directory path: find the latest checkpoint inside it.
        last_checkpoint = get_last_checkpoint(resume)
        if last_checkpoint is None:
            raise ValueError(
                f"No checkpoint found in '{resume}'. "
                "Ensure the directory contains at least one checkpoint, "
                "or unset `trainer.resume_from_checkpoint`."
            )
        logger.info(f"Resuming from checkpoint: {last_checkpoint} (scanned from {resume})")
    elif resume is not True:
        # resume is None: auto-detect from output_dir (existing behaviour).
        # transformers 5 removed `overwrite_output_dir` from TrainingArguments
        # entirely, so trainer_args_dict() (config.py) can no longer forward
        # it onto `targs` -- read it from the raw YAML instead, with the same
        # default (False) TrainingArguments used to apply.
        overwrite_output_dir = bool(cfg.trainer.get("overwrite_output_dir", False))
        if os.path.isdir(targs.output_dir) and targs.do_train and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(targs.output_dir)
            if last_checkpoint is None and len(os.listdir(targs.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({targs.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this "
                    "behavior, change `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
    # else: resume is True — HF Trainer will scan output_dir for the last checkpoint internally.

    # Load or create model
    logger.info("Loading model...")
    if model_cfg.ckpt is not None:
        ckpt_dir = Path(model_cfg.ckpt)
        logger.info(f"Loading model, config, and processor from checkpoint: {ckpt_dir}")

        config = MELTConfig.from_pretrained(ckpt_dir)

        # A checkpoint's config.json records no attention implementation: the
        # sub-configs are serialised without `_attn_implementation`, so loading
        # one falls back to transformers' default of sdpa and the YAML's
        # `model.decoder.attn_implementation` is silently ignored.  On an H100
        # torch dispatches sdpa to the cuDNN backend, whose per-call CPU
        # planning cost dominates incremental decoding -- profiled at 65 ms of
        # CPU per attention call against 13 us of GPU, which put generate() at
        # ~2.3 s per token and one eval batch at ~100 s (artemis job 328287)
        # where the same evaluation on a from-scratch run took ~8.6 s.
        #
        # MELTConfig deliberately does not propagate `_attn_implementation` to
        # its sub-configs (see configuration_melt.py), so passing
        # `attn_implementation=` to from_pretrained would not reach the
        # decoder.  Set it on the sub-config directly -- the same one
        # prepare_melt_config configures through `decoder_kwargs` on the
        # from-scratch path -- and hand the config to from_pretrained.
        requested_attn = decoder_cfg.get("attn_implementation", None)
        if requested_attn:
            config.text_decoder_config._attn_implementation = requested_attn
            logger.info(
                f"Text decoder attention implementation from config: {requested_attn}"
            )

        processor = MELTProcessor.from_pretrained(ckpt_dir)
        model = MELTForCausalLM.from_pretrained(ckpt_dir, config=config)
    else:
        config = prepare_melt_config(cfg, processor)
        model = MELTForCausalLM(config, load_backbones=True)

    logger.info("Tied model weights:")
    for tied_pair in find_tied_parameters(model):
        logger.info(f"  {tied_pair[0]} <--> {tied_pair[1]}")

    # If we added new tokens and the model did not have spare embedding entries,
    # we need to resize the token embeddings
    if len(processor.tokenizer) > config.vocab_size:
        logger.info(
            f"Resizing token embeddings from {config.vocab_size} to {len(processor.tokenizer)}"
        )
        model.text_decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    # prepare_melt_config() withholds pad_token_id when it falls outside the
    # decoder's original vocab_size (nn.Embedding bounds-checks it as padding_idx
    # at construction time, before this resize could run). Now that the table is
    # big enough, set the real id on both the free-standing config and the model.
    pad_token_id = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.pad_token])[0]
    if config.text_decoder_config.pad_token_id != pad_token_id:
        config.text_decoder_config.pad_token_id = pad_token_id
        model.text_decoder.config.pad_token_id = pad_token_id

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
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def _set_requires_grad(module: torch.nn.Module, value: bool):
        """Set requires_grad on all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = value

    # Apply freeze/unfreeze per component.
    # When LoRA is enabled get_peft_model freezes all base params, so here we
    # explicitly unfreeze any component whose freeze flag is false.
    adapter_freeze = adapter_cfg.freeze
    encoder_freeze = encoder_cfg.freeze
    decoder_freeze = decoder_cfg.freeze

    if adapter_freeze:
        logger.info("Freezing the adapter")
    elif lora_enabled:
        logger.info("Unfreezing the adapter (overriding LoRA freeze)")
    _set_requires_grad(model.audio_stack.adapter, not adapter_freeze)

    if encoder_freeze:
        logger.info("Freezing the encoder")
    elif lora_enabled:
        logger.info("Unfreezing the encoder (overriding LoRA freeze)")
    _set_requires_grad(model.audio_stack.encoder, not encoder_freeze)

    if decoder_freeze:
        logger.info("Freezing the decoder")
    elif lora_enabled:
        logger.info("Unfreezing the decoder (overriding LoRA freeze)")
    _set_requires_grad(model.text_decoder, not decoder_freeze)

    return model, last_checkpoint, config, processor


def main(cfg: DictConfig) -> None:
    """Run training from a loaded config.

    Args:
        cfg: Training configuration (OmegaConf DictConfig).
    """
    configure_logging()

    # Seed BEFORE anything consumes randomness -- above all before the model is
    # built further down, since the adapter is randomly initialised. HF's Trainer
    # calls set_seed itself, but only inside its __init__, which runs long after
    # our model exists, so it cannot be relied on for init reproducibility.
    #
    # The same seed on every rank is intended: FSDP shards a model that must be
    # identical across ranks. Data-order variation comes from Lhotse's own
    # shard_seed, so this does not make ranks iterate the same batches.
    seed = int(cfg.trainer.get("seed", 42))
    set_seed(seed)

    rank = ddp.get_global_rank()
    world_size = ddp.get_world_size()
    local_world_size = ddp.get_local_world_size()
    local_rank = ddp.get_local_rank()
    is_local_master = ddp.is_local_master()
    is_global_master = ddp.is_global_master()
    is_distributed = ddp.is_distributed()

    logger.info(f"Distributed setup: rank {rank} out of {world_size}, seed: {seed}")
    logger.info(
        f"world_size: {world_size}, local_world_size: {ddp.get_local_world_size()}"
        f" local_rank: {local_rank}, group_rank: {ddp.get_group_rank()}"
        f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
        f" is_distributed: {is_distributed}"
    )

    # CPU affinity as it stands *after* torch has initialised OpenMP, which is
    # where it can silently collapse: OMP_PROC_BIND without OMP_PLACES pins the
    # main thread to one core, and DataLoader workers forked from it inherit that
    # mask. A single usable CPU on a multi-CPU allocation means the data pipeline
    # is confined to one core no matter how many workers are configured.
    usable_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else -1
    logger.info(
        f"CPU affinity: {usable_cpus} usable, torch threads: {torch.get_num_threads()},"
        f" OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')},"
        f" OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND', 'unset')}"
    )
    if usable_cpus == 1 and int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) > 1:
        logger.warning(
            "Only 1 usable CPU despite SLURM_CPUS_PER_TASK="
            f"{os.environ.get('SLURM_CPUS_PER_TASK')}. Something has pinned this"
            " process; dataloader workers will inherit it and the GPUs will starve."
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

    # Create training arguments
    targs = Seq2SeqTrainingArguments(**trainer_args_dict(cfg))

    ##########################
    ## PROCESSOR SETUP
    ##########################
    processor = prepare_processor(cfg)

    ##########################
    ## MODEL PREPARATION
    ##########################
    model, last_checkpoint, config, processor = prepare_model(cfg, targs, processor)
    logger.info("Model prepared!")

    # Upload the fully-resolved config (after env-var expansion and CLI
    # overrides) as a wandb artifact for reproducibility. Write the file into
    # the output directory so it is easy to inspect locally.
    #
    # This runs AFTER prepare_model on purpose. prepare_model refuses to start
    # into an output directory that already holds something, and it runs on
    # every rank -- so writing this file beforehand made the global master
    # populate the directory that all four ranks were about to inspect, and a
    # fresh run with wandb enabled and overwrite_output_dir=false failed every
    # time. The check is about output from a *previous* run, so nothing of ours
    # may land in that directory until it has passed.
    if "wandb" in cfg.trainer.report_to and is_global_master:
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

    ##########################
    ## TRAINING
    ##########################
    compute_metrics = None
    if hasattr(cfg, "evaluation") and cfg.evaluation is not None:
        # TODO: we might extend this class to other metrics, its CPU-bound WER/CER computation for now
        train_evaluator = TrainingEvaluator(cfg.evaluation, processor)
        compute_metrics = train_evaluator

    # No explicit train_dataset/eval_dataset - they are handled by Lhotse
    trainer = MELTTrainer(
        model=model,
        args=targs,
        config=cfg,
        processor=processor,
        compute_metrics=compute_metrics
    )

    # Determine checkpoint to resume from.
    # - True  → pass through so HF Trainer auto-detects the last checkpoint in output_dir.
    # - str   → prepare_model already resolved it to the actual checkpoint path via
    #           get_last_checkpoint(); use that resolved path.
    # - None  → fall back to any checkpoint auto-detected from output_dir.
    checkpoint = None
    if targs.resume_from_checkpoint is True:
        checkpoint = True
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
    #
    # Left disabled on purpose, tracked in issue #91.  Enabling it is what would
    # make save_model() below actually write weights under SHARDED_STATE_DICT:
    # Trainer.save_model gates the FSDP save on the plugin reporting
    # FULL_STATE_DICT, so today the call is a silent no-op for weights.  For now
    # the weights are consolidated after the fact with utils/merge_fsdp_weight.py
    # and verify_saved_weights() below makes sure that step is never forgotten.

    logger.info("Saving model, processor, and config...")
    trainer.save_model()

    if is_global_master:
        processor.save_pretrained(targs.output_dir)
        config.save_pretrained(targs.output_dir)

        # Save config for reproducibility
        config_path = str(Path(targs.output_dir) / "training_config.yaml")
        save_config(cfg, config_path)
        logger.info(f"Saved training config to {config_path}")

        # Everything above writes regardless of whether the weights made it to
        # disk, so a run that saved nothing still leaves a plausible-looking
        # directory.  Check, and report loudly if the weights are missing --
        # without failing the job, which really did complete.  See issue #91.
        verify_saved_weights(targs.output_dir)


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
