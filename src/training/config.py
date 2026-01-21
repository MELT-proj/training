"""Configuration dataclasses for MELT training.

This module defines configuration dataclasses that can be used with tyro
for CLI argument parsing. Supports both flat CLI args (e.g., --learning_rate 0.001)
and hierarchical specification (e.g., --model.encoder.freeze).

Usage:
    python src/train.py --model.encoder.name facebook/w2v-bert-2.0
    python src/train.py --trainer.max_steps 1000 --trainer.learning_rate 2e-5
"""

from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


# =============================================================================
# Data Source Configuration
# =============================================================================


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    type: str = "lhotse_shar"
    """Type of data source: 'lhotse_shar' or 'lhotse_cuts'."""

    shar_path: str | None = None
    """Path to shar directory (for lhotse_shar type)."""

    cuts_path: str | None = None
    """Path to cuts file (for lhotse_cuts type)."""

    weight: float = 1.0
    """Weight for mixing multiple data sources."""

    tags: dict[str, str] = field(default_factory=lambda: {"task": "asr", "lang": "en"})
    """Tags to add to cuts (e.g., task, language)."""


@dataclass
class DatasetConfig:
    """Configuration for a dataset split (train or validation)."""

    # Data sources
    input_cfg: list[DataSourceConfig] = field(default_factory=list)
    """List of data source configurations."""

    # Batch settings
    batch_size: int | None = None
    """Fixed batch size (mutually exclusive with batch_duration)."""

    batch_duration: float | None = 120.0
    """Target batch duration in seconds for dynamic batching."""

    quadratic_duration: float | None = None
    """Quadratic duration constraint for attention-heavy models."""

    # Bucketing for efficient batching
    use_bucketing: bool = True
    """Whether to use duration bucketing for efficient batching."""

    num_buckets: int = 30
    """Number of duration buckets."""

    bucket_buffer_size: int = 10000
    """Buffer size for bucket sampling."""

    bucket_duration_bins: list[float] | None = None
    """Custom duration bins for bucketing (auto-estimated if None)."""

    # Sampling
    shuffle: bool = True
    """Whether to shuffle the data."""

    shuffle_buffer_size: int = 10000
    """Buffer size for shuffling."""

    drop_last: bool = False
    """Whether to drop the last incomplete batch."""

    seed: int = 42
    """Random seed for reproducibility."""

    shard_seed: str | int = "trng"
    """Seed for shard shuffling ('trng' for true random)."""

    # Duration filtering
    min_duration: float = 0.5
    """Minimum audio duration in seconds."""

    max_duration: float = 30.0
    """Maximum audio duration in seconds."""

    # DataLoader settings
    num_workers: int = 2
    """Number of data loading workers."""

    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per worker."""

    sample_rate: int = 16000
    """Target audio sample rate."""

    text_field: str = "text"
    """Field name for text in the data."""

    lang_field: str = "lang"
    """Field name for language in the data."""


@dataclass
class DataConfig:
    """Top-level data configuration."""

    sample_rate: int = 16000
    """Audio sample rate."""

    apply_chat_template: bool = False
    """Whether to apply chat template to text."""

    min_chars: int = 3
    """Minimum number of characters for valid text."""

    train_ds: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        input_cfg=[
            DataSourceConfig(
                type="lhotse_shar",
                shar_path="${LOCAL_DATASETS_DIR}/librispeech/clean/train.100",
                tags={"task": "asr", "lang": "en"},
            ),
            DataSourceConfig(
                type="lhotse_shar",
                shar_path="${LOCAL_DATASETS_DIR}/librispeech/clean/train.360",
                tags={"task": "asr", "lang": "en"},
            ),
        ],
        batch_duration=120.0,
        use_bucketing=True,
        shuffle=True,
    ))
    """Training dataset configuration."""

    validation_ds: DatasetConfig = field(default_factory=lambda: DatasetConfig(
        input_cfg=[
            DataSourceConfig(
                type="lhotse_shar",
                shar_path="${LOCAL_DATASETS_DIR}/librispeech/clean/validation",
                tags={"task": "asr", "lang": "en"},
            ),
        ],
        batch_size=8,
        batch_duration=None,
        use_bucketing=False,
        shuffle=False,
        num_workers=4,
    ))
    """Validation dataset configuration."""


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class EncoderConfig:
    """Audio encoder configuration."""

    name: str = "facebook/w2v-bert-2.0"
    """Pretrained encoder model name or path."""

    freeze: bool = True
    """Whether to freeze encoder weights."""


@dataclass
class DecoderConfig:
    """Text decoder configuration."""

    name: str = "Qwen/Qwen2.5-0.5B"
    """Pretrained decoder model name or path."""

    attn_implementation: str = "flash_attention_2"
    """Attention implementation to use."""

    audio_bos_token: str = "<|audio_bos|>"
    """Beginning of audio token."""

    audio_eos_token: str = "<|audio_eos|>"
    """End of audio token."""

    audio_token: str = "<|AUDIO|>"
    """Audio placeholder token."""

    freeze: bool = True
    """Whether to freeze decoder weights."""


@dataclass
class AdapterConfig:
    """Adapter configuration."""

    type: str = "mlp"
    """Adapter type (e.g., 'mlp', 'linear')."""

    freeze: bool = False
    """Whether to freeze adapter weights."""

    # add_adapter: bool = False
    # """Whether to add adapter layers."""


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    """Audio encoder configuration."""

    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    """Text decoder configuration."""

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    """Adapter configuration."""

    ckpt: str | None = None
    """Path to checkpoint to resume from."""


# =============================================================================
# Trainer Configuration
# =============================================================================


@dataclass
class TrainerConfig:
    """Training configuration compatible with HuggingFace TrainingArguments."""

    output_dir: str = "${OUTPUT_DIR}/LS_asr"
    """Output directory for checkpoints and logs."""

    overwrite_output_dir: bool = True
    """Whether to overwrite the output directory."""

    seed: int = 42
    """Random seed."""

    do_train: bool = True
    """Whether to run training."""

    do_eval: bool = True
    """Whether to run evaluation."""

    # Accelerate DataLoader behavior under Transformers Trainer.
    # If you hit errors about different-shaped batches being concatenated,
    # set `split_batches: true` (slice one big batch per step).
    # accelerator_config:
    #   split_batches: true

    # Batch and accumulation
    per_device_train_batch_size: int = 1
    """Training batch size per device (handled by Lhotse, set to 1)."""

    per_device_eval_batch_size: int = 1
    """Evaluation batch size per device (handled by Lhotse, set to 1)."""

    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps."""

    # Optimizer
    adam_beta1: float = 0.9
    """Adam beta1 parameter."""

    adam_beta2: float = 0.95
    """Adam beta2 parameter."""

    learning_rate: float = 2e-5
    """Learning rate."""

    # LR schedule
    lr_scheduler_type: str = "cosine"
    """Learning rate scheduler type."""

    warmup_steps: int = 2000
    """Number of warmup steps."""

    # Duration
    num_train_epochs: int | None = None
    """Number of training epochs."""

    max_steps: int | None = None
    """Maximum number of training steps (-1 for unlimited)."""

    compute_max_steps_from_epochs: bool = False
    """If True, compute max_steps from num_train_epochs and dataset duration.
    
    When enabled, max_steps will be calculated as:
        max_steps = num_train_epochs * (total_dataset_duration / batch_duration / gradient_accumulation_steps)
    
    This requires the data config to have batch_duration set.
    """

    # Logging
    logging_strategy: str = "steps"
    """Logging strategy ('steps' or 'epoch')."""

    logging_steps: int = 25
    """Log every N steps."""

    report_to: list[str] = field(default_factory=lambda: ["none"])
    """Reporting integrations (e.g., 'wandb', 'tensorboard')."""

    # Evaluation
    eval_strategy: str = "no"
    """Evaluation strategy ('no', 'steps', or 'epoch')."""

    eval_steps: int = 3000
    """Evaluate every N steps."""

    # Checkpointing
    save_strategy: str = "steps"
    """Checkpoint saving strategy ('steps' or 'epoch')."""

    save_steps: int = 1000
    """Save checkpoint every N steps."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep."""

    # Performance
    bf16: bool = True
    """Whether to use bfloat16 precision."""

    group_by_length: bool = False
    """Whether to group by length (handled by Lhotse bucketing)."""

    dataloader_num_workers: int = 0
    """DataLoader workers (handled by Lhotse)."""

    remove_unused_columns: bool = False
    """Whether to remove unused columns."""

    ddp_find_unused_parameters: bool = False
    """Whether to find unused parameters in DDP."""

    resume_from_checkpoint: str | None = None
    """Path to checkpoint to resume from."""


# =============================================================================
# Optimization Configuration
# =============================================================================


@dataclass
class OptimizationConfig:
    """Optimization configuration for different model components."""

    encoder_lr: float = 6e-6
    """Learning rate for encoder."""

    decoder_lr: float = 2e-5
    """Learning rate for decoder."""

    adapter_lr: float = 2e-4
    """Learning rate for adapter."""

    min_lr_scale: float = 0.1
    """Minimum learning rate scale for schedulers."""


# =============================================================================
# Top-level Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This is the main configuration class that combines all sub-configurations.
    It can be used with tyro for CLI argument parsing with hierarchical support.

    Example:
        python src/train.py --model.encoder.name facebook/w2v-bert-2.0
        python src/train.py --trainer.max_steps 1000
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    """Data configuration."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer configuration."""

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    """Optimization configuration."""

    # CLI-only arguments
    dry_run: bool = False
    """Run in dry-run mode (no actual training)."""

    config_file: str | None = None
    """Path to YAML config file to load as base configuration."""


# =============================================================================
# Utility Functions
# =============================================================================


def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Supports both $VAR and ${VAR} syntax, with optional defaults: ${VAR:-default}.
    """
    if not isinstance(value, str):
        return value

    # Handle ${VAR:-default} syntax
    def replace_with_default(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    # Replace ${VAR:-default} patterns
    value = re.sub(r'\$\{([^}:]+)(?::-([^}]*))?\}', replace_with_default, value)

    # Also expand standard $VAR and ${VAR} patterns
    value = os.path.expandvars(value)

    return value


def expand_env_vars_in_config(config: TrainingConfig) -> TrainingConfig:
    """Recursively expand environment variables in all string fields."""

    def expand_dict(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = expand_env_vars(v)
            elif isinstance(v, dict):
                result[k] = expand_dict(v)
            elif isinstance(v, list):
                result[k] = expand_list(v)
            else:
                result[k] = v
        return result

    def expand_list(lst: list) -> list:
        result = []
        for item in lst:
            if isinstance(item, str):
                result.append(expand_env_vars(item))
            elif isinstance(item, dict):
                result.append(expand_dict(item))
            elif isinstance(item, list):
                result.append(expand_list(item))
            else:
                result.append(item)
        return result

    config_dict = asdict(config)
    expanded = expand_dict(config_dict)
    return config_from_dict(expanded)


def config_to_dict(config: TrainingConfig) -> dict:
    """Convert a TrainingConfig to a plain dict."""
    return asdict(config)


def config_from_dict(d: dict) -> TrainingConfig:
    """Create a TrainingConfig from a dict."""

    def parse_data_source(src: dict) -> DataSourceConfig:
        kwargs = {}
        if "type" in src:
            kwargs["type"] = src["type"]
        if "shar_path" in src:
            kwargs["shar_path"] = src["shar_path"]
        if "cuts_path" in src:
            kwargs["cuts_path"] = src["cuts_path"]
        if "weight" in src:
            kwargs["weight"] = src["weight"]
        if "tags" in src:
            kwargs["tags"] = src["tags"]
        return DataSourceConfig(**kwargs)

    def parse_dataset_config(ds: dict) -> DatasetConfig:
        kwargs = {}
        if "input_cfg" in ds:
            kwargs["input_cfg"] = [parse_data_source(src) for src in ds["input_cfg"]]
        if "batch_size" in ds:
            kwargs["batch_size"] = ds["batch_size"]
        if "batch_duration" in ds:
            kwargs["batch_duration"] = ds["batch_duration"]
        if "quadratic_duration" in ds:
            kwargs["quadratic_duration"] = ds["quadratic_duration"]
        if "use_bucketing" in ds:
            kwargs["use_bucketing"] = ds["use_bucketing"]
        if "num_buckets" in ds:
            kwargs["num_buckets"] = ds["num_buckets"]
        if "bucket_buffer_size" in ds:
            kwargs["bucket_buffer_size"] = ds["bucket_buffer_size"]
        if "bucket_duration_bins" in ds:
            kwargs["bucket_duration_bins"] = ds["bucket_duration_bins"]
        if "shuffle" in ds:
            kwargs["shuffle"] = ds["shuffle"]
        if "shuffle_buffer_size" in ds:
            kwargs["shuffle_buffer_size"] = ds["shuffle_buffer_size"]
        if "drop_last" in ds:
            kwargs["drop_last"] = ds["drop_last"]
        if "seed" in ds:
            kwargs["seed"] = ds["seed"]
        if "shard_seed" in ds:
            kwargs["shard_seed"] = ds["shard_seed"]
        if "min_duration" in ds:
            kwargs["min_duration"] = ds["min_duration"]
        if "max_duration" in ds:
            kwargs["max_duration"] = ds["max_duration"]
        if "num_workers" in ds:
            kwargs["num_workers"] = ds["num_workers"]
        if "pin_memory" in ds:
            kwargs["pin_memory"] = ds["pin_memory"]
        if "prefetch_factor" in ds:
            kwargs["prefetch_factor"] = ds["prefetch_factor"]
        if "sample_rate" in ds:
            kwargs["sample_rate"] = ds["sample_rate"]
        if "text_field" in ds:
            kwargs["text_field"] = ds["text_field"]
        if "lang_field" in ds:
            kwargs["lang_field"] = ds["lang_field"]
        return DatasetConfig(**kwargs)

    # Parse model config
    model_dict = d.get("model", {})
    
    encoder_kwargs = {}
    if "encoder" in model_dict:
        enc_dict = model_dict["encoder"]
        if "name" in enc_dict:
            encoder_kwargs["name"] = enc_dict["name"]
        if "freeze" in enc_dict:
            encoder_kwargs["freeze"] = enc_dict["freeze"]
    encoder = EncoderConfig(**encoder_kwargs)
    
    decoder_kwargs = {}
    if "decoder" in model_dict:
        dec_dict = model_dict["decoder"]
        if "name" in dec_dict:
            decoder_kwargs["name"] = dec_dict["name"]
        if "attn_implementation" in dec_dict:
            decoder_kwargs["attn_implementation"] = dec_dict["attn_implementation"]
        if "audio_bos_token" in dec_dict:
            decoder_kwargs["audio_bos_token"] = dec_dict["audio_bos_token"]
        if "audio_eos_token" in dec_dict:
            decoder_kwargs["audio_eos_token"] = dec_dict["audio_eos_token"]
        if "audio_token" in dec_dict:
            decoder_kwargs["audio_token"] = dec_dict["audio_token"]
        if "freeze" in dec_dict:
            decoder_kwargs["freeze"] = dec_dict["freeze"]
    decoder = DecoderConfig(**decoder_kwargs)
    
    adapter_kwargs = {}
    if "adapter" in model_dict:
        adp_dict = model_dict["adapter"]
        if "type" in adp_dict:
            adapter_kwargs["type"] = adp_dict["type"]
        if "freeze" in adp_dict:
            adapter_kwargs["freeze"] = adp_dict["freeze"]
    adapter = AdapterConfig(**adapter_kwargs)
    
    model_kwargs = {
        "encoder": encoder,
        "decoder": decoder,
        "adapter": adapter,
    }
    if "ckpt" in model_dict:
        model_kwargs["ckpt"] = model_dict["ckpt"]
    model = ModelConfig(**model_kwargs)

    # Parse data config
    data_dict = d.get("data", {})
    data_kwargs = {}
    if "sample_rate" in data_dict:
        data_kwargs["sample_rate"] = data_dict["sample_rate"]
    if "apply_chat_template" in data_dict:
        data_kwargs["apply_chat_template"] = data_dict["apply_chat_template"]
    if "min_chars" in data_dict:
        data_kwargs["min_chars"] = data_dict["min_chars"]
    if "train_ds" in data_dict:
        data_kwargs["train_ds"] = parse_dataset_config(data_dict["train_ds"])
    if "validation_ds" in data_dict:
        data_kwargs["validation_ds"] = parse_dataset_config(data_dict["validation_ds"])
    data = DataConfig(**data_kwargs)

    # Parse trainer config
    trainer_dict = d.get("trainer", {})
    trainer_kwargs = {}
    if "output_dir" in trainer_dict:
        trainer_kwargs["output_dir"] = trainer_dict["output_dir"]
    if "overwrite_output_dir" in trainer_dict:
        trainer_kwargs["overwrite_output_dir"] = bool(trainer_dict["overwrite_output_dir"])
    if "seed" in trainer_dict:
        trainer_kwargs["seed"] = int(trainer_dict["seed"])
    if "do_train" in trainer_dict:
        trainer_kwargs["do_train"] = bool(trainer_dict["do_train"])
    if "do_eval" in trainer_dict:
        trainer_kwargs["do_eval"] = bool(trainer_dict["do_eval"])
    if "per_device_train_batch_size" in trainer_dict:
        trainer_kwargs["per_device_train_batch_size"] = int(trainer_dict["per_device_train_batch_size"])
    if "per_device_eval_batch_size" in trainer_dict:
        trainer_kwargs["per_device_eval_batch_size"] = int(trainer_dict["per_device_eval_batch_size"])
    if "gradient_accumulation_steps" in trainer_dict:
        trainer_kwargs["gradient_accumulation_steps"] = int(trainer_dict["gradient_accumulation_steps"])
    if "adam_beta1" in trainer_dict:
        trainer_kwargs["adam_beta1"] = float(trainer_dict["adam_beta1"])
    if "adam_beta2" in trainer_dict:
        trainer_kwargs["adam_beta2"] = float(trainer_dict["adam_beta2"])
    if "learning_rate" in trainer_dict:
        trainer_kwargs["learning_rate"] = float(trainer_dict["learning_rate"])
    if "lr_scheduler_type" in trainer_dict:
        trainer_kwargs["lr_scheduler_type"] = str(trainer_dict["lr_scheduler_type"])
    if "warmup_steps" in trainer_dict:
        trainer_kwargs["warmup_steps"] = int(trainer_dict["warmup_steps"])
    if "num_train_epochs" in trainer_dict:
        if trainer_dict["num_train_epochs"] is not None:
            trainer_kwargs["num_train_epochs"] = int(trainer_dict["num_train_epochs"])
    if "max_steps" in trainer_dict:
        if trainer_dict["max_steps"] is not None:
            trainer_kwargs["max_steps"] = int(trainer_dict["max_steps"])
    if "compute_max_steps_from_epochs" in trainer_dict:
        trainer_kwargs["compute_max_steps_from_epochs"] = bool(trainer_dict["compute_max_steps_from_epochs"])
    if "logging_strategy" in trainer_dict:
        trainer_kwargs["logging_strategy"] = str(trainer_dict["logging_strategy"])
    if "logging_steps" in trainer_dict:
        trainer_kwargs["logging_steps"] = int(trainer_dict["logging_steps"])
    if "report_to" in trainer_dict:
        report_to = trainer_dict["report_to"]
        if isinstance(report_to, str):
            report_to = [report_to]
        trainer_kwargs["report_to"] = report_to
    if "eval_strategy" in trainer_dict:
        trainer_kwargs["eval_strategy"] = str(trainer_dict["eval_strategy"])
    if "eval_steps" in trainer_dict:
        trainer_kwargs["eval_steps"] = int(trainer_dict["eval_steps"])
    if "save_strategy" in trainer_dict:
        trainer_kwargs["save_strategy"] = str(trainer_dict["save_strategy"])
    if "save_steps" in trainer_dict:
        trainer_kwargs["save_steps"] = int(trainer_dict["save_steps"])
    if "save_total_limit" in trainer_dict:
        trainer_kwargs["save_total_limit"] = int(trainer_dict["save_total_limit"])
    if "bf16" in trainer_dict:
        trainer_kwargs["bf16"] = bool(trainer_dict["bf16"])
    if "group_by_length" in trainer_dict:
        trainer_kwargs["group_by_length"] = bool(trainer_dict["group_by_length"])
    if "dataloader_num_workers" in trainer_dict:
        trainer_kwargs["dataloader_num_workers"] = int(trainer_dict["dataloader_num_workers"])
    if "remove_unused_columns" in trainer_dict:
        trainer_kwargs["remove_unused_columns"] = bool(trainer_dict["remove_unused_columns"])
    if "ddp_find_unused_parameters" in trainer_dict:
        trainer_kwargs["ddp_find_unused_parameters"] = bool(trainer_dict["ddp_find_unused_parameters"])
    if "resume_from_checkpoint" in trainer_dict:
        trainer_kwargs["resume_from_checkpoint"] = trainer_dict["resume_from_checkpoint"]
    trainer = TrainerConfig(**trainer_kwargs)

    # Parse optimization config
    opt_dict = d.get("optimization", {})
    opt_kwargs = {}
    if "encoder_lr" in opt_dict:
        opt_kwargs["encoder_lr"] = float(opt_dict["encoder_lr"])
    if "decoder_lr" in opt_dict:
        opt_kwargs["decoder_lr"] = float(opt_dict["decoder_lr"])
    if "adapter_lr" in opt_dict:
        opt_kwargs["adapter_lr"] = float(opt_dict["adapter_lr"])
    if "min_lr_scale" in opt_dict:
        opt_kwargs["min_lr_scale"] = float(opt_dict["min_lr_scale"])
    optimization = OptimizationConfig(**opt_kwargs)

    config_kwargs = {
        "model": model,
        "data": data,
        "trainer": trainer,
        "optimization": optimization,
    }
    if "dry_run" in d:
        config_kwargs["dry_run"] = d["dry_run"]
    if "config_file" in d:
        config_kwargs["config_file"] = d["config_file"]
    return TrainingConfig(**config_kwargs)


def load_config_from_yaml(config_file: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    return config_from_dict(config_dict)


def merge_configs(base: TrainingConfig, override: TrainingConfig) -> TrainingConfig:
    """Merge two configs, with override taking precedence for non-default values.

    This is used to merge a YAML config with CLI overrides.
    """
    base_dict = asdict(base)
    override_dict = asdict(override)

    def merge_dicts(base_d: dict, override_d: dict, defaults: dict) -> dict:
        result = base_d.copy()
        for key, value in override_d.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    # Recursively merge nested dicts
                    default_nested = defaults.get(key, {}) if isinstance(defaults.get(key), dict) else {}
                    result[key] = merge_dicts(result[key], value, default_nested)
                elif value != defaults.get(key):
                    # Override if value differs from default
                    result[key] = value
            else:
                result[key] = value
        return result

    # Get default config for comparison
    default_config = TrainingConfig()
    default_dict = asdict(default_config)

    merged_dict = merge_dicts(base_dict, override_dict, default_dict)
    return config_from_dict(merged_dict)


def save_config(config: TrainingConfig, path: str) -> None:
    """Save a TrainingConfig to a YAML file."""
    config_dict = asdict(config)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def trainer_args_dict(config: TrainingConfig) -> dict:
    """Extract a Transformers TrainingArguments-compatible dict from config."""
    trainer_cfg = config.trainer

    # Expand environment variables in output_dir
    output_dir = expand_env_vars(trainer_cfg.output_dir)
    output_dir = os.path.expanduser(output_dir)

    result = {
        "output_dir": output_dir,
        "overwrite_output_dir": trainer_cfg.overwrite_output_dir,
        "seed": trainer_cfg.seed,
        "do_train": trainer_cfg.do_train,
        "do_eval": trainer_cfg.do_eval,
        "per_device_train_batch_size": trainer_cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": trainer_cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": trainer_cfg.gradient_accumulation_steps,
        "adam_beta1": trainer_cfg.adam_beta1,
        "adam_beta2": trainer_cfg.adam_beta2,
        "learning_rate": trainer_cfg.learning_rate,
        "lr_scheduler_type": trainer_cfg.lr_scheduler_type,
        "warmup_steps": trainer_cfg.warmup_steps,
        "num_train_epochs": trainer_cfg.num_train_epochs,
        "max_steps": trainer_cfg.max_steps,
        "logging_strategy": trainer_cfg.logging_strategy,
        "logging_steps": trainer_cfg.logging_steps,
        "report_to": trainer_cfg.report_to,
        "eval_strategy": trainer_cfg.eval_strategy,
        "eval_steps": trainer_cfg.eval_steps,
        "save_strategy": trainer_cfg.save_strategy,
        "save_steps": trainer_cfg.save_steps,
        "save_total_limit": trainer_cfg.save_total_limit,
        "bf16": trainer_cfg.bf16,
        "group_by_length": trainer_cfg.group_by_length,
        "dataloader_num_workers": trainer_cfg.dataloader_num_workers,
        "remove_unused_columns": trainer_cfg.remove_unused_columns,
        "ddp_find_unused_parameters": trainer_cfg.ddp_find_unused_parameters,
    }

    if trainer_cfg.resume_from_checkpoint is not None:
        result["resume_from_checkpoint"] = trainer_cfg.resume_from_checkpoint

    return result


__all__ = [
    "DataSourceConfig",
    "DatasetConfig",
    "DataConfig",
    "EncoderConfig",
    "DecoderConfig",
    "AdapterConfig",
    "ModelConfig",
    "TrainerConfig",
    "OptimizationConfig",
    "TrainingConfig",
    "load_config_from_yaml",
    "save_config",
    "config_to_dict",
    "config_from_dict",
    "merge_configs",
    "trainer_args_dict",
    "expand_env_vars",
    "expand_env_vars_in_config",
]
