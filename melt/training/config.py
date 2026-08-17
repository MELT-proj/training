"""Configuration module for MELT training using OmegaConf.

This module provides a simplified configuration system using OmegaConf.
It supports:
- Loading configs from YAML files
- CLI overrides with hierarchical dot notation (e.g., --run.exp_name my_exp)
- Environment variable expansion via ${VAR} syntax
- Nested configuration structure matching the YAML layout

Config hierarchy:
- model: encoder, decoder, adapter settings
- data: train_ds, validation_ds settings
- trainer: TrainingArguments-compatible settings
- optimization: learning rate settings for different components
- run: run-specific settings (exp_name, dry_run, etc.)

Usage:
    python src/train.py --config config/train/asr.yaml
    python src/train.py --config config/train/asr.yaml --trainer.max_steps 1000
    python src/train.py --config config/train/asr.yaml --run.exp_name my_exp
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from omegaconf import DictConfig, OmegaConf

from transformers import TrainingArguments


# =============================================================================
# OmegaConf Custom Resolvers
# =============================================================================


def _register_resolvers():
    """Register custom OmegaConf resolvers for environment variable expansion."""
    # Register ${oc.env:VAR} resolver if not already registered
    # OmegaConf has built-in oc.env resolver, but we also support ${VAR} syntax
    # by using OmegaConf's native environment variable interpolation
    pass


_register_resolvers()


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = """
# =============================================================================
# Run Configuration (non-TrainingArguments settings)
# =============================================================================
run:
  exp_name: null          # Experiment name for wandb
  dry_run: false          # Run in dry-run mode (no actual training)
  config: null            # Path to config file (set via CLI)
  memory_profiling: false # Enable PyTorch CUDA memory snapshot on OOM (writes .pkl to output_dir)
  memory_preallocation: false # Run a max-length warmup forward+backward pass before training to preallocate CUDA buffers

# =============================================================================
# Model Configuration
# =============================================================================
model:
  encoder:
    name: facebook/w2v-bert-2.0
    freeze: true
    max_audio_seq_len: 1500  # for frames of 20ms, this is 30s

  decoder:
    name: Qwen/Qwen2.5-0.5B
    attn_implementation: flash_attention_2
    audio_bos_token: "<|audio_bos|>"
    audio_eos_token: "<|audio_eos|>"
    audio_token: "<|audio|>"
    freeze: true

  adapter:
    _type: mlp
    freeze: false
    hidden_size: 1024
    num_hidden_layers: 2
    intermediate_size: 4096
    hidden_act: gelu
    dropout: 0.1
    downsample_rate: 5
    window_size: 15
    num_adapter_layers: 1
    layerdrop: 0.0
    adapter_kernel_size: 3
    adapter_stride: 2
    mlp_hidden_size: null

  lora:
    enabled: false
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: null  # null = PEFT auto-detection; or list e.g. [q_proj, v_proj]
    bias: none

  ckpt: null  # Path to checkpoint to resume from

# =============================================================================
# Data Configuration
# =============================================================================
data:
  sample_rate: 16000
  apply_chat_template: false
  # Which assistant-span boundaries label masking looks for. Must match the
  # decoder's tokenizer: `chatml` for Qwen (2.5, 3, 3.5) and EuroLLM, `llama3`
  # for Llama 3.x, whose header format is not ChatML and would otherwise be
  # unfindable — leaving every token masked. Checked at startup.
  chat_template_config: chatml
  min_chars: 3
  # When true, a configured `text_field` that resolves to nothing raises
  # instead of silently falling back to the supervision text — but only when
  # a fallback actually exists to fall back to. Turn this on for any mix
  # containing ST sources, whose target (`custom.translation_en`) holds
  # different content from the supervision — the fallback turns such a sample
  # into an ASR pair wearing an ST label. A cut with no text anywhere (no
  # configured field, no supervision text, no `custom.text`) has nothing to
  # mislabel, so it is skipped like the non-strict case instead of raising.
  strict_text_field: false

  train_ds:
    input_cfg: []
    total_hours: null
    total_cuts: null
    force_estimate: null
    batch_size: null
    batch_duration: 120.0
    quadratic_duration: null
    lhotse_sampler_type: dynamic_bucketing  # One among: dynamic_bucketing, dynamic, bucketing
    num_buckets: 30
    buffer_size: 10000
    bucket_duration_bins: null
    shuffle: true
    drop_last: false
    seed: 42
    # 'randomized' resolves to seed + 100*worker_id + 100000*rank, so every
    # (rank, worker) walks its own stream but does so reproducibly. 'trng'
    # also separates the streams, but draws from the system entropy pool, and
    # a sampler checkpoint stores the string rather than the drawn value — a
    # resumed run therefore fast-forwards through a stream it has never seen.
    # Ignored once the sources are indexed: partitioning already separates the
    # streams, so shard order should be identical everywhere, and the loader
    # falls back to `seed` (with a warning) if this is still 'randomized'.
    shard_seed: randomized
    # How Shar sources are read. null auto-detects per source: indexed when the
    # .idx sidecars are present, streaming otherwise. true demands indexed and
    # fails on a source that is not; false forces streaming. Indexed reading
    # partitions by sample index across the (rank x worker) pool, making an
    # epoch exactly 100% of the data -- see issue #52.
    indexed: null
    # How often each dataloader worker records its position, in batches. The
    # snapshot is an iterator position rather than data, so 1 is cheap and means
    # a resumed run picks up at the exact batch instead of the last multiple.
    # Training only -- eval has no position worth resuming.
    snapshot_every_n_steps: 1
    min_duration: 0.5
    max_duration: 30.0
    max_tokens: null
    max_tps: null
    num_workers: 2
    pin_memory: true
    prefetch_factor: 2
    sample_rate: 16000
    text_field: text
    lang_field: lang

  validation_ds:
    input_cfg: []
    total_hours: null
    total_cuts: null
    force_estimate: null
    batch_size: 8
    batch_duration: null
    lhotse_sampler_type: dynamic
    num_buckets: 30
    shuffle: false
    drop_last: false
    seed: 42
    shard_seed: randomized     # as above: same separation, but reproducible
    min_duration: 0.5
    max_duration: 30.0
    max_tokens: null
    max_tps: null
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
    sample_rate: 16000
    text_field: text
    lang_field: lang

# =============================================================================
# Trainer Configuration (TrainingArguments-compatible)
# =============================================================================
trainer:
  output_dir: ./outputs
  overwrite_output_dir: true
  seed: 42
  do_train: true
  do_eval: true
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  lr_scheduler_type: cosine
  warmup_steps: 2000
  num_train_epochs: null
  max_steps: null
  logging_strategy: steps
  logging_steps: 25
  report_to:
    - none
  eval_strategy: "no"
  eval_steps: 3000
  save_strategy: steps
  save_steps: 1000
  save_total_limit: 5
  bf16: true
  group_by_length: false
  # Eval-only: the train path takes its worker count from data.train_ds.num_workers.
  # 4 is the point where added workers stop paying off on a multi-GPU node.
  dataloader_num_workers: 4
  remove_unused_columns: false
  ddp_find_unused_parameters: false
  resume_from_checkpoint: null

# =============================================================================
# Optimization Configuration
# =============================================================================
optimization:
  adam_beta1: 0.9
  adam_beta2: 0.95
  encoder_lr: 6.0e-6
  decoder_lr: 2.0e-5
  adapter_lr: 2.0e-4
  min_lr_scale: 0.1
"""


def get_default_config() -> DictConfig:
    """Get the default configuration as a DictConfig."""
    return OmegaConf.create(DEFAULT_CONFIG)


# =============================================================================
# Environment Variable Expansion
# =============================================================================


def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Supports both $VAR and ${VAR} syntax.
    Also supports ${VAR:-default} for default values.
    """
    if not isinstance(value, str):
        return value

    # Handle ${VAR:-default} syntax
    def replace_with_default(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    # Replace ${VAR:-default} patterns
    value = re.sub(r"\$\{([^}:]+)(?::-([^}]*))?\}", replace_with_default, value)

    # Also expand standard $VAR and ${VAR} patterns
    value = os.path.expandvars(value)

    return value


def expand_env_vars_in_config(cfg: DictConfig) -> DictConfig:
    """Recursively expand environment variables in all string fields.

    Args:
        cfg: OmegaConf DictConfig to expand.

    Returns:
        A new DictConfig with environment variables expanded.
    """
    # Convert to container (dict/list) for manipulation
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

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

    expanded = expand_dict(cfg_dict)
    return OmegaConf.create(expanded)


# =============================================================================
# Config Loading and Parsing
# =============================================================================


def load_config(config_path: str | None = None, cli_args: list[str] | None = None) -> DictConfig:
    """Load configuration from YAML file and CLI arguments.

    The configuration is built in layers:
    1. Default configuration
    2. YAML file (if provided)
    3. CLI overrides

    Args:
        config_path: Path to YAML configuration file.
        cli_args: List of CLI arguments in dotlist format (e.g., ["trainer.max_steps=100"]).

    Returns:
        Merged DictConfig with all configurations applied.
    """
    # Start with defaults
    cfg = get_default_config()

    # Load YAML config if provided
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        yaml_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    # Apply CLI overrides
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def parse_args_and_load_config() -> DictConfig:
    """Parse command-line arguments and load configuration.

    Supports:
    - --config <path>: Path to YAML config file
    - Hierarchical overrides: --trainer.max_steps 100, --run.exp_name my_exp

    Returns:
        Merged DictConfig with all configurations applied.
    """
    # First pass: extract --config argument
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--help", "-h", action="store_true", help="Show help message")

    known_args, remaining = parser.parse_known_args()

    if known_args.help:
        print(__doc__)
        print("\nUsage: python src/train.py --config <config.yaml> [overrides...]")
        print("\nOverrides use dot notation: --trainer.max_steps 100 --run.exp_name my_exp")
        print("\nConfiguration sections:")
        print("  run.*          : Run settings (exp_name, dry_run)")
        print("  model.*        : Model settings (encoder, decoder, adapter)")
        print("  data.*         : Data settings (train_ds, validation_ds)")
        print("  trainer.*      : TrainingArguments settings")
        print("  optimization.* : Learning rate settings")
        sys.exit(0)

    # Convert remaining args to dotlist format
    # --key value or --key=value -> key=value
    dotlist = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key = arg[2:]  # Remove --
            if "=" in key:
                # --key=value format
                dotlist.append(key)
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                # --key value format
                value = remaining[i + 1]
                dotlist.append(f"{key}={value}")
                i += 1
            else:
                # --key with no value (boolean flag)
                dotlist.append(f"{key}=true")
        i += 1

    # Load config
    cfg = load_config(config_path=known_args.config, cli_args=dotlist)

    # Store the config path in the config
    if known_args.config:
        cfg.run.config = known_args.config

    return cfg


def load_config_from_yaml(config_path: str) -> DictConfig:
    """Load configuration from a YAML file only (no defaults applied).

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        DictConfig loaded from the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


# =============================================================================
# Config Utilities
# =============================================================================


def save_config(cfg: DictConfig, path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        cfg: Configuration to save.
        path: Path to save the YAML file.
    """
    # Resolve any interpolations before saving
    resolved = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        OmegaConf.save(OmegaConf.create(resolved), f)


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert configuration to a plain dictionary.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        Plain dictionary representation.
    """
    return OmegaConf.to_container(cfg, resolve=True)


def trainer_args_dict(cfg: DictConfig) -> dict:
    """Extract TrainingArguments-compatible dict from config.

    The trainer section of the config maps directly to HuggingFace
    TrainingArguments. This function extracts and validates those fields.

    Args:
        cfg: Full configuration DictConfig.

    Returns:
        Dictionary suitable for TrainingArguments(**dict).
    """
    trainer_cfg = cfg.get("trainer", {})

    # Convert to dict and expand env vars in output_dir
    result = OmegaConf.to_container(trainer_cfg, resolve=True)

    # Expand environment variables in output_dir
    if "output_dir" in result:
        result["output_dir"] = expand_env_vars(result["output_dir"])
        result["output_dir"] = os.path.expanduser(result["output_dir"])

    # Handle exp_name from run section (previously was in trainer)
    # exp_name is not a TrainingArgument, so we don't include it

    # Get valid TrainingArguments keys
    valid_keys = set(TrainingArguments(output_dir=".").to_dict().keys())

    # Filter to only valid TrainingArguments
    result = {k: v for k, v in result.items() if k in valid_keys}

    return result


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "DictConfig",
    "OmegaConf",
    "get_default_config",
    "load_config",
    "load_config_from_yaml",
    "parse_args_and_load_config",
    "save_config",
    "config_to_dict",
    "trainer_args_dict",
    "expand_env_vars",
    "expand_env_vars_in_config",
]
