"""Save MELT processor and model config from a training OmegaConf.

This utility parses the same training config format used by `src/training/train.py`,
creates a `MELTProcessor` and `MELTConfig` from it, and saves both to
`trainer.output_dir`.

Usage examples:
    python infra/save_training_artifacts.py --config config/train/asr.yaml
    python infra/save_training_artifacts.py --config config/train/asr.yaml --trainer.output_dir ./outputs/run-1
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoFeatureExtractor, AutoTokenizer, TrainingArguments

from src.logging_utils import configure_logging, get_logger
from src.modeling import MELTConfig, MELTProcessor
from src.training.config import (
    DictConfig,
    expand_env_vars_in_config,
    parse_args_and_load_config,
    save_config,
    trainer_args_dict,
)


logger = get_logger(__name__)


def prepare_processor(cfg: DictConfig) -> MELTProcessor:
    """Build the MELT processor from training config."""
    logger.info("Loading processor for encoder=%s, decoder=%s", cfg.model.encoder.name, cfg.model.decoder.name)
    return MELTProcessor(
        feature_extractor=AutoFeatureExtractor.from_pretrained(cfg.model.encoder.name),
        tokenizer=AutoTokenizer.from_pretrained(cfg.model.decoder.name, use_fast=True),
        config=cfg.model,
    )


def prepare_melt_config(cfg: DictConfig, processor: MELTProcessor) -> MELTConfig:
    """Build MELTConfig from training config and processor special tokens."""
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    decoder_cfg = model_cfg.decoder
    adapter_cfg = model_cfg.adapter

    max_audio_seq_len = encoder_cfg.get("max_audio_seq_len", 1500)

    config = MELTConfig(
        audio_encoder=encoder_cfg.name,
        text_decoder=decoder_cfg.name,
        adapter_config=adapter_cfg,
        decoder_kwargs={"attn_implementation": decoder_cfg.get("attn_implementation", "sdpa")},
        max_audio_seq_len=max_audio_seq_len,
    )

    config.audio_bos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_bos_token])[0]
    config.audio_eos_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_eos_token])[0]
    config.pad_token_id = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.pad_token])[0]
    config.audio_token_id = processor.tokenizer.convert_tokens_to_ids([processor.audio_token])[0]
    config.audio_encoder_config.max_audio_seq_len = max_audio_seq_len

    return config


def main() -> None:
    """Parse config, prepare artifacts, and save to trainer.output_dir."""
    configure_logging()

    cfg = parse_args_and_load_config()
    cfg = expand_env_vars_in_config(cfg)

    targs = TrainingArguments(**trainer_args_dict(cfg))
    output_dir = Path(targs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = prepare_processor(cfg)
    melt_config = prepare_melt_config(cfg, processor)

    logger.info("Saving processor and MELT config to %s", output_dir)
    processor.save_pretrained(output_dir)
    melt_config.save_pretrained(output_dir)

    config_path = output_dir / "training_config.yaml"
    save_config(cfg, str(config_path))
    logger.info("Saved training config to %s", config_path)


if __name__ == "__main__":
    main()
