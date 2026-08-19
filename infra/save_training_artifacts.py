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

from transformers import Seq2SeqTrainingArguments

from src.logging_utils import configure_logging, get_logger
from src.training.config import (
    expand_env_vars_in_config,
    parse_args_and_load_config,
    save_config,
    trainer_args_dict,
)
from src.training.setup import prepare_melt_config, prepare_processor


logger = get_logger(__name__)


def main() -> None:
    """Parse config, prepare artifacts, and save to trainer.output_dir."""
    configure_logging()

    cfg = parse_args_and_load_config()
    cfg = expand_env_vars_in_config(cfg)

    targs = Seq2SeqTrainingArguments(**trainer_args_dict(cfg))
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
