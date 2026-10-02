"""MELT self-distillation entrypoint (online AZeroS-style modality alignment).

Same run as ``melt.training.train`` -- same config sections, data pipeline,
model construction, freezing, checkpointing and W&B setup -- with the loss
swapped for :class:`~melt.training.self_distill.MELTSelfDistillTrainer`'s.
See ``melt/training/self_distill.py`` for the method and ``docs/self_distillation.md``
for the design notes.

The config is layered, so a method overlay can sit on top of a campaign
config without copying its ~600 lines of data mixture:

    defaults  <-  base_config (named inside the overlay)  <-  overlay  <-  CLI

Usage:
    python -m melt.training.train_self_distill \\
        --config projects/ablation-campaign/self-distill.yaml --distill.lmbda 0
"""

import argparse
import inspect
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..logging_utils import configure_logging, get_logger
from . import train as standard_train
from .config import expand_env_vars_in_config, get_default_config
from .self_distill import MELTSelfDistillTrainer, SelfDistillConfig, _generalized_jsd_loss


logger = get_logger(__name__)


def _to_dotlist(args: list[str]) -> list[str]:
    """``--a.b value`` / ``--a.b=value`` / ``--flag`` -> OmegaConf dotlist.

    Mirrors ``config.parse_args_and_load_config``; ``tests/test_self_distill.py``
    pins the two to the same result so they cannot drift apart.
    """
    dotlist = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                dotlist.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                dotlist.append(f"{key}={args[i + 1]}")
                i += 1
            else:
                dotlist.append(f"{key}=true")
        i += 1
    return dotlist


def load_layered_config(argv: list[str]) -> DictConfig:
    """Build the run config from an overlay that names its ``base_config``.

    ``base_config`` is resolved relative to the overlay file, so the overlay
    can live next to the campaign config it extends. An overlay without one is
    an ordinary single-file config.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True)
    known, remaining = parser.parse_known_args(argv)

    overlay_path = Path(known.config)
    overlay = OmegaConf.load(overlay_path)
    layers = [get_default_config()]
    base_config = overlay.pop("base_config", None)
    if base_config is not None:
        base_path = (overlay_path.parent / base_config).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"base_config {base_config!r} (from {overlay_path}) not found at {base_path}")
        layers.append(OmegaConf.load(base_path))
    layers.append(overlay)
    dotlist = _to_dotlist(remaining)
    if dotlist:
        layers.append(OmegaConf.from_dotlist(dotlist))

    cfg = OmegaConf.merge(*layers)
    cfg.run.config = str(overlay_path)
    cfg.run.base_config = str(base_path) if base_config is not None else None
    return cfg


def install_self_distill_trainer() -> None:
    """Make ``train.main`` construct :class:`MELTSelfDistillTrainer`.

    ``train.main`` constructs its trainer through the module-level name
    ``MELTTrainer``; rebinding that name in this process is what swaps the
    objective while leaving the standard entrypoint untouched. The source check
    makes the swap fail loudly, instead of silently training the gold-transcript
    objective, if ``train.main`` ever stops constructing it that way.
    """
    if "MELTTrainer(" not in inspect.getsource(standard_train.main):
        raise RuntimeError(
            "melt.training.train.main no longer constructs `MELTTrainer(...)`; "
            "train_self_distill cannot swap in its trainer."
        )
    standard_train.MELTTrainer = MELTSelfDistillTrainer


def main(cfg: DictConfig) -> None:
    """Run ``train.main`` with the self-distillation trainer."""
    # Validate, and import TRL, before any model loads.
    SelfDistillConfig.from_config(cfg.get("distill"))
    _generalized_jsd_loss()
    install_self_distill_trainer()
    standard_train.main(cfg)


if __name__ == "__main__":
    configure_logging()
    cfg = expand_env_vars_in_config(load_layered_config(sys.argv[1:]))
    if cfg.run.get("dry_run", False):
        logger.info("Dry run mode - config parsed successfully")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    else:
        main(cfg)
