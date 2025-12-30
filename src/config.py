"""Configuration utilities for MELT training.

This project uses hierarchical YAML configs loaded via OmegaConf.

Per the simplified configuration scheme, Python keeps configuration as an
OmegaConf ``DictConfig`` and only defines a minimal CLI ``TrainingArgs``
dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class TrainingArgs:
    """CLI args for training."""

    config_file: str
    dry_run: bool = False


def load_config(config_file: str, dotlist_overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config and apply dotlist overrides (e.g. ``trainer.max_steps=10``)."""

    cfg = OmegaConf.load(config_file)
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))

    # Ensure common environment variables used in the YAML are at least set to defaults
    # so tests and non-cluster runs don't fail during interpolation resolution.
    os.environ.setdefault("SCRATCH", os.path.expanduser("~"))
    os.environ.setdefault("LOCAL_DATASETS_DIR", str(Path.home()))

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Save a resolved config to YAML."""

    OmegaConf.save(config=OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), f=path)


def as_dict(cfg: DictConfig) -> dict[str, object]:
    """Convert config to a resolved plain dict."""

    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise TypeError("Expected config to resolve to a dict")
    return out


def trainer_args_dict(cfg: DictConfig) -> dict[str, object]:
    """Extract a Transformers TrainingArguments-compatible dict from ``cfg.trainer``."""

    if "trainer" not in cfg:
        raise ValueError("Config missing required key: trainer")

    tcfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not isinstance(tcfg, dict):
        raise TypeError("cfg.trainer must be a mapping")

    output_dir = os.path.expandvars(str(tcfg.get("output_dir", "")))
    tcfg["output_dir"] = os.path.expanduser(output_dir)

    report_to = tcfg.get("report_to")
    if isinstance(report_to, str):
        tcfg["report_to"] = [report_to]

    # Transformers expects an AcceleratorConfig instance (not a raw dict).
    # This controls accelerate DataLoader behavior (e.g. dispatch_batches/split_batches).
    accelerator_config = tcfg.get("accelerator_config")
    if isinstance(accelerator_config, dict):
        from transformers.trainer_pt_utils import AcceleratorConfig

        tcfg["accelerator_config"] = AcceleratorConfig(**accelerator_config)

    return tcfg


__all__ = [
    "TrainingArgs",
    "load_config",
    "save_config",
    "as_dict",
    "trainer_args_dict",
]
