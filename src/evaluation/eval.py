"""MELT evaluation entry-point.

Iterates over each test dataset declared in the eval YAML, generates
transcriptions with the model, computes metrics, and writes JSON-lines
results.

Usage:
    python -m src.evaluation.eval --config config/eval/asr.yaml
    python -m src.evaluation.eval --config config/eval/asr.yaml --model.path /new/ckpt
    python -m src.evaluation.eval --config config/eval/asr.yaml --data.test_ds.0.max_cuts 16
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..logging_utils import configure_logging, get_logger
from .backends import ModelBackend, build_backend
from .datasets import EvalSample, build_eval_dataset
from .metrics import METRIC_REGISTRY, get_metric

logger = get_logger(__name__)


# =============================================================================
# Default eval config (minimal — most values should come from YAML)
# =============================================================================

DEFAULT_EVAL_CONFIG = """
run:
  exp_name: melt_eval
  dry_run: false

model:
  backend: melt            # "melt" | future: "hf_pipeline"
  path: null               # local checkpoint path (required)
  encoder:
    name: facebook/w2v-bert-2.0
  decoder:
    name: Qwen/Qwen2.5-0.5B
    audio_token: "<|audio|>"
    audio_bos_token: "<|audio_bos|>"
    audio_eos_token: "<|audio_eos|>"

data:
  test_ds: []              # list of test dataset configs

generation_args:
  max_new_tokens: 256

output:
  results_file: eval_results.jsonl
  report_to:
    - none                 # "wandb" to enable wandb logging
  skip_existing: true      # skip already-evaluated (model, dataset) pairs
"""


# =============================================================================
# Config loading (mirrors src/training/config.py patterns)
# =============================================================================


def get_default_eval_config() -> DictConfig:
    return OmegaConf.create(DEFAULT_EVAL_CONFIG)


def load_eval_config(
    config_path: str | None = None,
    cli_args: list[str] | None = None,
) -> DictConfig:
    """Load eval configuration from defaults + YAML + CLI overrides."""
    cfg = get_default_eval_config()

    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Eval config file not found: {config_path}")
        yaml_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def parse_eval_args() -> DictConfig:
    """Parse CLI and return the merged eval config."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to eval YAML config")
    parser.add_argument("-h", "--help", action="store_true")

    known, remaining = parser.parse_known_args()

    if known.help:
        print(__doc__)
        sys.exit(0)

    # Convert remaining CLI args to OmegaConf dotlist
    dotlist: list[str] = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                dotlist.append(key)
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                dotlist.append(f"{key}={remaining[i + 1]}")
                i += 1
            else:
                dotlist.append(f"{key}=true")
        i += 1

    cfg = load_eval_config(config_path=known.config, cli_args=dotlist)
    if known.config:
        OmegaConf.update(cfg, "run.config", known.config)
    return cfg


# =============================================================================
# Collation helpers
# =============================================================================


def collate_eval_samples(
    samples: list[EvalSample],
    processor,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Collate a list of :class:`EvalSample` into a batch ready for ``model.generate()``.

    Returns:
        (batch_dict, references, sample_ids)
    """
    texts = [processor.audio_token for _ in samples]
    audios = [[s.audio] for s in samples]
    references = [s.reference for s in samples]
    sample_ids = [s.sample_id for s in samples]

    batch = processor(
        text=texts,
        audio=audios,
        padding=True,
        return_tensors="pt",
    )

    return batch, references, sample_ids


# =============================================================================
# Result I/O
# =============================================================================


def _model_key(cfg: DictConfig) -> str:
    """Derive a short model identifier for dedup in results."""
    return cfg.model.get("path", "unknown") or "unknown"


def load_existing_results(path: str) -> list[dict]:
    """Load all JSON-line records from *path* (empty list if missing)."""
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def was_already_evaluated(
    records: list[dict],
    model_key: str,
    dataset_name: str,
) -> bool:
    """Check whether a (model, dataset) pair already appears in *records*."""
    for r in records:
        if r.get("model") == model_key and r.get("dataset") == dataset_name:
            return True
    return False


def append_result(path: str, record: dict) -> None:
    """Append a single JSON-line record to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Core evaluation loop
# =============================================================================


def evaluate_dataset(
    ds_cfg: DictConfig,
    backend: ModelBackend,
    generation_args: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Generate hypotheses for every sample in one test dataset.

    Args:
        ds_cfg: Per-dataset config node.
        backend: The model backend to call ``.generate()`` on.
        generation_args: Keyword arguments forwarded to generation.

    Returns:
        (hypotheses, references, sample_ids)
    """
    dataset = build_eval_dataset(ds_cfg)
    max_cuts = int(ds_cfg.get("max_cuts", 8))

    # Build a simple sequential torch DataLoader over the lazy iterable
    torch_ds = _IterableWrapper(dataset)
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=max_cuts,
        collate_fn=lambda items: items,  # keep as list[EvalSample]
        num_workers=int(ds_cfg.get("num_workers", 0)),
        pin_memory=False,
    )

    all_hyps: list[str] = []
    all_refs: list[str] = []
    all_ids: list[str] = []

    processor = backend.processor

    for samples in tqdm(loader, desc=ds_cfg.get("name", "eval"), dynamic_ncols=True):
        batch, refs, ids = collate_eval_samples(samples, processor)
        hyps = backend.generate(batch, generation_args=generation_args)

        all_hyps.extend(hyps)
        all_refs.extend(refs)
        all_ids.extend(ids)

    return all_hyps, all_refs, all_ids


class _IterableWrapper(torch.utils.data.IterableDataset):
    """Thin wrapper that turns an :class:`EvalDataset` into a torch IterableDataset."""

    def __init__(self, eval_dataset):
        self._ds = eval_dataset

    def __iter__(self):
        yield from self._ds


# =============================================================================
# Main
# =============================================================================


def main(cfg: DictConfig) -> None:
    """Run evaluation from a fully resolved config."""
    configure_logging()

    logger.info("Eval config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if not cfg.data.test_ds:
        logger.warning("No test datasets specified in data.test_ds — nothing to do.")
        return

    model_key = _model_key(cfg)
    results_file = cfg.output.results_file
    skip_existing = cfg.output.get("skip_existing", True)
    existing_records = load_existing_results(results_file) if skip_existing else []

    # Resolve generation args
    generation_args = OmegaConf.to_container(cfg.get("generation_args", {}), resolve=True) or {}

    # ---- wandb ---------------------------------------------------------------
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    use_wandb = "wandb" in cfg.output.get("report_to", [])

    if use_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "melt-eval"),
            config=dict_cfg,
            name=cfg.run.get("exp_name", None),
        )

    # ---- model ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = build_backend(cfg.model, device=device)

    # ---- iterate over test datasets ------------------------------------------
    for ds_cfg in cfg.data.test_ds:
        ds_name = ds_cfg.get("name", "unnamed")

        if skip_existing and was_already_evaluated(existing_records, model_key, ds_name):
            logger.info("Skipping already-evaluated: model=%s dataset=%s", model_key, ds_name)
            continue

        logger.info("Evaluating dataset: %s", ds_name)
        hyps, refs, _ids = evaluate_dataset(ds_cfg, backend, generation_args)

        # -- compute metrics ---------------------------------------------------
        metric_names: list[str] = list(ds_cfg.get("metrics", []))
        all_metrics: dict[str, float] = {}
        for m_name in metric_names:
            metric = get_metric(m_name)
            result = metric.compute(hypotheses=hyps, references=refs)
            all_metrics.update(result)
            logger.info("  %s: %s", m_name, result)

        # -- write result record -----------------------------------------------
        record = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "model": model_key,
            "dataset": ds_name,
            "num_samples": len(refs),
            "metrics": all_metrics,
            "generation_args": generation_args,
        }
        append_result(results_file, record)
        logger.info("Result appended to %s", results_file)

        # -- wandb logging -----------------------------------------------------
        if use_wandb:
            log_payload = {f"{ds_name}/{k}": v for k, v in all_metrics.items()}
            wandb.log(log_payload)

    if use_wandb:
        wandb.finish()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    configure_logging()

    cfg = parse_eval_args()

    if cfg.run.get("dry_run", False):
        logger.info("Dry run — config parsed successfully")
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    else:
        main(cfg)
