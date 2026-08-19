"""OOM canary script for IWSLT 2026 metric training.

Finds the top-N longest training cuts by scanning SHAR manifests (no audio I/O),
loads them, and runs a full training epoch over them using MELTTrainerForRegression.
Because it calls trainer.train() the trainer is fully initialised (including
current_gradient_accumulation_steps) before any forward/backward pass.

Usage:
    python -m projects.iwslt26-metric.canary --config projects/iwslt26-metric/config.yaml
    python -m projects.iwslt26-metric.canary --config projects/iwslt26-metric/config.yaml --num_cuts 20
"""

import gzip
import heapq
import json
import os
from collections import defaultdict
from glob import glob

import torch
import wandb
from lhotse import CutSet
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainingArguments, TrainerCallback

from melt.logging_utils import configure_logging, get_logger
from melt.modeling import MELTProcessor
from melt.training.config import (
    expand_env_vars_in_config,
    parse_args_and_load_config,
    trainer_args_dict,
)
from melt.training.data.audio.lhotse import SpeechTextQEDataset
from melt.training.trainer import count_trainable_parameters

from .train import prepare_model
from .trainer import MELTTrainerForRegression

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cut collection
# ---------------------------------------------------------------------------

def _apply_tags(cut, tags: dict):
    """Mirror _add_tags_to_cut: flat-merge into cut.custom and set cut.tags."""
    if not tags:
        return cut
    if cut.custom is None:
        cut.custom = {}
    cut.custom.update(tags)
    cut.tags = tags  # Lhotse __setattr__ stores this as cut.custom["tags"]
    return cut


def collect_top_cuts(input_cfg, num_cuts: int) -> list:
    """Return top-N longest cuts from SHAR/lhotse_cuts sources in *input_cfg*.

    Step 1: scan gzipped JSONL manifests (metadata only, no audio) with a
            min-heap to track the global top-N by duration.
    Step 2: reload those specific shards with audio and inject per-source tags.

    Returns a list of Cut objects sorted longest-first.
    """
    # min-heap entries: (duration, cut_id, source_type, source_path, manifest_file)
    min_heap: list = []
    source_tags: dict[str, dict] = {}  # source_path -> tags dict

    logger.info("[Canary] Scanning training manifests (metadata only, no audio)...")

    for source_cfg in input_cfg:
        source_type = getattr(source_cfg, "type", "lhotse_shar")
        tags = getattr(source_cfg, "tags", None)
        tags_dict: dict = dict(tags) if tags is not None else {}

        if source_type == "lhotse_shar":
            shar_path = os.path.expandvars(str(source_cfg.shar_path))
            source_tags[shar_path] = tags_dict
            for mf in sorted(glob(os.path.join(shar_path, "cuts.*.jsonl.gz"))):
                try:
                    with gzip.open(mf, "rt", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            cut_data = json.loads(line)
                            duration = float(cut_data.get("duration", 0))
                            cut_id = cut_data.get("id", "")
                            entry = (duration, cut_id, "lhotse_shar", shar_path, mf)
                            if len(min_heap) < num_cuts:
                                heapq.heappush(min_heap, entry)
                            elif duration > min_heap[0][0]:
                                heapq.heapreplace(min_heap, entry)
                except Exception as exc:
                    logger.warning("[Canary] Could not read manifest %s: %s", mf, exc)

        elif source_type == "lhotse_cuts":
            cuts_path = os.path.expandvars(str(source_cfg.cuts_path))
            source_tags[cuts_path] = tags_dict
            if not os.path.exists(cuts_path):
                logger.warning("[Canary] cuts_path not found: %s", cuts_path)
                continue
            try:
                for cut in CutSet.from_file(cuts_path):
                    entry = (cut.duration, cut.id, "lhotse_cuts", cuts_path, cuts_path)
                    if len(min_heap) < num_cuts:
                        heapq.heappush(min_heap, entry)
                    elif cut.duration > min_heap[0][0]:
                        heapq.heapreplace(min_heap, entry)
            except Exception as exc:
                logger.warning("[Canary] Could not read cuts file %s: %s", cuts_path, exc)

    if not min_heap:
        logger.warning("[Canary] No cuts found — check input_cfg paths.")
        return []

    min_dur = min_heap[0][0]
    max_dur = max(e[0] for e in min_heap)
    logger.info(
        "[Canary] Top-%d cuts found: duration range %.1f–%.1f s",
        num_cuts, min_dur, max_dur,
    )

    # --- Step 2: load cuts with audio from only the relevant shards ---
    shar_shards: dict = defaultdict(list)
    cutfile_ids: dict = defaultdict(list)

    for _, cut_id, source_type, source_path, mf in min_heap:
        if source_type == "lhotse_shar":
            shar_shards[(source_path, mf)].append(cut_id)
        else:
            cutfile_ids[source_path].append(cut_id)

    collected = []

    for (shar_path, mf), target_ids in shar_shards.items():
        target_ids_set = set(target_ids)
        tags = source_tags.get(shar_path, {})
        mf_name = os.path.basename(mf)
        suffix = mf_name[len("cuts."):-len(".jsonl.gz")]
        audio_tars = sorted(glob(os.path.join(shar_path, f"*.{suffix}.tar")))
        if not audio_tars:
            logger.warning("[Canary] No audio tars for shard %s in %s", suffix, shar_path)
            continue
        fields: dict = {"cuts": [mf]}
        for tar_path in audio_tars:
            tar_name = os.path.basename(tar_path)
            field_name = tar_name[: tar_name.index(f".{suffix}.tar")]
            fields[field_name] = [tar_path]
        try:
            for cut in CutSet.from_shar(fields=fields):
                if cut.id in target_ids_set:
                    collected.append(_apply_tags(cut, tags))
        except Exception as exc:
            logger.warning("[Canary] Failed to load shard %s: %s", mf, exc)

    for cuts_path, target_ids in cutfile_ids.items():
        target_ids_set = set(target_ids)
        tags = source_tags.get(cuts_path, {})
        try:
            for cut in CutSet.from_file(cuts_path):
                if cut.id in target_ids_set:
                    collected.append(_apply_tags(cut, tags))
        except Exception as exc:
            logger.warning("[Canary] Failed to load cuts from %s: %s", cuts_path, exc)

    collected.sort(key=lambda c: c.duration, reverse=True)
    logger.info(
        "[Canary] Loaded %d cuts with audio (longest: %.1fs)",
        len(collected),
        collected[0].duration if collected else 0.0,
    )
    return collected[:num_cuts]


# ---------------------------------------------------------------------------
# Canary dataset / dataloader
# ---------------------------------------------------------------------------

class _PrebuiltBatchDataset(Dataset):
    """Wraps a list of pre-built batches so DataLoader can iterate them."""

    def __init__(self, batches: list):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class _GpuMemCallback(TrainerCallback):
    """Log GPU memory (allocated / reserved / peak) after every training step."""

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            logger.info(
                "[Canary] GPU %d | step %d | allocated=%.2f GB  reserved=%.2f GB  peak=%.2f GB",
                i, state.global_step, allocated, reserved, peak,
            )


class CanaryTrainer(MELTTrainerForRegression):
    """MELTTrainerForRegression that trains only on pre-selected canary cuts."""

    def __init__(self, canary_cuts: list, **kwargs):
        self._canary_cuts = canary_cuts
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        logger.info(
            "[Canary] Building dataloader from %d pre-selected cuts.", len(self._canary_cuts)
        )
        dataset = SpeechTextQEDataset(
            processor=self.processor,
            config=self.config.data,
            is_train=True,
            return_labels=True,
        )

        batches = []
        for cut in self._canary_cuts:
            batch = dataset[CutSet.from_cuts([cut])]
            if batch is not None:
                batches.append(batch)
            else:
                logger.warning(
                    "[Canary] Dataset returned None for cut id=%s (%.1fs) — skipping.",
                    cut.id, cut.duration,
                )

        if not batches:
            raise RuntimeError("[Canary] All canary cuts were skipped — nothing to train on.")

        logger.info("[Canary] %d/%d cuts produced valid batches.", len(batches), len(self._canary_cuts))

        # Each item is already a full batch dict; collate_fn just unwraps the list-of-one.
        return DataLoader(
            _PrebuiltBatchDataset(batches),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=0,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        # Canary run is training-only; return an empty dataloader if eval is requested.
        return DataLoader([], batch_size=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: DictConfig, num_cuts: int = 10) -> None:
    configure_logging()

    logger.info("[Canary] Loading processor from %s", cfg.model.ckpt)
    processor = MELTProcessor.from_pretrained(cfg.model.ckpt)

    # Use the same model preparation as the main training script:
    # loads the checkpoint, applies LoRA, freezes components per config.
    targs_for_prepare = Seq2SeqTrainingArguments(**trainer_args_dict(cfg))
    model, _, _ = prepare_model(cfg, targs_for_prepare, processor)
    trainable_str = count_trainable_parameters(model)
    logger.info("[Canary] Trainable parameters: %s", trainable_str)

    # Find top-N longest cuts
    canary_cuts = collect_top_cuts(cfg.data.train_ds.input_cfg, num_cuts=num_cuts)
    if not canary_cuts:
        logger.error("[Canary] No canary cuts collected — aborting.")
        return

    # Build training args: one epoch over the canary cuts, no eval, no saving
    targs_dict = trainer_args_dict(cfg)
    targs_dict.update(
        {
            "max_steps": len(canary_cuts),
            "num_train_epochs": 1,
            "do_eval": False,
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_steps": 1,
            "report_to": ["wandb"],
        }
    )
    targs = Seq2SeqTrainingArguments(**targs_dict)

    # Initialise wandb before the trainer so the full stdout is captured and
    # the run carries the "canary" tag. Mirrors the pattern in train.py.
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "melt"),
        name=f"canary-{cfg.run.get('exp_name', 'run')}",
        tags=["canary"],
        config={"num_cuts": num_cuts, "model_ckpt": str(cfg.model.ckpt)},
    )

    trainer = CanaryTrainer(
        canary_cuts=canary_cuts,
        model=model,
        args=targs,
        config=cfg,
        processor=processor,
        callbacks=[_GpuMemCallback()],
    )

    logger.info("[Canary] Starting canary training run (%d steps)...", len(canary_cuts))
    trainer.train()
    wandb.finish()
    logger.info("[Canary] Canary run complete — model survived all %d cuts.", len(canary_cuts))


if __name__ == "__main__":
    configure_logging()

    import argparse

    ap = argparse.ArgumentParser(description="OOM canary run for IWSLT metric training")
    ap.add_argument("--num_cuts", type=int, default=10, help="Number of longest cuts to test")
    # Remaining args (--config, overrides) are consumed by parse_args_and_load_config
    known, _ = ap.parse_known_args()

    cfg = parse_args_and_load_config()
    cfg = expand_env_vars_in_config(cfg)

    logger.info("[Canary] Config loaded. Running canary with num_cuts=%d", known.num_cuts)
    logger.info("[Canary] Config:\n%s", OmegaConf.to_yaml(cfg))

    main(cfg, num_cuts=known.num_cuts)
