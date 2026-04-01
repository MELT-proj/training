"""Finalize an output directory from a training run that crashed mid-training.

The script restores the two artefacts that HuggingFace Trainer writes only at
the very end of a successful run:

  1. ``config.json``  — rebuilt from the training YAML if missing.
  2. ``model.safetensors`` — merged from the latest FSDP checkpoint if missing.

Usage:
    python projects/iwslt26-metric/infra/finalize_crashed_run.py \\
        /path/to/output/MELT_QE_v1.1 \\
        --config projects/iwslt26-metric/config.yaml

    # Dry-run to see what would happen without writing anything:
    python projects/iwslt26-metric/infra/finalize_crashed_run.py \\
        /path/to/output/MELT_QE_v1.1 \\
        --config projects/iwslt26-metric/config.yaml \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from accelerate.utils import merge_fsdp_weights
from omegaconf import OmegaConf

# Add repo root to path so melt imports work when run as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import LoraConfig, TaskType

from melt.training.config import expand_env_vars_in_config
from melt.training.setup import prepare_melt_config, prepare_processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the checkpoint-N subfolder with the largest N, or None."""
    checkpoints = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
                checkpoints.append((step, p))
            except ValueError:
                pass
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: x[0])[1]


def _restore_config_and_processor(output_dir: Path, cfg, dry_run: bool) -> bool:
    """Rebuild and save model config and processor files from the training YAML.

    When LoRA is enabled in the config, saves ``adapter_config.json`` (PEFT)
    instead of ``config.json`` (MELTConfig), since the folder is meant to be
    loaded as a PEFT adapter rather than a full model checkpoint.

    Returns True if any file was (or would be) written.
    """
    lora_cfg = cfg.model.get("lora", None)
    lora_enabled = lora_cfg is not None and lora_cfg.get("enabled", False)

    if lora_enabled:
        config_file = "adapter_config.json"
    else:
        config_file = "config.json"

    config_path = output_dir / config_file
    # Processor presence is indicated by tokenizer.json (always written by save_pretrained)
    processor_path = output_dir / "tokenizer.json"

    config_missing = not config_path.exists()
    processor_missing = not processor_path.exists()

    if not config_missing and not processor_missing:
        print(f"  [{config_file}]  already present — skipping.")
        print("  [processor]              already present — skipping.")
        return False

    # Processor is needed for both paths (token IDs for MELTConfig, and to save)
    processor = prepare_processor(cfg)

    if config_missing:
        if lora_enabled:
            print(f"  [{config_file}]  missing — rebuilding LoRA adapter config from YAML...")
            target_modules = list(lora_cfg.target_modules) if lora_cfg.target_modules is not None else None
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
                target_modules=target_modules,
                bias=lora_cfg.bias,
                task_type=TaskType.SEQ_CLS
            )
            if dry_run:
                print(f"  [{config_file}]  DRY-RUN: would save to {config_path}")
            else:
                peft_config.save_pretrained(str(output_dir))
                print(f"  [{config_file}]  saved to {config_path}")
        else:
            print(f"  [{config_file}]  missing — rebuilding MELTConfig from YAML...")
            melt_config = prepare_melt_config(cfg, processor)
            # MELTForSequenceClassification is instantiated with num_labels=1
            # (see train.py: from_pretrained(..., text_decoder_kwargs={"num_labels": 1}))
            melt_config.text_decoder_config.num_labels = 1
            if dry_run:
                print(f"  [{config_file}]  DRY-RUN: would save to {config_path}")
            else:
                melt_config.save_pretrained(str(output_dir))
                print(f"  [{config_file}]  saved to {config_path}")
    else:
        print(f"  [{config_file}]  already present — skipping.")

    if processor_missing:
        print("  [processor]              missing — saving processor files...")
        if dry_run:
            print(f"  [processor]              DRY-RUN: would save to {output_dir}")
        else:
            processor.save_pretrained(str(output_dir))
            print(f"  [processor]              saved to {output_dir}")
    else:
        print("  [processor]              already present — skipping.")

    return config_missing or processor_missing


def _restore_weights(output_dir: Path, dry_run: bool) -> bool:
    """Merge FSDP shards from the latest checkpoint into model.safetensors.

    Returns True if the file was (or would be) written.
    """
    weights_path = output_dir / "model.safetensors"
    if weights_path.exists():
        print("  [model.safetensors]  already present — skipping.")
        return False

    print("  [model.safetensors]  missing — looking for latest checkpoint...")

    latest_ckpt = _find_latest_checkpoint(output_dir)
    if latest_ckpt is None:
        print("  [model.safetensors]  ERROR: no checkpoint-N subfolders found in", output_dir)
        return False

    fsdp_dir = latest_ckpt / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        print(f"  [model.safetensors]  ERROR: FSDP shard dir not found: {fsdp_dir}")
        return False

    print(f"  [model.safetensors]  latest checkpoint : {latest_ckpt.name}")
    print(f"  [model.safetensors]  FSDP shards dir  : {fsdp_dir}")

    if dry_run:
        print(f"  [model.safetensors]  DRY-RUN: would merge {fsdp_dir} -> {weights_path}")
    else:
        merge_fsdp_weights(
            checkpoint_dir=str(fsdp_dir),
            output_path=str(output_dir),
            safe_serialization=True,
            remove_checkpoint_dir=False,
        )
        print(f"  [model.safetensors]  saved to {weights_path}")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Path to the training output directory to finalize.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config used for this run.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing any files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"ERROR: output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFinalizing run directory: {output_dir}")
    if args.dry_run:
        print("  (dry-run mode — no files will be written)\n")

    # Load and merge training config with defaults
    from melt.training.config import load_config
    cfg = load_config(config_path=args.config)
    cfg = expand_env_vars_in_config(cfg)

    lora_enabled = cfg.model.get("lora", {}) and cfg.model.lora.get("enabled", False)
    config_label = "adapter_config.json" if lora_enabled else "config.json"
    print(f"\n--- Step 1: {config_label} + processor ---")
    config_restored = _restore_config_and_processor(output_dir, cfg, dry_run=args.dry_run)

    print("\n--- Step 2: model.safetensors ---")
    weights_restored = _restore_weights(output_dir, dry_run=args.dry_run)

    print("\n--- Summary ---")
    if not config_restored and not weights_restored:
        print("  Nothing to do — output directory already has all artefacts.")
    else:
        status = "Would restore" if args.dry_run else "Restored"
        if config_restored:
            print(f"  {status}: {config_label} + processor files")
        if weights_restored:
            print(f"  {status}: model.safetensors")
    print()


if __name__ == "__main__":
    main()
