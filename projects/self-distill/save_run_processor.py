"""Save a run's processor and training config before the run finishes.

Trainer checkpoints (``checkpoint-N``) carry weights and config but no
processor or tokenizer files, and the run root only gets them at the final
save, so a mid-run checkpoint cannot be evaluated by anything that loads the
processor from its path (melt-eval, alignment_probe.py). This rebuilds the
processor exactly as train.py does (``prepare_processor`` on the run's resolved
config) and checks its MELT token ids against a checkpoint's config.

Usage (training image, CPU; HF_HOME pointing at the cache the run used):
    python projects/self-distill/save_run_processor.py \\
        --run-config /path/to/outputs/<run>/resolved_config.json \\
        --check-ckpt /path/to/outputs/<run>/checkpoint-1200 --out /path/to/outputs/<run>/run-processor
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from melt.training.setup import prepare_processor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check-ckpt", default=None, help="a checkpoint whose config.json the token ids must match")
    args = p.parse_args()

    cfg = OmegaConf.load(args.run_config)
    processor = prepare_processor(cfg)
    ids = {"audio_token_id": processor.audio_token_id, "audio_bos_token_id": processor.audio_bos_token_id,
           "audio_eos_token_id": processor.audio_eos_token_id, "pad_token_id": processor.tokenizer.pad_token_id,
           "vocab": len(processor.tokenizer)}
    if args.check_ckpt:
        ckpt = json.load(open(Path(args.check_ckpt) / "config.json"))
        for key in ("audio_token_id", "audio_bos_token_id", "audio_eos_token_id"):
            if key in ckpt and ckpt[key] != ids[key]:
                raise SystemExit(f"{key}: rebuilt processor has {ids[key]}, checkpoint config has {ckpt[key]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(out)
    OmegaConf.save(cfg, out / "training_config.yaml")
    print(json.dumps({"out": str(out), **ids}, indent=1))


if __name__ == "__main__":
    main()
