"""Write a melt-eval spec that scores exactly the utterances a run's in-training eval scores.

Phase 2, step 1c (README.md). Per named validation set the trainer evaluates a
seeded random subset of ``max_samples`` cuts drawn across all of that set's
sources (``materialize_cuts_for_eval``), while melt-eval's freezer subsamples
each source with its own shuffle, so by default the two never score the same
clips. This reproduces the trainer's selection with the trainer's own
functions and pins it in melt-eval through ``reference_map``: a cut missing from
the map is dropped at freeze time. The references are the trainer's own
(``get_text_from_cut`` with the source's text_field, then ``strip().lower()``).

Usage (any environment where ``melt`` imports; CPU only, reads manifests, no audio):

    python projects/self-distill/eval_parity_spec.py \\
        --run-config /path/to/outputs/<run>/resolved_config.json --out-dir /path/to/spec
    melteval freeze /path/to/spec/spec.yaml -o /path/to/frozen-set
"""

import argparse
import json
import os

import yaml
from omegaconf import OmegaConf

from melt.training.data.audio.lhotse.dataloader import (
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.data.audio.lhotse.helpers import get_text_from_cut


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-config", required=True, help="the run's resolved_config.json (or training_config.yaml)")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--container-root",
        default="/workspace/shar",
        help="dataset root as the run saw it; rewritten to $LOCAL_DATASETS_DIR to read the manifests here",
    )
    args = p.parse_args()

    host_root = os.environ["LOCAL_DATASETS_DIR"].rstrip("/")
    container_root = args.container_root.rstrip("/")
    cfg = OmegaConf.load(args.run_config)
    for src in cfg.data.validation_ds.input_cfg:
        if str(src.shar_path).startswith(container_root + "/"):
            src.shar_path = host_root + str(src.shar_path)[len(container_root):]

    eval_cfg = resolve_eval_data_config(cfg.data)
    named = split_eval_config_by_name(eval_cfg) or {"all": eval_cfg}
    strict = bool(eval_cfg.get("strict_text_field", False))
    os.makedirs(args.out_dir, exist_ok=True)

    spec_sources, counts = [], {}
    for name, sub in named.items():
        refs = {}
        for cut in materialize_cuts_for_eval(sub):
            tags = cut.tags if getattr(cut, "tags", None) else {}
            text = get_text_from_cut(cut, tags.get("text_field") or sub.get("text_field", "text"), strict=strict)
            if text and text.strip():
                refs[str(cut.id)] = text.strip().lower()
        ref_path = os.path.abspath(os.path.join(args.out_dir, f"refs_{name}.json"))
        with open(ref_path, "w") as f:
            json.dump(refs, f, ensure_ascii=False, indent=1)
        counts[name] = len(refs)
        for src in OmegaConf.to_container(sub.input_cfg, resolve=True):
            tags = dict(src.get("tags") or {})
            rel = os.path.relpath(src["shar_path"], host_root)
            spec_sources.append({
                "type": "lhotse_shar",
                "shar_path": "${LOCAL_DATASETS_DIR}/" + rel,
                "reference_map": ref_path,
                "tags": {"task": tags.get("task", "asr"), "lang": tags.get("lang", ""), "dataset_id": name},
            })

    spec = {"name": f"parity-{os.path.basename(os.path.dirname(os.path.abspath(args.run_config)))}", "seed": 0,
            "input_cfg": spec_sources}
    with open(os.path.join(args.out_dir, "spec.yaml"), "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    print(json.dumps({"utterances_per_set": counts, "sources": len(spec_sources)}, indent=1))


if __name__ == "__main__":
    main()
