"""Stream F, native tail: which ASR cuts of ga/mt/sl/hr/lv/et/lt/da get noisy copies.

Reads the ASR leaves of a rendered Fondue config (the annealing one), keeps those of the tail languages whose Shar
exists under the datasets root, and builds for each a bool mask over the Shar rows: the cuts that pass every `array`
rule of `data.train_ds.cut_filter` that matches the leaf, exactly as `cut_filter.py` applies them at train time
(`<scores_root>/<shar path relative to datasets_root>/<array>.u16.npy`, one uint16 per cut in Shar order, value =
round(10000 * score), 65535 = no score, decided by the rule's `missing`). The noisy copies carry no `wer_band` tag,
so only cuts that already pass are copied.

Leaves that exist only on MN5 (VoxPopuli-Mosel) are listed as `skipped`; their audio is not on nyx.

Output: `<out>/plan.json` (one entry per leaf: source dir, mask, tags for the new leaf, cuts, kept, hours) and
`<out>/masks/<name>.npy`. Run inside synth.sif.
"""

import json
import os
from pathlib import Path

import numpy as np
import tyro
import yaml


DATASETS = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar"
SCORES = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar-asr-scores"
TAIL = ("ga", "mt", "sl", "hr", "lv", "et", "lt", "da")
SCORE_SCALE, NULL_SCORE = 10_000, 65_535
DROP_TAGS = ("scored", "wer_band")  # the copies are pre-filtered, so no cut_filter rule may match them


def leaves(node: dict, outer: dict, out: list) -> None:
    tags = {**outer, **(node.get("tags") or {})}
    if node.get("type") == "lhotse_shar":
        out.append((node["shar_path"], tags))
    for child in node.get("input_cfg") or []:
        leaves(child, tags, out)


def matching_rules(rules: dict, tags: dict) -> dict:
    """The array rules whose `match` is a subset of the leaf's merged tags (a `false` rule exempts nothing here)."""
    hit = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict) or "array" not in rule:
            continue
        if all(tags.get(k) == v for k, v in (rule.get("match") or {}).items()):
            hit[name] = rule
    return hit


def keep_mask(rules: dict, scores_dir: Path) -> tuple[np.ndarray, list[str]]:
    mask = None
    for rule in rules.values():
        a = np.load(scores_dir / f"{rule['array']}.u16.npy")
        lo = int(round(float(rule.get("min", 0.0)) * SCORE_SCALE))
        hi = int(round(float(rule["max"]) * SCORE_SCALE)) if rule.get("max") is not None else NULL_SCORE - 1
        ok = (a >= lo) & (a <= hi) & (a != NULL_SCORE)
        if rule.get("missing", "drop") == "keep":
            ok |= a == NULL_SCORE
        mask = ok if mask is None else mask & ok
    if mask is None:
        raise ValueError("no array rule matches this leaf")
    return mask, sorted(rules)


def hours_of(shar_dir: Path, mask: np.ndarray) -> float:
    secs, row = 0.0, 0
    for cuts in sorted(p for p in shar_dir.glob("cuts.*.jsonl") if not p.name.endswith(".idx")):
        for line in open(cuts):
            if mask[row]:
                secs += json.loads(line)["duration"]
            row += 1
    assert row == len(mask), f"{shar_dir}: {row} rows, mask has {len(mask)}"
    return secs / 3600


def main(
    config: str,
    out: str,
    datasets_root: str = DATASETS,
    scores_root: str = SCORES,
    langs: tuple[str, ...] = TAIL,
    min_cuts: int = 50,
) -> None:
    cfg = yaml.safe_load(open(config))
    rules = cfg["data"]["train_ds"]["cut_filter"]["rules"]
    found: list = []
    for child in cfg["data"]["train_ds"]["input_cfg"]:
        leaves(child, {}, found)
    plan, skipped = [], []
    (Path(out) / "masks").mkdir(parents=True, exist_ok=True)
    for shar_path, tags in found:
        if tags.get("task") != "asr" or tags.get("lang") not in langs:
            continue
        rel = shar_path.replace("${oc.env:LOCAL_DATASETS_DIR}/", "", 1)
        src = Path(datasets_root) / rel
        if not src.is_dir():
            skipped.append({"rel": rel, "lang": tags["lang"], "reason": "not on this host"})
            continue
        hit = matching_rules(rules, tags)
        mask, used = keep_mask(hit, Path(scores_root) / rel)
        kept = int(mask.sum())
        name = "asr_" + rel.removesuffix("/train").replace("/", "_") + "_noisy"
        if kept < min_cuts:
            skipped.append({"rel": rel, "lang": tags["lang"], "reason": f"only {kept} cuts pass"})
            continue
        np.save(Path(out) / "masks" / f"{name}.npy", mask)
        plan.append(
            {
                "name": name, "lang": tags["lang"], "rel": rel, "src_dir": str(src), "rules": used,
                "mask": str(Path(out) / "masks" / f"{name}.npy"), "cuts": len(mask), "kept": kept,
                "hours_kept": round(hours_of(src, mask), 3),
                "tags": {k: v for k, v in tags.items() if k not in DROP_TAGS},
            }
        )  # fmt: skip
    Path(out, "plan.json").write_text(
        json.dumps({"config": os.path.abspath(config), "plan": plan, "skipped": skipped}, indent=1)
    )
    print(f"{'lang':4} {'leaf':46} {'cuts':>7} {'kept':>7} {'hours':>8}")
    for p in plan:
        print(f"{p['lang']:4} {p['rel']:46} {p['cuts']:7d} {p['kept']:7d} {p['hours_kept']:8.2f}")
    print(f"\ntotal kept hours (one copy): {sum(p['hours_kept'] for p in plan):.1f} in {len(plan)} leaves")
    for s in skipped:
        print("skipped:", s)


if __name__ == "__main__":
    tyro.cli(main)
