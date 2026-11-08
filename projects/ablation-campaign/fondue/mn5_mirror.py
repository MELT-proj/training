#!/usr/bin/env python3
"""Lay out a datasets root and a scores root on nyx that look like MN5's, for the builder.

MN5 is the truth for what Fondue reads, but its manifests are too large to copy and nyx is
where the builder runs. Every corpus except two is the same on both, so the mirror links
nyx's copy; the two that differ are laid out from what MN5 has:

* ``yodas3``: only YODAS3_LANGS, the languages left on MN5;
* ``voxpopuli_mosel``: on MN5 only. Each leaf is an empty directory, its score arrays are
  copied from MN5 (small), and its hours and cut counts come from ``mosel_stats.py``
  (run on MN5) into the hours cache, so nothing reads its manifests here.

The hours cache is keyed by path, so the nyx entries are re-keyed under the mirror.

    python3 projects/ablation-campaign/fondue/mn5_mirror.py --out <dir> \\
        --mosel-scores <copy of MN5's shar-asr-scores/voxpopuli_mosel>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_fondue_mix import MOSEL_LANGS, QUALITY, REPO_ROOT, YODAS3_LANGS


NYX = Path("/mnt/scratch-nyx/giuseppe/melt/melt-data")
DIFFERS = ("yodas3", "voxpopuli_mosel")


def link_tree(src: Path, dst: Path, skip: tuple[str, ...]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.name not in skip and not (dst / p.name).exists():
            (dst / p.name).symlink_to(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mosel-scores", type=Path, required=True)
    ap.add_argument("--mosel-stats", type=Path, default=QUALITY / "mosel_wer_dist.json")
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "infra" / ".mix_weights_hours.json")
    args = ap.parse_args()
    shar, scores = args.out / "shar", args.out / "shar-asr-scores"

    link_tree(NYX / "shar", shar, DIFFERS)
    link_tree(NYX / "shar-asr-scores", scores, DIFFERS)
    for lang in YODAS3_LANGS:
        for root, nyx in ((shar, NYX / "shar"), (scores, NYX / "shar-asr-scores")):
            (root / "yodas3").mkdir(exist_ok=True)
            if (nyx / "yodas3" / lang).exists() and not (root / "yodas3" / lang).exists():
                (root / "yodas3" / lang).symlink_to(nyx / "yodas3" / lang)
    stats = {s["lang"]: s for s in json.loads(args.mosel_stats.read_text())}
    if set(stats) != set(MOSEL_LANGS):
        raise SystemExit(f"mosel stats cover {sorted(stats)}, the builder expects {MOSEL_LANGS}")
    for lang in MOSEL_LANGS:
        (shar / "voxpopuli_mosel" / lang / "train").mkdir(parents=True, exist_ok=True)
    if not (scores / "voxpopuli_mosel").exists():
        (scores / "voxpopuli_mosel").symlink_to(args.mosel_scores.resolve())

    cache = json.loads(args.cache.read_text())
    mirrored = {k.replace(str(NYX / "shar"), str(shar), 1): v for k, v in cache.items() if k.startswith(str(NYX / "shar"))}
    for lang, s in stats.items():
        mirrored[str(shar / "voxpopuli_mosel" / lang / "train")] = {
            "hours": s["hours"], "cuts": s["cuts"], "shards": s["shards"], "sample": None,
        }  # fmt: skip
    out_cache = args.out / "mix_weights_hours.json"
    out_cache.write_text(json.dumps(mirrored, indent=2))
    print(f"Mirror at {args.out}: datasets root {shar}, scores root {scores}, hours cache {out_cache}")


if __name__ == "__main__":
    main()
