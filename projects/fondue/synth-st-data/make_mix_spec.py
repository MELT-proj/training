"""Write the leaf spec that `build_fondue_mix.py synth` adds to Fondue's anneal config.

One entry per training leaf: a Shar directory of SpeechBlocks-Synth-v1 read for one target language, with the hours and
cuts that survive its QE bound, so the builder needs nothing from this package but this file.

- ST leaves (streams A-E and the noisy copies of A/C/E): one per (corpus, target) in `hours_by_leaf.csv`. The bound is the
  `metricx_max` of its (stream, src, tgt) row in the thresholds table; a cut survives when
  `custom.synth.metricx_xxl.<tgt>` is at most that (inclusive).
- ASR leaves: the noisy copies of the native tail (`leaves.json` of the noisy build, kind `asr`), each joining the Fondue
  entry of its language.

Inputs come from `collection_hours.py` and `noise_all.py`. Run anywhere with the standard library.
"""

import csv
import hashlib
import json
import time
from pathlib import Path

import tyro


SYNTH = "/mnt/scratch-nyx/giuseppe/synth"
NAME = "SpeechBlocks-Synth-v1"


def md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main(
    hours_by_leaf: str = f"{SYNTH}/reports/speechblocks-synth-v1/hours_by_leaf.csv",
    thresholds: str = f"{SYNTH}/reports/qe/thresholds_keep75.csv",
    noisy_leaves: str = f"{SYNTH}/shar-noisy/leaves.json",
    out: str = f"{SYNTH}/reports/speechblocks-synth-v1/mix_spec.json",
) -> None:
    bounds = {
        (r["stream"], r["src"], r["tgt"]): (float(r["metricx_max"]), float(r["keep"]))
        for r in csv.DictReader(open(thresholds, newline=""))
    }
    st_leaves, used = [], set()
    for r in csv.DictReader(open(hours_by_leaf, newline="")):
        if int(r["kept_pairs"]) == 0:
            continue
        key = (r["stream"], r["src"], r["tgt"])
        used.add(key)
        st_leaves.append(
            {
                "name": r["corpus"], "group": r["group"], "stream": r["stream"], "src": r["src"], "tgt": r["tgt"],
                "pairs": int(r["pairs"]), "pair_s": float(r["pair_s"]),
                "kept_pairs": int(r["kept_pairs"]), "kept_s": float(r["kept_pair_s"]),
            }
        )  # fmt: skip
    asr_leaves = []
    for leaf in json.load(open(noisy_leaves)):
        if leaf["kind"] != "asr":
            continue
        asr_leaves.append(
            {
                "name": leaf["name"], "lang": leaf["tags"]["lang"], "source": leaf["rel"], "cuts": leaf["cuts"],
                "s": round(leaf["hours"] * 3600, 1), "copies": leaf["copies"], "tags": leaf["tags"],
            }
        )  # fmt: skip
    spec = {
        "name": NAME,
        "built": time.strftime("%Y-%m-%d"),
        "inputs": {p: md5(p) for p in (hours_by_leaf, thresholds, noisy_leaves)},
        "rules": [
            {"stream": s, "src": a, "tgt": t, "bound": bounds[(s, a, t)][0], "keep": bounds[(s, a, t)][1]}
            for (s, a, t) in sorted(used)
        ],
        "st_leaves": sorted(st_leaves, key=lambda x: (x["src"], x["tgt"], x["name"])),
        "asr_leaves": sorted(asr_leaves, key=lambda x: (x["lang"], x["name"])),
    }
    Path(out).write_text(json.dumps(spec, indent=1) + "\n")
    hours = sum(x["kept_s"] for x in st_leaves) / 3600
    print(f"{out}: {len(st_leaves)} ST leaves ({hours:,.1f} kept h), {len(asr_leaves)} ASR leaves, {len(used)} rules")


if __name__ == "__main__":
    tyro.cli(main)
