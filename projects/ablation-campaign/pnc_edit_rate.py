#!/usr/bin/env python3
"""How far does a leaf's `custom.pnc_text` move from its original transcript?

The truecase/PNC pass (preprocessing/data-utils/truecase_pnc) is an LLM
rewriting text, and it edits words as well as casing and punctuation. This
measures that per leaf, so a leaf whose rewrite cannot be trusted can be
trained on the original text instead (06-fondue.md, PNC trust rule).

Words are compared after lower-casing and stripping punctuation. Edits are
split into:

* ``ins+del``   words the model added or dropped;
* ``bigSub``    substitutions that are not near-spellings (diacritic-stripped
                character similarity < 0.7): a different word, not a spelling fix;
* ``smallSub``  near-spelling substitutions (orthography, OCR repairs);
* ``ITN``       replaced by digits only (number rewriting, a style change).

``content`` = ins+del + bigSub, as a percentage of original words, is the
quantity the threshold applies to: it is the part that can put words in the
transcript that were not there. Spelling repair and digits are reported but do
not count. Measured 2026-09-25: a threshold of 1% separates MLS nl/pl,
People's Speech and FLEURS ga from every other leaf.

Usage::

    python3 pnc_edit_rate.py --shar-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --leaf mls_sidon/polish/train --leaf fleurs/de_de/train --threshold 1.0

    # a run_pnc.py --output_dir sidecar, paired with its shard in file order
    python3 pnc_edit_rate.py --shar-root ... --leaf fleurs/ga_ie/train \\
        --sidecar /path/to/cuts.000000.pnc.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
import unicodedata
from pathlib import Path


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def strip_marks(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def edit_counts(pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Count original words and the four edit classes over (original, pnc) pairs."""
    n = {"words": 0, "ins_del": 0, "big": 0, "small": 0, "itn": 0}
    for original, pnc in pairs:
        a, b = words(original), words(pnc)
        n["words"] += len(a)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "insert":
                n["ins_del"] += j2 - j1
            elif tag == "delete":
                n["ins_del"] += i2 - i1
            elif tag == "replace":
                x, y = " ".join(a[i1:i2]), " ".join(b[j1:j2])
                if re.fullmatch(r"[\d ]+", y):
                    n["itn"] += i2 - i1
                elif difflib.SequenceMatcher(None, strip_marks(x), strip_marks(y)).ratio() < 0.7:
                    n["big"] += max(i2 - i1, j2 - j1)
                else:
                    n["small"] += max(i2 - i1, j2 - j1)
    return n


def leaf_pairs(leaf: Path, sidecar: Path | None, cap: int, nfiles: int) -> list[tuple[str, str]]:
    """(original, pnc) pairs from the first ``nfiles`` shards of a leaf, at most ``cap``."""
    side = None
    if sidecar:
        with open(sidecar) as fh:
            side = [json.loads(line)["pnc_text"] for line in fh]
    pairs: list[tuple[str, str]] = []
    k = 0
    for shard in sorted(glob.glob(str(leaf / "cuts.*.jsonl")))[:nfiles]:
        with open(shard) as fh:
            for line in fh:
                cut = json.loads(line)
                text = (cut.get("supervisions") or [{}])[0].get("text") or ""
                pnc = side[k] if side else (cut.get("custom") or {}).get("pnc_text")
                k += 1
                if text.strip() and pnc:
                    pairs.append((text, pnc))
                if len(pairs) >= cap:
                    return pairs
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shar-root", type=Path, required=True)
    ap.add_argument("--leaf", action="append", required=True, help="leaf relative to the root; repeatable")
    ap.add_argument("--sidecar", type=Path, default=None, help="pair the (single) leaf with a run_pnc.py sidecar")
    ap.add_argument("--cap", type=int, default=3000, help="max cuts per leaf")
    ap.add_argument("--nfiles", type=int, default=2, help="shards read per leaf")
    ap.add_argument("--threshold", type=float, default=1.0, help="content-edit %% at or above which pnc_text is not trusted")
    args = ap.parse_args()
    if args.sidecar and len(args.leaf) != 1:
        ap.error("--sidecar needs exactly one --leaf")

    rows = []
    for rel in args.leaf:
        c = edit_counts(leaf_pairs(args.shar_root / rel, args.sidecar, args.cap, args.nfiles))
        w = max(c["words"], 1)
        rows.append((rel, 100 * (c["ins_del"] + c["big"]) / w, 100 * c["ins_del"] / w, 100 * c["big"] / w,
                     100 * c["small"] / w, 100 * c["itn"] / w))
    print(f"{'leaf':40s} {'content%':>8s} {'ins+del%':>8s} {'bigSub%':>8s} {'smallSub%':>9s} {'ITN%':>6s}  verdict")
    for rel, content, insdel, big, small, itn in sorted(rows, key=lambda r: -r[1]):
        verdict = "RAW (do not trust)" if content >= args.threshold else "pnc_text ok"
        print(f"{rel:40s} {content:8.2f} {insdel:8.2f} {big:8.2f} {small:9.2f} {itn:6.2f}  {verdict}")


if __name__ == "__main__":
    main()
