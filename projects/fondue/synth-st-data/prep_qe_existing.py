"""Stream D: prepare reference-free QE of translations that already exist in a Shar source.

Writes, for one Shar directory, a unit pool (`<units>/<name>.pool.jsonl`: id, text = the SOURCE
text, shard, idx, src_lang, targets=[tgt]) and the matching pre-made MT chunks
(`<out>/<name>/mt.NNNN.jsonl`, same chunking as run_mt_qe.py) holding the existing translation, so
`run_mt_qe.py --stage qe` scores them without any generation. Source and translation come from
dotted paths into the cut (`supervisions.0.text`, `custom.translation_en`, `custom.sentence`).
"""

import json
import os
from glob import glob
from pathlib import Path

import tyro

from common import write_jsonl_atomic


def get_path(obj, dotted: str):
    for k in dotted.split("."):
        if isinstance(obj, list) and k.isdigit():
            obj = obj[int(k)] if int(k) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


def main(
    shar_dir: str,
    name: str,
    src_lang: str,
    tgt_lang: str,
    src_field: str,
    mt_field: str,
    units_dir: str,
    out_dir: str,
    chunk: int = 50000,
) -> None:
    rows, mt = [], []
    for path in sorted(glob(os.path.join(shar_dir, "cuts.*.jsonl"))):
        shard = os.path.basename(path)
        with open(path) as f:
            for idx, line in enumerate(f):
                cut = json.loads(line)
                src, tr = get_path(cut, src_field), get_path(cut, mt_field)
                if not src or not tr or not str(src).strip() or not str(tr).strip():
                    continue
                rows.append({"id": cut["id"], "text": str(src).strip(), "shard": shard, "idx": idx,
                             "duration": round(cut["duration"], 3), "src_lang": src_lang, "targets": [tgt_lang]})  # fmt: skip
                mt.append(
                    {"id": cut["id"], "src_lang": src_lang, "tgt": tgt_lang, "mt": str(tr).strip(), "finish": "stop"}
                )
    assert len({r["id"] for r in rows}) == len(rows), f"{shar_dir}: cut ids repeat; a unit needs unique ids"
    write_jsonl_atomic(Path(units_dir) / f"{name}.pool.jsonl", rows)
    for k in range(0, len(mt), chunk):
        write_jsonl_atomic(Path(out_dir) / name / f"mt.{k // chunk:04d}.jsonl", mt[k : k + chunk])
    print(f"{name}: {len(rows)} cuts, {(len(mt) + chunk - 1) // chunk} chunks")


if __name__ == "__main__":
    tyro.cli(main)
