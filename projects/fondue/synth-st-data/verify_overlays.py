"""Check every overlay under `shar-out` against its source Shar and the unit outputs.

Per shard: same number of cuts, same ids in the same order, audio tar link resolves, manifest index exists and has one
offset per cut. Per corpus: the translations written equal the `final` rows (SYNTH_COMPLETE), every `ok` flag has its
`translation_<tgt>` and score, and no text is empty. Run inside synth.sif.
"""

import glob
import gzip
import json
import os
from multiprocessing import Pool

import tyro

from overlay_all import resolve, source


def ids(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def check_shard(args: tuple[str, str]) -> dict:
    src_cuts, out_cuts = args
    res = {"cuts": 0, "ok": 0, "bad": []}
    n_idx = os.path.getsize(out_cuts + ".idx") // 8 if os.path.exists(out_cuts + ".idx") else -1
    src_iter = ids(src_cuts)
    for cut in ids(out_cuts):
        s = next(src_iter, None)
        res["cuts"] += 1
        if s is None or s["id"] != cut["id"]:
            res["bad"].append(f"{out_cuts}: id order differs at cut {res['cuts']}")
            break
        syn = cut.get("custom", {}).get("synth", {})
        for tgt in syn.get("ok", {}):
            res["ok"] += 1
            t = cut["custom"].get(f"translation_{tgt}")
            if not t or not t.strip():
                res["bad"].append(f"{cut['id']}: empty translation_{tgt}")
            if tgt not in syn.get("metricx_xxl", {}):
                res["bad"].append(f"{cut['id']}: no metricx_xxl.{tgt}")
    if next(src_iter, None) is not None:
        res["bad"].append(f"{out_cuts}: fewer cuts than source")
    if n_idx not in (-1, res["cuts"]) and n_idx != res["cuts"] + 1:
        res["bad"].append(f"{out_cuts}: index has {n_idx} entries for {res['cuts']} cuts")
    if n_idx == -1:
        res["bad"].append(f"{out_cuts}: no index")
    tar = os.path.join(
        os.path.dirname(out_cuts), os.path.basename(out_cuts).replace("cuts.", "recording.").replace(".jsonl", ".tar")
    )
    if not os.path.exists(tar):
        res["bad"].append(f"{tar}: dangling")
    res["bad"] = res["bad"][:5]
    return res


def main(out_root: str = "/mnt/scratch-nyx/giuseppe/synth/shar-out", jobs: int = 8) -> None:
    total_bad = 0
    for corpus_dir in sorted(glob.glob(f"{out_root}/*")):
        corpus = os.path.basename(corpus_dir)
        done = os.path.join(corpus_dir, "train", "SYNTH_COMPLETE")
        if not os.path.exists(done):
            print(f"{corpus}: NOT COMPLETE")
            total_bad += 1
            continue
        want = sum(json.load(open(done))["translations"].values())
        src = resolve(source(corpus)[0])
        if src.startswith("fleurs:") or "fleurs" in corpus:
            from fleurs_gold import FLEURS_DIR

            src = f"/mnt/scratch-nyx/giuseppe/melt/melt-data/shar/fleurs/{FLEURS_DIR[corpus.split('_', 1)[1]]}/train"
        pairs = [
            (os.path.join(src, os.path.basename(p)), p) for p in sorted(glob.glob(f"{corpus_dir}/train/cuts.*.jsonl"))
        ]
        with Pool(jobs) as pool:
            res = pool.map(check_shard, pairs)
        got = sum(r["ok"] for r in res)
        bad = [b for r in res for b in r["bad"]]
        # FLEURS repeats cut ids (several speakers read one sentence, text identical): all repeats share the translation
        if got != want and not (corpus.startswith("fleurs_") and got > want):
            bad.append(f"translations {got} != SYNTH_COMPLETE {want}")
        total_bad += len(bad)
        print(f"{corpus}: {len(pairs)} shards, {sum(r['cuts'] for r in res)} cuts, {got} translations, "
              f"{'OK' if not bad else 'FAIL ' + '; '.join(bad[:3])}", flush=True)  # fmt: skip
    print("FAILED" if total_bad else "ALL OK")


if __name__ == "__main__":
    tyro.cli(main)
