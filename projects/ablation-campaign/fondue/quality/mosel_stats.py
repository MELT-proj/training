"""Per-language stats of VoxPopuli-Mosel on MN5, for the Fondue builder (run on a gp node).

For each <root>/voxpopuli_mosel/<lang>/train: cut count, hours, shard count, the share of
cuts with a blank supervision text, the custom keys seen, whether the WER array's
cut-id md5 matches the manifests' row order, and the WER hours histogram (0.05 bins,
30 bins, overflow in the last) over scored cuts, as asr_wer_dist.json stores it.

    python mosel_stats.py <datasets_root> <scores_root> <out.json>
"""

import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

STEP, BINS, SCALE, NULL = 0.05, 30, 10_000, 65_535


def leaf_stats(args: tuple[str, str, str]) -> dict:
    lang, root, scores_root = args
    leaf = Path(root) / "voxpopuli_mosel" / lang / "train"
    shards = sorted(leaf.glob("cuts.*.jsonl"))
    ids, durs, blank, keys = [], [], 0, Counter()
    for shard in shards:
        with open(shard) as f:
            for line in f:
                c = json.loads(line)
                ids.append(c["id"])
                durs.append(c["duration"])
                sups = c.get("supervisions") or []
                if not sups or not (sups[0].get("text") or "").strip():
                    blank += 1
                keys.update((c.get("custom") or {}).keys())
    durs = np.asarray(durs)
    out = {
        "lang": lang,
        "cuts": len(ids),
        "hours": float(durs.sum() / 3600),
        "shards": len(shards),
        "blank": blank / max(len(ids), 1),
        "custom_keys": dict(keys.most_common()),
    }
    sdir = Path(scores_root) / "voxpopuli_mosel" / lang / "train"
    if (sdir / "wer.u16.npy").exists():
        meta = json.loads((sdir / "meta.json").read_text())
        wer = np.load(sdir / "wer.u16.npy")
        out["rows"] = int(wer.shape[0])
        out["md5_ok"] = hashlib.md5("\n".join(ids).encode()).hexdigest() == meta["cut_ids_md5"]
        scored = wer != NULL
        out["null_share"] = float(1 - scored.mean()) if len(wer) else None
        if wer.shape[0] == len(ids):
            idx = np.minimum((wer[scored] / SCALE / STEP).astype(int), BINS - 1)
            out["wer_h"] = [float(x) for x in np.bincount(idx, weights=durs[scored] / 3600, minlength=BINS)]
            out["wer_n"] = [int(x) for x in np.bincount(idx, minlength=BINS)]
    return out


def main() -> None:
    root, scores_root, dest = sys.argv[1:4]
    langs = sorted(p.name for p in (Path(root) / "voxpopuli_mosel").iterdir())
    with Pool(len(langs)) as pool:
        stats = pool.map(leaf_stats, [(lang, root, scores_root) for lang in langs])
    Path(dest).write_text(json.dumps(stats, indent=1))
    for s in stats:
        print(
            f"{s['lang']}: {s['cuts']:9d} cuts {s['hours']:9.1f} h {s['shards']:4d} shards "
            f"blank {s['blank']:.4f} md5_ok {s.get('md5_ok')} null {s.get('null_share')}"
        )


if __name__ == "__main__":
    main()
