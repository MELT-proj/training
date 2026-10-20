#!/usr/bin/env python3
"""Read the smoke's per-batch cut-id logs and report, per ASR source and step window,
how many delivered cuts sit inside each phase's band, plus ST/exempt pass-through.

    python check_smoke.py --cut-ids-dir DIR --stats-root .../shar-stats \
        --scores-root .../shar-asr-scores --config smoke.yaml
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from melt.training.data.audio.lhotse.asr_score_filter import (
    NULL_SCORE,
    compile_phases,
    resolve_source_filter,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-ids-dir", required=True, type=Path)
    ap.add_argument("--stats-root", required=True, type=Path)
    ap.add_argument("--scores-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--window", type=int, default=5, help="step window width")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["data"]["train_ds"]
    g = cfg["asr_filter"]
    prefix = "${oc.env:LOCAL_DATASETS_DIR}/"
    sources = {}  # leaf -> (task, bands per phase)
    index = {}  # cut_id -> (leaf, row)
    for e in cfg["input_cfg"]:
        leaf = e["shar_path"].removeprefix(prefix)
        task = e["tags"]["task"]
        spec = resolve_source_filter(g, [e["asr_filter"]]) if (task == "asr" and "asr_filter" in e) else (
            resolve_source_filter(g, []) if task == "asr" else None)
        bands = None
        if spec is not None:
            scores = np.load(args.scores_root / leaf / "wer.u16.npy")
            bands = (scores, compile_phases(spec["phases"], scores))
        sources[leaf] = (task, bands)
        ids = pq.read_table(args.stats_root / leaf / "cuts.parquet", columns=["cut_id"]).column(0).to_pylist()
        for row, cid in enumerate(ids):
            index[cid] = (leaf, row)

    # (leaf, window) -> [n, in_phase0_band, in_phase1_band, null]
    table = defaultdict(lambda: [0, 0, 0, 0])
    for f in sorted(args.cut_ids_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            win = rec["asr_filter_step"] // args.window * args.window
            for cid in rec["cut_ids"]:
                hit = index.get(cid)
                if hit is None:
                    continue
                leaf, row = hit
                task, bands = sources[leaf]
                t = table[(leaf, task, win)]
                t[0] += 1
                if bands is not None:
                    scores, phases = bands
                    v = int(scores[row])
                    if v == NULL_SCORE:
                        t[3] += 1
                    for k in (0, 1):
                        if k < len(phases) and phases[k][1] <= v <= phases[k][2]:
                            t[1 + k] += 1

    print(f"{'source':36s} {'task':4s} {'step':>5s} {'cuts':>6s} {'in band 0':>10s} {'in band 1':>10s} {'no score':>9s}")
    for (leaf, task, win), (n, b0, b1, nul) in sorted(table.items()):
        if task == "asr" and sources[leaf][1] is not None:
            print(f"{leaf:36s} {task:4s} {win:5d} {n:6d} {b0 / n:10.0%} {b1 / n:10.0%} {nul:9d}")
        else:
            print(f"{leaf:36s} {task:4s} {win:5d} {n:6d} {'-':>10s} {'-':>10s} {'-':>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
