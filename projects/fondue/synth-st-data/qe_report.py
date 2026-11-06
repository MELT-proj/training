"""Per-direction MetricX statistics over all scored units: quantile table, hours, histograms (CSV + PNG).

One row per (stream, src -> tgt). Hours are source-audio hours of the scored cuts (from the unit pools).
This is what the filter thresholds are chosen from; nothing here filters. Run inside synth.sif.
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import tyro


QS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
BINS = np.arange(0, 25.5, 0.5)


def stream_of(unit: str) -> str:
    for prefix, stream in (("c_", "C"), ("y2lr_", "C"), ("fleurs_", "B"), ("tts_", "E"), ("d_", "D")):
        if unit.startswith(prefix):
            return stream
    return "A"


def main(
    synth: str = "/mnt/scratch-nyx/giuseppe/synth",
    out_dir: str = "/mnt/scratch-nyx/giuseppe/synth/reports/qe",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    outs = {"A": "out-a", "B": "out-b", "C": "out-c", "D": "out-d", "E": "out-e"}
    vals: dict[tuple, list] = defaultdict(list)
    dur: dict[tuple, float] = defaultdict(float)
    extra: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])  # n, lid mismatch, len_ratio outside [0.5, 2]
    for pool in sorted(glob.glob(f"{synth}/units/*.pool.jsonl")):
        unit = os.path.basename(pool)[: -len(".pool.jsonl")]
        s = stream_of(unit)
        finals = sorted(glob.glob(f"{synth}/{outs[s]}/{unit}/final.*.jsonl"))
        if not finals:
            continue
        seg = {}
        for line in open(pool):
            r = json.loads(line)
            seg[r["id"]] = r["duration"]
        for path in finals:
            for line in open(path):
                r = json.loads(line)
                key = (s, r["src_lang"], r["tgt"])
                vals[key].append(r["metricx_xxl"])
                dur[key] += seg[r["id"]]
                e = extra[key]
                e[0] += 1
                e[1] += r["lid"] != r["tgt"]
                e[2] += not 0.5 <= r["len_ratio"] <= 2.0
        print(unit, flush=True)
    rows, hist = [], []
    for key in sorted(vals):
        a = np.array(vals[key])
        n, lid_bad, len_bad = extra[key]
        rows.append(
            {"stream": key[0], "src": key[1], "tgt": key[2], "pairs": len(a), "hours": round(dur[key] / 3600, 1),
             "mean": round(a.mean(), 3), **{f"q{q}": round(float(np.percentile(a, q)), 3) for q in QS},
             "lid_mismatch": round(lid_bad / n, 4), "len_ratio_out": round(len_bad / n, 4)}
        )  # fmt: skip
        h, _ = np.histogram(a, bins=BINS)
        hist.append(
            {"stream": key[0], "src": key[1], "tgt": key[2], **{f"{b:.1f}": int(c) for b, c in zip(BINS[:-1], h)}}
        )
    pd.DataFrame(rows).to_csv(f"{out_dir}/quantiles.csv", index=False)
    pd.DataFrame(hist).to_csv(f"{out_dir}/histograms.csv", index=False)
    plot(vals, out_dir)
    print(f"{len(rows)} directions -> {out_dir}")


def plot(vals: dict, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[str, list] = defaultdict(list)
    for key in sorted(vals):
        groups[f"{key[0]}_{key[1] if key[0] != 'A' else 'en'}"].append(key)
    for name, keys in groups.items():
        cols = 4
        fig, axes = plt.subplots(
            (len(keys) + cols - 1) // cols, cols, figsize=(14, 2.6 * ((len(keys) + cols - 1) // cols)), squeeze=False
        )
        for ax, key in zip(axes.ravel(), keys):
            a = np.array(vals[key])
            ax.hist(a, bins=BINS, color="#4c78a8")
            ax.axvline(np.median(a), color="#e45756", lw=1)
            ax.set_title(f"{key[1]}→{key[2]}  n={len(a):,}  med={np.median(a):.1f}", fontsize=9)
            ax.set_yticks([])
        for ax in axes.ravel()[len(keys) :]:
            ax.axis("off")
        fig.suptitle(f"MetricX-24 error score by direction ({name}); lower is better", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/hist_{re.sub(r'[^A-Za-z0-9_]', '', name)}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)
