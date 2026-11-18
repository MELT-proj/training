"""Histogram SPITE's BLASER 2.0 and xCOMET-XL scores, weighted by audio hours, for the threshold artifact.

Writes st.json: per (corpus, target) group, 2-D histograms BLASER x xCOMET for both BLASER variants
(audio vs English source text, audio vs translation), in cuts and hours, plus quantiles.
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = "/mnt/scratch-nyx/giuseppe/melt/melt-data/spite_metadata"
CORPORA = {"cv16": "SPITE CV16", "gigaspeech": "SPITE GigaSpeech"}
TARGETS = ["de", "es", "fr", "it", "ko", "nl", "pt", "ru", "zh"]
# BLASER: 0.1-wide bins over [3.0, 6.0); bin 0 is < 3.0, the last is >= 6.0.
B_EDGES = np.round(np.arange(3.0, 6.0001, 0.1), 2)
# xCOMET is saturated near 1: fine bins at the top. Bin i holds [X_EDGES[i-1], X_EDGES[i]); bin 0 is < 0.5.
X_EDGES = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98,
                    0.985, 0.99, 0.995, 0.999])
Q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]


def hist2(b: np.ndarray, x: np.ndarray, w: np.ndarray | None) -> list[list[float]]:
    bi = np.searchsorted(B_EDGES, b, side="right")
    xi = np.searchsorted(X_EDGES, x, side="right")
    h = np.zeros((len(B_EDGES) + 1, len(X_EDGES) + 1))
    np.add.at(h, (bi, xi), 1 if w is None else w)
    return h


def main(out: str) -> None:
    groups = []
    for corpus, name in CORPORA.items():
        f = pq.ParquetFile(f"{ROOT}/{corpus}.parquet")
        base = f.read(columns=["blaser2_src", "audio_length"]).to_pandas()
        bsrc = base.blaser2_src.to_numpy()
        hrs = base.audio_length.to_numpy() / 3600
        for tgt in TARGETS:
            t = f.read(columns=[f"blaser2_mt_{tgt}", f"xcomet_xl_{tgt}"]).to_pandas()
            bmt = t[f"blaser2_mt_{tgt}"].to_numpy()
            xc = t[f"xcomet_xl_{tgt}"].to_numpy()
            ok = np.isfinite(bsrc) & np.isfinite(bmt) & np.isfinite(xc)
            g = {"ds": corpus, "name": name, "tgt": tgt, "n": int(ok.sum()), "h": round(float(hrs[ok].sum()), 2),
                 "missing": int((~ok).sum()),
                 "q_bsrc": np.round(np.quantile(bsrc[ok], Q), 3).tolist(),
                 "q_bmt": np.round(np.quantile(bmt[ok], Q), 3).tolist(),
                 "q_xc": np.round(np.quantile(xc[ok], Q), 4).tolist()}
            for key, b in (("src", bsrc), ("mt", bmt)):
                g[f"{key}_n"] = hist2(b[ok], xc[ok], None).astype(int).tolist()
                g[f"{key}_h"] = np.round(hist2(b[ok], xc[ok], hrs[ok]), 3).tolist()
            groups.append(g)
            print(corpus, tgt, g["n"], g["h"], g["q_bsrc"][4], g["q_xc"][4], file=sys.stderr)
    json.dump({"b_edges": B_EDGES.tolist(), "x_edges": X_EDGES.tolist(), "quantiles": Q, "groups": groups},
              open(out, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
