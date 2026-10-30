#!/usr/bin/env python3
"""Turn the ASR-WER sidecars into the compact score arrays the training loader joins.

Input: ``<asr_root>/<leaf>/asr_wer.parquet`` (one row per row of the leaf's Shar
manifests, in order; columns ``row_idx``, ``cut_id``, ``asr_status``, ``asr_wer``,
``asr_cer``, ``asr_model``). Output, one directory per leaf under ``--out_root``:

    wer.u16.npy, cer.u16.npy   uint16, round(10000 * score); 65535 = no score
    meta.json                  rows, md5 of the cut-id column, model, source mtime

New files only; the sidecars and the Shar manifests are never touched. Scores
above 6.5534 saturate (such a cut is far outside any band anyway). With
``--stats_root``, each leaf's ``cut_id`` column is also compared with
``<stats_root>/<leaf>/cuts.parquet`` (the row order of the manifests).

    python infra/build_asr_scores.py --asr_root .../shar-asr-wer --out_root .../shar-asr-scores
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SCALE = 10_000
NULL = 65_535


def quantise(values: np.ndarray) -> np.ndarray:
    """float array with NaN for missing -> uint16 (NULL for missing)."""
    out = np.full(values.shape, NULL, dtype=np.uint16)
    ok = ~np.isnan(values)
    out[ok] = np.minimum(np.rint(values[ok].astype(np.float64) * SCALE), NULL - 1).astype(np.uint16)
    return out


def build_leaf(parquet: Path, out_dir: Path, stats_parquet: Path | None = None) -> dict:
    table = pq.read_table(
        parquet, columns=["row_idx", "cut_id", "asr_status", "asr_wer", "asr_cer", "asr_model"]
    )
    row_idx = table.column("row_idx").to_numpy()
    if not np.array_equal(row_idx, np.arange(len(row_idx))):
        raise ValueError(f"{parquet}: row_idx is not 0..n-1 in order")
    ids = table.column("cut_id").to_pylist()
    if stats_parquet is not None:
        stats_ids = pq.read_table(stats_parquet, columns=["cut_id"]).column(0).to_pylist()
        if stats_ids != ids:
            raise ValueError(f"{parquet}: cut_id column differs from {stats_parquet}")
    status = np.array(table.column("asr_status").to_pylist())
    arrays = {}
    for metric in ("wer", "cer"):
        col = table.column(f"asr_{metric}").to_numpy(zero_copy_only=False).astype(np.float64)
        col[status != "scored"] = np.nan  # null where there is no score, whatever the file holds
        arrays[metric] = quantise(col)
    models = sorted({m for m in table.column("asr_model").to_pylist() if m})
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, arr in arrays.items():
        np.save(out_dir / f"{metric}.u16.npy", arr)
    meta = {
        "rows": len(ids),
        "cut_ids_md5": hashlib.md5("\n".join(ids).encode()).hexdigest(),
        "models": models,
        "scale": SCALE,
        "null": NULL,
        "source": str(parquet),
        "source_mtime": parquet.stat().st_mtime,
        "scored": int((arrays["wer"] != NULL).sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--stats_root", type=Path, default=None)
    ap.add_argument("--leaf_glob", default="**/asr_wer.parquet")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    files = sorted(args.asr_root.glob(args.leaf_glob))
    if not files:
        print(f"no asr_wer.parquet under {args.asr_root}", file=sys.stderr)
        return 1
    for f in files:
        leaf = f.parent.relative_to(args.asr_root)
        if args.dry_run:
            print("would build", leaf)
            continue
        stats = args.stats_root / leaf / "cuts.parquet" if args.stats_root else None
        meta = build_leaf(f, args.out_root / leaf, stats)
        print(f"{leaf}: {meta['rows']} rows, {meta['scored']} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
