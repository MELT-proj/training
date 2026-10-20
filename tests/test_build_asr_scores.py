"""infra/build_asr_scores.py: sidecar parquet -> uint16 score arrays."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


_spec = importlib.util.spec_from_file_location(
    "build_asr_scores", Path(__file__).resolve().parents[1] / "infra" / "build_asr_scores.py"
)
bas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bas)


def _sidecar(path: Path, status, wer, cer, ids=None):
    n = len(status)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "row_idx": pa.array(range(n), pa.int32()),
                "cut_id": ids or [f"c{i}" for i in range(n)],
                "asr_status": status,
                "asr_wer": pa.array(wer, pa.float32()),
                "asr_cer": pa.array(cer, pa.float32()),
                "asr_model": ["parakeet"] * n,
            }
        ),
        path,
    )


def test_round_trip_and_nulls(tmp_path):
    src = tmp_path / "in" / "x" / "asr_wer.parquet"
    _sidecar(
        src,
        ["scored", "scored", "no_text", "scored", "scored"],
        [0.0, 0.1, 0.5, None, 12.0],
        [0.0, 0.05, 0.5, 0.2, 1.0],
    )
    meta = bas.build_leaf(src, tmp_path / "out" / "x")
    wer = np.load(tmp_path / "out" / "x" / "wer.u16.npy")
    cer = np.load(tmp_path / "out" / "x" / "cer.u16.npy")
    assert wer.dtype == np.uint16
    # 0.1 stored as float32 is 0.10000000149; it must land exactly on 1000.
    assert wer.tolist() == [0, 1000, 65535, 65535, 65534]
    assert cer.tolist() == [0, 500, 65535, 2000, 10000]
    assert meta["rows"] == 5 and meta["scored"] == 3
    assert json.loads((tmp_path / "out" / "x" / "meta.json").read_text())["rows"] == 5


def test_row_alignment_checks(tmp_path):
    src = tmp_path / "in" / "x" / "asr_wer.parquet"
    _sidecar(src, ["scored"] * 2, [0.0, 0.1], [0.0, 0.1])
    stats = tmp_path / "stats" / "x" / "cuts.parquet"
    stats.parent.mkdir(parents=True)
    pq.write_table(pa.table({"cut_id": ["c0", "other"]}), stats)
    with pytest.raises(ValueError, match="differs"):
        bas.build_leaf(src, tmp_path / "out" / "x", stats)
