"""Tests for infra/compute_mix_weights.py's --skip-cut measurement."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

import compute_mix_weights as cmw  # noqa: E402

NO_TRANSCRIPT = (
    "not (cut.get('supervisions') and "
    "(cut['supervisions'][0].get('text') or '').strip())"
)


@pytest.fixture
def shard(tmp_path):
    rows = [
        {"id": "a", "duration": 10, "supervisions": [{"text": "hello"}]},
        {"id": "b", "duration": 5, "supervisions": [{"text": "  "}]},
        {"id": "c", "duration": 2, "supervisions": []},
    ]
    path = tmp_path / "cuts.000000.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_no_skip_counts_everything(shard):
    assert cmw.shard_seconds(shard) == (17.0, 0.0, 0)


def test_skip_cut_drops_cuts_without_transcript(shard):
    assert cmw.shard_seconds(shard, NO_TRANSCRIPT) == (10.0, 7.0, 2)


def test_skip_cut_fails_loudly_on_a_missing_key(shard):
    with pytest.raises(RuntimeError, match="--skip-cut raised"):
        cmw.shard_seconds(shard, "cut['nope']")
