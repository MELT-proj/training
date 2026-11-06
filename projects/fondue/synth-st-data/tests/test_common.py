import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_prompt, is_degenerate, ledger_append, read_jsonl, write_jsonl_atomic  # noqa: E402
from run_mt_qe import expand_pairs  # noqa: E402


def test_prompt_is_instruction_first():
    msg = build_prompt("en", "lv", "Hello there.")[0]["content"]
    assert msg.startswith("Translate the following English text into Latvian. Output only the translation.")
    assert msg.endswith("Hello there.")


def test_jsonl_roundtrip_is_atomic(tmp_path):
    rows = [{"id": "a", "t": "ž"}, {"id": "b", "t": "x"}]
    p = tmp_path / "d" / "o.jsonl"
    write_jsonl_atomic(p, rows)
    assert read_jsonl(p) == rows
    assert [f.name for f in p.parent.iterdir()] == ["o.jsonl"]


def test_degenerate():
    assert is_degenerate("A fine sentence here.", "")
    assert is_degenerate("Short.", "x" * 200)
    assert is_degenerate("A sentence of some length here ok.", " ".join(["la la la"] * 10))
    assert not is_degenerate("The vote took place today.", "Balsojums notika šodien.")


def test_expand_pairs_uses_row_targets_and_skips_self():
    pool = [{"id": "a", "targets": ["lv", "mt"]}, {"id": "b", "src_lang": "lv", "targets": ["lv", "en"]}]
    pairs = expand_pairs(pool)
    assert [(r["id"], t) for r, t in pairs] == [("a", "lv"), ("a", "mt"), ("b", "en")]
    assert [(r["id"], t) for r, t in expand_pairs([{"id": "c"}], ["ga"])] == [("c", "ga")]


def test_ledger_header_once(tmp_path):
    p = tmp_path / "l.tsv"
    ledger_append(p, unit="u", stage="mt", n=1, gpu_seconds=2, gpus=1, status="ok")
    ledger_append(p, unit="u", stage="qe", n=1, gpu_seconds=3, gpus=1, status="ok")
    lines = p.read_text().splitlines()
    assert lines[0].startswith("unit\t") and len(lines) == 3


def test_assign_targets_dense_all_random_weighted_and_stable():
    from build_pool import assign_targets
    from common import EN2X_TARGETS

    assert assign_targets("x", True, 8, EN2X_TARGETS) == EN2X_TARGETS
    assert assign_targets("x", False, 8, EN2X_TARGETS) == assign_targets("x", False, 8, EN2X_TARGETS)
    counts = dict.fromkeys(EN2X_TARGETS, 0)
    for i in range(4000):
        t = assign_targets(f"id{i}", False, 8, EN2X_TARGETS)
        assert len(set(t)) == 8
        for x in t:
            counts[x] += 1
    assert counts["mt"] > 1.3 * counts["pl"] and counts["ga"] > 1.3 * counts["cs"]


def test_chunks_cover_all_pairs():
    from run_mt_qe import chunks

    assert [len(c) for c in chunks(list(range(10)), 4)] == [4, 4, 2]
