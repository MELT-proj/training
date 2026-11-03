import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overlay import overlay_shard, set_translation  # noqa: E402


def make_source(tmp_path, n=3):
    src = tmp_path / "src"
    src.mkdir()
    (src / "cuts.000000.jsonl").write_text(
        "".join(json.dumps({"id": f"c{i}", "duration": 1.0, "custom": {"pnc_text": f"t{i}"}}) + "\n" for i in range(n))
    )
    (src / "recording.000000.tar").write_bytes(b"tar")
    return src


def test_overlay_roundtrip_order_links_and_idempotence(tmp_path):
    src, out = make_source(tmp_path), tmp_path / "out"

    def annotate(cut):
        if cut["id"] != "c1":  # c1 stays untranslated: no ok flag, dropped at training time
            set_translation(cut, "lv", "tulkojums " + cut["id"], {"xcomet_xxl": 0.9})

    num, n = overlay_shard(str(src / "cuts.000000.jsonl"), str(src), str(out), annotate)
    assert (num, n) == ("000000", 3)
    cuts = [json.loads(line) for line in (out / "cuts.000000.jsonl").read_text().splitlines()]
    assert [c["id"] for c in cuts] == ["c0", "c1", "c2"]  # same order as the source
    assert cuts[0]["custom"]["pnc_text"] == "t0"  # original fields untouched
    assert cuts[0]["custom"]["synth"]["ok"] == {"lv": 1}
    assert cuts[0]["custom"]["synth"]["xcomet_xxl"] == {"lv": 0.9}
    assert "synth" not in cuts[1]["custom"] and "translation_lv" not in cuts[1]["custom"]
    link = out / "recording.000000.tar"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.read_bytes() == b"tar"
    assert (out / "cuts.000000.jsonl.idx").exists()
    assert overlay_shard(str(src / "cuts.000000.jsonl"), str(src), str(out), annotate)[1] == -1  # skipped
