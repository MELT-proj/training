"""Write cuts-only Shar overlays from MT+QE unit outputs (streams A, C, E).

For one source corpus:
  1. partition: every `final.NNNN.jsonl` of its units is split by source shard into
     `<work>/<name>/shard.<shard>.jsonl` (rebuilt from scratch each run);
  2. overlay: each source shard that has rows is rewritten with `custom.translation_<tgt>` and
     `custom.synth` (see `overlay.set_translation`), audio tars symlinked.

Only shards present in the pool are written. Cuts without a surviving translation stay in the
manifest (the tar is positional) without `custom.synth.ok.<tgt>`, and are dropped at training
time by a `field` rule.

Run inside synth.sif.
"""

import glob
import json
import os
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import tyro

from common import MT_MODEL, PROMPT_VERSION, QE_MODEL
from overlay import overlay_shard, set_translation


def partition(unit_names: list[str], unit_dir: str, out_dir: str, work: str) -> dict[str, int]:
    """Split unit finals into per-shard sidecars. Returns {shard file: rows}."""
    shard_of: dict[str, str] = {}
    for u in unit_names:
        for line in open(os.path.join(unit_dir, f"{u}.pool.jsonl")):
            r = json.loads(line)
            shard_of[r["id"]] = r["shard"]
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    handles: dict[str, object] = {}
    counts: dict[str, int] = {}
    for u in unit_names:
        for path in sorted(glob.glob(os.path.join(out_dir, u, "final.*.jsonl"))):
            for line in open(path):
                shard = shard_of[json.loads(line)["id"]]
                if shard not in handles:
                    handles[shard] = open(os.path.join(work, f"{shard}.sidecar"), "a")
                handles[shard].write(line)
                counts[shard] = counts.get(shard, 0) + 1
    for h in handles.values():
        h.close()
    return counts


def make_annotator(sidecar: str, stream: str, src_text_field: str, tts: dict | None = None):
    rows: dict[str, list[dict]] = {}
    if os.path.exists(sidecar):
        for line in open(sidecar):
            r = json.loads(line)
            rows.setdefault(r["id"], []).append(r)

    def annotate(cut: dict) -> None:
        for r in rows.get(cut["id"], []):
            fields = {
                "xcomet_xxl": r["xcomet_xxl"], "xcomet_spans": r["xcomet_spans"],
                "lid": r["lid"], "lid_p": r["lid_p"], "len_ratio": r["len_ratio"],
            }  # fmt: skip
            if r.get("entity_copy_rate") is not None:
                fields["entity_copy_rate"] = r["entity_copy_rate"]
            if "rt_chrf" in r:
                fields["rt_chrf"] = r["rt_chrf"]
            set_translation(cut, r["tgt"], r["text"], fields)
        if cut["id"] in rows:
            s = cut["custom"]["synth"]
            r0 = rows[cut["id"]][0]
            s.update(
                {"stream": stream, "mt_model": MT_MODEL, "qe_model": QE_MODEL, "prompt_version": PROMPT_VERSION,
                 "src_lang": r0["src_lang"], "src_text_field": src_text_field, "n_entities": r0["n_entities"]}
            )  # fmt: skip
            if tts:
                s["tts"] = tts

    return annotate


def overlay_one(shard_path: str, src_dir: str, out_dir: str, work: str, stream: str, src_text_field: str) -> tuple:
    name = os.path.basename(shard_path)
    annotate = make_annotator(os.path.join(work, f"{name}.sidecar"), stream, src_text_field)
    return overlay_shard(shard_path, src_dir, out_dir, annotate)


def main(
    src_dir: str,
    units: list[str],
    unit_dir: str,
    unit_out: str,
    out_dir: str,
    work: str,
    stream: str = "en2x",
    src_text_field: str = "custom.pnc_text",
    jobs: int = 8,
) -> None:
    counts = partition(units, unit_dir, unit_out, work)
    fn = partial(
        overlay_one, src_dir=src_dir, out_dir=out_dir, work=work, stream=stream, src_text_field=src_text_field
    )
    with Pool(jobs) as pool:
        res = pool.map(fn, [os.path.join(src_dir, s) for s in sorted(counts)])
    print(f"{out_dir}: {len(res)} shards, {sum(counts.values())} translations")
    Path(out_dir, "SYNTH_COMPLETE").write_text(json.dumps({"stream": stream, "translations": counts}))


if __name__ == "__main__":
    tyro.cli(main)
