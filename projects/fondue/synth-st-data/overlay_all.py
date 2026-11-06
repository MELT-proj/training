"""Write the cuts-only overlays for every corpus whose units are fully scored (all streams but F).

Maps each unit family to its source Shar directory, checks that the pool rows really point at that
directory (cut id at the recorded line), and calls `write_shar` once per corpus. Corpora whose units
are not all `DONE` are skipped unless `--partial`. Run inside synth.sif.
"""

import glob
import json
import os
import re
from pathlib import Path

import tyro

from fleurs_gold import FLEURS_DIR
from write_shar import main as write_shar


SHAR = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar"
SYNTH = "/mnt/scratch-nyx/giuseppe/synth"
GRANARY = {
    "bg": "Bulgarian", "da": "Danish", "et": "Estonian", "fi": "Finnish", "hr": "Croatian", "hu": "Hungarian",
    "lt": "Lithuanian", "lv": "Latvian", "sk": "Slovak", "sv": "Swedish", "uk": "Ukrainian",
}  # fmt: skip
EN = {"giga_en": ("gigaspeech/train", "custom.pnc_text"), "granary_en": ("yodas-granary/English/asr_only", "custom.pnc_text"),
      "vp_en": ("voxpopuli/en/train", "custom.pnc_text"), "cv22_en": ("cv22_sidon/en/train", "custom.metadata.sentence")}  # fmt: skip


def source(corpus: str) -> tuple[str, str, str]:
    """(source dir relative to SHAR or absolute, source text field, stream) for a unit family."""
    if corpus in EN:
        return *EN[corpus], "A"
    if corpus == "tts_en":
        return f"{SYNTH}/e/tts_en/train", "text", "E"
    if m := re.fullmatch(r"fleurs_(\w+)", corpus):
        return f"fleurs:{m[1]}", "supervisions.0.text", "B"
    if m := re.fullmatch(r"c_cv22_(\w+)", corpus):
        return f"cv22:{m[1]}", "custom.metadata.sentence", "C"
    if m := re.fullmatch(r"c_vp_(\w+)", corpus):
        return f"voxpopuli/{m[1]}/train", "custom.pnc_text", "C"
    if m := re.fullmatch(r"y2lr_cv_(\w+)", corpus):
        return f"cv22_sidon/{m[1]}/train", "custom.metadata.sentence", "C"
    if m := re.fullmatch(r"y2lr_vo_(\w+)", corpus):
        return f"voxpopuli/{m[1]}/train", "custom.pnc_text", "C"
    if m := re.fullmatch(r"d_granary_(\w+)", corpus):
        return f"yodas-granary/{GRANARY[m[1]]}/ast", "supervisions.0.text", "D"
    if m := re.fullmatch(r"d_covost_en_(\w+)", corpus):
        return f"covost2/en_{'sv-SE' if m[1] == 'sv' else m[1]}/train", "supervisions.0.text", "D"
    raise KeyError(corpus)


def resolve(rel: str) -> str:
    if rel.startswith("/"):
        return rel
    if rel.startswith("cv22:"):
        lang = rel[5:]
        d = {"ga": "ga-IE", "sv": "sv-SE"}.get(lang, lang)
        return f"{SHAR}/cv22_sidon/{d}/train"
    return f"{SHAR}/{rel}"


def check_pointers(src: str, pool: str) -> None:
    """The unit pool must point at this directory: first, middle and last rows hold the recorded cut id."""
    rows = [json.loads(line) for line in open(pool)]
    for r in (rows[0], rows[len(rows) // 2], rows[-1]):
        line = open(os.path.join(src, r["shard"])).read().split("\n")[r["idx"]]
        assert json.loads(line)["id"] == r["id"], f"{pool}: {r['id']} is not line {r['idx']} of {src}/{r['shard']}"


def main(
    only: list[str] = [],
    units_dir: str = f"{SYNTH}/units",
    out_root: str = f"{SYNTH}/shar-out",
    work_root: str = f"{SYNTH}/overlay-work",
    jobs: int = 8,
    partial: bool = False,
) -> None:  # fmt: skip
    families: dict[str, list[str]] = {}
    for pool in sorted(glob.glob(f"{units_dir}/*.pool.jsonl")):
        unit = os.path.basename(pool)[: -len(".pool.jsonl")]
        families.setdefault(re.sub(r"-\d{3}$", "", unit), []).append(unit)
    for corpus, units in families.items():
        if only and corpus not in only:
            continue
        rel, field, stream = source(corpus)
        outs = {"A": "out-a", "B": "out-b", "C": "out-c", "D": "out-d", "E": "out-e"}[stream]
        unit_out = f"{SYNTH}/{outs}"
        undone = [u for u in units if not os.path.exists(f"{unit_out}/{u}/DONE")]
        if undone and not partial:
            print(f"SKIP {corpus}: {len(undone)}/{len(units)} units not DONE")
            continue
        src = resolve(rel)
        if rel.startswith("fleurs:"):
            lang = rel[7:]
            src = f"{SHAR}/fleurs/{FLEURS_DIR[lang]}/train"
        for u in units:
            check_pointers(src, f"{units_dir}/{u}.pool.jsonl")
        out = Path(out_root) / corpus / "train"
        out.mkdir(parents=True, exist_ok=True)
        write_shar(
            src_dir=src, units=units, unit_dir=units_dir, unit_out=unit_out, out_dir=str(out),
            work=f"{work_root}/{corpus}", stream={"A": "en2x", "B": "fleurs", "C": "lr", "D": "existing", "E": "tts"}[stream],
            src_text_field=field, jobs=jobs,
        )  # fmt: skip


if __name__ == "__main__":
    tyro.cli(main)
