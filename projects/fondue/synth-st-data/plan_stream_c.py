"""Stream C pools: native LR transcripts -> en/de/fr/es/it + neighbour language; de/fr/es/it audio -> the 18 LRs.

Runs build_pool.py once per (corpus, language). Run inside synth.sif. WER bands are Fondue's
ANNEAL bounds (build_fondue_mix.py: WER_BY_LANG, then WER_BY_CORPUS, then the default).
"""

import glob
import os
import subprocess
import sys

import tyro

from common import EN2X_TARGETS


SHAR = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar"
SCORES = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar-asr-scores"
NEIGHBOUR = {"lv": "lt", "lt": "lv", "sk": "cs", "cs": "sk", "hr": "sl", "sl": "hr",
             "fi": "et", "et": "fi", "da": "sv", "sv": "da", "bg": "uk", "uk": "bg"}  # fmt: skip
Y = ["de", "fr", "es", "it"]
CV22_DIR = {"ga": "ga-IE", "sv": "sv-SE"}
CV22_LR = ["bg", "cs", "da", "el", "et", "fi", "ga", "hu", "lt", "lv", "mt", "pl", "ro", "sk", "sl", "sv", "uk"]
VP_LR = ["cs", "et", "fi", "hr", "hu", "lt", "pl", "ro", "sk", "sl"]
WER_LANG_ANNEAL = {
    **dict.fromkeys(("ga", "sl", "et", "lv", "el"), 0.2),
    **dict.fromkeys(("mt", "lt", "sv", "sk"), 0.15),
}
WER_CORPUS_ANNEAL = {"cv22_sidon": 0.05, "voxpopuli": 0.05}


def run(corpus: str, name: str, shar_rel: str, lang: str, text_field: str, targets: list[str] | None,
        n_random: int, stride: int, out_dir: str) -> None:  # fmt: skip
    shar_dir = os.path.join(SHAR, shar_rel)
    if not glob.glob(os.path.join(shar_dir, "cuts.*.jsonl")):
        print("SKIP (no shards):", shar_dir)
        return
    if not os.path.exists(os.path.join(SCORES, shar_rel, "wer.u16.npy")):
        print("SKIP (no WER array, Fondue excludes it too):", shar_rel)
        return
    wer = WER_LANG_ANNEAL.get(lang) or WER_CORPUS_ANNEAL[corpus]
    cmd = [sys.executable, "build_pool.py", "--shar-dir", shar_dir, "--out-dir", out_dir, "--corpus", name,
           "--text-field", text_field, "--scores-dir", os.path.join(SCORES, shar_rel), "--wer-max", str(wer),
           "--src-lang", lang, "--shard-stride", str(stride), "--n-random", str(n_random), "--workers", "4"]  # fmt: skip
    if targets:
        cmd += ["--targets", *targets]
    else:
        cmd += ["--pool-targets", *EN2X_TARGETS]
    print(" ".join(cmd[2:8]), "wer<", wer, flush=True)
    subprocess.run(cmd, check=True)


def main(out_dir: str = "/mnt/scratch-nyx/giuseppe/synth/pool", y_hours_shards: int = 5) -> None:
    for lang in CV22_LR:
        d = CV22_DIR.get(lang, lang)
        tg = ["en", *Y] + ([NEIGHBOUR[lang]] if lang in NEIGHBOUR else [])
        run(
            "cv22_sidon",
            f"c_cv22_{lang}",
            f"cv22_sidon/{d}/train",
            lang,
            "custom.metadata.sentence",
            tg,
            0,
            1,
            out_dir,
        )
    for lang in VP_LR:
        tg = ["en", *Y] + ([NEIGHBOUR[lang]] if lang in NEIGHBOUR else [])
        run("voxpopuli", f"c_vp_{lang}", f"voxpopuli/{lang}/train", lang, "custom.pnc_text", tg, 0, 1, out_dir)
    # Y -> LR: ~400 h of audio per Y, each cut into ~5 tail-weighted targets (~100 h per pair direction)
    for y in Y:
        for corpus, field, rel in (("cv22_sidon", "custom.metadata.sentence", f"cv22_sidon/{y}/train"),
                                   ("voxpopuli", "custom.pnc_text", f"voxpopuli/{y}/train")):  # fmt: skip
            n = len(glob.glob(os.path.join(SHAR, rel, "cuts.*.jsonl")))
            run(corpus, f"y2lr_{corpus[:2]}_{y}", rel, y, field, None, 5, max(1, n // y_hours_shards), out_dir)


if __name__ == "__main__":
    tyro.cli(main)
