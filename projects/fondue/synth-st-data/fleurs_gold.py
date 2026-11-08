"""Stream B: gold LR->Y translations from FLEURS' n-way parallelism, plus the split-overlap audit.

FLEURS cut ids carry no FLoRes sentence id, so each FLEURS transcript is mapped to its
FLoRes-200 dev/devtest sentence by normalised text (lowercase, no punctuation or symbols,
collapsed spaces) -- the same sentence in every language shares that id.

Audit (06-fondue.md 3.1): for every language, the FLoRes ids of the FLEURS train split must
not occur in its test or validation split. Written to `<out>/audit.json`; the join is
refused for a language that fails it.

Join: a train cut of language L whose sentence id also occurs in the FLEURS **train** split of
Y (en de fr es it) gets `translation_<Y>` = the FLoRes text of that sentence in Y (cased,
punctuated, human). Restricting to Y's train ids keeps every Y sentence out of Y's evaluation
splits. Output: cuts-only overlays `<out_shar>/synth_fleurs_gold/<locale>/train`.

Run inside synth.sif.
"""

import glob
import json
import os
import tarfile
import unicodedata
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import tyro

from overlay import overlay_shard, set_translation


FLORES = {
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "bg": "bul_Cyrl", "cs": "ces_Latn", "da": "dan_Latn", "el": "ell_Grek", "fi": "fin_Latn",
    "ga": "gle_Latn", "hr": "hrv_Latn", "hu": "hun_Latn", "lt": "lit_Latn", "mt": "mlt_Latn",
    "pl": "pol_Latn", "ro": "ron_Latn", "sk": "slk_Latn", "uk": "ukr_Cyrl", "sv": "swe_Latn",
    "lv": "lvs_Latn", "et": "est_Latn", "sl": "slv_Latn", "nl": "nld_Latn", "pt": "por_Latn", "ru": "rus_Cyrl",
}  # fmt: skip
FLEURS_DIR = {
    "bg": "bg_bg", "cs": "cs_cz", "da": "da_dk", "de": "de_de", "el": "el_gr", "en": "en_us",
    "es": "es_419", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "ga": "ga_ie", "hr": "hr_hr",
    "hu": "hu_hu", "it": "it_it", "lt": "lt_lt", "lv": "lv_lv", "mt": "mt_mt", "nl": "nl_nl",
    "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "sv": "sv_se",
    "uk": "uk_ua", "ru": "ru_ru",
}  # fmt: skip
Y_LANGS = ["en", "de", "fr", "es", "it"]
LR_LANGS = [
    "bg", "cs", "da", "el", "fi", "ga", "hr", "hu", "lt", "mt",
    "pl", "ro", "sk", "uk", "sv", "lv", "et", "sl",
]  # fmt: skip
SPLITS = ("train", "validation", "test")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in s)
    return " ".join(s.split())


def load_flores(tar_path: str) -> dict[str, dict[str, str]]:
    """{lang: {sentence_id: text}}; the id is `<split>:<line>` so dev and devtest never clash."""
    out: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path) as tf:
        for lang, code in FLORES.items():
            d = {}
            for split in ("dev", "devtest"):
                f = tf.extractfile(f"./flores200_dataset/{split}/{code}.{split}")
                for i, line in enumerate(f.read().decode("utf-8").splitlines()):
                    d[f"{split}:{i}"] = line.strip()
            out[lang] = d
    return out


def fleurs_ids(shar_root: str, lang: str, split: str, text_to_id: dict[str, str]) -> list[tuple[str, int, str | None]]:
    """(shard file, position, flores id or None) for every cut of one FLEURS split."""
    rows = []
    d = os.path.join(shar_root, "fleurs", FLEURS_DIR[lang], split)
    for path in sorted(glob.glob(os.path.join(d, "cuts.*.jsonl"))):
        with open(path) as f:
            for i, line in enumerate(f):
                text = json.loads(line)["supervisions"][0]["text"]
                rows.append((os.path.basename(path), i, text_to_id.get(norm(text))))
    return rows


def audit_and_index(shar_root: str, flores: dict) -> tuple[dict, dict]:
    audit, index = {}, {}
    for lang in FLORES:
        text_to_id = {norm(t): i for i, t in flores[lang].items()}
        per = {s: fleurs_ids(shar_root, lang, s, text_to_id) for s in SPLITS}
        ids = {s: {r[2] for r in per[s] if r[2]} for s in SPLITS}
        audit[lang] = {
            **{f"{s}_cuts": len(per[s]) for s in SPLITS},
            **{f"{s}_matched": sum(r[2] is not None for r in per[s]) for s in SPLITS},
            **{f"{s}_ids": len(ids[s]) for s in SPLITS},
            "train_test_overlap": len(ids["train"] & ids["test"]),
            "train_validation_overlap": len(ids["train"] & ids["validation"]),
        }
        index[lang] = per["train"]
    return audit, index


def join_language(
    lang: str, rows: list, flores: dict, train_ids: dict[str, set], shar_root: str, out_shar: str
) -> dict:  # fmt: skip
    by_shard: dict[str, dict[int, str]] = {}
    for shard, i, fid in rows:
        if fid:
            by_shard.setdefault(shard, {})[i] = fid
    src_dir = os.path.join(shar_root, "fleurs", FLEURS_DIR[lang], "train")
    out_dir = os.path.join(out_shar, "synth_fleurs_gold", FLEURS_DIR[lang], "train")
    counts = dict.fromkeys(Y_LANGS, 0)
    for path in sorted(glob.glob(os.path.join(src_dir, "cuts.*.jsonl"))):
        name = os.path.basename(path)
        state = {"i": -1}
        mapping = by_shard.get(name, {})

        def annotate(cut: dict) -> None:
            state["i"] += 1
            fid = mapping.get(state["i"])
            if not fid:
                return
            for y in Y_LANGS:
                if y != lang and fid in train_ids[y]:
                    set_translation(cut, y, flores[y][fid], {"gold": 1})
                    counts[y] += 1
            if "synth" in cut.get("custom", {}):
                cut["custom"]["synth"].update(
                    {"stream": "fleurs_gold", "src_lang": lang, "flores_id": fid,
                     "src_text_field": "supervisions.0.text", "licence": "CC-BY-SA-4.0 (FLoRes-200) / CC-BY-4.0 (FLEURS)"}
                )  # fmt: skip

        overlay_shard(path, src_dir, out_dir, annotate)
    return counts


def main(
    flores_tar: str = "/mnt/scratch-nyx/giuseppe/synth/flores/flores200.tar.gz",
    shar_root: str = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar",
    out_shar: str = "/mnt/scratch-nyx/giuseppe/synth/shar",
    report_dir: str = "/mnt/scratch-nyx/giuseppe/synth/reports",
    jobs: int = 6,
) -> None:
    flores = load_flores(flores_tar)
    audit, index = audit_and_index(shar_root, flores)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    (Path(report_dir) / "fleurs_audit.json").write_text(json.dumps(audit, indent=1))
    bad = [lang for lang, a in audit.items() if a["train_test_overlap"] or a["train_validation_overlap"]]
    print("audit: languages with train/eval sentence overlap:", bad or "none")
    low = {lang: round(a["train_matched"] / max(a["train_cuts"], 1), 3) for lang, a in audit.items()}
    print("train match rate:", low)
    if bad:
        raise SystemExit("audit failed: not writing the join")
    train_ids = {lang: {r[2] for r in rows if r[2]} for lang, rows in index.items()}
    fn = partial(join_language, flores=flores, train_ids=train_ids, shar_root=shar_root, out_shar=out_shar)
    with Pool(jobs) as pool:
        res = pool.starmap(fn, [(lang, index[lang]) for lang in LR_LANGS])
    summary = dict(zip(LR_LANGS, res))
    (Path(report_dir) / "fleurs_gold_counts.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    tyro.cli(main)
