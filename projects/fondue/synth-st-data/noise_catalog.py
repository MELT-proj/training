"""Stream F: build the licence-filtered noise / RIR catalog (`catalog.json`) with disjoint train and eval splits.

Licence policy (user, 2026-11-02): attribution-only terms that allow derivative work. Allowed:
CC0 / public domain, CC BY (any version, recorded per file), Apache-2.0 (the OpenSLR RIRs). Excluded
everywhere: SA, ND, NC, Sampling+ and anything unparseable. `--allow` narrows the set (e.g. to CC-BY-4.0
and CC0 only); the per-category counts of what was dropped go into `catalog.json["excluded"]`.

Sources: MUSAN (noise + a music subset), DEMAND (CC BY 4.0), FSD50K (per-clip licences, selected labels),
OpenSLR 26/28 RIRs. Splits are disjoint by recording environment where one exists:
DEMAND by environment, RIRs by room, FSD50K by its own dev/eval split (disjoint uploaders), MUSAN by
file id hash. WHAM!/ESC-50 (non-commercial) are not used: every needed category is covered.
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import tyro


ROOT = "/mnt/scratch-nyx/giuseppe/synth/noise"
DEFAULT_ALLOW = ("CC0-1.0", "Public-Domain", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0")

DEMAND_CATEGORY = {
    "DKITCHEN": "domestic", "DLIVING": "domestic", "DWASHING": "domestic",
    "NFIELD": "nature", "NPARK": "nature", "NRIVER": "nature",
    "OHALLWAY": "office", "OMEETING": "office", "OOFFICE": "office",
    "PCAFETER": "public_place", "PRESTO": "public_place", "PSTATION": "public_place",
    "SPSQUARE": "street", "STRAFFIC": "street",
    "TBUS": "transport", "TCAR": "transport", "TMETRO": "metro_train",
}  # fmt: skip
DEMAND_EVAL = {"DWASHING", "NPARK", "OMEETING", "PSTATION", "TBUS"}

FSD_CATEGORY = {
    "dog": ["Dog", "Bark", "Growling"],
    "traffic": ["Traffic_noise_and_roadway_noise", "Car", "Car_passing_by", "Bus", "Truck", "Motorcycle", "Siren"],
    "metro_train": ["Train", "Subway_and_metro_and_underground", "Rail_transport"],
    "crowd": ["Crowd", "Chatter", "Cheering", "Applause", "Laughter"],
    "ambience": ["Rain", "Raindrop", "Wind", "Thunderstorm", "Water", "Stream", "Ocean", "Waves_and_surf",
                 "Fire", "Crackle", "Typing", "Computer_keyboard", "Walk_and_footsteps", "Insect", "Frog", "Cricket"],
}  # fmt: skip
FSD_LICENCE = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
}


def is_eval(key: str, mod: int = 10) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % mod == 0


def musan(root: str, excluded: Counter) -> list[dict]:
    out = []
    m = Path(root) / "musan" / "musan"
    for f in sorted((m / "noise" / "free-sound").glob("*.wav")):  # README: all Public Domain on Freesound
        out.append({"corpus": "musan/noise/free-sound", "file_id": f.stem, "path": str(f), "licence": "Public-Domain",
                    "category": "ambience_misc"})  # fmt: skip
    lic = (m / "noise" / "sound-bible" / "LICENSE").read_text(errors="ignore")
    for blk in re.split(r"\n(?=noise-sound-bible-\d+\n)", lic):
        mm = re.match(r"(noise-sound-bible-\d+)", blk)
        if not mm:
            continue
        lm = re.search(r"License:\s*(.+)", blk)
        raw = lm.group(1).strip() if lm else ""
        norm = {"Public Domain": "Public-Domain", "Attribution 3.0": "CC-BY-3.0"}.get(raw, f"other:{raw}")
        f = m / "noise" / "sound-bible" / f"{mm.group(1)}.wav"
        if norm.startswith("other") or not f.exists():
            excluded[f"musan/sound-bible {norm}"] += 1
            continue
        out.append({"corpus": "musan/noise/sound-bible", "file_id": f.stem, "path": str(f), "licence": norm,
                    "category": "effects"})  # fmt: skip
    # music: only entries whose per-file licence is attribution-only (hd-classical) or CC BY 3.0 for the whole dir (rfm)
    hd = (m / "music" / "hd-classical" / "LICENSE").read_text(errors="ignore")
    for blk in re.split(r"\n=+\n", hd):
        mm = re.search(r"(music-hd-\d+)", blk)
        lm = re.search(r"License:\s*(.+)", blk)
        if not mm:
            continue
        raw = lm.group(1).strip() if lm else ""
        norm = {"Public Domain": "Public-Domain", "CC BY 3.0": "CC-BY-3.0"}.get(raw, f"other:{raw}")
        f = m / "music" / "hd-classical" / f"{mm.group(1)}.wav"
        if norm.startswith("other") or not f.exists():
            excluded[f"musan/hd-classical {norm}"] += 1
            continue
        out.append({"corpus": "musan/music/hd-classical", "file_id": f.stem, "path": str(f), "licence": norm,
                    "category": "music"})  # fmt: skip
    for f in sorted((m / "music" / "rfm").glob("*.wav")):  # dir LICENSE: CC BY 3.0 (Kevin MacLeod)
        out.append({"corpus": "musan/music/rfm", "file_id": f.stem, "path": str(f), "licence": "CC-BY-3.0",
                    "category": "music"})  # fmt: skip
    for d in ("fma", "fma-western-art", "jamendo"):  # mixed SA / ND / BY per album: not used
        excluded[f"musan/music/{d} (mixed SA/ND)"] += len(list((m / "music" / d).glob("*.wav")))
    for e in out:
        e["split"] = "eval" if is_eval(e["file_id"]) else "train"
    return out


def demand(root: str) -> list[dict]:
    out = []
    for env, cat in DEMAND_CATEGORY.items():
        f = Path(root) / "demand" / "x" / env / "ch01.wav"
        if f.exists():
            out.append({"corpus": "DEMAND", "file_id": f"{env}/ch01", "path": str(f), "licence": "CC-BY-4.0",
                        "category": cat, "split": "eval" if env in DEMAND_EVAL else "train"})  # fmt: skip
    return out


def fsd50k(root: str, excluded: Counter, max_per_cat_split: int = 600) -> list[dict]:
    base = Path(root) / "fsd50k"
    audio = {"dev": base / "audio" / "dev", "eval": base / "audio" / "eval"}
    if not audio["dev"].exists():
        return []
    cat_of = {lab: c for c, labs in FSD_CATEGORY.items() for lab in labs}
    out = []
    for part, split in (("dev", "train"), ("eval", "eval")):
        info = json.load(open(base / "meta" / "FSD50K.metadata" / f"{part}_clips_info_FSD50K.json"))
        per_cat: Counter = Counter()
        for row in csv.DictReader(open(base / "meta" / "FSD50K.ground_truth" / f"{part}.csv")):
            cats = [cat_of[x] for x in row["labels"].split(",") if x in cat_of]
            if not cats or "Speech" in row["labels"].split(","):
                continue
            lic = FSD_LICENCE.get(info[row["fname"]]["license"])
            if lic is None:
                excluded[f"fsd50k {info[row['fname']]['license'].split('/licenses/')[-1]}"] += 1
                continue
            f = audio[part] / f"{row['fname']}.wav"
            if not f.exists() or per_cat[cats[0]] >= max_per_cat_split:
                continue
            per_cat[cats[0]] += 1
            out.append({"corpus": "FSD50K", "file_id": row["fname"], "path": str(f), "licence": lic,
                        "category": cats[0], "split": split, "uploader": info[row["fname"]].get("uploader")})  # fmt: skip
    return out


def rirs(root: str) -> list[dict]:
    out = []
    base = Path(root) / "rir" / "x"
    for kind in ("smallroom", "mediumroom", "largeroom"):
        rooms = sorted(p for p in (base / "simulated_rirs_16k" / kind).iterdir() if p.is_dir())
        for room in rooms:
            wavs = sorted(room.glob("*.wav"))[:3]  # a few source positions per room
            for f in wavs:
                out.append({"corpus": f"openslr26/{kind}", "file_id": f"{room.name}/{f.stem}", "path": str(f),
                            "licence": "Apache-2.0", "category": kind,
                            "split": "eval" if is_eval(room.name) else "train"})  # fmt: skip
    real = base / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    for f in sorted(real.glob("*.wav")):
        out.append({"corpus": "openslr28/real", "file_id": f.stem, "path": str(f), "licence": "Apache-2.0",
                    "category": "real", "split": "eval" if is_eval(f.stem, 5) else "train"})  # fmt: skip
    return out


def main(root: str = ROOT, out: str = f"{ROOT}/catalog.json", allow: list[str] | None = None) -> None:
    allow_set = set(allow or DEFAULT_ALLOW)
    excluded: Counter = Counter()
    noise = musan(root, excluded) + demand(root) + fsd50k(root, excluded)
    rir = rirs(root)
    keep_n = [n for n in noise if n["licence"] in allow_set]
    keep_r = [r for r in rir if r["licence"] in allow_set]
    for n in noise:
        if n["licence"] not in allow_set:
            excluded[f"{n['corpus']} {n['licence']} (not in --allow)"] += 1
    cat = {
        "allow": sorted(allow_set),
        "noise": {s: [n for n in keep_n if n["split"] == s] for s in ("train", "eval")},
        "rir": {s: [r for r in keep_r if r["split"] == s] for s in ("train", "eval")},
        "excluded": dict(excluded),
    }
    Path(out).write_text(json.dumps(cat, indent=1))
    for s in ("train", "eval"):
        c = Counter((n["category"], n["licence"]) for n in cat["noise"][s])
        print(s, "noise", len(cat["noise"][s]), "rir", len(cat["rir"][s]))
        for (k, lic), v in sorted(c.items()):
            print(f"   {k:14s} {lic:14s} {v}")
    print("excluded:", dict(excluded))


if __name__ == "__main__":
    tyro.cli(main)
