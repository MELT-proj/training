"""Hours in the synthetic collection, measured from the written manifests (overlays and noisy copies).

Three measures, because "hours" is ambiguous when one recording carries up to 18 translations:

- audio hours: seconds of distinct cuts (a cut counts once however many targets it carries);
- pair-hours: seconds summed over (cut, target) pairs, i.e. what a training run sees, since every target is its own
  training example;
- kept: the pairs the QE thresholds would let through. A pair is kept when `custom.synth.metricx_xxl.<tgt>` is at most
  `metricx_max` of its (stream, src, tgt) row in `thresholds_keep75.csv` (the bound is inclusive, as in the training
  `cut_filter`).

Reads every `cuts.*.jsonl` of `shar-out/*/train` (streams A-E; D only adds scores to existing data) and of
`shar-noisy/*_noisy/train` (stream F: noisy copies of A/C/E cuts, plus noisy copies of native ASR cuts). Writes
`hours.json`, `hours_by_corpus.csv`, `hours_by_leaf.csv` (one row per corpus x target, in seconds) and
`hours_by_direction.csv` to `--out`. Run inside synth.sif.
"""

import csv
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import tyro

from overlay_all import source


SYNTH = "/mnt/scratch-nyx/giuseppe/synth"
STREAM_LETTER = {"en2x": "A", "fleurs": "B", "lr": "C", "existing": "D", "tts": "E"}
# What the stream means, for the report.
STREAM_NOTE = {
    "A": "en -> 18 targets over English audio (MT + QE)",
    "B": "FLEURS transcripts -> en/de/fr/es/it (MT + QE)",
    "C": "LR -> en/Y, neighbour pairs, Y -> LR (MT + QE)",
    "D": "QE scores on existing ST data (no new translation)",
    "E": "LLM text -> TTS audio -> MT + QE (new audio)",
}


def thresholds(path: str) -> dict[tuple[str, str, str], float]:
    with open(path) as f:
        return {(r["stream"], r["src"], r["tgt"]): float(r["metricx_max"]) for r in csv.DictReader(f)}


def scan_shard(args: tuple[str, dict]) -> dict:
    """Aggregate one manifest: per-direction pair counts/seconds, per-cut totals."""
    path, thr = args
    cuts = n_sec = tr_cuts = tr_sec = kept_cuts = kept_sec = 0.0
    pairs: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0, 0.0])
    unknown: dict[str, int] = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            cut = json.loads(line)
            dur = cut["duration"]
            cuts += 1
            n_sec += dur
            synth = cut.get("custom", {}).get("synth", {})
            targets = list(synth.get("ok", {}))
            if not targets:
                continue
            letter = STREAM_LETTER[synth["stream"]]
            tr_cuts += 1
            tr_sec += dur
            any_kept = False
            for tgt in targets:
                key = (letter, synth["src_lang"], tgt)
                row = pairs[key]
                row[0] += 1
                row[1] += dur
                bound = thr.get(key)
                if bound is None:
                    unknown["/".join(key)] += 1
                elif synth["metricx_xxl"][tgt] <= bound:
                    row[2] += 1
                    row[3] += dur
                    any_kept = True
            if any_kept:
                kept_cuts += 1
                kept_sec += dur
    return {
        "cuts": cuts, "sec": n_sec, "tr_cuts": tr_cuts, "tr_sec": tr_sec, "kept_cuts": kept_cuts, "kept_sec": kept_sec,
        "pairs": dict(pairs), "unknown": dict(unknown),
    }  # fmt: skip


def scan_corpus(
    shar_dir: str, thr: dict, pool: Pool
) -> dict:  # fmt: skip
    shards = sorted(glob.glob(f"{shar_dir}/cuts.*.jsonl"))
    parts = pool.map(scan_shard, [(s, thr) for s in shards])
    out = {"shards": len(shards), "cuts": 0, "sec": 0.0, "tr_cuts": 0, "tr_sec": 0.0, "kept_cuts": 0, "kept_sec": 0.0}
    pairs: dict[tuple, list[float]] = defaultdict(lambda: [0, 0.0, 0, 0.0])
    unknown: dict[str, int] = defaultdict(int)
    for p in parts:
        for k in ("cuts", "sec", "tr_cuts", "tr_sec", "kept_cuts", "kept_sec"):
            out[k] += p[k]
        for key, v in p["pairs"].items():
            for i in range(4):
                pairs[key][i] += v[i]
        for k, n in p["unknown"].items():
            unknown[k] += n
    out["pairs"] = dict(pairs)
    out["unknown"] = dict(unknown)
    out["pair_n"] = sum(v[0] for v in pairs.values())
    out["pair_sec"] = sum(v[1] for v in pairs.values())
    out["kept_pair_n"] = sum(v[2] for v in pairs.values())
    out["kept_pair_sec"] = sum(v[3] for v in pairs.values())
    return out


def hours(sec: float) -> float:
    return round(sec / 3600, 3)


def main(
    synth: str = SYNTH,
    thresholds_csv: str = f"{SYNTH}/reports/qe/thresholds_keep75.csv",
    out: str = f"{SYNTH}/reports/speechblocks-synth-v1",
    jobs: int = 16,
    skip_noisy: bool = False,
) -> None:  # fmt: skip
    thr = thresholds(thresholds_csv)
    os.makedirs(out, exist_ok=True)
    corpora: dict[str, dict] = {}
    with Pool(jobs) as pool:
        for d in sorted(glob.glob(f"{synth}/shar-out/*")):
            name = os.path.basename(d)
            corpora[name] = {"group": source(name)[2], **scan_corpus(f"{d}/train", thr, pool)}
            print(f"{name}: {corpora[name]['cuts']:,.0f} cuts", flush=True)
        if not skip_noisy:
            for d in sorted(glob.glob(f"{synth}/shar-noisy/*_noisy")):
                name = os.path.basename(d)
                stats_path = f"{d}/noise_stats.json"
                if not os.path.exists(stats_path):
                    print(f"{name}: no noise_stats.json yet, skipped", flush=True)
                    continue
                stats = json.load(open(stats_path))
                res = scan_corpus(f"{d}/train", thr, pool)
                res["group"] = "F-asr" if stats["stream"] == "asr_tail" else "F"
                res["stats_hours"] = stats["hours"]
                res["stats_noisy_cuts"] = stats["noisy_cuts"]
                corpora[name] = res
                print(f"{name}: {res['cuts']:,.0f} cuts", flush=True)
    summary: dict[str, dict] = {}
    for name, c in corpora.items():
        s = summary.setdefault(
            c["group"],
            {"corpora": 0, "cuts": 0, "audio_sec": 0.0, "translated_cuts": 0, "translated_sec": 0.0,
             "pairs": 0, "pair_sec": 0.0, "kept_pairs": 0, "kept_pair_sec": 0.0, "kept_audio_sec": 0.0},
        )  # fmt: skip
        s["corpora"] += 1
        s["cuts"] += c["cuts"]
        s["audio_sec"] += c["sec"]
        s["translated_cuts"] += c["tr_cuts"]
        s["translated_sec"] += c["tr_sec"]
        s["pairs"] += c["pair_n"]
        s["pair_sec"] += c["pair_sec"]
        s["kept_pairs"] += c["kept_pair_n"]
        s["kept_pair_sec"] += c["kept_pair_sec"]
        s["kept_audio_sec"] += c["kept_sec"]
    # The noisy-copy groups must agree with the numbers the builder recorded.
    mismatches = [
        f"{n}: manifests {hours(c['sec'])} h vs stats {c['stats_hours']} h"
        for n, c in corpora.items()
        if "stats_hours" in c and abs(hours(c["sec"]) - c["stats_hours"]) > 0.01 * max(c["stats_hours"], 1) + 0.01
    ]
    unknown = {n: c["unknown"] for n, c in corpora.items() if c["unknown"]}
    result = {
        "name": "SpeechBlocks-Synth-v1",
        "thresholds": thresholds_csv,
        "streams": {
            g: {
                "note": STREAM_NOTE.get(g, "noisy copies" if g.startswith("F") else ""),
                "corpora": s["corpora"], "cuts_in_manifests": int(s["cuts"]),
                "manifest_audio_h": hours(s["audio_sec"]),
                "translated_cuts": int(s["translated_cuts"]), "translated_audio_h": hours(s["translated_sec"]),
                "pairs": int(s["pairs"]), "pair_h": hours(s["pair_sec"]),
                "kept_pairs": int(s["kept_pairs"]), "kept_pair_h": hours(s["kept_pair_sec"]),
                "kept_audio_h": hours(s["kept_audio_sec"]),
            }
            for g, s in sorted(summary.items())
        },
        "stats_mismatches": mismatches,
        "pairs_without_threshold": unknown,
    }  # fmt: skip
    with open(f"{out}/hours.json", "w") as f:
        json.dump(result, f, indent=1)
    with open(f"{out}/hours_by_corpus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "group", "cuts", "manifest_audio_h", "translated_cuts", "translated_audio_h", "pairs",
                    "pair_h", "kept_pairs", "kept_pair_h"])  # fmt: skip
        for n, c in corpora.items():
            w.writerow([n, c["group"], int(c["cuts"]), hours(c["sec"]), int(c["tr_cuts"]), hours(c["tr_sec"]),
                        int(c["pair_n"]), hours(c["pair_sec"]), int(c["kept_pair_n"]), hours(c["kept_pair_sec"])])  # fmt: skip
    # One row per training leaf (corpus x target): seconds, not hours, so a mean cut length stays exact for tiny leaves.
    with open(f"{out}/hours_by_leaf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "group", "stream", "src", "tgt", "pairs", "pair_s", "kept_pairs", "kept_pair_s"])
        for n, c in corpora.items():
            for (letter, src, tgt), v in sorted(c["pairs"].items()):
                w.writerow([n, c["group"], letter, src, tgt, v[0], round(v[1], 1), v[2], round(v[3], 1)])
    direction: dict[tuple, list] = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for c in corpora.values():
        for (letter, src, tgt), v in c["pairs"].items():
            for i in range(4):
                direction[(c["group"], letter, src, tgt)][i] += v[i]
    with open(f"{out}/hours_by_direction.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "stream", "src", "tgt", "pairs", "pair_h", "kept_pairs", "kept_pair_h"])
        for (group, letter, src, tgt), v in sorted(direction.items()):
            w.writerow([group, letter, src, tgt, v[0], hours(v[1]), v[2], hours(v[3])])
    print(json.dumps(result["streams"], indent=1))
    print("stats mismatches:", mismatches or "none")
    print("pairs without a threshold:", unknown or "none")


if __name__ == "__main__":
    tyro.cli(main)
