"""Stream F: check a noisy Shar against its source and write listening pairs.

Manifest checks, on every cut of `noisy_dir`: unique ids, the full `custom.synth.noise` record, SNR inside [0, 20] dB,
noise and RIR files that belong to the recorded split of the catalog with the recorded licence (and a licence in the
catalog's allow list), the recorded copy and row inside the run's limits, and the row count equal to `noise_stats.json`.

Audio checks, on `n` cuts spread over the noise categories: the source cut is found by its recorded row, the mix is
re-created from the metadata alone (same seed, same noise file, SNR, RIR and offset) and compared with the audio stored in
the Shar (max sample difference; FLAC quantisation allows ~2 LSB); the labels (supervision text, every `custom` field
except `synth.noise`) must equal the source's; the duration and the 16 kHz rate must match; without an RIR the measured SNR
of (stored - gain * clean) must be within 0.5 dB of the recorded one.

Output: `<out>/report.json`, and a self-contained `<out>/listen.html` (clean and noisy audio embedded, noise metadata
next to each pair) so the pairs can be heard anywhere. Exit status 1 if any check fails. Run inside synth.sif.
"""

import base64
import bisect
import io
import json
import os
import random
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import tyro
from lhotse import CutSet
from lhotse.shar.readers.indexed import LazyIndexedSharIterator

from noise_mix import (
    READER_FIELDS,
    SNR_RANGE,
    count_rows,
    get_path,
    list_shards,
    mix,
    open_text,
    pick,
    rms,
    seed_key,
    to_mono16k,
)


NOISE_KEYS = {
    "corpus", "file_id", "licence", "category", "snr_db", "rir_corpus", "rir_id", "rir_licence", "seed",
    "split", "source_cut", "source_row", "copy",
}  # fmt: skip


def read_row(shards: list[tuple[str, str]], offsets: list[int], row: int):
    """The cut at global row `row` of a Shar directory, audio included."""
    from lhotse import CutSet

    i = bisect.bisect_right(offsets, row) - 1
    cuts_path, tar_path = shards[i]
    for k, cut in enumerate(CutSet.from_shar(fields={"cuts": [cuts_path], "recording": [tar_path]})):
        if k == row - offsets[i]:
            return cut
    raise IndexError(row)


def labels(custom: dict | None, added_stream: str) -> dict:
    """Everything a training leaf reads from `custom`, without what the reader or the mixer add."""
    c = json.loads(json.dumps(custom or {}))
    for k in READER_FIELDS:  # the reader adds them to every cut it loads, source and copy alike
        c.pop(k, None)
    if "synth" in c:
        c["synth"].pop("noise", None)
        if c["synth"].get("stream") == added_stream and set(c["synth"]) <= {"version", "stream"}:
            c.pop("synth")  # the mixer created this record for a source that had none
    return c


def flac_b64(x) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="FLAC")
    return base64.b64encode(buf.getvalue()).decode()


def main(
    noisy_dir: str, src_dir: str, catalog: str, out: str, n: int = 24, seed: int = 0, max_seconds: float = 12.0
) -> None:
    import numpy as np

    stats = json.load(open(Path(noisy_dir) / "noise_stats.json"))
    cat = json.load(open(catalog))
    import hashlib

    same_catalog = hashlib.md5(open(catalog, "rb").read()).hexdigest() == stats["catalog_md5"]
    by_row = bool(stats["keep_mask"])
    split = stats["split"]
    files = {s: {(f["corpus"], f["file_id"]): f for f in cat["noise"][s]} for s in cat["noise"]}
    rirs = {s: {(f["corpus"], f["file_id"]): f for f in cat["rir"][s]} for s in cat["rir"]}
    shards = list_shards(str(Path(noisy_dir) / stats.get("subdir", "train")))
    problems: list[str] = []

    def bad(msg: str) -> None:
        if len(problems) < 50:
            problems.append(msg)

    # ---- manifest checks over every cut
    ids, rows, per_cat, per_licence = set(), 0, Counter(), Counter()
    sample_pool: dict[str, list[tuple[int, int, str, float]]] = defaultdict(
        list
    )  # category -> (shard, local, id, dur)
    offsets, total = [], 0
    for si, (cuts_path, _) in enumerate(shards):
        offsets.append(total)
        with open_text(cuts_path) as f:
            for li, line in enumerate(f):
                d = json.loads(line)
                total += 1
                if d["id"] in ids:
                    bad(f"duplicate id {d['id']}")
                ids.add(d["id"])
                noise = get_path(d.get("custom") or {}, "synth.noise")
                if any(k in (d.get("custom") or {}) for k in READER_FIELDS):
                    bad(f"{d['id']}: reader bookkeeping ({', '.join(READER_FIELDS)}) written into the manifest")
                if not isinstance(noise, dict) or set(noise) != NOISE_KEYS:
                    bad(f"{d['id']}: noise record {sorted(noise) if isinstance(noise, dict) else noise}")
                    continue
                if not SNR_RANGE[0] <= noise["snr_db"] <= SNR_RANGE[1] and stats["snr_db"] is None:
                    bad(f"{d['id']}: snr {noise['snr_db']} outside {SNR_RANGE}")
                nf = files.get(noise["split"], {}).get((noise["corpus"], noise["file_id"]))
                if nf is None or nf["licence"] != noise["licence"] or nf["category"] != noise["category"]:
                    bad(
                        f"{d['id']}: noise {noise['corpus']}/{noise['file_id']} not in the {noise['split']} catalog split"
                    )
                if noise["rir_id"] is not None:
                    rf = rirs.get(noise["split"], {}).get((noise["rir_corpus"], noise["rir_id"]))
                    if rf is None or rf["licence"] != noise["rir_licence"]:
                        bad(f"{d['id']}: rir {noise['rir_corpus']}/{noise['rir_id']} not in the catalog")
                if noise["licence"] not in cat["allow"] or noise["split"] != split:
                    bad(f"{d['id']}: licence {noise['licence']} or split {noise['split']} not allowed here")
                if not 0 <= noise["copy"] < stats["copies"]:
                    bad(f"{d['id']}: copy {noise['copy']} of {stats['copies']}")
                per_cat[noise["category"]] += 1
                per_licence[noise["licence"]] += 1
                if d["duration"] <= max_seconds:
                    sample_pool[noise["category"]].append((si, li, d["id"], d["duration"]))
    rows = total
    if rows != stats["noisy_cuts"]:
        bad(f"{rows} cuts on disk, noise_stats.json says {stats['noisy_cuts']}")

    # ---- the training loader's view: lhotse's indexed reader must accept the directory and agree with the manifests
    shar_dir = str(Path(noisy_dir) / stats.get("subdir", "train"))
    stray = sorted(p for p in os.listdir(shar_dir) if not p.startswith(("cuts.", "recording.")))
    if stray:  # lhotse takes every file of a Shar directory for a field
        bad(f"files in the Shar directory that lhotse would read as fields: {stray}")
    indexed = None
    if not LazyIndexedSharIterator.supports_configuration(in_dir=shar_dir):
        bad("lhotse's indexed reader (the one the training loader uses) refuses the directory")
    else:
        indexed = CutSet.from_shar(in_dir=shar_dir, shuffle_shards=False).data
        if len(indexed) != rows:
            bad(f"the indexed reader sees {len(indexed)} cuts, the manifests have {rows}")

    # ---- audio checks on a stratified sample
    rng = random.Random(seed)
    cats = sorted(sample_pool)
    picked = []
    for c in cats:
        picked += rng.sample(sample_pool[c], min(len(sample_pool[c]), max(1, n // max(1, len(cats)))))
    picked = picked[:n] if len(picked) > n else picked
    src_shards = list_shards(src_dir)
    src_offsets = [sum(count_rows(c) for c, _ in src_shards[:i]) for i in range(len(src_shards))] if picked else []
    pairs, diffs, snr_err, relabel = [], [], [], 0
    for si, li, cid, _ in picked:
        noisy = read_row([shards[si]], [0], li)
        if indexed is not None and indexed[offsets[si] + li].id != cid:
            bad(f"{cid}: random access through the .idx returns a different cut")
        noise = noisy.custom["synth"]["noise"]
        src = read_row(src_shards, src_offsets, noise["source_row"])
        clean = to_mono16k(src.load_audio(), src.sampling_rate)
        stored = to_mono16k(noisy.load_audio(), noisy.sampling_rate)
        if src.id != noise["source_cut"]:
            bad(f"{cid}: row {noise['source_row']} is {src.id}, record says {noise['source_cut']}")
        if noisy.sampling_rate != 16000 or abs(noisy.duration - src.duration) > 1e-3 or len(stored) != len(clean):
            bad(f"{cid}: rate/duration/length differ from the source")
        if labels(noisy.custom, stats["stream"]) != labels(src.custom, stats["stream"]) or [
            s.text for s in noisy.supervisions
        ] != [s.text for s in src.supervisions]:
            relabel += 1
            bad(f"{cid}: labels differ from the source")
        rec = None
        if same_catalog:
            key = seed_key(src.id, noise["source_row"], noise["copy"], by_row)
            r, nz, snr, rir = pick(cat, split, key, stats["snr_db"])
            if (nz["file_id"], snr, rir["file_id"] if rir else None) != (
                noise["file_id"],
                noise["snr_db"],
                noise["rir_id"],
            ):
                bad(f"{cid}: the seed does not reproduce the recorded noise/SNR/RIR")
            rec = mix(clean, nz["path"], rir["path"] if rir else None, snr, r)
            diffs.append(float(np.abs(rec - stored).max()))
        if noise["rir_id"] is None:
            g = float(stored @ clean / (clean @ clean + 1e-12))
            snr_err.append(abs(20 * np.log10(rms(g * clean) / rms(stored - g * clean)) - noise["snr_db"]))
        pairs.append((noisy, src, clean, stored))
    if diffs and max(diffs) > 2.5 / 32768:
        bad(f"re-created mix differs from the stored audio by up to {max(diffs):.6f}")
    if snr_err and max(snr_err) > 0.5:
        bad(f"measured SNR is off by up to {max(snr_err):.2f} dB")

    # ---- listening page
    rowsh = []
    for noisy, src, clean, stored in pairs:
        nz = noisy.custom["synth"]["noise"]
        tr = next(
            (f"{k[12:]}: {v}" for k, v in sorted((noisy.custom or {}).items()) if k.startswith("translation_")), ""
        )
        text = (noisy.supervisions[0].text if noisy.supervisions else "") or (noisy.custom or {}).get("pnc_text", "")
        text = text or get_path(noisy.custom or {}, "metadata.sentence") or ""
        rir = f"{nz['rir_corpus']}/{nz['rir_id']}" if nz["rir_id"] else "none"
        rowsh.append(
            f"<tr><td><b>{escape(noisy.id)}</b><br>{escape(str(text))[:300]}<br><i>{escape(tr)[:300]}</i></td>"
            f"<td>{escape(nz['category'])}<br>{escape(nz['corpus'])}/{escape(str(nz['file_id']))}<br>{escape(nz['licence'])}"
            f"<br>SNR {nz['snr_db']} dB<br>RIR {escape(rir)}</td>"
            f"<td>clean<br><audio controls preload=none src='data:audio/flac;base64,{flac_b64(clean)}'></audio></td>"
            f"<td>noisy<br><audio controls preload=none src='data:audio/flac;base64,{flac_b64(stored)}'></audio></td></tr>"
        )
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "listen.html").write_text(
        "<!doctype html><meta charset=utf-8><title>noisy copies</title><style>body{font:14px sans-serif;margin:1em}"
        "td{border-bottom:1px solid #ccc;padding:6px;vertical-align:top}</style>"
        f"<h3>{escape(Path(noisy_dir).name)}: {len(pairs)} pairs, one per row</h3><table>{''.join(rowsh)}</table>"
    )
    report = {
        "noisy_dir": noisy_dir, "cuts": rows, "ids_unique": len(ids) == rows, "categories": dict(sorted(per_cat.items())),
        "licences": dict(per_licence), "audio_checked": len(pairs), "catalog_unchanged": same_catalog,
        "max_recreate_diff": max(diffs) if diffs else None, "max_snr_error_db": max(snr_err) if snr_err else None,
        "label_mismatches": relabel, "problems": problems, "ok": not problems,
    }  # fmt: skip
    Path(out, "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "categories"}))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    tyro.cli(main)
