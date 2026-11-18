"""Stream F: noisy / reverberant copies of Shar cuts, with full provenance.

For every selected cut: SNR ~ U(0, 20) dB, 50% chance of convolving the speech with a room impulse
response, noise drawn from the TRAIN or the EVAL noise split of `catalog.json` (built by
noise_catalog.py; the two splits share no file and no recording environment). Everything random is
seeded from md5(cut id [+ row] [+ copy] + salt), so a rerun reproduces the same copy. The new cut keeps the
original's `custom` (translations, scores) and gets

  custom.synth.noise = {corpus, file_id, licence, category, snr_db, rir_corpus, rir_id, rir_licence, seed,
                        split, source_cut, source_row, copy}

Input: a Shar directory (cuts.NNNNNN.jsonl[.gz] + recording.NNNNNN.tar, symlinks fine). Output: a new indexed Shar
(uncompressed manifests, `.idx` files, 16 kHz mono FLAC) under `<out>/train`, plus `<out>/noise_stats.json`.

Which cuts get a copy:
  --fraction F        cuts whose hash falls under F (default 0.3), and with `--require-field` only those that have
                      that dotted field (for example `synth.ok`, so cuts without a translation are skipped)
  --keep-mask M.npy   instead, the rows set in a bool array aligned with the Shar order (positional, like cut_filter's
                      score arrays). Ids may repeat (FLEURS), so the row goes into the seed and into the new id
  --copies N          N independently seeded copies of every selected cut
  --dry-run           count the selected cuts and hours, write nothing
Run inside synth.sif.
"""

import glob
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import tyro


SNR_RANGE = (0.0, 20.0)
RIR_PROB = 0.5
SALT = "synth-noise-v1"
SR = 16000
MIN_SAMPLES = SR // 10  # below 0.1 s a noisy copy teaches nothing
READER_FIELDS = ("shard_origin", "shar_epoch")  # lhotse's Shar reader adds these to `custom` on read


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def get_path(obj, dotted: str):
    for k in dotted.split("."):
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def to_mono16k(x, sr: int):
    """(samples,) or (channels, samples) at `sr` -> mono float32 at 16 kHz."""
    import numpy as np
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=0)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


@lru_cache(maxsize=128)
def load_mono16k(path: str):
    """A noise or RIR file as mono float32 at 16 kHz. Cached per process: callers must not modify the array."""
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return to_mono16k(x.T, sr)


def rms(x) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def mix(speech, noise_path: str, rir_path: str | None, snr_db: float, rng: random.Random):
    """Reverberate (optionally), add noise at `snr_db` against the (reverberated) speech, avoid clipping."""
    import numpy as np
    from scipy.signal import fftconvolve

    x = speech
    if rir_path:
        rir = load_mono16k(rir_path)
        rir = rir[np.argmax(np.abs(rir)) :]  # drop the pre-direct-path lead-in
        rir = rir / (np.linalg.norm(rir) + 1e-9)
        y = fftconvolve(x, rir)[: len(x)].astype("float32")
        x = y * (rms(speech) / rms(y))  # keep the level of the dry speech
    n = load_mono16k(noise_path)
    if len(n) < len(x):
        n = np.tile(n, int(np.ceil(len(x) / len(n))))
    off = rng.randrange(0, len(n) - len(x) + 1)
    n = n[off : off + len(x)]
    n = n * (rms(x) / (rms(n) * 10 ** (snr_db / 20)))
    out = x + n
    peak = float(np.abs(out).max())
    return (out * (0.99 / peak) if peak > 0.99 else out).astype("float32")


def by_category(catalog: dict, split: str) -> dict[str, list[dict]]:
    """The split's noise files grouped by category (sorted, so a seed picks the same file on every run)."""
    cache = catalog.setdefault("_by_category", {})
    if split not in cache:
        cats: dict[str, list[dict]] = {}
        for n in catalog["noise"][split]:
            cats.setdefault(n["category"], []).append(n)
        cache[split] = {c: cats[c] for c in sorted(cats)}
    return cache[split]


def pick(catalog: dict, split: str, key: str, snr_db: float | None = None):
    """Deterministic (rng, noise file, SNR, RIR or None) for a seed key; `snr_db` fixes the SNR.

    The SNR is always drawn, so a fixed-SNR variant of a key keeps the same noise file, RIR and offset.
    """
    rng = random.Random(h(key + SALT))
    cats = by_category(catalog, split)
    cat = rng.choice(list(cats))  # category-balanced, then a file inside it
    noise = rng.choice(cats[cat])
    draw = round(rng.uniform(*SNR_RANGE), 2)
    snr = draw if snr_db is None else float(snr_db)
    rir = rng.choice(catalog["rir"][split]) if rng.random() < RIR_PROB else None
    return rng, noise, snr, rir


def selected(cut_id: str, fraction: float) -> bool:
    return h(cut_id + SALT + "sel") / 16**12 < fraction


def keep_cut(cfg: dict, cut_id: str, custom: dict | None, row: int, mask) -> bool:
    if mask is not None:
        return bool(mask[row])
    if not selected(cut_id, cfg["fraction"]):
        return False
    return not cfg["require_field"] or bool(get_path(custom or {}, cfg["require_field"]))


def seed_key(cut_id: str, row: int, copy: int, by_row: bool) -> str:
    return cut_id + (f"#r{row}" if by_row else "") + (f"#c{copy}" if copy else "")


def noisy_id(cut_id: str, row: int, copy: int, tag: str, by_row: bool) -> str:
    return f"{cut_id}{f'-r{row}' if by_row else ''}-{tag}{copy + 1}"


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def list_shards(src_dir: str) -> list[tuple[str, str]]:
    cuts = sorted(
        glob.glob(os.path.join(src_dir, "cuts.*.jsonl")) + glob.glob(os.path.join(src_dir, "cuts.*.jsonl.gz"))
    )
    return [(c, os.path.join(src_dir, f"recording.{os.path.basename(c).split('.')[1]}.tar")) for c in cuts]


def count_rows(path: str) -> int:
    with open_text(path) as f:
        return sum(1 for _ in f)


def dry(args: tuple) -> tuple[int, int, float]:
    """Rows, selected cuts and selected hours of one shard, from the manifest alone."""
    import numpy as np

    cuts_path, offset, cfg = args
    mask = np.load(cfg["keep_mask"], mmap_mode="r") if cfg["keep_mask"] else None
    rows = sel = 0
    secs = 0.0
    with open_text(cuts_path) as f:
        for i, line in enumerate(f):
            rows += 1
            d = json.loads(line)
            if keep_cut(cfg, d["id"], d.get("custom"), offset + i, mask):
                sel += 1
                secs += d.get("duration", 0.0)
    return rows, sel, secs / 3600


def work(args: tuple) -> dict:
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.shar.writers import SharWriter

    wid, shards, cfg = args
    catalog = json.load(open(cfg["catalog"]))
    mask = np.load(cfg["keep_mask"]) if cfg["keep_mask"] else None
    by_row = mask is not None
    wdir = Path(cfg["out"]) / f"_w{wid:02d}"
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    st: Counter = Counter()
    cats: Counter = Counter()
    snr_bins: Counter = Counter()
    secs = 0.0
    done = False
    with SharWriter(str(wdir), fields={"recording": "flac"}, shard_size=5000, compress_jsonl=False) as w:
        for cuts_path, tar_path, offset in shards:
            if done:
                break
            for i, cut in enumerate(CutSet.from_shar(fields={"cuts": [cuts_path], "recording": [tar_path]})):
                row = offset + i
                st["rows"] += 1
                if not keep_cut(cfg, cut.id, cut.custom, row, mask):
                    continue
                if cfg["limit"] and st["cuts"] >= cfg["limit"]:
                    done = True
                    break
                try:
                    speech = to_mono16k(cut.load_audio(), cut.sampling_rate)
                except Exception:  # unreadable audio is counted and reported, never hidden
                    st["load_error"] += 1
                    continue
                if len(speech) < MIN_SAMPLES:
                    st["too_short"] += 1
                    continue
                st["cuts"] += 1
                for copy in range(cfg["copies"]):
                    key = seed_key(cut.id, row, copy, by_row)
                    rng, noise, snr, rir = pick(catalog, cfg["split"], key, cfg["snr_db"])
                    noisy = mix(speech, noise["path"], rir["path"] if rir else None, snr, rng)
                    buf = io.BytesIO()
                    sf.write(buf, noisy, SR, format="FLAC")
                    nid = noisy_id(cut.id, row, copy, cfg["tag"], by_row)
                    rec = Recording.from_bytes(buf.getvalue(), recording_id=nid)
                    custom = json.loads(json.dumps(cut.custom or {}))
                    for k in READER_FIELDS:  # attached by the Shar reader, not part of the manifest
                        custom.pop(k, None)
                    synth = custom.setdefault("synth", {"version": 1})
                    synth.setdefault("stream", cfg["stream"])
                    synth["noise"] = {
                        "corpus": noise["corpus"], "file_id": noise["file_id"], "licence": noise["licence"],
                        "category": noise["category"], "snr_db": snr,
                        "rir_corpus": rir["corpus"] if rir else None, "rir_id": rir["file_id"] if rir else None,
                        "rir_licence": rir["licence"] if rir else None, "seed": h(key + SALT),
                        "split": cfg["split"], "source_cut": cut.id, "source_row": row, "copy": copy,
                    }  # fmt: skip
                    sups = [
                        replace(s, id=f"{s.id}-{cfg['tag']}{copy + 1}", recording_id=nid) for s in cut.supervisions
                    ]
                    w.write(
                        MonoCut(
                            id=nid,
                            start=0,
                            duration=rec.duration,
                            channel=0,
                            recording=rec,
                            supervisions=sups,
                            custom=custom,
                        )  # fmt: skip
                    )
                    st["written"] += 1
                    st["rir"] += rir is not None
                    cats[noise["category"]] += 1
                    snr_bins[int(snr // 5) * 5] += 1
                    secs += rec.duration
    return {"wid": wid, "stats": dict(st), "cats": dict(cats), "snr_bins": dict(snr_bins), "hours": secs / 3600}


def finalize(out: str, subdir: str = "train") -> int:
    """Renumber `_wNN/*` shards into `<out>/<subdir>/{cuts,recording}.NNNNNN.*` (files moved, not copied)."""
    dest = Path(out) / subdir
    dest.mkdir(parents=True, exist_ok=True)
    k = 0
    for wdir in sorted(Path(out).glob("_w*")):
        for cuts in sorted(c for c in wdir.glob("cuts.*.jsonl*") if not c.name.endswith(".idx")):
            num = cuts.name.split(".")[1]
            suffix = cuts.name[len(f"cuts.{num}") :]
            shutil.move(cuts, dest / f"cuts.{k:06d}{suffix}")
            shutil.move(wdir / f"recording.{num}.tar", dest / f"recording.{k:06d}.tar")
            for idx in (f"cuts.{num}{suffix}.idx", f"recording.{num}.tar.idx"):
                if (wdir / idx).exists():
                    shutil.move(wdir / idx, dest / idx.replace(num, f"{k:06d}", 1))
            k += 1
        shutil.rmtree(wdir)
    return k


def main(
    src_dir: str,
    out: str,
    catalog: str,
    split: str = "train",
    fraction: float = 0.3,
    require_field: str = "",
    keep_mask: str = "",
    copies: int = 1,
    snr_db: float | None = None,
    tag: str = "noisy",
    stream: str = "noisy",
    limit: int = 0,
    workers: int = 8,
    dry_run: bool = False,
    overwrite: bool = False,
    subdir: str = "train",
) -> dict:
    t0 = time.time()
    shards = list_shards(src_dir)
    if not shards:
        raise SystemExit(f"no cuts.*.jsonl[.gz] under {src_dir}")
    if not dry_run and (Path(out) / subdir).exists() and not overwrite:
        raise SystemExit(f"{out}/{subdir} exists; pass --overwrite to replace it")
    if not dry_run and overwrite:
        shutil.rmtree(Path(out) / subdir, ignore_errors=True)
    cfg = {
        "out": out, "catalog": catalog, "split": split, "fraction": fraction, "require_field": require_field,
        "keep_mask": keep_mask, "copies": copies, "snr_db": snr_db, "tag": tag, "stream": stream,
        "limit": math.ceil(limit / workers) if limit else 0,
    }  # fmt: skip
    with Pool(workers) as pool:
        rows = pool.map(count_rows, [c for c, _ in shards])
        offsets = [sum(rows[:i]) for i in range(len(rows))]
        if dry_run:
            res = pool.map(dry, [(c, o, cfg) for (c, _), o in zip(shards, offsets, strict=True)])
            report = {
                "rows": sum(r[0] for r in res), "selected_cuts": sum(r[1] for r in res),
                "selected_hours": round(sum(r[2] for r in res), 2), "copies": copies,
                "noisy_hours": round(copies * sum(r[2] for r in res), 2),
            }  # fmt: skip
            print(json.dumps(report))
            return report
        items = [(c, t, o) for (c, t), o in zip(shards, offsets, strict=True)]
        parts = [(w, items[w::workers], cfg) for w in range(workers)]
        res = pool.map(work, parts)
    n_shards = finalize(out, subdir)
    tot: Counter = Counter()
    cats: Counter = Counter()
    bins: Counter = Counter()
    for r in res:
        tot.update(r["stats"])
        cats.update(r["cats"])
        bins.update({int(k): v for k, v in r["snr_bins"].items()})
    report = {
        "src_dir": src_dir, "out": out, "subdir": subdir, "shards": n_shards, "split": split, "copies": copies,
        "stream": stream,
        "fraction": None if keep_mask else fraction, "keep_mask": keep_mask or None,
        "require_field": require_field or None, "snr_db": snr_db,
        "rows": tot["rows"], "source_cuts": tot["cuts"], "noisy_cuts": tot["written"],
        "hours": round(sum(r["hours"] for r in res), 3), "with_rir": tot["rir"],
        "load_errors": tot["load_error"], "too_short": tot["too_short"],
        "categories": dict(sorted(cats.items())), "snr_bins": dict(sorted(bins.items())),
        "catalog_md5": hashlib.md5(open(catalog, "rb").read()).hexdigest(),
        "code_revision": os.environ.get("SYNTH_CODE_REVISION", ""), "seconds": round(time.time() - t0),
    }  # fmt: skip
    Path(out, "noise_stats.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    tyro.cli(main)
