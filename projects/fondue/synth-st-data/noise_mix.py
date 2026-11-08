"""Stream F: noisy / reverberant copies of Shar cuts, with full provenance.

For every selected cut: SNR ~ U(0, 20) dB, 50% chance of convolving the speech with a room impulse
response, noise drawn from the TRAIN or the EVAL noise split of `catalog.json` (built by
noise_catalog.py; the two splits share no file and no recording environment). Everything random is
seeded from md5(cut id + salt), so a rerun reproduces the same copy. The new cut keeps the original's
`custom` (translations, scores) and gets

  custom.synth.noise = {corpus, file_id, licence, category, snr_db, rir_corpus, rir_id, seed, split}

Input: a Shar directory (cuts.NNNNNN.jsonl[.gz] + recording.NNNNNN.tar, symlinks fine). Output: a new Shar
with its own FLAC audio under `<out>/train`. `--select` keeps cuts whose hash falls under `fraction`
(and, with `--require-field`, only cuts that have that dotted field). Run inside synth.sif.
"""

import glob
import hashlib
import io
import json
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import tyro


SNR_RANGE = (0.0, 20.0)
RIR_PROB = 0.5
SALT = "synth-noise-v1"


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def get_path(obj, dotted: str):
    for k in dotted.split("."):
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def load_mono16k(path: str):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        from math import gcd

        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    return x


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


def pick(catalog: dict, split: str, cut_id: str):
    """Deterministic noise file (and RIR) for a cut; None for the RIR with probability 1 - RIR_PROB."""
    rng = random.Random(h(cut_id + SALT))
    noises = catalog["noise"][split]
    cats = sorted({n["category"] for n in noises})
    cat = rng.choice(cats)  # category-balanced, then a file inside it
    noise = rng.choice([n for n in noises if n["category"] == cat])
    snr = round(rng.uniform(*SNR_RANGE), 2)
    rir = rng.choice(catalog["rir"][split]) if rng.random() < RIR_PROB else None
    return rng, noise, snr, rir


def work(args: tuple) -> tuple[int, int]:
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.shar.writers import SharWriter

    wid, shards, out, catalog_path, split, fraction, require_field, tag = args
    catalog = json.load(open(catalog_path))
    wdir = Path(out) / f"_w{wid:02d}"
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    n = 0
    with SharWriter(str(wdir), fields={"recording": "flac"}, shard_size=5000) as w:
        for cuts_path, tar_path in shards:
            for cut in CutSet.from_shar(fields={"cuts": [cuts_path], "recording": [tar_path]}):
                if h(cut.id + SALT + "sel") / 16**12 >= fraction:
                    continue
                if require_field and not get_path(cut.custom or {}, require_field):
                    continue
                rng, noise, snr, rir = pick(catalog, split, cut.id)
                speech = cut.load_audio()[0]
                noisy = mix(speech, noise["path"], rir["path"] if rir else None, snr, rng)
                buf = io.BytesIO()
                sf.write(buf, noisy, 16000, format="FLAC")
                nid = f"{cut.id}-{tag}"
                rec = Recording.from_bytes(buf.getvalue(), recording_id=nid)
                custom = json.loads(json.dumps(cut.custom or {}))
                synth = custom.setdefault("synth", {"version": 1})
                synth["noise"] = {
                    "corpus": noise["corpus"], "file_id": noise["file_id"], "licence": noise["licence"],
                    "category": noise["category"], "snr_db": snr,
                    "rir_corpus": rir["corpus"] if rir else None, "rir_id": rir["file_id"] if rir else None,
                    "rir_licence": rir["licence"] if rir else None, "seed": h(cut.id + SALT), "split": split,
                    "source_cut": cut.id,
                }  # fmt: skip
                sups = list(cut.supervisions)
                for s in sups:
                    s.recording_id, s.id = nid, f"{s.id}-{tag}"
                w.write(MonoCut(id=nid, start=0, duration=rec.duration, channel=0, recording=rec,
                                supervisions=sups, custom=custom))  # fmt: skip
                n += 1
    return wid, n


def finalize(out: str) -> int:
    dest = Path(out) / "train"
    dest.mkdir(parents=True, exist_ok=True)
    k = 0
    for wdir in sorted(Path(out).glob("_w*")):
        for cuts in sorted(wdir.glob("cuts.*.jsonl*")):
            num = cuts.name.split(".")[1]
            suffix = cuts.name[len(f"cuts.{num}") :]
            shutil.move(cuts, dest / f"cuts.{k:06d}{suffix}")
            shutil.move(wdir / f"recording.{num}.tar", dest / f"recording.{k:06d}.tar")
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
    tag: str = "noisy",
    workers: int = 8,
) -> None:
    cuts = sorted(
        glob.glob(os.path.join(src_dir, "cuts.*.jsonl")) + glob.glob(os.path.join(src_dir, "cuts.*.jsonl.gz"))
    )
    shards = [(c, os.path.join(src_dir, f"recording.{os.path.basename(c).split('.')[1]}.tar")) for c in cuts]
    parts = [(w, shards[w::workers], out, catalog, split, fraction, require_field, tag) for w in range(workers)]
    with Pool(workers) as pool:
        res = pool.map(work, parts)
    print(f"{sum(r[1] for r in res)} noisy cuts in {finalize(out)} shards -> {out}/train")


if __name__ == "__main__":
    tyro.cli(main)
