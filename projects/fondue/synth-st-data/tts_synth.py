"""Stream E, step 2: synthesise English utterances with Kokoro-82M (Apache-2.0) into a Shar source.

Voices: all 28 American/British English voices of the model, assigned round-robin by utterance
(so accents and genders mix), speed uniform in [0.9, 1.1] seeded per utterance. Audio: 24 kHz
-> 16 kHz mono FLAC. Cuts keep the text as `supervisions[0].text` and `custom.pnc_text`, and carry
`custom.synth = {stream: tts_en, tts: {model, licence, voice, speed, text_prompt_domain, ...}}`.
Each worker writes its own Shar directory under `<out>/_w<NN>`; `finalize` renumbers all shards
into one `<out>/train`. Run inside synth.sif with PYTHONPATH containing en_core_web_sm.
"""

import hashlib
import io
import json
import os
import random
import re
import shutil
from multiprocessing import Pool
from pathlib import Path

import tyro


MODEL = "hexgrad/Kokoro-82M"
LICENCE = "Apache-2.0"
MIN_S, MAX_S = 3.0, 25.0


def voices(hub_snapshot: str) -> list[str]:
    names = sorted(p[:-3] for p in os.listdir(os.path.join(hub_snapshot, "voices")) if re.match(r"^[ab][fm]_", p))
    return names


def work(args: tuple) -> tuple[int, int, int]:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    from lhotse import MonoCut, Recording, SupervisionSegment
    from lhotse.shar.writers import SharWriter
    from scipy.signal import resample_poly

    wid, rows, out, voice_list, threads = args
    torch.set_num_threads(threads)
    pipes = {c: KPipeline(lang_code=c, repo_id=MODEL) for c in "ab"}
    wdir = Path(out) / f"_w{wid:02d}"
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    n_ok = n_skip = 0
    with SharWriter(str(wdir), fields={"recording": "flac"}, shard_size=5000) as w:
        for r in rows:
            uid = hashlib.md5(r["text"].encode()).hexdigest()[:16]
            v = voice_list[int(uid, 16) % len(voice_list)]
            speed = round(random.Random(uid).uniform(0.9, 1.1), 3)
            try:
                segs = [np.asarray(a) for _, _, a in pipes[v[0]](r["text"], voice=v, speed=speed)]
                audio = np.concatenate(segs)
            except Exception:
                n_skip += 1
                continue
            audio16 = resample_poly(audio, 2, 3).astype("float32")  # 24k -> 16k
            dur = len(audio16) / 16000
            if not (MIN_S <= dur <= MAX_S) or float(np.abs(audio16).max()) < 0.01:
                n_skip += 1
                continue
            buf = io.BytesIO()
            sf.write(buf, audio16, 16000, format="FLAC")
            cid = f"tts_en_{uid}"
            rec = Recording.from_bytes(buf.getvalue(), recording_id=cid)
            sup = SupervisionSegment(cid, cid, 0.0, rec.duration, text=r["text"], language="en")
            custom = {
                "pnc_text": r["text"],
                "synth": {
                    "version": 1, "stream": "tts_en",
                    "tts": {"model": MODEL, "licence": LICENCE, "voice": v, "speed": speed,
                            "text_prompt_domain": r["domain"], "scenario": r["scenario"], "register": r["register"],
                            "text_model": "utter-project/EuroLLM-9B-Instruct"},
                },
            }  # fmt: skip
            w.write(MonoCut(id=cid, start=0, duration=rec.duration, channel=0, recording=rec,
                            supervisions=[sup], custom=custom))  # fmt: skip
            n_ok += 1
    return wid, n_ok, n_skip


def finalize(out: str) -> int:
    """Renumber `_wNN/*` shards into `<out>/train/{cuts,recording}.NNNNNN.*` (files moved, not copied)."""
    dest = Path(out) / "train"
    dest.mkdir(parents=True, exist_ok=True)
    k = 0
    for wdir in sorted(Path(out).glob("_w*")):
        for cuts in sorted(wdir.glob("cuts.*.jsonl*")):
            num = cuts.name.split(".")[1]
            suffix = cuts.name[len(f"cuts.{num}") :]
            shutil.move(cuts, dest / f"cuts.{k:06d}{suffix}")
            shutil.move(wdir / f"recording.{num}.tar", dest / f"recording.{k:06d}.tar")
            if (wdir / f"recording.{num}.tar.idx").exists():
                shutil.move(wdir / f"recording.{num}.tar.idx", dest / f"recording.{k:06d}.tar.idx")
            k += 1
        shutil.rmtree(wdir)
    return k


def main(inp: str, out: str, hub_snapshot: str, workers: int = 12, threads: int = 2, limit: int = 0) -> None:
    rows = [json.loads(line) for line in open(inp)]
    if limit:
        rows = rows[:limit]
    vl = voices(hub_snapshot)
    parts = [(w, rows[w::workers], out, vl, threads) for w in range(workers)]
    with Pool(workers) as pool:
        res = pool.map(work, parts)
    shards = finalize(out)
    print(f"{sum(r[1] for r in res)} utterances, {sum(r[2] for r in res)} skipped, {shards} shards, {len(vl)} voices")


if __name__ == "__main__":
    tyro.cli(main)
