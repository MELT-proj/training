import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noise_mix import mix, pick, rms  # noqa: E402


def test_mix_hits_requested_snr_and_is_reproducible(tmp_path):
    sr = 16000
    speech = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype("float32")
    noise = np.random.default_rng(0).normal(0, 0.1, sr * 2).astype("float32")
    npath = tmp_path / "n.wav"
    sf.write(npath, noise, sr)
    out = mix(speech, str(npath), None, 10.0, random.Random(1))
    resid = out - speech
    assert abs(20 * np.log10(rms(speech) / rms(resid)) - 10.0) < 0.3
    again = mix(speech, str(npath), None, 10.0, random.Random(1))
    assert np.allclose(out, again) and np.abs(out).max() <= 1.0


def test_mix_with_rir_keeps_length_and_level(tmp_path):
    sr = 16000
    speech = np.random.default_rng(1).normal(0, 0.1, sr * 2).astype("float32")
    rir = np.zeros(2000, dtype="float32")
    rir[10], rir[900] = 1.0, 0.4
    sf.write(tmp_path / "n.wav", np.random.default_rng(2).normal(0, 0.1, sr).astype("float32"), sr)
    sf.write(tmp_path / "r.wav", rir, sr)
    out = mix(speech, str(tmp_path / "n.wav"), str(tmp_path / "r.wav"), 20.0, random.Random(0))
    assert len(out) == len(speech) and 0.5 < rms(out) / rms(speech) < 2.0


def test_pick_is_deterministic_and_respects_split():
    cat = {
        "noise": {"train": [{"category": c, "file_id": f"{c}{i}", "corpus": "x", "licence": "l", "path": "p"}
                            for c in "ab" for i in range(3)],
                  "eval": [{"category": "a", "file_id": "e0", "corpus": "x", "licence": "l", "path": "p"}]},
        "rir": {"train": [{"file_id": "r", "corpus": "x", "licence": "l", "path": "p"}],
                "eval": [{"file_id": "re", "corpus": "x", "licence": "l", "path": "p"}]},
    }  # fmt: skip
    a = pick(cat, "train", "cut1")[1:]
    assert a == pick(cat, "train", "cut1")[1:]
    assert pick(cat, "eval", "cut1")[1]["file_id"] == "e0"
    assert 0 <= a[1] <= 20
    rirs = [pick(cat, "train", f"c{i}")[3] for i in range(400)]
    assert 0.35 < sum(r is not None for r in rirs) / 400 < 0.65
