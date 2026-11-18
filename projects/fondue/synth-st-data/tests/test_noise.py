import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noise_mix import main, mix, pick, rms, to_mono16k  # noqa: E402


NOISE_KEYS = {
    "corpus", "file_id", "licence", "category", "snr_db", "rir_corpus", "rir_id", "rir_licence", "seed",
    "split", "source_cut", "source_row", "copy",
}  # fmt: skip


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


def test_copies_get_their_own_draw_and_a_fixed_snr_keeps_the_rest():
    cat = {
        "noise": {"train": [{"category": c, "file_id": f"{c}{i}", "corpus": "x", "licence": "l", "path": "p"}
                            for c in "abc" for i in range(5)]},
        "rir": {"train": [{"file_id": f"r{i}", "corpus": "x", "licence": "l", "path": "p"} for i in range(5)]},
    }  # fmt: skip
    first = pick(cat, "train", "cut1")
    second = pick(cat, "train", "cut1#c1")
    assert first[2] != second[2]  # the SNR is continuous: two independent draws differ
    fixed = pick(cat, "train", "cut1", snr_db=5.0)
    assert fixed[2] == 5.0 and fixed[1] == first[1] and fixed[3] == first[3]


def test_to_mono16k_downmixes_and_resamples():
    stereo = np.stack([np.ones(8000, dtype="float32"), -np.ones(8000, dtype="float32")])
    assert np.allclose(to_mono16k(stereo, 16000), 0.0) and to_mono16k(stereo, 16000).shape == (8000,)
    assert to_mono16k(np.zeros(8000, dtype="float32"), 8000).shape == (16000,)  # 1 s at 8 kHz -> 1 s at 16 kHz
    assert to_mono16k(np.zeros(48000, dtype="float32"), 48000).shape == (16000,)


def make_catalog(tmp_path):
    sr = 16000
    files = {}
    for k, name in enumerate(("a0", "a1", "b0", "e0")):
        files[name] = str(tmp_path / f"{name}.wav")
        sf.write(files[name], np.random.default_rng(k).normal(0, 0.1, sr * 3).astype("float32"), sr)
    rir = np.zeros(1600, dtype="float32")
    rir[5], rir[800] = 1.0, 0.3
    sf.write(tmp_path / "r0.wav", rir, sr)

    def noise(name, category, split):
        return {"corpus": "x", "file_id": name, "licence": "CC0-1.0", "category": category, "path": files[name],
                "split": split}  # fmt: skip

    cat = {
        "noise": {"train": [noise("a0", "a", "train"), noise("a1", "a", "train"), noise("b0", "b", "train")],
                  "eval": [noise("e0", "a", "eval")]},
        "rir": {s: [{"corpus": "y", "file_id": "r0", "licence": "Apache-2.0", "path": str(tmp_path / "r0.wav")}]
                for s in ("train", "eval")},
    }  # fmt: skip
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(cat))
    return str(path)


def make_shar(tmp_path, ids=("c0", "c1", "c2", "c3"), translated=("c0", "c2", "c3"), sr=16000, dur=2.0):
    from lhotse import MonoCut, Recording, SupervisionSegment
    from lhotse.shar.writers import SharWriter

    src = tmp_path / "src"
    src.mkdir()
    with SharWriter(str(src), fields={"recording": "flac"}, shard_size=2, compress_jsonl=False) as w:
        for i, cid in enumerate(ids):
            x = (0.2 * np.sin(2 * np.pi * (200 + 50 * i) * np.arange(int(sr * dur)) / sr)).astype("float32")
            buf = io.BytesIO()
            sf.write(buf, x, sr, format="FLAC")
            rec = Recording.from_bytes(buf.getvalue(), recording_id=f"{cid}-{i}")
            sup = SupervisionSegment(f"{cid}-{i}", rec.id, 0.0, rec.duration, text=f"text {cid}", language="en")
            custom = {"pnc_text": f"Text {cid}."}
            if cid in translated:
                custom |= {"translation_lv": f"tulkojums {cid}",
                           "synth": {"version": 1, "stream": "A", "ok": {"lv": 1}, "metricx_xxl": {"lv": 1.5}}}  # fmt: skip
            w.write(MonoCut(id=cid, start=0, duration=rec.duration, channel=0, recording=rec, supervisions=[sup],
                            custom=custom))  # fmt: skip
    return src


def read_out(out):
    from lhotse import CutSet

    d = Path(out) / "train"
    cuts = sorted(str(p) for p in d.glob("cuts.*.jsonl"))
    tars = sorted(str(p) for p in d.glob("recording.*.tar"))
    assert cuts and len(cuts) == len(tars)
    for p in cuts:
        assert Path(p + ".idx").exists()  # indexed layout, the one MN5 reads
    for p in tars:
        assert Path(p + ".idx").exists()
    return list(CutSet.from_shar(fields={"cuts": cuts, "recording": tars}))


def test_noisy_copies_keep_labels_record_provenance_and_reproduce(tmp_path):
    src, cat = make_shar(tmp_path), make_catalog(tmp_path)
    args = {
        "src_dir": str(src),
        "catalog": cat,
        "fraction": 1.0,
        "require_field": "synth.ok",
        "copies": 2,
        "workers": 1,
    }
    rep = main(out=str(tmp_path / "o1"), **args)
    cuts = read_out(tmp_path / "o1")
    assert rep["source_cuts"] == 3 and rep["noisy_cuts"] == 6 and len(cuts) == 6  # c1 has no translation
    assert len({c.id for c in cuts}) == 6 and {c.id[-7:] for c in cuts} == {"-noisy1", "-noisy2"}
    for c in cuts:
        assert c.custom["translation_lv"] == "tulkojums " + c.custom["synth"]["noise"]["source_cut"]
        assert c.custom["synth"]["ok"] == {"lv": 1} and c.custom["synth"]["stream"] == "A"
        noise = c.custom["synth"]["noise"]
        assert set(noise) == NOISE_KEYS and 0 <= noise["snr_db"] <= 20 and noise["split"] == "train"
        assert noise["file_id"] in {"a0", "a1", "b0"} and noise["licence"] == "CC0-1.0"  # never the eval file e0
        assert (noise["rir_id"] is None) == (noise["rir_corpus"] is None) == (noise["rir_licence"] is None)
        assert c.supervisions[0].text == "text " + noise["source_cut"] and c.supervisions[0].recording_id == c.id
        assert c.load_audio().shape == (1, 32000) and abs(c.duration - 2.0) < 1e-3
    by_source = {}
    for c in cuts:
        by_source.setdefault(c.custom["synth"]["noise"]["source_cut"], []).append(c.custom["synth"]["noise"])
    assert all(a["snr_db"] != b["snr_db"] for a, b in by_source.values())  # the two copies differ
    assert json.loads((tmp_path / "o1" / "noise_stats.json").read_text())["noisy_cuts"] == 6
    for manifest in (tmp_path / "o1" / "train").glob("cuts.*.jsonl"):  # the reader's bookkeeping stays out of the file
        for line in manifest.read_text().splitlines():
            assert "shard_origin" not in line and "shar_epoch" not in line
    main(out=str(tmp_path / "o2"), **args)  # same seeds, same audio
    again = read_out(tmp_path / "o2")
    assert [c.custom["synth"]["noise"] for c in cuts] == [c.custom["synth"]["noise"] for c in again]
    assert all(np.array_equal(a.load_audio(), b.load_audio()) for a, b in zip(cuts, again, strict=True))
    try:
        main(out=str(tmp_path / "o2"), **args)
    except SystemExit as e:
        assert "exists" in str(e)
    else:
        raise AssertionError("an existing output must not be overwritten silently")


def test_keep_mask_is_positional_and_survives_repeated_cut_ids(tmp_path):
    src = make_shar(tmp_path, ids=("c0", "c0", "c1", "c1"), translated=())  # FLEURS-like: ids repeat
    mask = tmp_path / "mask.npy"
    np.save(mask, np.array([True, False, True, True]))
    main(src_dir=str(src), out=str(tmp_path / "o"), catalog=make_catalog(tmp_path), keep_mask=str(mask), workers=1)
    cuts = read_out(tmp_path / "o")
    assert sorted(c.id for c in cuts) == ["c0-r0-noisy1", "c1-r2-noisy1", "c1-r3-noisy1"]
    assert sorted(c.custom["synth"]["noise"]["source_row"] for c in cuts) == [0, 2, 3]
    assert {c.custom["synth"]["stream"] for c in cuts} == {"noisy"}  # no synth dict on the source: created here
    seeds = {c.custom["synth"]["noise"]["seed"] for c in cuts}
    assert len(seeds) == 3  # the repeated id c1 (rows 2 and 3) still gets two different draws


def test_dry_run_counts_without_writing(tmp_path):
    src = make_shar(tmp_path)
    rep = main(src_dir=str(src), out=str(tmp_path / "o"), catalog="unused", fraction=1.0, require_field="synth.ok",
               copies=2, workers=1, dry_run=True)  # fmt: skip
    assert rep["rows"] == 4 and rep["selected_cuts"] == 3 and rep["noisy_hours"] == round(2 * 3 * 2.0 / 3600, 2)
    assert not (tmp_path / "o").exists()


def test_other_sampling_rates_are_resampled_not_pitch_shifted(tmp_path):
    src = make_shar(tmp_path, sr=8000, dur=2.0)
    main(src_dir=str(src), out=str(tmp_path / "o"), catalog=make_catalog(tmp_path), fraction=1.0, workers=1)
    cuts = read_out(tmp_path / "o")
    assert cuts and all(c.sampling_rate == 16000 and abs(c.duration - 2.0) < 1e-3 for c in cuts)


def test_eval_choice_takes_distinct_ids_once_and_is_stable(tmp_path):
    from noisy_fleurs_eval import choose

    src = make_shar(tmp_path, ids=("c0", "c0", "c1", "c2", "c3"), translated=())
    mask = choose(str(src), 3)
    assert mask.dtype == bool and mask.shape == (5,) and mask.sum() == 3
    assert not (mask[0] and mask[1])  # the repeated id c0 counts once, at its first row
    assert (choose(str(src), 3) == mask).all()
    assert choose(str(src), 10).sum() == 4  # fewer distinct ids than asked for: all of them
