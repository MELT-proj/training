"""Select English segments from whole Shar shards and tag them with spaCy entities.

Unit of work = one Shar shard (`cuts.NNNNNN.jsonl`): the pool keeps every cut of
the shard that passes the ASR-WER band, has a cased source text and a sensible
duration. Selecting whole shards (not scattered cuts) matters because the audio
tars are positional: the overlay manifest has to keep every cut of a shard, and
the cuts we did not translate are dropped at training time by a `cut_filter`
`field` rule. Dense shards keep that waste small.

Per shard it writes `<out>/<corpus>/pool.NNNNNN.jsonl` (idempotent) with: id,
shard, idx (position in the shard), text, duration, ents, n_entities,
entity_dense and `targets` (the languages this cut is translated into), plus a
`<corpus>/shards.tsv` line with the shard's total cut count.

Run inside synth.sif (spaCy en_core_web_lg).
"""

import glob
import hashlib
import json
import os
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from common import EN2X_TARGETS, write_jsonl_atomic


ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "DATE"}
# Share of the non-dense target draws per language. The tail gets more: the
# extremely low-resource ones (mt, ga) double, the other very small ones 1.5x.
TARGET_WEIGHT = {"mt": 2.0, "ga": 2.0, "lt": 1.5, "lv": 1.5, "et": 1.5, "sl": 1.5, "hr": 1.5}


def get_path(obj: dict, dotted: str):
    for k in dotted.split("."):
        if isinstance(obj, list) and k.isdigit():
            obj = obj[int(k)] if int(k) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


def assign_targets(cut_id: str, dense: bool, n_random: int, targets: list[str]) -> list[str]:
    """All targets for entity-dense cuts, else `n_random` drawn without replacement, tail-weighted.

    Seeded from the cut id so reruns give the same assignment.
    """
    if dense:
        return list(targets)
    rng = random.Random(int(hashlib.md5(cut_id.encode()).hexdigest()[:12], 16))
    left = list(targets)
    out = []
    for _ in range(min(n_random, len(left))):
        t = rng.choices(left, weights=[TARGET_WEIGHT.get(x, 1.0) for x in left])[0]
        left.remove(t)
        out.append(t)
    return sorted(out)


def process_shard(
    shard_path: str, corpus: str, out_dir: str, text_field: str, scores: np.ndarray | None,
    offset: int, wer_max: float, min_dur: float, max_dur: float, n_random: int,
    src_lang: str = "en", targets: tuple = (), pool_targets: tuple = tuple(EN2X_TARGETS),
) -> tuple[str, int, int]:  # fmt: skip
    import spacy

    name = os.path.basename(shard_path).split(".")[1]
    out = Path(out_dir) / corpus / f"pool.{name}.jsonl"
    if out.exists():
        return name, -1, -1
    nlp = None
    if src_lang == "en":  # entity stratification is English-only (spaCy en_core_web_lg)
        nlp = process_shard.nlp = getattr(process_shard, "nlp", None) or spacy.load(
            "en_core_web_lg", disable=["parser", "lemmatizer", "attribute_ruler"]
        )
    cand, n_total = [], 0
    with open(shard_path) as f:
        for idx, line in enumerate(f):
            n_total += 1
            cut = json.loads(line)
            if scores is not None:
                s = scores[offset + idx]
                if s == 65535 or s >= int(wer_max * 10000):
                    continue
            text = get_path(cut, text_field)
            if not text or len(text.split()) < 3 or not (min_dur <= cut["duration"] <= max_dur):
                continue
            cand.append(
                {"id": cut["id"], "shard": os.path.basename(shard_path), "idx": idx,
                 "text": text.strip(), "duration": round(cut["duration"], 3)}
            )  # fmt: skip
    docs = nlp.pipe((c["text"] for c in cand), batch_size=256) if nlp else [None] * len(cand)
    for c, doc in zip(cand, docs):
        ents = [(e.text, e.label_) for e in doc.ents if e.label_ in ENT_LABELS] if doc is not None else []
        c["ents"], c["n_entities"], c["entity_dense"] = ents, len(ents), len(ents) >= 2
        c["src_lang"] = src_lang
        if targets:  # explicit list: every cut gets all of them
            c["targets"] = [t for t in targets if t != src_lang]
        else:  # entity-dense: all pool targets; else n_random drawn, tail-weighted
            c["targets"] = assign_targets(c["id"], c["entity_dense"], n_random, list(pool_targets))
    write_jsonl_atomic(out, cand)
    with open(Path(out_dir) / corpus / "shards.tsv", "a") as f:
        f.write(f"{os.path.basename(shard_path)}\t{n_total}\t{len(cand)}\n")
    return name, n_total, len(cand)


def main(
    shar_dir: str,
    out_dir: str,
    corpus: str,
    text_field: str = "custom.pnc_text",
    scores_dir: str = "",
    wer_max: float = 0.05,
    shard_stride: int = 1,
    counts_file: str = "",
    max_shards: int = 0,
    min_dur: float = 2.0,
    max_dur: float = 30.0,
    n_random: int = 8,
    src_lang: str = "en",
    targets: list[str] | None = None,
    pool_targets: list[str] | None = None,
    workers: int = 8,
) -> None:
    """Pool for one corpus. `shard_stride` takes every k-th shard to spread over the corpus."""
    shards = sorted(glob.glob(os.path.join(shar_dir, "cuts.*.jsonl")))
    scores = np.load(os.path.join(scores_dir, "wer.u16.npy")) if scores_dir else None
    if counts_file:  # `wc -l` output for the shards (skips reading a huge corpus just to count)
        by_name = {l.split()[1]: int(l.split()[0]) for l in open(counts_file) if l.strip()}
        counts = [by_name[os.path.basename(p)] for p in shards]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        assert scores is None or len(scores) == offsets[-1], f"wer array {len(scores)} != {offsets[-1]} cuts"
    else:
        counts = [sum(1 for _ in open(p)) for p in shards]  # offsets into the WER array
        offsets = np.concatenate([[0], np.cumsum(counts)])
        if scores is not None:
            assert len(scores) == offsets[-1], f"wer array {len(scores)} != {offsets[-1]} cuts in {shar_dir}"
    pick = list(range(0, len(shards), shard_stride))
    if max_shards:
        pick = pick[:max_shards]
    fn = partial(
        process_shard, corpus=corpus, out_dir=out_dir, text_field=text_field, scores=scores,
        wer_max=wer_max, min_dur=min_dur, max_dur=max_dur, n_random=n_random, src_lang=src_lang,
        targets=tuple(targets or ()), pool_targets=tuple(pool_targets or EN2X_TARGETS),
    )  # fmt: skip
    (Path(out_dir) / corpus).mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        res = pool.starmap(_call, [(fn, shards[i], int(offsets[i])) for i in pick])
    wrong = [r[0] for i, r in zip(pick, res) if r[1] >= 0 and r[1] != int(offsets[i + 1] - offsets[i])]
    if wrong:
        raise RuntimeError(f"shards {wrong} hold a different number of cuts than counted: WER offsets are wrong")
    done = [r for r in res if r[1] >= 0]
    print(f"{corpus}: {len(pick)} shards ({len(pick) - len(done)} already done), "
          f"{sum(r[2] for r in done)} kept of {sum(r[1] for r in done)} cuts")  # fmt: skip


def _call(fn, shard_path: str, offset: int):
    return fn(shard_path, offset=offset)


if __name__ == "__main__":
    tyro.cli(main)
