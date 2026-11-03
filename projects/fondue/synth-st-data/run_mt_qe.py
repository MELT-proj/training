"""One work unit of machine translation + reference-free QE.

A unit is a pool JSONL (`make_units.py`); every row has `id`, `text`, `src_lang`
(default en), `targets`, and optionally `ents`/`n_entities`. Two stages, each its
own process so vLLM's GPU memory is certainly released before XCOMET loads:

  --stage mt  EuroLLM-9B-Instruct via vLLM, greedy, one candidate per source,
              instruction-first prompt. Directions with Maltese on either side
              also get a back-translation into the source language.
  --stage qe  XCOMET-XXL (source + MT, no reference), LID, length ratio, entity
              copy rate, round-trip chrF; degenerate outputs go to dropped.*.

The pairs are processed in chunks; a chunk whose output exists is skipped, so a
job killed at its wall limit loses at most one chunk. Every stage appends its
GPU-seconds to the ledger TSV. Run inside synth.sif, offline.
"""

import os
import time
from pathlib import Path

import tyro

from common import (
    MT_MODEL,
    NOT_IN_XLMR,
    PROMPT_VERSION,
    QE_MODEL,
    build_prompt,
    is_degenerate,
    ledger_append,
    read_jsonl,
    write_jsonl_atomic,
)


COPY_LABELS = {"PERSON", "ORG", "PRODUCT", "WORK_OF_ART"}


def expand_pairs(pool: list[dict], default_targets: list[str] | None = None) -> list[tuple[dict, str]]:
    """(row, target) pairs. A row's own `targets` wins over `default_targets`."""
    pairs = []
    for row in pool:
        tg = row.get("targets") or default_targets
        if not tg:
            raise ValueError(f"row {row['id']} has no targets and none were given")
        pairs.extend((row, t) for t in tg if t != row.get("src_lang", "en"))
    return pairs


def chunks(pairs: list, size: int) -> list[list]:
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


def uses_mt_xlmr_gap(src: str, tgt: str) -> bool:
    return src in NOT_IN_XLMR or tgt in NOT_IN_XLMR


def stage_mt(unit_dir: Path, parts: list, ledger: str, unit: str, gpu_mem: float) -> None:
    from vllm import LLM, SamplingParams

    todo = [(k, c) for k, c in enumerate(parts) if not (unit_dir / f"mt.{k:04d}.jsonl").exists()]
    if not todo:
        return
    t0 = time.time()
    llm = LLM(MT_MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=gpu_mem)
    ledger_append(ledger, unit=unit, stage="mt_load", n=0, gpu_seconds=round(time.time() - t0), gpus=1, status="ok")
    sp = SamplingParams(temperature=0, max_tokens=768)
    for k, part in todo:
        t1 = time.time()
        outs = llm.chat([build_prompt(r.get("src_lang", "en"), t, r["text"]) for r, t in part], sp)
        rows = [
            {"id": r["id"], "src_lang": r.get("src_lang", "en"), "tgt": t,
             "mt": o.outputs[0].text.strip(), "finish": o.outputs[0].finish_reason}
            for (r, t), o in zip(part, outs)
        ]  # fmt: skip
        rt_idx = [i for i, (r, t) in enumerate(part) if uses_mt_xlmr_gap(r.get("src_lang", "en"), t) and rows[i]["mt"]]
        if rt_idx:
            bt = llm.chat(
                [build_prompt(part[i][1], part[i][0].get("src_lang", "en"), rows[i]["mt"]) for i in rt_idx], sp
            )
            for i, o in zip(rt_idx, bt):
                rows[i]["rt"] = o.outputs[0].text.strip()
        write_jsonl_atomic(unit_dir / f"mt.{k:04d}.jsonl", rows)
        ledger_append(
            ledger, unit=unit, stage="mt_gen", n=len(rows), gpu_seconds=round(time.time() - t1), gpus=1, status="ok"
        )


def predict_with_oom_retry(model, samples: list[dict], batch: int):
    """XCOMET-XXL in fp32 leaves ~20 GB of an H100-64GB for activations: halve the batch on OOM."""
    import torch

    while True:
        try:
            return model.predict(samples, batch_size=batch, gpus=1, progress_bar=False)
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            print(f"qe: CUDA OOM, retrying with batch {batch}", flush=True)


def stage_qe(unit_dir: Path, by_id: dict, ledger: str, unit: str, batch: int, qe_dtype: str, n_parts: int) -> None:
    import fasttext
    import sacrebleu
    import torch
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    todo = [k for k in range(n_parts) if not (unit_dir / f"final.{k:04d}.jsonl").exists()]
    if not todo:
        return
    t0 = time.time()
    ckpt = os.path.join(snapshot_download(QE_MODEL, local_files_only=True), "checkpoints/model.ckpt")
    model = load_from_checkpoint(ckpt)
    if qe_dtype == "bf16":
        model = model.to(torch.bfloat16)
    ledger_append(ledger, unit=unit, stage="qe_load", n=0, gpu_seconds=round(time.time() - t0), gpus=1, status="ok")
    lid = fasttext.load_model(os.path.join(os.environ.get("SYNTH_ASSETS", "/opt/assets"), "lid.176.bin"))

    for k in todo:
        mt_rows = read_jsonl(unit_dir / f"mt.{k:04d}.jsonl")
        keep, dropped = [], []
        for r in mt_rows:
            src = by_id[r["id"]]["text"]
            (dropped if is_degenerate(src, r["mt"]) or r["finish"] != "stop" else keep).append(r)
        t1 = time.time()
        order = sorted(range(len(keep)), key=lambda i: len(keep[i]["mt"]))  # little padding per batch
        samples = [{"src": by_id[keep[i]["id"]]["text"], "mt": keep[i]["mt"]} for i in order]
        res = predict_with_oom_retry(model, samples, batch)
        spans = res.metadata.error_spans
        out = [None] * len(keep)
        for j, i in enumerate(order):
            r, seg = keep[i], by_id[keep[i]["id"]]
            labels, probs = lid.predict(r["mt"].replace("\n", " "), k=1)
            ents = seg.get("ents", [])
            names = [e for e, lab in ents if lab in COPY_LABELS]
            row = {
                "id": r["id"], "tgt": r["tgt"], "src_lang": r["src_lang"], "text": r["mt"],
                "xcomet_xxl": round(float(res.scores[j]), 5),
                "xcomet_spans": [
                    {"text": s["text"], "start": s["start"], "end": s["end"],
                     "severity": s["severity"], "confidence": round(float(s["confidence"]), 4)}
                    for s in spans[j]
                ],
                "lid": labels[0].replace("__label__", ""), "lid_p": round(float(probs[0]), 4),
                "len_ratio": round(len(r["mt"]) / max(len(seg["text"]), 1), 3),
                "n_entities": seg.get("n_entities", 0),
                "entity_copy_rate": round(sum(e in r["mt"] for e in names) / len(names), 3) if names else None,
                "mt_model": MT_MODEL, "qe_model": QE_MODEL, "prompt_version": PROMPT_VERSION,
            }  # fmt: skip
            if "rt" in r:
                row["rt_chrf"] = round(sacrebleu.sentence_chrf(r["rt"], [seg["text"]]).score / 100, 4)
            out[i] = row
        write_jsonl_atomic(unit_dir / f"dropped.{k:04d}.jsonl", dropped)
        write_jsonl_atomic(unit_dir / f"final.{k:04d}.jsonl", out)  # last: marks the chunk done
        ledger_append(
            ledger, unit=unit, stage="qe", n=len(out), gpu_seconds=round(time.time() - t1), gpus=1, status="ok"
        )


def main(
    pool: str,
    out_dir: str,
    unit: str,
    stage: str,  # mt | qe
    default_targets: list[str] | None = None,
    limit: int = 0,
    chunk: int = 50000,
    ledger: str = "ledger.tsv",
    gpu_mem: float = 0.88,
    qe_batch: int = 16,
    qe_dtype: str = "fp32",
    dry_run: bool = False,
) -> None:
    unit_dir = Path(out_dir) / unit
    rows = read_jsonl(pool)
    if limit:
        rows = rows[:limit]
    parts = chunks(expand_pairs(rows, default_targets), chunk)
    n_done = sum(
        (unit_dir / f"{'mt' if stage == 'mt' else 'final'}.{k:04d}.jsonl").exists() for k in range(len(parts))
    )
    print(f"unit={unit} stage={stage} rows={len(rows)} chunks={len(parts)} done={n_done}")
    if dry_run:
        return
    unit_dir.mkdir(parents=True, exist_ok=True)
    if stage == "mt":
        stage_mt(unit_dir, parts, ledger, unit, gpu_mem)
    else:
        stage_qe(unit_dir, {r["id"]: r for r in rows}, ledger, unit, qe_batch, qe_dtype, len(parts))
        if all((unit_dir / f"final.{k:04d}.jsonl").exists() for k in range(len(parts))):
            (unit_dir / "DONE").write_text(f"{len(parts)} chunks\n")


if __name__ == "__main__":
    tyro.cli(main)
