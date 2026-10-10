"""Summarise melt-eval logs from the step-2 battery: one block per (run, set, format).

ASR: WER and CER per language and per source, recomputed from the samples with
the trainer's normaliser (``BasicTextNormalizer`` + jiwer, which matched
melt-eval's own scorer exactly in the parity check), the means over the five
trained languages and over the rest, and the trainer's runaway fraction (a
hypothesis more than ``RUNAWAY_LENGTH_RATIO`` times the reference's words).
ST: melt-eval's own per-language metrics, with the same seen/unseen means.

Run where both ``inspect_ai`` and ``melt`` import (the melt-eval venv with the
training checkout on PYTHONPATH); CPU only:

    python projects/self-distill/eval/summarize_evals.py <file.eval> ... [--json out.json]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import jiwer
from inspect_ai.log import read_eval_log

from melt.evaluation import BasicTextNormalizer
from melt.training.metrics import RUNAWAY_LENGTH_RATIO

SEEN = ("en", "de", "fr", "es", "it")


def run_tag(model: str) -> str:
    """``melt//.../S2-opd-from-GOLDW-final-...`` -> ``S2-opd``; a GOLDW root -> ``GOLDW-final``."""
    name = Path(model.split("//", 1)[-1]).name
    if name.startswith("S2-"):
        return name.split("-from-", 1)[0]
    return name.split("-", 1)[0] + ("-final" if not name.startswith("checkpoint") else "")


def set_name(frozen: str) -> str:
    path = Path(frozen)
    return path.parent.name if path.name == "frozen" else path.name


def asr_scores(pairs: list[tuple[str, str]]) -> dict:
    """Corpus WER/CER, runaway fraction and pair counts for normalised (ref, hyp) pairs."""
    kept = [(r, h) for r, h in pairs if r]
    refs, hyps = [r for r, _ in kept], [h for _, h in kept]
    runaway = sum(len(h.split()) > RUNAWAY_LENGTH_RATIO * len(r.split()) for r, h in kept)
    return {
        "n": len(kept),
        "empty_refs": len(pairs) - len(kept),
        "wer": jiwer.wer(refs, hyps) if kept else float("nan"),
        "cer": jiwer.cer(refs, hyps) if kept else float("nan"),
        "runaway": runaway / len(kept) if kept else float("nan"),
    }


def mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def summarize(path: str, norm: BasicTextNormalizer) -> dict:
    log = read_eval_log(path)
    args = log.eval.task_args or {}
    out = {
        "file": str(path),
        "run": run_tag(str(log.eval.model)),
        "model": str(log.eval.model),
        "set": set_name(str(args.get("frozen_set") or "")),
        "format": Path(args["format_config"]).stem if args.get("format_config") else "own",
        "task": args.get("task_filter"),
        "metrics": {m: v.value for sc in (log.results.scores if log.results else []) for m, v in sc.metrics.items()},
    }
    if out["task"] == "asr":
        by_lang, by_source = defaultdict(list), defaultdict(list)
        for s in log.samples or []:
            meta = s.metadata or {}
            pair = (norm(str(s.target)).strip(), norm(s.output.completion if s.output else "").strip())
            by_lang[meta.get("lang", "?")].append(pair)
            by_source[(meta.get("lang", "?"), meta.get("dataset_id", "?"))].append(pair)
        out["lang"] = {lang: asr_scores(p) for lang, p in sorted(by_lang.items())}
        out["source"] = {f"{lang}/{src}": asr_scores(p) for (lang, src), p in sorted(by_source.items())}
        for group, langs in (("seen", [l for l in out["lang"] if l in SEEN]), ("unseen", [l for l in out["lang"] if l not in SEEN])):
            for key in ("wer", "cer"):
                out[f"{group}_{key}"] = mean([out["lang"][l][key] for l in langs])
    else:
        per_lang = defaultdict(dict)
        for name, value in out["metrics"].items():
            m = re.fullmatch(r"(bleu|chrf|chrF|comet)_([a-z]{2,3})(?:_[a-z]{2,3})?", name)
            if m:
                per_lang[m.group(2)][m.group(1).lower()] = value
        out["lang"] = dict(sorted(per_lang.items()))
        for group, langs in (("seen", [l for l in out["lang"] if l in SEEN]), ("unseen", [l for l in out["lang"] if l not in SEEN])):
            for key in ("bleu", "chrf"):
                out[f"{group}_{key}"] = mean([out["lang"][l].get(key, float("nan")) for l in langs])
    return out


def show(r: dict) -> None:
    head = f"{r['run']} | {r['set']} | format {r['format']}"
    if r["task"] == "asr":
        langs = " ".join(f"{l} {v['wer']:.3f}" for l, v in r["lang"].items() if l in SEEN)
        print(f"{head} | WER {langs} | seen {r['seen_wer']:.3f}"
              + (f" | unseen WER {r['unseen_wer']:.3f} CER {r['unseen_cer']:.3f}" if r["unseen_wer"] == r["unseen_wer"] else "")
              + f" | max runaway {max(v['runaway'] for v in r['lang'].values()):.3f}")
        sources = sorted({k.split("/", 1)[1] for k in r["source"]})
        if len(sources) > 1:
            for src in sources:
                cells = " ".join(f"{k.split('/')[0]} {v['wer']:.3f}" for k, v in r["source"].items() if k.endswith("/" + src))
                print(f"    {src}: {cells}")
    else:
        langs = " ".join(f"{l} {v.get('chrf', float('nan')):.1f}" for l, v in r["lang"].items() if l in SEEN)
        print(f"{head} | chrF {langs} | seen chrF {r['seen_chrf']:.1f} BLEU {r['seen_bleu']:.1f}"
              f" | unseen chrF {r['unseen_chrf']:.1f} BLEU {r['unseen_bleu']:.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    norm = BasicTextNormalizer()
    rows = [summarize(path, norm) for path in args.logs]
    for r in sorted(rows, key=lambda r: (r["set"], r["run"], r["format"])):
        show(r)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
