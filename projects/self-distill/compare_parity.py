"""Compare a run's in-training eval with melt-eval on the same checkpoint and clips.

Phase 2, step 1c. Inputs: the training job's log, the eval step whose weights
the checkpoint holds, and the ``.eval`` log melt-eval wrote on the frozen set
from eval_parity_spec.py. Reports, per language:

* WER as the trainer logged it, as melt-eval's scorer reported it, and
  recomputed here from melt-eval's raw outputs with the trainer's normaliser
  (``BasicTextNormalizer`` + jiwer), so a normalisation difference cannot pass
  for a model difference;
* the share of the trainer's logged hypotheses (10 per language) that melt-eval
  reproduces verbatim, matched on the reference text.

Run where both ``inspect_ai`` and ``melt`` import (the melt-eval venv with the
training checkout on PYTHONPATH); CPU only.
"""

import argparse
import ast
import json
import re
from collections import defaultdict

import jiwer
from inspect_ai.log import read_eval_log

from melt.evaluation import BasicTextNormalizer


def trainer_side(log_path: str, step: int) -> tuple[dict, dict]:
    """(per-language WER, per-language [(REF, HYP)]) logged at *step*."""
    text = open(log_path, errors="replace").read()
    steps = []
    for s in re.findall(r"\[eval_asr_en\] step (\d+) ", text):
        if s not in steps:
            steps.append(s)
    rounds, current = [], {}
    for m in re.finditer(r"(\{'eval_asr_[a-z]{2}_loss'[^}]*\})", text):
        row = ast.literal_eval(m.group(1))
        lang = next(k for k in row if k.endswith("_loss"))[len("eval_asr_"):-len("_loss")]
        current[lang] = float(row[f"eval_asr_{lang}_wer"])
        if len(current) == 5:
            rounds.append(current)
            current = {}
    if str(step) not in steps:
        raise SystemExit(f"step {step} not among the logged evals {steps}")
    wer = rounds[steps.index(str(step))]
    pairs, lang, ref = defaultdict(list), None, None
    for line in text.splitlines():
        m = re.search(r"\[eval_asr_(\w\w)\] step (\d+) ", line)
        if m:
            lang = m.group(1) if int(m.group(2)) == step else None
            continue
        if lang and "REF:" in line:
            ref = line.split("REF:", 1)[1].strip()
        elif lang and "HYP:" in line and ref is not None:
            pairs[lang].append((ref, line.split("HYP:", 1)[1].strip()))
            ref = None
    return wer, pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer-log", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--eval-log", required=True, help="melt-eval .eval file")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    norm = BasicTextNormalizer()
    trainer_wer, logged = trainer_side(args.trainer_log, args.step)
    log = read_eval_log(args.eval_log)
    metrics = {m: v.value for sc in log.results.scores for m, v in sc.metrics.items()}
    by_lang = defaultdict(list)
    for s in log.samples:
        by_lang[(s.metadata or {}).get("lang", "?")].append((str(s.target).strip(), s.output.completion.strip()))

    report = {}
    for lang in sorted(trainer_wer):
        rows = by_lang.get(lang, [])
        refs = [norm(r).strip() for r, _ in rows]
        hyps = [norm(h).strip() for _, h in rows]
        recomputed = jiwer.wer(refs, hyps) if rows else float("nan")
        # The trainer logs each string with its whitespace collapsed (_one_line).
        outputs = {" ".join(r.split()): " ".join(h.split()) for r, h in rows}
        matched = [(h, outputs[r]) for r, h in logged.get(lang, []) if r in outputs]
        report[lang] = {
            "n_melteval": len(rows),
            "wer_trainer": trainer_wer[lang],
            "wer_melteval_scorer": metrics.get(f"wer_{lang}"),
            "wer_melteval_recomputed": recomputed,
            "logged_found": f"{len(matched)}/{len(logged.get(lang, []))}",
            "verbatim_same": sum(a == b for a, b in matched) / max(1, len(matched)),
            "normalised_same": sum(norm(a) == norm(b) for a, b in matched) / max(1, len(matched)),
        }
    for lang, r in report.items():
        print(f"{lang}: " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
