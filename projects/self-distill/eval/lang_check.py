"""Which language did each FLEURS translation come out in?

    python lang_check.py <refs dir> <file.eval> ... [--lowercase] [--json out.json]

FLEURS is sentence-parallel, so an output's language is the candidate whose version of the same
sentence id it matches best (sentence chrF). <refs dir> holds ``fleurs-<lang>-reference-dev.json``
for all 24 languages (id -> sentence); the source transcript is a candidate too (copying instead
of translating). English is not a candidate when the source is English: for en->X an English
output is a copy of the source. Only the FLEURS ST sets are read (``fleurs24-st-x*`` and
``fleurs24-st-enx``); ga as a source is skipped, as the scores of the other sets do.

Per (model, set, key) it counts the outputs on the requested target, in English (X->Y with a
non-English source), copying the source, in some other language, and empty. ``key`` is the target
for en->X (one source) and the source for X->Y; the group is ``seen`` for the trained languages
(en/de/fr/es/it) and ``unseen`` for the rest.

With ``--lowercase`` hypotheses, references and sources are lowercased first, as in
``summarize_evals.py --lowercase``: SFT models write lowercase and FLEURS is cased.

Run in the melt-eval venv (inspect_ai + sacrebleu); CPU only.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log
from sacrebleu.metrics import CHRF
from summarize_evals import SEEN, run_tag


CANDIDATES = (
    "en", "es", "fr", "de", "it", "nl", "pt", "pl", "hu", "fi", "cs", "ro",
    "sv", "hr", "sk", "bg", "lv", "da", "lt", "el", "et", "sl", "ga", "mt",
)  # fmt: skip
chrf = CHRF()


def load_refs(refs_dir: str, lowercase: bool) -> dict[str, dict[str, str]]:
    """The 24 FLEURS dev references, id -> sentence, per language."""
    refs = {}
    for lang in CANDIDATES:
        data = json.loads((Path(refs_dir) / f"fleurs-{lang}-reference-dev.json").read_text(encoding="utf-8"))
        refs[lang] = {str(k): (v.lower() if lowercase else v) for k, v in data.items()}
    return refs


def new_cell() -> dict:
    return {"n": 0, "on": 0, "en": 0, "copy": 0, "empty": 0, "other": Counter()}


def count_log(
    path: str, refs: dict[str, dict[str, str]], lowercase: bool, cells: dict, pooled: dict, per_key: dict
) -> None:
    """Add one log's outputs to ``cells[(model, set, target)][group]``, ``per_key[(model, set, key)]`` and,
    for en->X, ``pooled[(model, set)][group]``."""
    log = read_eval_log(path)
    frozen = str((log.eval.task_args or {}).get("frozen_set") or "")
    enx = "fleurs24-st-enx" in frozen
    if not enx and "fleurs24-st-x" not in frozen:
        return
    model, set_name = str(log.eval.model), Path(frozen).name
    for s in log.samples or []:
        m = s.metadata or {}
        src, tgt, sid = m.get("src_lang"), m.get("tgt_lang"), str(m.get("cut_id"))
        if src == "ga":
            continue
        hyp = (s.output.completion if s.output else "").strip()
        source_text = m.get("source_text") or ""
        if lowercase:
            hyp, source_text = hyp.lower(), source_text.lower()
        scores = {
            lang: chrf.sentence_score(hyp, [refs[lang][sid]]).score
            for lang in CANDIDATES
            if sid in refs[lang] and lang != src
        }
        if source_text:
            scores["SOURCE"] = chrf.sentence_score(hyp, [source_text]).score
        best = max(scores, key=scores.get) if scores and hyp else "EMPTY"
        key = tgt if enx else src
        group = "seen" if key in SEEN else "unseen"
        for cell in (
            cells[(model, set_name, tgt)][group],
            per_key[(model, set_name, key)],
            *([pooled[(model, set_name)][group]] if enx else []),
        ):
            cell["n"] += 1
            cell["on"] += best == tgt
            cell["en"] += best == "en" and tgt != "en"
            cell["copy"] += best == "SOURCE"
            cell["empty"] += best == "EMPTY"
            if best not in (tgt, "SOURCE", "EMPTY") and not (best == "en" and tgt != "en"):
                cell["other"][best] += 1


def language_shares(paths: list[str], refs_dir: str, lowercase: bool = False) -> tuple[dict, dict, dict]:
    """``(cells, pooled, per_key)`` for the given logs; see :func:`count_log`."""
    refs = load_refs(refs_dir, lowercase)
    cells = defaultdict(lambda: defaultdict(new_cell))  # (model, set, target) -> group -> cell
    pooled = defaultdict(lambda: defaultdict(new_cell))  # (model, set) -> group -> cell, en->X only
    per_key = defaultdict(new_cell)  # (model, set, key) -> cell, one source (X->Y) or one target (en->X)
    for path in paths:
        count_log(path, refs, lowercase, cells, pooled, per_key)
    return cells, pooled, per_key


def to_rows(per_key: dict) -> list[dict]:
    """One JSON-friendly row per (model, set, key): the counts of :func:`new_cell` plus the group."""
    return [
        {
            "model": model,
            "run": run_tag(model),
            "set": set_name,
            "key": key,
            "group": "seen" if key in SEEN else "unseen",
            **{k: (dict(v) if k == "other" else v) for k, v in c.items()},
        }
        for (model, set_name, key), c in sorted(per_key.items())
    ]


def show(cells: dict, pooled: dict) -> None:
    print("share of outputs on target / in English / copying the source / in another language / empty;")
    print(
        "X->Y sets: trained sources | unseen sources; en->X sets: trained targets (de/fr/es/it) | the other 19 targets"
    )
    for (model, set_name, tgt), groups in sorted(cells.items()):
        line = f"{run_tag(model):24s} {set_name:20s} ->{tgt}:"
        for group in ("seen", "unseen"):
            c = groups.get(group)
            if c and c["n"]:
                n = c["n"]
                line += (
                    f"  {group} n={n:4d} on {c['on'] / n:.2f} en {c['en'] / n:.2f} copy {c['copy'] / n:.2f}"
                    f" other {sum(c['other'].values()) / n:.2f} empty {c['empty'] / n:.2f}"
                )
        print(line)
    for (model, set_name), groups in sorted(pooled.items()):
        print(f"\n{run_tag(model)} {set_name}, pooled over targets:")
        for group, label in (("seen", "trained targets (de/fr/es/it)"), ("unseen", "other targets")):
            c = groups.get(group)
            if c and c["n"]:
                n = c["n"]
                off = ", ".join(f"{k} {v}" for k, v in c["other"].most_common(6)) or "-"
                print(
                    f"  {label:30s} n={n:5d} on target {c['on'] / n:.3f}  copy source {c['copy'] / n:.3f}"
                    f"  other language {sum(c['other'].values()) / n:.3f}  empty {c['empty'] / n:.3f}  (other language: {off})"
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("refs_dir")
    p.add_argument("logs", nargs="+")
    p.add_argument("--lowercase", action="store_true", help="compare lowercased text (see above)")
    p.add_argument("--json", default=None, help="write the per-cell counts here")
    args = p.parse_args()
    cells, pooled, per_key = language_shares(args.logs, args.refs_dir, args.lowercase)
    show(cells, pooled)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"lowercase": args.lowercase, "rows": to_rows(per_key)}, f, indent=1)


if __name__ == "__main__":
    main()
