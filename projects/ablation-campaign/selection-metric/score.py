#!/usr/bin/env python3
"""Selection metric for MA-stage checkpoints, computed from melt-eval JSON logs.

The definition lives in ``README.md`` next to this file; the group membership and
weights below are copied from it and are the only place they are hard-coded.
This script never runs a model and never imports inspect_ai: it reads the
``--log-format json`` logs melt-eval writes, and nothing else.

Each *checkpoint directory* holds that checkpoint's JSON logs, in any layout
below it (one sub-directory per set is what the MN5 launch uses). It must
contain exactly one log for every in-domain set and for the FLEURS set. Missing
sets, missing languages, duplicated logs, incomplete logs and logs that were not
scored with the ASR scorer all abort the run: a language is never dropped
silently.

Per-language CER is taken as melt-eval reports it, ``cer_<lang>`` under
``results.scores[0].metrics`` (corpus-level, grouped by language), not
re-derived from samples.

Usage::

    python score.py CKPT_DIR [CKPT_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# The metric, as defined in README.md. Change it there first, then here.
# ---------------------------------------------------------------------------

CER_CLIP = 1.0

TRAIN_LANGS = ("en", "de", "es", "fr", "it")

#: group name -> (languages, source set, weight or None for "reported only").
#: ID is scored on the per-language in-domain sets; every other group on FLEURS.
GROUPS: dict[str, tuple[tuple[str, ...], str, float | None]] = {
    "ID": (TRAIN_LANGS, "id", 1.0),
    "OOD-train": (TRAIN_LANGS, "fleurs", 1.5),
    "OOD-related": (("pt", "ro", "nl", "da", "sv"), "fleurs", 0.6),
    "OOD-latin": (
        ("cs", "sk", "pl", "hr", "sl", "hu", "fi", "et", "lt", "lv", "ga", "mt"),
        "fleurs",
        None,
    ),
    "OOD-script": (("bg", "el", "ru", "uk"), "fleurs", None),
}

#: Frozen-set names, as the basename of ``task_args.frozen_set`` in each log.
FLEURS_SET = "selection-fleurs26-dev200"
ID_SET_TEMPLATE = "selection-id-{lang}"

SCORER_METRIC = "corpus_cer"  # present only when the log was scored with task_filter=asr


class ScoringError(RuntimeError):
    """A log or a language the metric needs is missing or unusable."""


# ---------------------------------------------------------------------------
# Reading logs
# ---------------------------------------------------------------------------


def read_log_summary(path: Path) -> dict:
    """Load a JSON log and return its set name and per-language CER.

    Returns:
        ``{"set": basename of the frozen set, "cer": {lang: cer}, "path": path}``.

    Raises:
        ScoringError: If the log is not a completed, ASR-scored melt-eval log.
    """
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"{path}: not readable as a JSON log ({exc})") from exc

    if log.get("status") != "success":
        raise ScoringError(f"{path}: log status is {log.get('status')!r}, not 'success'")
    results = log.get("results") or {}
    if results.get("completed_samples") != results.get("total_samples"):
        raise ScoringError(
            f"{path}: {results.get('completed_samples')} of {results.get('total_samples')} samples completed"
        )

    frozen = ((log.get("eval") or {}).get("task_args") or {}).get("frozen_set")
    if not frozen:
        raise ScoringError(f"{path}: no eval.task_args.frozen_set, cannot tell which set this is")

    scores = results.get("scores") or []
    if not scores:
        raise ScoringError(f"{path}: no scores in results")
    metrics = scores[0].get("metrics") or {}
    if SCORER_METRIC not in metrics:
        raise ScoringError(
            f"{path}: no {SCORER_METRIC!r} metric; this log was not scored with -T task_filter=asr"
        )

    cer = {
        name[len("cer_"):]: float(metric["value"])
        for name, metric in metrics.items()
        if name.startswith("cer_")
    }
    return {"set": Path(str(frozen).rstrip("/")).name, "cer": cer, "path": path}


def load_checkpoint(directory: Path) -> dict[str, dict[str, float]]:
    """Read every JSON log under *directory*, keyed by set name.

    Returns:
        ``{set_name: {lang: cer}}`` for the sets the metric uses.

    Raises:
        ScoringError: On no logs, a set with two logs, or a needed set with none.
    """
    logs = sorted(directory.rglob("*.json"))
    if not logs:
        raise ScoringError(f"{directory}: no *.json logs found (were they written with --log-format json?)")

    needed = {FLEURS_SET} | {ID_SET_TEMPLATE.format(lang=lang) for lang in TRAIN_LANGS}
    by_set: dict[str, dict] = {}
    for path in logs:
        summary = read_log_summary(path)
        if summary["set"] not in needed:
            continue  # a log for some other set living in the same tree
        if summary["set"] in by_set:
            raise ScoringError(
                f"{directory}: two logs for set {summary['set']!r}: "
                f"{by_set[summary['set']]['path']} and {path}"
            )
        by_set[summary["set"]] = summary

    missing = sorted(needed - set(by_set))
    if missing:
        raise ScoringError(f"{directory}: no log for set(s): {', '.join(missing)}")
    return {name: summary["cer"] for name, summary in by_set.items()}


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def group_values(sets: dict[str, dict[str, float]], group: str) -> dict[str, float]:
    """Raw (unclipped) CER for each language of *group*.

    Raises:
        ScoringError: If any language the group needs is absent from its logs.
    """
    langs, source, _ = GROUPS[group]
    values: dict[str, float] = {}
    missing: list[str] = []
    for lang in langs:
        cer = sets[FLEURS_SET if source == "fleurs" else ID_SET_TEMPLATE.format(lang=lang)]
        if lang in cer:
            values[lang] = cer[lang]
        else:
            missing.append(lang)
    if missing:
        raise ScoringError(f"group {group}: no CER for language(s) {', '.join(missing)} in the logs")
    return values


def score_checkpoint(sets: dict[str, dict[str, float]]) -> dict:
    """Compute per-language CER, the group medians and the weighted score.

    Returns:
        ``{"languages": {group: {lang: (raw, clipped)}}, "medians": {group: m},
        "score": s}``. Lower is better.
    """
    languages: dict[str, dict[str, tuple[float, float]]] = {}
    medians: dict[str, float] = {}
    for group in GROUPS:
        raw = group_values(sets, group)
        languages[group] = {lang: (v, min(v, CER_CLIP)) for lang, v in raw.items()}
        medians[group] = statistics.median(clipped for _, clipped in languages[group].values())

    weighted = {g: w for g, (_, _, w) in GROUPS.items() if w is not None}
    score = sum(w * medians[g] for g, w in weighted.items()) / sum(weighted.values())
    return {"languages": languages, "medians": medians, "score": score}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_checkpoint(name: str, result: dict) -> str:
    """Per-language CERs, group medians and score for one checkpoint."""
    lines = [f"== {name}"]
    for group, langs in result["languages"].items():
        cells = " ".join(
            f"{lang}={clipped:.3f}{'*' if raw > CER_CLIP else ''}" for lang, (raw, clipped) in langs.items()
        )
        lines.append(f"  {group:<12} median {result['medians'][group]:.4f}   {cells}")
    lines.append(f"  score {result['score']:.4f}   (* = clipped at {CER_CLIP:g})")
    return "\n".join(lines)


def format_table(results: dict[str, dict]) -> str:
    """Table across checkpoints, sorted by score (lower first)."""
    header = ["checkpoint", *GROUPS, "score"]
    rows = [
        [name, *(f"{r['medians'][g]:.4f}" for g in GROUPS), f"{r['score']:.4f}"]
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["score"])
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    fmt = lambda row: "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
    weights = " + ".join(f"{w:g}*{g}" for g, (_, _, w) in GROUPS.items() if w is not None)
    return "\n".join([fmt(header), *(fmt(row) for row in rows), "", f"score = ({weights}) / {sum(w for *_, w in GROUPS.values() if w is not None):g}; lower is better"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint result directories.")
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}
    for directory in args.checkpoints:
        name = directory.resolve().name
        if name in results:
            print(f"ERROR: two checkpoint directories are named {name!r}; the table needs unique names", file=sys.stderr)
            return 1
        try:
            results[name] = score_checkpoint(load_checkpoint(directory))
        except ScoringError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    for name, result in results.items():
        print(format_checkpoint(name, result))
        print()
    print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
