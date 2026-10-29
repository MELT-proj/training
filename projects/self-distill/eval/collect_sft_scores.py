"""Summarise the SFT battery's eval logs for ``sft_tables.py``.

    python collect_sft_scores.py [--logs-dir DIR] [--out-dir DIR] [--refs-dir DIR]

Picks the melt-eval logs that scored a step-0 start (GOLDW-final = A, S3-azeros_anchor = B) or an SFT
run's checkpoint or final model under ``format-sft-random``, keeps the newest successful log per
(model, set), and writes into ``--out-dir``:

- ``summaries.json``: ``summarize_evals.py --lowercase`` rows (ST scored case-insensitively);
- ``langshare.json``: ``lang_check.py --lowercase`` counts for the FLEURS ST sets.

Per-log results are cached in ``--out-dir/cache`` under the log's file name (a log never changes once
written); delete the directory to recompute. Run in the melt-eval venv with the training checkout on
PYTHONPATH, like ``summarize_evals.py``; CPU only. Then: ``python sft_tables.py summaries.json langshare.json``.
"""

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log
from lang_check import count_log, load_refs, new_cell, to_rows
from sft_tables import classify_model
from summarize_evals import summarize

from melt.evaluation import BasicTextNormalizer


DEFAULT_LOGS = "/mnt/scratch-artemis/giuseppe/melt-data/eval-logs"
DEFAULT_OUT = "/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring/summaries"
DEFAULT_REFS = "/mnt/scratch-artemis/giuseppe/tmp/fleurs-refs24"


def select_logs(logs_dir: str, since: str, classify=classify_model) -> list[tuple[str, dict, str, str]]:
    """``(path, classification, set name, model)`` of the newest successful SFT-format log per model and set.

    ``classify`` maps a log's model path to ``{cond, init, seed, step}`` or None to skip it: ``sft_tables.classify_model``
    for the A-against-B comparison, ``tier1_tables.classify_arm`` for the MA-utility arms.
    """
    newest: dict[tuple, tuple] = {}
    for path in sorted(Path(logs_dir).glob("*.eval")):
        if path.name[: len(since)] < since:
            continue
        log = read_eval_log(str(path), header_only=True)
        args = log.eval.task_args or {}
        if "format-sft-random" not in str(args.get("format_config") or "") or log.status != "success":
            continue
        info = classify(str(log.eval.model))
        if info is None:
            continue
        set_name = Path(str(args.get("frozen_set") or "")).name
        key = (info["cond"], info["init"], info["seed"], info["step"], set_name)
        if key in newest:
            print(f"note: {set_name} of {info} was scored twice; keeping the newer log {path.name}")
        newest[key] = (str(path), info, set_name, str(log.eval.model))  # sorted by name = by time: the later wins
    return list(newest.values())


def cached(cache: Path, name: str, compute):
    """``compute()``'s JSON-able result, stored under ``cache/name``."""
    file = cache / name
    if file.exists():
        return json.loads(file.read_text())
    result = compute()
    file.write_text(json.dumps(result))
    return result


def collect(logs_dir: str, out_dir: Path, refs_dir: str, since: str, classify=classify_model) -> None:
    """Write ``summaries.json`` and ``langshare.json`` for the logs ``classify`` accepts (see ``select_logs``)."""
    out = Path(out_dir)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    selected = select_logs(logs_dir, since, classify)
    print(f"{len(selected)} logs selected")
    norm, refs = BasicTextNormalizer(), None
    summaries, langshare = [], []
    for path, info, set_name, model in selected:
        name = Path(path).name
        summaries.append(
            cached(cache, name + ".summary.json", lambda path=path: summarize(path, norm, lowercase=True))
        )
        if "-st-" in set_name:
            if refs is None:
                refs = load_refs(refs_dir, lowercase=True)

            def compute_shares(path=path):
                from collections import defaultdict

                per_key = defaultdict(new_cell)
                count_log(
                    path,
                    refs,
                    True,
                    defaultdict(lambda: defaultdict(new_cell)),
                    defaultdict(lambda: defaultdict(new_cell)),
                    per_key,
                )
                return to_rows(per_key)

            langshare += cached(cache, name + ".lang.json", compute_shares)
    (out / "summaries.json").write_text(json.dumps(summaries, indent=1))
    (out / "langshare.json").write_text(json.dumps({"lowercase": True, "rows": langshare}, indent=1))
    print(
        f"wrote {out / 'summaries.json'} ({len(summaries)} rows) and {out / 'langshare.json'} ({len(langshare)} rows)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", default=DEFAULT_LOGS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--refs-dir", default=DEFAULT_REFS, help="fleurs-<lang>-reference-dev.json for the 24 languages")
    p.add_argument("--since", default="2026-10-11", help="ignore logs started before this date (file name prefix)")
    args = p.parse_args()
    collect(args.logs_dir, Path(args.out_dir), args.refs_dir, args.since)


if __name__ == "__main__":
    main()
