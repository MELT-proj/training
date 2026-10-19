"""Collate the SD24 comparison: every model of step 6 on the FLEURS-24 dev sets, under both prompt formats.

    python collect_sd24_scores.py [--logs-dir DIR] [--out-dir DIR] [--sft-summaries FILE] [--detail]

The rows of the table (``MODELS``) are the alignment starts (A for Qwen3.5-2B, A' for EuroLLM), the full-model
reference (SFT-EU24), the gold-CE control (CE24) and the self-distilled adapters (SD24 with the instruction mix,
SD-SIFT24 with AZeroS's recipe), each for Qwen and for EuroLLM where it exists. Each is scored under
``fmt-repeatfirst`` (the SD24 training instructions) and ``fmt-sft-random`` (the random task templates SFT-EU24
trained on). The melt-eval logs are found by the metadata the launchers set (``sft_run``, ``sft_label``);
the step-0 start A and the SFT-EU24 final under the random templates come from the SFT scoring's
``summaries.json`` (``render_sft_tables.sh``). The newest successful log per (run, label, set) wins. A model not
scored yet shows ``-``.

Scores are the ones of ``summarize_evals.py --lowercase`` (training and the SFT models write lowercase; ASR is
normalised anyway). Writes ``sd24_summaries.json`` (every row used) and ``sd24_tables.md`` into ``--out-dir`` and
prints the tables. Run in the melt-eval venv with the training checkout on PYTHONPATH, like ``summarize_evals.py``;
CPU only.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log
from lang_check import count_log, load_refs, new_cell
from sft_tables import SETS, classify_model
from summarize_evals import SEEN, summarize

from melt.evaluation import BasicTextNormalizer


DEFAULT_LOGS = "/mnt/scratch-artemis/giuseppe/melt-data/eval-logs"
DEFAULT_OUT = "/mnt/scratch-nyx/giuseppe/melt/outputs-sd24/scoring"
DEFAULT_SFT_SUMMARIES = "/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring/summaries/summaries.json"
DEFAULT_REFS = "/mnt/scratch-artemis/giuseppe/tmp/fleurs-refs24"

SFT_EU24 = "SFT-EU24-from-goldw-bd60-ga4-8g-s1337"
FORMATS = ("repeatfirst", "sft-random")

# (key, label, run name for the melt-eval metadata ``sft_run``). The label of a log is ``fmt-<format>``, except the
# SFT scoring's own ``final`` (random templates), which the SFT summaries supply.
MODELS = [
    ("A", "A: Qwen start (standard MA, 1,000 h x 5 languages)", "GOLDW-final-A"),
    ("SFT-EU24", "SFT-EU24: full-model fine-tune from A (reference)", SFT_EU24),
    ("CE24", "CE24: adapter only, gold CE (control)", "SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337"),
    ("SD24", "SD24: adapter only, instruction mix", "SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337"),
    ("SD-SIFT24", "SD-SIFT24: adapter only, AZeroS", "SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337"),
    ("A-EuroLLM", "A': EuroLLM start (standard MA, 1,000 h x 5 languages)", "EuroLLM-A"),
    (
        "SD24-EuroLLM",
        "SD24-EuroLLM: adapter only, instruction mix",
        "SD24-azeros-from-EuroLLM-A-mix025-2700h-bd64-8g-s1337",
    ),
    ("SD-SIFT24-EuroLLM", "SD-SIFT24-EuroLLM: adapter only, AZeroS", "SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337"),
]
NOISE = {"wer": 0.003, "chrf": 1.0}


def select_logs(logs_dir: str, since: str) -> dict[tuple[str, str, str], tuple[str, str]]:
    """``{(sft_run, sft_label, set): (path, model path)}``, the newest successful log of each."""
    newest: dict = {}
    for path in sorted(Path(logs_dir).glob("*.eval")):  # sorted by name = by start time: the later wins
        if path.name[: len(since)] < since:
            continue
        log = read_eval_log(str(path), header_only=True)
        meta = log.eval.metadata or {}
        if "sft_run" not in meta or log.status != "success":
            continue
        set_name = Path(str((log.eval.task_args or {}).get("frozen_set") or "")).name
        newest[(meta["sft_run"], meta.get("sft_label", ""), set_name)] = (str(path), str(log.eval.model))
    return newest


def sft_rows(sft_summaries: str) -> tuple[dict[tuple[str, str], dict], dict[str, dict[str, float]]]:
    """The SFT scoring's rows under the random templates: A at step 0 and SFT-EU24 final.

    Returns ``({(model key, set): row}, {model key: {group: share of en->X outputs in the requested language}})``;
    the shares come from ``langshare.json`` next to the summaries.
    """
    rows: dict = {}
    summaries = Path(sft_summaries)
    if not summaries.exists():
        return rows, {}
    key_of: dict[str, str] = {}
    for row in json.loads(summaries.read_text()):
        info = classify_model(row["model"])
        if info is None or row["status"] != "success":
            continue
        if info["cond"] is None and info["init"] == "goldw":
            key_of[row["model"]] = "A"
        elif info["cond"] == "EU24" and info["init"] == "goldw" and info["final"]:
            key_of[row["model"]] = "SFT-EU24"
        else:
            continue
        rows[(key_of[row["model"]], row["set"])] = row
    counts: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # key -> group -> [on, n]
    share_file = summaries.with_name("langshare.json")
    if share_file.exists():
        for row in json.loads(share_file.read_text())["rows"]:
            if row["set"] == SETS["enx"] and row["model"] in key_of:
                cell = counts[key_of[row["model"]]][row["group"]]
                cell[0] += row["on"]
                cell[1] += row["n"]
    return rows, {k: {g: on / n if n else None for g, (on, n) in groups.items()} for k, groups in counts.items()}


def headline(rows: dict[str, dict], shares: dict) -> dict:
    """The table cells of one model and format from its rows (``rows[set key]``) and the en->X language shares."""
    asr, xen, enx = rows.get("asr"), rows.get("xen"), rows.get("enx")
    cells = {
        "wer_hr": asr and asr["seen_wer"],
        "wer_other": asr and asr["unseen_wer"],
        "cer_other": asr and asr["unseen_cer"],
        "xen_hr": xen and xen["seen_chrf"],
        "xen_other": xen and xen["unseen_chrf"],
        "xen_copy_hr": xen and xen["seen_chrf_vs_source"],
        "enx_hr": enx and enx["seen_chrf"],
        "enx_other": enx and enx["unseen_chrf"],
    }
    cells["enx_on_hr"], cells["enx_on_other"] = shares.get("seen"), shares.get("unseen")
    return cells


def fmt(value: float | None, digits: int = 3, pct: bool = False) -> str:
    if value is None or value != value:
        return "-"
    return f"{100 * value:.0f}%" if pct else f"{value:.{digits}f}"


COLUMNS = [
    ("wer_hr", "WER HR5", 3, False),
    ("wer_other", "WER other-19", 3, False),
    ("cer_other", "CER other-19", 3, False),
    ("xen_hr", "X->en HR4", 1, False),
    ("xen_other", "X->en other", 1, False),
    ("enx_hr", "en->X HR4", 1, False),
    ("enx_other", "en->X other", 1, False),
    ("enx_on_hr", "in target lang. HR4", 0, True),
    ("enx_on_other", "in target lang. other", 0, True),
]


def markdown(table: dict[tuple[str, str], dict]) -> str:
    head = ["model", "prompts"] + [c[1] for c in COLUMNS]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for key, label, _ in MODELS:
        for fmt_name in FORMATS:
            cells = table.get((key, fmt_name))
            if cells is None:
                continue
            out.append("| " + " | ".join([key, fmt_name] + [fmt(cells.get(c), d, p) for c, _, d, p in COLUMNS]) + " |")
    return "\n".join(out)


def detail(rows_by_model: dict[tuple[str, str], dict[str, dict]]) -> str:
    """Per-language values: ASR (WER for the five trained languages, CER for the rest), X->en and en->X chrF."""
    blocks = []
    for (key, fmt_name), rows in rows_by_model.items():
        for set_key, value in (("asr", "wer/cer"), ("xen", "chrf"), ("enx", "chrf")):
            row = rows.get(set_key)
            if row is None:
                continue
            if set_key == "asr":
                cells = {lang: (v["wer"] if lang in SEEN else v["cer"]) for lang, v in row["lang"].items()}
            else:
                cells = {lang: v["chrf"] for lang, v in row["lang"].items()}
            digits = 3 if set_key == "asr" else 1
            blocks.append(
                f"{key} | {fmt_name} | {set_key} ({value}): "
                + " ".join(f"{k} {v:.{digits}f}" for k, v in sorted(cells.items()))
            )
    return "\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", default=DEFAULT_LOGS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--sft-summaries", default=DEFAULT_SFT_SUMMARIES)
    p.add_argument("--refs-dir", default=DEFAULT_REFS)
    p.add_argument("--since", default="2026-10-11", help="ignore logs started before this date (file name prefix)")
    p.add_argument("--detail", action="store_true", help="also print the per-language values")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = select_logs(args.logs_dir, args.since)
    from_sft, sft_shares = sft_rows(args.sft_summaries)
    norm, refs = BasicTextNormalizer(), load_refs(args.refs_dir, lowercase=True)
    rows_by_model, table = {}, {}
    for key, _, run in MODELS:
        for fmt_name in FORMATS:
            rows, shares = {}, {}
            for set_key, set_name in SETS.items():
                if set_key == "indomain":
                    continue
                hit = logs.get((run, f"fmt-{fmt_name}", set_name))
                if hit is not None:
                    rows[set_key] = summarize(hit[0], norm, lowercase=True)
                    if set_key == "enx":
                        pooled = defaultdict(lambda: defaultdict(new_cell))
                        count_log(
                            hit[0],
                            refs,
                            True,
                            defaultdict(lambda: defaultdict(new_cell)),
                            pooled,
                            defaultdict(new_cell),
                        )
                        for group, cell in next(iter(pooled.values())).items():
                            shares[group] = cell["on"] / cell["n"] if cell["n"] else None
                elif fmt_name == "sft-random" and (key, set_name) in from_sft:
                    rows[set_key] = from_sft[(key, set_name)]
                    if set_key == "enx":
                        shares = sft_shares.get(key, {})
            if rows:
                rows_by_model[(key, fmt_name)] = rows
                table[(key, fmt_name)] = headline(rows, shares)
    text = markdown(table)
    (out / "sd24_tables.md").write_text(text + "\n")
    (out / "sd24_summaries.json").write_text(
        json.dumps({f"{k}|{f}": v for (k, f), v in rows_by_model.items()}, indent=1)
    )
    print(text)
    print(f"\nnoise floor: about {NOISE['chrf']} chrF and {NOISE['wer']} WER (seed replicas of the 5-language runs)")
    if args.detail:
        print()
        print(detail(rows_by_model))
    missing = [f"{key}/{fmt_name}" for key, _, _ in MODELS for fmt_name in FORMATS if (key, fmt_name) not in table]
    if missing:
        print("\nnot scored yet: " + ", ".join(missing))


if __name__ == "__main__":
    main()
