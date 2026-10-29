#!/usr/bin/env python3
"""EU24 comparison report: FLEURS test WER/CER, chrF++ and spBLEU by resource tier, and the selection metric.

    python eu24_report.py --out-dir DIR [--logs-dir DIR] [--label eu24-repeatfirst] [--since 2026-10-15T17]
                          [--models A SFT-EU24 ...] [--asr-set S] [--xen-set S] [--enx-set S]

Reads melt-eval ``.eval`` logs (selected by the metadata ``sft_run`` / ``sft_label`` the launcher sets, newest
successful log per run and set) and writes ``eu24_report.md``, ``eu24_report.json`` and ``eu24_per_language.csv``
into DIR. Nothing is recomputed from audio.

* ASR: per-language WER and CER as melt-eval logs them (``wer_<lang>`` / ``cer_<lang>``, normaliser ``basic``,
  the completion as it stands).
* ST: BLEU, spBLEU, chrF and chrF++ per direction from ``melteval.st_metrics`` (PR #28), cased and lowercased.
  chrF++ is word order 2, spBLEU is BLEU over the flores200 SentencePiece model, chrF is word order 0.
* The selection metric: ``projects/ablation-campaign/selection-metric/score.py`` is imported and applied to the CER of
  the six sets it names (never re-implemented here). A variant that takes the three FLEURS groups from the FLEURS
  test ASR log instead of FLEURS dev is added to the JSON only.
* Tiers: by hours of transcribed audio held in the MELT pools (``TIERS`` below). A group's number is the unweighted
  mean over its languages; a group with a missing language is reported as missing, never averaged over fewer.

Run in the melt-eval venv with PYTHONPATH holding the melt-eval checkout of PR #28 (``melteval.st_metrics``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log


SCORE_PY = str(Path(__file__).resolve().parents[2] / "ablation-campaign" / "selection-metric" / "score.py")
DEFAULT_LOGS = "/mnt/scratch-artemis/giuseppe/melt-data/eval-logs"

# Hours of transcribed audio per language in the MELT training pools (CV22, MLS, VoxPopuli, Granary, FLEURS train),
# as recorded per ASR entry in projects/self-distill/sft/SFT-EU24.yaml ("capped, of N h" / "all of its data").
HOURS = {
    "en": 144001.2, "es": 28200.0, "fr": 14370.7, "de": 11128.2, "it": 6734.1, "nl": 3028.3, "pt": 2100.9,
    "pl": 930.7, "hu": 273.8, "fi": 193.7, "cs": 160.3, "ro": 130.6, "sv": 110.3,
    "sk": 54.2, "bg": 35.2, "lv": 30.5, "da": 29.0, "lt": 28.4, "el": 27.0, "et": 24.3, "hr": 21.2, "sl": 15.3,
    "ga": 12.7, "mt": 12.3,
}  # fmt: skip
TIER_NAMES = ("high", "mid", "low")
TIERS = {
    "high": [lang for lang, h in HOURS.items() if h >= 1000],
    "mid": [lang for lang, h in HOURS.items() if 100 <= h < 1000],
    "low": [lang for lang, h in HOURS.items() if h < 100],
}
EU24 = [lang for tier in TIER_NAMES for lang in TIERS[tier]]
HR5 = ["en", "de", "es", "fr", "it"]
TIER_RULE = "high: >= 1,000 h of transcribed audio in the MELT pools; mid: 100-1,000 h; low: < 100 h"

# (key, row label, melt-eval metadata sft_run)
MODELS = [
    ("A", "A (Qwen start, standard training)", "GOLDW-final-A"),
    ("SFT-EU24", "SFT-EU24 (full fine-tune, reference)", "SFT-EU24-from-goldw-bd60-ga4-8g-s1337"),
    ("CE24", "CE24 (adapter, gold CE, control)", "SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337"),
    ("SD24", "SD24 (adapter, instruction-mixed SD)", "SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337"),
    ("SD-SIFT24", "SD-SIFT24 (adapter, AZeroS recipe)", "SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337"),
    ("A-EuroLLM", "A' (EuroLLM start, standard training)", "EuroLLM-A"),
    ("SD24-EuroLLM", "SD24 on EuroLLM", "SD24-azeros-from-EuroLLM-A-mix025-2700h-bd64-8g-s1337"),
    ("SD-SIFT24-EuroLLM", "SD-SIFT24 on EuroLLM", "SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337"),
]
ST_METRICS = ("chrfpp", "spbleu", "bleu", "chrf")
METRIC_LABEL = {"chrfpp": "chrF++", "spbleu": "spBLEU", "bleu": "BLEU (13a)", "chrf": "chrF"}


def load_selection_module():
    spec = importlib.util.spec_from_file_location("selection_score", SCORE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover(logs_dir: str, label: str, since: str) -> dict[tuple[str, str], Path]:
    """Newest successful log per (sft_run, frozen set name) carrying ``sft_label == label``."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(Path(logs_dir).glob("*.eval")):
        if path.name < since:
            continue
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception as exc:  # a log a running job is still writing is not a complete zip yet
            print(f"skipping unreadable log {path.name}: {type(exc).__name__}: {exc}")
            continue
        meta = log.eval.metadata or {}
        if meta.get("sft_label") != label or log.status != "success":
            continue
        frozen = Path(str((log.eval.task_args or {}).get("frozen_set") or "")).name
        found[(str(meta.get("sft_run")), frozen)] = path
    return found


def asr_numbers(path: Path) -> dict:
    log = read_eval_log(str(path), header_only=True)
    metrics = log.results.scores[0].metrics
    return {
        "wer": {k[4:]: v.value for k, v in metrics.items() if k.startswith("wer_")},
        "cer": {k[4:]: v.value for k, v in metrics.items() if k.startswith("cer_")},
        "pooled_wer": metrics["corpus_wer_raw"].value,
        "pooled_cer": metrics["corpus_cer"].value,
        "samples": log.results.completed_samples,
        "log": path.name,
    }


def st_numbers(path: Path, direction: str) -> dict:
    """Per-direction metrics, cased and lowercased: ``{case: {lang: {metric: value}}}`` (lang = the non-English side)."""
    from melteval.st_metrics import score_log

    out: dict = {"log": path.name}
    # Casing habit of the model: the share of translations written without a single capital letter.
    completions = [(s.output.completion if s.output else "") for s in (read_eval_log(str(path)).samples or [])]
    out["no_capitals_share"] = sum(c == c.lower() for c in completions) / max(1, len(completions))
    for case, lowercase in (("cased", False), ("lower", True)):
        result = score_log(path, group_by="dataset_id", lowercase=lowercase)
        per_lang: dict[str, dict] = {}
        for group in result["groups"]:
            match = re.fullmatch(r"fleurs-([a-z]{2,3})_([a-z]{2,3})", group["group"])
            if not match:
                raise ValueError(f"{path.name}: unexpected dataset_id {group['group']!r}")
            lang = match.group(1) if direction == "xen" else match.group(2)
            per_lang[lang] = {m: group[m] for m in ST_METRICS} | {"n": group["n"], "n_empty": group["n_empty"]}
        out[case] = per_lang
        out[f"{case}_signatures"] = result["signatures"]
        out["n_scored"] = result["n_scored"]
        out["n_errors"] = result["n_errors"]
    return out


def selection_inputs(found: dict, run: str, sel, test_asr: dict | None):
    """CER dicts for the six selection sets of *run*, or ``None`` when one of them is missing."""
    sets: dict[str, dict[str, float]] = {}
    samples: dict[str, list] = {}
    needed = [sel.FLEURS_SET] + [sel.ID_SET_TEMPLATE.format(lang=lang) for lang in sel.TRAIN_LANGS]
    for name in needed:
        path = found.get((run, name))
        if path is None:
            return None
        numbers = asr_numbers(path)
        sets[name] = numbers["cer"]
        log = read_eval_log(str(path))
        samples[name] = [
            {"scores": {k: {"value": v.value} for k, v in (s.scores or {}).items()}, "metadata": s.metadata}
            for s in (log.samples or [])
        ]
    summaries = {name: {"cer": sets[name], "runaway": sel.runaway_stats(samples[name])} for name in needed}
    return sets, summaries


def mean_over(values: dict[str, float | None], langs: list[str]) -> float | None:
    picked = [values.get(lang) for lang in langs]
    if any(v is None for v in picked):
        return None
    return statistics.fmean(picked)


def tier_table_values(values: dict[str, float | None], langs: list[str]) -> dict[str, float | None]:
    """All / high / mid / low means of *values* over *langs* (the languages the direction covers)."""
    out = {"all": mean_over(values, langs)}
    for tier in TIER_NAMES:
        out[tier] = mean_over(values, [lang for lang in TIERS[tier] if lang in langs])
    return out


def fmt(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def render(report: dict, models: list[tuple[str, str, str]]) -> str:
    lines: list[str] = []
    add = lines.append
    meta = report["meta"]
    split = meta.get("split", "test")
    add(f"# EU24 results on FLEURS {split}")
    add("")
    label_note = ", ".join(f"{k}: `{v}`" for k, v in meta.get("label_map", {}).items())
    add(
        f"Generated {meta['generated']}. Prompt format `{meta['label']}`"
        + (f" ({label_note})" if label_note else "")
        + f". Eval code: melt-eval PR #28 @ `{meta['eval_code']}`. Sets: {meta['sets_note']}. {meta.get('sizes_note', '')}"
    )
    add("")
    add("## Language tiers")
    add("")
    add(
        f"Rule: {TIER_RULE}. Hours are the transcribed audio held per language in the MELT pools (CV22, MLS, VoxPopuli, "
        "Granary, FLEURS train), the number SFT-EU24's data mix was built from."
    )
    add("")
    for tier in TIER_NAMES:
        entries = ", ".join(
            f"{lang} ({HOURS[lang]:,.0f} h)" if HOURS[lang] >= 100 else f"{lang} ({HOURS[lang]:.0f} h)"
            for lang in TIERS[tier]
        )
        add(f"- **{tier}** ({len(TIERS[tier])}): {entries}")
    add("")

    have = [m for m in models if m[0] in report["models"]]

    def header(first: str, columns: list[str]) -> list[str]:
        return [f"| {first} | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]

    # --- WER per language ---------------------------------------------------------------------------------------
    add(f"## 1. WER on FLEURS {split}, per language (%)")
    add("")
    lines += header("lang (tier)", [m[0] for m in have])
    for tier in TIER_NAMES:
        for lang in TIERS[tier]:
            cells = [fmt(report["models"][m[0]].get("asr", {}).get("wer", {}).get(lang), 100) for m in have]
            add(f"| {lang} ({tier}) | " + " | ".join(cells) + " |")
    add("")

    # --- unified metric -----------------------------------------------------------------------------------------
    has_selection = any(report["models"][m[0]].get("selection") for m in have)
    add("## 2. Selection metric (the campaign's single ranking score, lower is better)")
    add("")
    if not has_selection:
        add(
            "Not computed for this table: the metric is defined on its own frozen sets (in-domain held-out CV22 / MLS / "
            "VoxPopuli for the five trained languages, and FLEURS dev with 200 utterances per language), not on the FLEURS-24 "
            "sets used here. Its numbers follow separately."
        )
        add("")
    if has_selection:
        with_variant = any(
            report["models"][m[0]].get("selection", {}).get("score_fleurs_test_variant") is not None for m in have
        )
        add(
            "Weighted median of per-language CER clipped at 1: (1.0 ID + 1.5 OOD-train + 0.6 OOD-related) / 3.1. ID = the five "
            "trained languages on held-out CV22 / MLS / VoxPopuli (600 utterances each); the OOD groups are FLEURS dev, 200 "
            "utterances per language (ru and uk included in the reported-only groups; the models were not trained on them)."
            + (
                " The last column replaces FLEURS dev by the FLEURS test log for the weighted groups (not the official definition)."
                if with_variant
                else ""
            )
        )
        add("")
        columns = ["score", "ID", "OOD-train", "OOD-related", "OOD-latin*", "OOD-script*", "runaways ID / FLEURS"]
        lines += header("model", columns + (["score, FLEURS-test variant"] if with_variant else []))
        for key, label, _ in have:
            s_ = report["models"][key].get("selection")
            if not s_:
                add(f"| {key} | " + " | ".join("-" for _ in columns + (["v"] if with_variant else [])) + " |")
                continue
            med = s_["medians"]
            rw = s_.get("runaway")
            rw_text = (
                "n/a"
                if rw is None
                else f"{rw['ID']['runaways']}/{rw['ID']['samples']} / {rw['FLEURS']['runaways']}/{rw['FLEURS']['samples']}"
            )
            row = (
                f"| {key} | **{s_['score']:.3f}** | {med['ID']:.3f} | {med['OOD-train']:.3f} | {med['OOD-related']:.3f} | "
                f"{med['OOD-latin']:.3f} | {med['OOD-script']:.3f} | {rw_text} |"
            )
            if with_variant:
                row += f" {fmt(s_.get('score_fleurs_test_variant'), 1, 3)} |"
            add(row)
        add("")
        add("*reported only, not in the score.")
        add("")

    # --- WER / CER by tier --------------------------------------------------------------------------------------
    add(f"## 3. WER and CER on FLEURS {split} by resource tier (%, macro-average over languages)")
    add("")
    cols = ["WER all-24", "high", "mid", "low", "trained-5", "CER all-24", "high", "mid", "low", "trained-5"]
    lines += header("model", cols)
    for key, label, _ in have:
        asr = report["models"][key].get("asr")
        if not asr:
            add(f"| {key} | " + " | ".join("-" for _ in cols) + " |")
            continue
        cells = []
        for metric in ("wer", "cer"):
            t = asr["tiers"][metric]
            cells += [
                fmt(t["all"], 100),
                fmt(t["high"], 100),
                fmt(t["mid"], 100),
                fmt(t["low"], 100),
                fmt(t["hr5"], 100),
            ]
        add(f"| {key} | " + " | ".join(cells) + " |")
    add("")

    # --- ST by tier ---------------------------------------------------------------------------------------------
    add(f"## 4. Translation: chrF++ and spBLEU on FLEURS {split} by resource tier (macro-average over directions)")
    add("")
    add(
        "chrF++ = character 6-grams plus word 1- and 2-grams (sacrebleu `nc:6|nw:2`); spBLEU = BLEU over the flores200 "
        "SentencePiece model. **Lowercased scoring is the primary one**: the standard-training models (A, SFT-EU24, CE24) "
        "write their translations without any capital letter (share below), a habit inherited from their training targets, so "
        "cased scoring would measure that habit instead of the translation. SD models inherit the decoder's natural casing. "
        "The cased scores follow for reference."
    )
    add("")
    lines += header("share of translations with no capital letter", ["X→en", "en→X"])
    for key, label, _ in have:
        m = report["models"][key]
        cells = [
            fmt((m.get(d) or {}).get("no_capitals_share"), 100, 0)
            + ("%" if (m.get(d) or {}).get("no_capitals_share") is not None else "")
            for d in ("xen", "enx")
        ]
        add(f"| {key} | " + " | ".join(cells) + " |")
    add("")
    panel = 0
    for case, case_label in (("lower", "lowercased (primary)"), ("cased", "cased")):
        for metric in ("chrfpp", "spbleu"):
            panel += 1
            add(f"### 4.{panel} {METRIC_LABEL[metric]}, {case_label}")
            add("")
            cols = ["X→en all-23", "high", "mid", "low", "en→X all-23", "high", "mid", "low"]
            lines += header("model", cols)
            for key, label, _ in have:
                m = report["models"][key]
                cells = []
                for direction in ("xen", "enx"):
                    t = (m.get(direction) or {}).get("tiers", {}).get(case, {}).get(metric)
                    cells += [fmt(t and t["all"]), fmt(t and t["high"]), fmt(t and t["mid"]), fmt(t and t["low"])]
                add(f"| {key} | " + " | ".join(cells) + " |")
            add("")

    # --- ST per language ----------------------------------------------------------------------------------------
    add("## 5. Translation per language (lowercased)")
    add("")
    for direction, title in (("xen", "X→en"), ("enx", "en→X")):
        for metric in ("chrfpp", "spbleu"):
            add(f"### {METRIC_LABEL[metric]}, {title}")
            add("")
            lines += header(
                "lang (tier)",
                [m[0] for m in have if m[0] in report["models"] and report["models"][m[0]].get(direction)],
            )
            shown = [m for m in have if report["models"][m[0]].get(direction)]
            for tier in TIER_NAMES:
                for lang in TIERS[tier]:
                    if lang == "en":
                        continue
                    cells = [fmt(report["models"][m[0]][direction]["lower"].get(lang, {}).get(metric)) for m in shown]
                    add(f"| {lang} ({tier}) | " + " | ".join(cells) + " |")
            add("")

    add("## Provenance")
    add("")
    for key, label, run in have:
        m = report["models"][key]
        logs = {k: v["log"] for k, v in m.items() if isinstance(v, dict) and "log" in v}
        add(f"- **{key}** (`{run}`): " + ", ".join(f"{k}: `{v}`" for k, v in logs.items()))
    sig = meta.get("signatures")
    if sig:
        add("")
        add("sacrebleu signatures: " + "; ".join(f"{k}: `{v}`" for k, v in sig.items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS)
    parser.add_argument("--label", default="eu24-repeatfirst")
    parser.add_argument("--since", default="2026-10-15T17")
    parser.add_argument("--models", nargs="*", default=[m[0] for m in MODELS])
    parser.add_argument("--asr-set", default="fleurs24-asr-test")
    parser.add_argument("--xen-set", default="fleurs24-st-xen-test")
    parser.add_argument("--enx-set", default="fleurs24-st-enx-test")
    parser.add_argument("--eval-code", default="cf6f9e8")
    parser.add_argument("--split", default="test", help="split name used in the headings (dev or test)")
    parser.add_argument("--sizes-note", default="", help="free text about set sizes, shown under the title")
    parser.add_argument("--label-map", nargs="*", default=[], help="KEY=LABEL overrides of --label for single models")
    parser.add_argument(
        "--selection-label", default=None, help="sft_label of the selection-metric logs (default: --label)"
    )
    args = parser.parse_args()
    label_map = dict(item.split("=", 1) for item in args.label_map)

    sel = load_selection_module()
    models = [m for m in MODELS if m[0] in args.models]
    found: dict[tuple[str, str], Path] = {}
    for label in {args.label, *label_map.values()}:
        by_label = discover(args.logs_dir, label, args.since)
        for (run, frozen), path in by_label.items():
            owner = next((m[0] for m in MODELS if m[2] == run), None)
            if label == label_map.get(owner, args.label):
                found[(run, frozen)] = path
    if args.selection_label and args.selection_label != args.label:
        for (run, frozen), path in discover(args.logs_dir, args.selection_label, args.since).items():
            if frozen.startswith("selection-"):
                found[(run, frozen)] = path
    print(
        f"{len(found)} logs found for label {args.label!r} (overrides: {label_map or 'none'}; selection label {args.selection_label or args.label!r})"
    )

    report: dict = {"meta": {}, "models": {}}
    signatures: dict[str, str] = {}
    for key, _, run in models:
        entry: dict = {}
        if (run, args.asr_set) in found:
            asr = asr_numbers(found[(run, args.asr_set)])
            asr["tiers"] = {}
            for metric in ("wer", "cer"):
                t = tier_table_values(asr[metric], EU24)
                t["hr5"] = mean_over(asr[metric], HR5)
                asr["tiers"][metric] = t
            entry["asr"] = asr
        for direction, set_name in (("xen", args.xen_set), ("enx", args.enx_set)):
            if (run, set_name) not in found:
                continue
            st = st_numbers(found[(run, set_name)], direction)
            langs = [lang for lang in EU24 if lang != "en"]
            st["tiers"] = {
                case: {
                    metric: tier_table_values({lang: st[case].get(lang, {}).get(metric) for lang in langs}, langs)
                    for metric in ST_METRICS
                }
                for case in ("cased", "lower")
            }
            entry[direction] = st
            signatures.update(
                {f"{case}-{k}": v[0] for case in ("cased", "lower") for k, v in st[f"{case}_signatures"].items() if v}
            )
        inputs = selection_inputs(found, run, sel, entry.get("asr"))
        if inputs is not None:
            sets, summaries = inputs
            result = sel.score_checkpoint(sets)
            selection = {
                "score": result["score"],
                "medians": result["medians"],
                "languages": {g: {l: v for l, (v, _) in d.items()} for g, d in result["languages"].items()},
            }
            selection["runaway"] = sel.runaway_report(summaries)
            if "asr" in entry and args.split == "test":
                variant_sets = dict(sets)
                variant_sets[sel.FLEURS_SET] = entry["asr"]["cer"]
                try:
                    selection["score_fleurs_test_variant"] = sel.score_checkpoint(variant_sets)["score"]
                except sel.ScoringError:
                    selection["score_fleurs_test_variant"] = None
            entry["selection"] = selection
        if entry:
            report["models"][key] = entry

    report["meta"] = {
        "generated": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        "label": args.label,
        "label_map": label_map,
        "split": args.split,
        "sizes_note": args.sizes_note,
        "eval_code": args.eval_code,
        "sets_note": f"{args.asr_set}, {args.xen_set}, {args.enx_set}",
        "signatures": signatures,
        "tiers": TIERS,
        "hours": HOURS,
        "tier_rule": TIER_RULE,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eu24_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out / "eu24_report.md").write_text(render(report, models))
    with (out / "eu24_per_language.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task", "case", "lang", "tier", "metric", "value"])
        tier_of = {lang: tier for tier in TIER_NAMES for lang in TIERS[tier]}
        for key, entry in report["models"].items():
            for metric in ("wer", "cer"):
                for lang, v in entry.get("asr", {}).get(metric, {}).items():
                    writer.writerow([key, "asr", "normalised", lang, tier_of.get(lang, "other"), metric, v])
            for direction in ("xen", "enx"):
                for case in ("cased", "lower"):
                    for lang, d in entry.get(direction, {}).get(case, {}).items():
                        for metric in ST_METRICS:
                            writer.writerow(
                                [key, direction, case, lang, tier_of.get(lang, "other"), metric, d[metric]]
                            )
    print(f"wrote {out}/eu24_report.md, .json, eu24_per_language.csv for models: {', '.join(report['models'])}")


if __name__ == "__main__":
    main()
