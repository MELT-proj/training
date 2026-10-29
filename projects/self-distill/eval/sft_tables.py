"""A-against-B tables for the SFT comparison, per data condition and scored step.

    python sft_tables.py summaries.json langshare.json [--json tables.json] [--markdown tables.md]

A is the run that starts from GOLDW-final (standard alignment), B the run that starts from
S3-azeros_anchor (self-distilled alignment); the two differ only in the adapter they start from. Inputs
are the files ``collect_sft_scores.py`` writes: ``summarize_evals.py --lowercase`` rows and the
``lang_check.py`` counts. Step 0 (both starts before fine-tuning) is the first row of every
condition. The x-axis is the optimizer step; each step takes 1,920 s of audio nominal, about 85% of
it filled, so the "audio h" column is an upper bound.

"HR" is the five high-resource languages en, de, fr, es, it (de, fr, es, it for the translation
directions that have English on one side): the only languages the alignment saw and the only ones the
HR5 condition fine-tunes on. "Other" is the remaining 19 (23 minus 4 for translation). In HR5 that is
exactly trained against untrained; in EU24 the other languages have training data too, so for EU24 each
translation family gets one more table that splits by whether the condition's mix (``../sft/SFT-<cond>.yaml``)
holds data for the language or direction.

Per step and per condition:

- ASR: WER over the five HR languages and CER over the other 19;
- X->en chrF: HR sources (de, fr, es, it) and the other 19;
- en->X chrF: HR targets (de, fr, es, it) and the other 19, with the share of outputs in the
  requested language (``lang_check.py``);
- in-domain ASR WER (``sd-dev-per-source``, scored at step 0 and for the final model only);
- the B-A delta of every language, and a sign test across languages: in how many B is better, in how
  many A is better (ties dropped), and the two-sided exact binomial p-value.

ST is scored case-insensitively (SFT trains on lowercased targets); ASR WER and CER normalise case anyway.
A "*" on a delta marks one above twice the noise floor measured between distillation replicas (1 chrF, 0.003 WER).
Standard library only.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


SEEN = ("en", "de", "fr", "es", "it")
SECONDS_PER_STEP = 1920
FINAL_STEP = 11499
NOISE = {"wer": 0.003, "chrf": 1.0}
SETS = {
    "asr": "fleurs24-asr-dev",
    "xen": "fleurs24-st-xen-dev",
    "enx": "fleurs24-st-enx-dev",
    "indomain": "sd-dev-per-source",
}
MIX_DIR = Path(__file__).resolve().parents[1] / "sft"
STEP0_MODELS = {
    "goldw": "GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g",
    "sd-azeros": "S3-azeros_anchor-from-GOLDW-final-mix05-bd30-lr2e4-3000st-s43-2g",
}
RUN_RE = re.compile(r"^SFT-(?P<cond>HR5|EU24)-from-(?P<init>goldw|sd-azeros)-.*-s(?P<seed>\d+)$")


def classify_model(model: str) -> dict | None:
    """``{cond, init, seed, step}`` for an SFT run's checkpoint or final model, ``cond`` None for a step-0 start,
    or None for any other model. The final model has ``step`` 11499 and ``final`` True."""
    path = Path(model.split("//", 1)[-1])
    is_checkpoint = path.name.startswith("checkpoint-")
    run = path.parent.name if is_checkpoint else path.name
    for init, name in STEP0_MODELS.items():
        if run == name and not is_checkpoint:
            return {"cond": None, "init": init, "seed": None, "step": 0, "final": False}
    match = RUN_RE.match(run)
    if not match:
        return None
    step = int(path.name.split("-", 1)[1]) if is_checkpoint else FINAL_STEP
    return {
        "cond": match["cond"],
        "init": match["init"],
        "seed": int(match["seed"]),
        "step": step,
        "final": not is_checkpoint,
    }


# --- pulling numbers out of the summaries -------------------------------------------------------------------


def mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def per_language(row: dict, key: str) -> dict[str, float]:
    """``{language: value}`` of ``key`` ("wer", "cer", "chrf") from a summary row."""
    return {lang: v[key] for lang, v in row["lang"].items()}


def asr_values(row: dict) -> dict[str, float]:
    """WER for the trained languages, CER for the others: the quantity compared per language."""
    return {lang: (v["wer"] if lang in SEEN else v["cer"]) for lang, v in row["lang"].items()}


def share_values(counts: dict[str, dict]) -> dict[str, float]:
    """``{target: share of outputs in the requested language}`` from the ``lang_check.py`` rows of one model."""
    return {key: c["on"] / c["n"] for key, c in counts.items() if c["n"]}


def group_mean(values: dict[str, float], trained: bool) -> float:
    """Mean over the HR languages (``trained`` True) or the others."""
    return mean([v for lang, v in values.items() if (lang in SEEN) == trained])


ENTRY_RE = re.compile(r"^\s*# (?:ASR (\w+)|ST (\w+)->(\w+)): [\d,.]+ h", re.MULTILINE)


def training_data_languages(cond: str, mix_dir: Path = MIX_DIR) -> dict[str, set[str]] | None:
    """Languages with training data in a condition's mix: ASR languages, X->en sources, en->X targets.

    Read from the entry comments ``build_sft_mix.py`` renders into ``SFT-<cond>.yaml`` (``# ASR de: 197.7 h ...``,
    ``# ST de->en: ...``, ``# ST en->de: ...``); None when the file is not there.
    """
    path = mix_dir / f"SFT-{cond}.yaml"
    if not path.is_file():
        return None
    languages: dict[str, set[str]] = {"asr": set(), "xen": set(), "enx": set()}
    for asr, src, tgt in ENTRY_RE.findall(path.read_text()):
        if asr:
            languages["asr"].add(asr)
        elif tgt == "en":
            languages["xen"].add(src)
        elif src == "en":
            languages["enx"].add(tgt)
    return languages


def split_means(values: dict[str, float] | None, with_data: set[str]) -> tuple[float | None, float | None]:
    """Mean over the languages in ``with_data`` and over the rest; None for a model without values."""
    if not values:
        return None, None
    return (
        mean([v for lang, v in values.items() if lang in with_data]),
        mean([v for lang, v in values.items() if lang not in with_data]),
    )


# --- loading ------------------------------------------------------------------------------------------------


def load(summaries_path: str, langshare_path: str | None) -> dict:
    """``data[(cond, seed)][step][init] = {"asr": row, "xen": row, "enx": row, "share": {target: share}}``.

    Step-0 rows are shared by every condition: they sit under ``data[(None, None)][0]``.
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in json.loads(Path(summaries_path).read_text()):
        info = classify_model(row["model"])
        if info is None or row["status"] != "success":
            continue
        for name, set_name in SETS.items():
            if row["set"] == set_name:
                data[(info["cond"], info["seed"])][info["step"]][info["init"]][name] = row
    if langshare_path:
        counts: dict = defaultdict(lambda: defaultdict(dict))
        for row in json.loads(Path(langshare_path).read_text())["rows"]:
            if row["set"] == SETS["enx"]:
                counts[row["model"]][row["key"]] = row
        for model, cells in counts.items():
            info = classify_model(model)
            if info is not None:
                data[(info["cond"], info["seed"])][info["step"]][info["init"]]["share"] = share_values(cells)
    return data


# --- statistics ---------------------------------------------------------------------------------------------


def sign_test(delta: dict[str, float], higher_is_better: bool) -> dict:
    """B better / A better / ties over languages, and the two-sided exact binomial p-value (ties dropped)."""
    better = sum(1 for d in delta.values() if d == d and (d > 0) == higher_is_better and d != 0)
    worse = sum(1 for d in delta.values() if d == d and (d < 0) == higher_is_better and d != 0)
    ties = sum(1 for d in delta.values() if d == d and d == 0)
    n = better + worse
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2**n)
    return {"b_better": better, "a_better": worse, "ties": ties, "n": n, "p": p}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """B - A per language, over the languages both have."""
    return {lang: b[lang] - a[lang] for lang in a if lang in b}


# --- the tables ----------------------------------------------------------------------------------------------


def fmt(value: float | None, digits: int) -> str:
    return "-" if value is None or value != value else f"{value:.{digits}f}"


def fmt_delta(a: float | None, b: float | None, digits: int, noise: float | None = None) -> str:
    if a is None or b is None or a != a or b != b:
        return "-"
    d = b - a
    star = "*" if noise is not None and abs(d) > 2 * noise else ""
    return f"{d:+.{digits}f}{star}"


def fmt_sign(test: dict | None) -> str:
    if test is None or test["n"] + test["ties"] == 0:
        return "-"
    p = "<.001" if test["p"] < 0.001 else f"={test['p']:.2f}".replace("=0.", "=.")
    return f"{test['b_better']}/{test['a_better']}, p{p}" + (f" ({test['ties']} tied)" if test["ties"] else "")


def step_label(step: int) -> str:
    return "0" if step == 0 else ("final" if step == FINAL_STEP else str(step))


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_block(data: dict, cond: str | None, seed: int | None, mix_dir: Path = MIX_DIR) -> dict:
    """All numbers of one condition: per step, the metrics of A and B, their deltas and the sign tests."""
    steps = sorted(set(data[(None, None)]) | set(data[(cond, seed)]))
    mix = training_data_languages(cond, mix_dir) if cond else None
    block: dict = {
        "cond": cond,
        "seed": seed,
        "steps": [],
        "training_data": {k: sorted(v) for k, v in mix.items()} if mix else None,
    }
    for step in steps:
        source = data[(None, None)] if step == 0 else data[(cond, seed)]
        cells = source.get(step, {})
        entry: dict = {"step": step, "audio_hours_nominal": step * SECONDS_PER_STEP / 3600}
        per_lang: dict = {}
        for fam in ("asr", "xen", "enx", "share"):
            vals = {}
            for init in ("goldw", "sd-azeros"):
                cell = cells.get(init, {}).get(fam)
                if cell is None:
                    vals[init] = None
                elif fam == "asr":
                    vals[init] = asr_values(cell)
                elif fam == "share":
                    vals[init] = cell
                else:
                    vals[init] = per_language(cell, "chrf")
            per_lang[fam] = vals
        entry["per_language"] = per_lang
        # summary numbers (A = goldw, B = sd-azeros)
        summary = {}
        for init, label in (("goldw", "A"), ("sd-azeros", "B")):
            asr = cells.get(init, {}).get("asr")
            indomain = cells.get(init, {}).get("indomain")
            summary[label] = {
                "wer5": mean([v["wer"] for lang, v in asr["lang"].items() if lang in SEEN]) if asr else None,
                "cer19": mean([v["cer"] for lang, v in asr["lang"].items() if lang not in SEEN]) if asr else None,
                "indomain_wer": indomain["seen_wer"] if indomain else None,
                "xen_hr": group_mean(per_lang["xen"][init], True) if per_lang["xen"][init] else None,
                "xen_other": group_mean(per_lang["xen"][init], False) if per_lang["xen"][init] else None,
                "enx_hr": group_mean(per_lang["enx"][init], True) if per_lang["enx"][init] else None,
                "enx_other": group_mean(per_lang["enx"][init], False) if per_lang["enx"][init] else None,
                "share_hr": group_mean(per_lang["share"][init], True) if per_lang["share"][init] else None,
                "share_other": group_mean(per_lang["share"][init], False) if per_lang["share"][init] else None,
            }
            if mix:  # the same families split by whether this condition's mix has data for the language or direction
                for fam, key in (("xen", "xen"), ("enx", "enx"), ("share", "enx")):
                    with_data, without = split_means(per_lang[fam][init], set(mix[key]))
                    summary[label][f"{fam}_with_data"], summary[label][f"{fam}_without_data"] = with_data, without
        entry["summary"] = summary
        # per-language deltas and sign tests
        entry["delta"], entry["sign"] = {}, {}
        for fam, higher in (("asr", False), ("xen", True), ("enx", True), ("share", True)):
            a, b = per_lang[fam]["goldw"], per_lang[fam]["sd-azeros"]
            if a is None or b is None:
                continue
            d = delta(a, b)
            entry["delta"][fam] = d
            entry["sign"][fam] = {
                "all": sign_test(d, higher),
                "hr": sign_test({k: v for k, v in d.items() if k in SEEN}, higher),
                "other": sign_test({k: v for k, v in d.items() if k not in SEEN}, higher),
            }
            if mix and fam != "asr":
                key = "xen" if fam == "xen" else "enx"
                entry["sign"][fam]["with_data"] = sign_test({k: v for k, v in d.items() if k in mix[key]}, higher)
                entry["sign"][fam]["without_data"] = sign_test(
                    {k: v for k, v in d.items() if k not in mix[key]}, higher
                )
        block["steps"].append(entry)
    return block


FAMILIES = {
    "asr": "ASR (WER for the 5 HR languages, CER for the other 19)",
    "xen": "X->en chrF",
    "enx": "en->X chrF",
    "share": "en->X share of outputs in the requested language",
}


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def differs_from_hr(block: dict, key: str) -> bool:
    """True when the condition's training data for ``key`` ("xen" or "enx") is not just the HR languages."""
    mix = block.get("training_data")
    return bool(mix) and set(mix[key]) != {lang for lang in SEEN if lang != "en"}


def render_block(block: dict) -> str:
    where = "Step 0, before fine-tuning" if block["cond"] is None else f"{block['cond']}, seed {block['seed']}"
    out = [f"## {where}: A = starts from GOLDW-final, B = starts from S3-azeros_anchor", ""]
    out.append(
        "x = optimizer step; one step is 1,920 s of audio nominal (about 85% filled). "
        "FLEURS dev, 100 per language or direction; ST scored case-insensitively. HR = en, de, fr, es, it. "
        "`*` marks a delta above twice the noise floor (1 chrF, 0.003 WER)."
    )
    steps = block["steps"]

    def row(entry: dict, cells: list[str]) -> list[str]:
        return [step_label(entry["step"]), f"{entry['audio_hours_nominal']:.0f}", *cells]

    def pair_table(title: str, columns: list[tuple[str, str, int, float | None]]) -> list[str]:
        """Rows of A, B and B-A for each ``(label, summary key, digits, noise)`` column group."""
        header = ["step", "audio h"]
        for label, _, _, _ in columns:
            header += [f"A {label}", f"B {label}", "B-A"]
        rows = []
        for e in steps:
            a, b = e["summary"]["A"], e["summary"]["B"]
            cells = []
            for _, key, digits, noise in columns:
                cells += [fmt(a[key], digits), fmt(b[key], digits), fmt_delta(a[key], b[key], digits, noise)]
            rows.append(row(e, cells))
        return ["", title, "", markdown_table(header, rows)]

    out += pair_table(
        "### ASR: WER on the 5 HR languages, CER on the other 19",
        [("WER", "wer5", 3, NOISE["wer"]), ("CER", "cer19", 3, None)],
    )
    indomain_steps = [
        e
        for e in steps
        if e["summary"]["A"]["indomain_wer"] is not None or e["summary"]["B"]["indomain_wer"] is not None
    ]
    if indomain_steps:
        rows = []
        for e in indomain_steps:
            a, b = e["summary"]["A"]["indomain_wer"], e["summary"]["B"]["indomain_wer"]
            rows.append(row(e, [fmt(a, 3), fmt(b, 3), fmt_delta(a, b, 3, NOISE["wer"])]))
        title = "### In-domain ASR WER (CV22, MLS, VoxPopuli dev; the 5 HR languages; step 0 and the final model only)"
        out += ["", title, "", markdown_table(["step", "audio h", "A WER", "B WER", "B-A"], rows)]
    out += pair_table(
        "### X->en chrF: HR sources (de, fr, es, it) and the other 19",
        [("HR", "xen_hr", 1, NOISE["chrf"]), ("other", "xen_other", 1, NOISE["chrf"])],
    )
    out += pair_table(
        "### en->X chrF: HR targets (de, fr, es, it) and the other 19",
        [("HR", "enx_hr", 1, NOISE["chrf"]), ("other", "enx_other", 1, NOISE["chrf"])],
    )

    rows = []
    for e in steps:
        a, b = e["summary"]["A"], e["summary"]["B"]
        rows.append(row(e, [pct(a["share_hr"]), pct(b["share_hr"]), pct(a["share_other"]), pct(b["share_other"])]))
    out += [
        "",
        "### en->X: share of outputs in the requested language",
        "",
        markdown_table(["step", "audio h", "A HR", "B HR", "A other", "B other"], rows),
    ]

    mix = block.get("training_data")
    if differs_from_hr(block, "xen"):
        n = (len(mix["xen"]), 23 - len(mix["xen"]))
        out += pair_table(
            f"### X->en chrF by training data in {block['cond']}: sources with data ({n[0]}) and without ({n[1]})",
            [("with data", "xen_with_data", 1, NOISE["chrf"]), ("without", "xen_without_data", 1, NOISE["chrf"])],
        )
    if differs_from_hr(block, "enx"):
        n = (len(mix["enx"]), 23 - len(mix["enx"]))
        out += pair_table(
            f"### en->X chrF by training data in {block['cond']}: targets with data ({n[0]}) and without ({n[1]})",
            [("with data", "enx_with_data", 1, NOISE["chrf"]), ("without", "enx_without_data", 1, NOISE["chrf"])],
        )
        rows = []
        for e in steps:
            a, b = e["summary"]["A"], e["summary"]["B"]
            rows.append(
                row(
                    e,
                    [
                        pct(a["share_with_data"]),
                        pct(b["share_with_data"]),
                        pct(a["share_without_data"]),
                        pct(b["share_without_data"]),
                    ],
                )
            )
        out += [
            "",
            f"### en->X share of outputs in the requested language by training data in {block['cond']}",
            "",
            markdown_table(["step", "audio h", "A with data", "B with data", "A without", "B without"], rows),
        ]

    out += [
        "",
        "### Sign tests across languages: languages where B is better / where A is better (ties dropped), two-sided exact p",
        "",
    ]
    names = {"asr": "ASR", "xen": "X->en", "enx": "en->X", "share": "in-language"}
    for fam in FAMILIES:
        counts = {"all": 24 if fam == "asr" else 23, "hr": 5 if fam == "asr" else 4, "other": 19}
        groups = ["all", "hr", "other"]
        if mix and fam != "asr" and differs_from_hr(block, "xen" if fam == "xen" else "enx"):
            key = "xen" if fam == "xen" else "enx"
            counts |= {"with_data": len(mix[key]), "without_data": 23 - len(mix[key])}
            groups += ["with_data", "without_data"]
        header = ["step", "audio h"] + [f"{names[fam]} {g.replace('_', ' ')} ({counts[g]})" for g in groups]
        rows = [row(e, [fmt_sign(e["sign"].get(fam, {}).get(g)) for g in groups]) for e in steps]
        out += [markdown_table(header, rows), ""]
    out.pop()

    for fam, title in FAMILIES.items():
        langs = sorted(
            {lang for e in steps for lang in e["delta"].get(fam, {})}, key=lambda lang: (lang not in SEEN, lang)
        )
        if not langs:
            continue
        digits = 3 if fam == "asr" else (2 if fam == "share" else 1)
        rows = [
            [lang + (" (HR)" if lang in SEEN else "")]
            + [fmt(e["delta"].get(fam, {}).get(lang), digits) for e in steps]
            for lang in langs
        ]
        direction = "negative = B better" if fam == "asr" else "positive = B better"
        out += [
            "",
            f"### Per-language B-A: {title} ({direction})",
            "",
            markdown_table(["language"] + [step_label(e["step"]) for e in steps], rows),
        ]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries")
    p.add_argument("langshare", nargs="?", default=None)
    p.add_argument("--json", default=None, help="write the numbers behind the tables here")
    p.add_argument("--markdown", default=None, help="write the tables here as well")
    p.add_argument("--cond", default=None, help="only HR5 or EU24")
    args = p.parse_args()

    data = load(args.summaries, args.langshare)
    blocks = [
        build_block(data, cond, seed)
        for (cond, seed) in sorted(k for k in data if k[0] is not None and (args.cond is None or k[0] == args.cond))
    ]
    if not blocks and (None, None) in data:
        print("no SFT checkpoint scored yet: step 0 only\n")
        blocks = [build_block(data, None, None)]
    text = "\n\n".join(render_block(b) for b in blocks)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(blocks, indent=1) + "\n")


if __name__ == "__main__":
    main()
