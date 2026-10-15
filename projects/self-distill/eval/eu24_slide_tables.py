#!/usr/bin/env python3
"""Compact, slide-sized tables of the EU24 comparison, built from the JSON of two eu24_report.py runs.

usage: eu24_slide_tables.py REPEATFIRST_JSON SFTRANDOM_JSON [OUT_MD]
Everything is read from the JSON; no number is typed in here.
"""

import json
import sys
from pathlib import Path


main = json.loads(Path(sys.argv[1]).read_text())
rand = json.loads(Path(sys.argv[2]).read_text())

ROWS = [  # key, label, decoder
    ("A", "A (start)", "Qwen3.5-2B"),
    ("SFT-EU24", "SFT-EU24 (full fine-tune)", "Qwen3.5-2B"),
    ("CE24", "CE24 (adapter, gold CE)", "Qwen3.5-2B"),
    ("SD24", "SD24 (adapter, self-distilled)", "Qwen3.5-2B"),
    ("SD-SIFT24", "SD-SIFT24 (adapter, AZeroS)", "Qwen3.5-2B"),
    ("A-EuroLLM", "A' (start)", "EuroLLM-1.7B-Instruct-2512"),
    ("SD24-EuroLLM", "SD24 (adapter, self-distilled)", "EuroLLM-1.7B-Instruct-2512"),
    ("SD-SIFT24-EuroLLM", "SD-SIFT24 (adapter, AZeroS)", "EuroLLM-1.7B-Instruct-2512"),
]


def f1(x, scale=1.0):
    return "-" if x is None else f"{x * scale:.1f}"


def rows(report, cells):
    out = []
    for key, label, decoder in ROWS:
        entry = report["models"].get(key)
        if entry is None:
            continue
        out.append(f"| {decoder} | {label} | " + " | ".join(cells(entry)) + " |")
    return out


def st(entry, direction, case, metric, tier):
    d = entry.get(direction)
    return None if d is None else d["tiers"][case][metric][tier]


lines = []
add = lines.append

add("**Table 1. ASR on FLEURS dev, % (macro-average over the languages of each group; lower is better)**")
add("")
add(
    "| decoder | model | WER all 24 | WER high (7) | WER mid (6) | WER low (11) | WER 5 trained languages | CER all 24 | CER high | CER mid | CER low |"
)
add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
lines += rows(
    main,
    lambda e: (
        [f1(e["asr"]["tiers"]["wer"][t], 100) for t in ("all", "high", "mid", "low", "hr5")]
        + [f1(e["asr"]["tiers"]["cer"][t], 100) for t in ("all", "high", "mid", "low")]
    ),
)
add("")

for metric, name in (("chrfpp", "chrF++"), ("spbleu", "spBLEU")):
    add(
        f"**Table {2 if metric == 'chrfpp' else 3}. {name} on FLEURS dev, lowercased (macro-average over directions; higher is better)**"
    )
    add("")
    add(
        "| decoder | model | X→en all 23 | high (6) | mid (6) | low (11) | en→X all 23 | high (6) | mid (6) | low (11) |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines += rows(
        main,
        lambda e, m=metric: [
            f1(st(e, d, "lower", m, t)) for d in ("xen", "enx") for t in ("all", "high", "mid", "low")
        ],
    )
    add("")

add("**Table 4. The same two metrics, cased (all 23 directions)**")
add("")
add(
    "| decoder | model | chrF++ X→en | chrF++ en→X | spBLEU X→en | spBLEU en→X | translations with no capital letter, X→en / en→X |"
)
add("|---|---|---:|---:|---:|---:|---:|")
lines += rows(
    main,
    lambda e: [
        f1(st(e, "xen", "cased", "chrfpp", "all")),
        f1(st(e, "enx", "cased", "chrfpp", "all")),
        f1(st(e, "xen", "cased", "spbleu", "all")),
        f1(st(e, "enx", "cased", "spbleu", "all")),
        "-"
        if e.get("xen") is None
        else f"{e['xen']['no_capitals_share'] * 100:.0f}% / {e['enx']['no_capitals_share'] * 100:.0f}%",
    ],
)
add("")

add(
    "**Table 5. Same models under the random task templates instead of the repeat-first prompts (lowercased; repeat-first → random)**"
)
add("")
add("| decoder | model | WER all 24 (%) | chrF++ X→en | chrF++ en→X | spBLEU X→en | spBLEU en→X |")
add("|---|---|---:|---:|---:|---:|---:|")


def pair(x, y):
    return f"{x} → {y}"


for key, label, decoder in ROWS:
    a, b = main["models"].get(key), rand["models"].get(key)
    if a is None or b is None:
        continue
    add(
        f"| {decoder} | {label} | "
        + pair(f1(a["asr"]["tiers"]["wer"]["all"], 100), f1(b["asr"]["tiers"]["wer"]["all"], 100))
        + " | "
        + " | ".join(
            pair(f1(st(a, d, "lower", m, "all")), f1(st(b, d, "lower", m, "all")))
            for m in ("chrfpp", "spbleu")
            for d in ("xen", "enx")
        )
        + " |"
    )
add("")

add("**Table 6. The campaign's selection metric (single ranking score; lower is better)**")
add("")
add(
    "| decoder | model | score | ID (5 trained languages, held-out CV22/MLS/VoxPopuli) | OOD-train (same 5, FLEURS dev) | OOD-related (pt ro nl da sv) | OOD-latin* | OOD-script* | runaway clips: in-domain / FLEURS |"
)
add("|---|---|---:|---:|---:|---:|---:|---:|---:|")


def sel_cells(e):
    s_ = e.get("selection")
    if not s_:
        return ["-"] * 7
    m = s_["medians"]
    rw = s_.get("runaway")
    rw_text = (
        "n/a"
        if rw is None
        else f"{rw['ID']['runaways']}/{rw['ID']['samples']} / {rw['FLEURS']['runaways']}/{rw['FLEURS']['samples']}"
    )
    return [
        f"**{s_['score']:.3f}**",
        f"{m['ID']:.3f}",
        f"{m['OOD-train']:.3f}",
        f"{m['OOD-related']:.3f}",
        f"{m['OOD-latin']:.3f}",
        f"{m['OOD-script']:.3f}",
        rw_text,
    ]


lines += rows(main, sel_cells)
add("")
add(
    "score = (1.0 ID + 1.5 OOD-train + 0.6 OOD-related) / 3.1; each group is the median over its languages of corpus CER clipped at 1. *reported only, not in the score (OOD-latin: the other 12 Latin-script EU languages; OOD-script: bg el ru uk, of which ru and uk were never trained)."
)
add("")

text = "\n".join(lines) + "\n"
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(text)
print(text)
