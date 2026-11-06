"""Shared helpers for the synthetic-ST pipeline: languages, prompt, ledger, JSONL IO."""

import json
import os
import time
from pathlib import Path


PROMPT_VERSION = "v1-instruction-first"
MT_MODEL = "utter-project/EuroLLM-9B-Instruct"
QE_MODEL = "google/metricx-24-hybrid-xxl-v2p6"  # Apache-2.0; reference-free (QE) mode, error score in [0, 25], lower is better
QE_MODEL_BF16 = QE_MODEL + "-bfloat16"  # Google's own half-memory release (fits a 48 GB A6000)
QE_TOKENIZER = "google/mt5-xl"

LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "el": "Greek", "fi": "Finnish",
    "ga": "Irish", "hr": "Croatian", "hu": "Hungarian", "lt": "Lithuanian",
    "mt": "Maltese", "pl": "Polish", "ro": "Romanian", "sk": "Slovak",
    "uk": "Ukrainian", "sv": "Swedish", "lv": "Latvian", "et": "Estonian",
    "sl": "Slovenian", "nl": "Dutch", "pt": "Portuguese", "ru": "Russian",
}  # fmt: skip
EN2X_TARGETS = [
    "bg", "cs", "da", "el", "fi", "ga", "hr", "hu", "lt", "mt",
    "pl", "ro", "sk", "uk", "sv", "lv", "et", "sl",
]  # fmt: skip
# Maltese is the weakest-resourced target (and its QE the least validated): such pairs also get a
# round-trip chrF as an independent check.
NOT_IN_XLMR = {"mt"}

MAX_NEW_TOKENS_FACTOR = 4  # generous cap relative to source tokens


def build_prompt(src_lang: str, tgt_lang: str, text: str) -> list[dict]:
    """Instruction-first chat messages (instruction before the source text)."""
    instr = (
        f"Translate the following {LANG_NAMES[src_lang]} text into "
        f"{LANG_NAMES[tgt_lang]}. Output only the translation."
    )
    return [{"role": "user", "content": f"{instr}\n\n{text}"}]


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: str | Path, rows: list[dict]) -> None:
    """Write rows to `path` through a temp file so a killed job leaves no partial output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(tmp, path)


LEDGER_COLS = ["unit", "stage", "n", "gpu_seconds", "gpus", "status", "host", "ts"]


def ledger_append(ledger: str | Path, **row) -> None:
    """Append one TSV row; the header is written on first use. One line per write (atomic enough on GPFS)."""
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    row.setdefault("host", os.uname().nodename)
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(ledger, "a") as f:
        if new:
            f.write("\t".join(LEDGER_COLS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in LEDGER_COLS) + "\n")


def is_degenerate(src: str, mt: str) -> bool:
    """Empty, absurd length ratio, or a short n-gram repeated many times."""
    mt = mt.strip()
    if not mt:
        return True
    ratio = len(mt) / max(len(src.strip()), 1)
    if ratio > 4.0 or ratio < 0.2:
        return True
    toks = mt.split()
    if len(toks) >= 12:
        grams = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]
        if max(grams.count(g) for g in set(grams)) >= 6:
            return True
    return False
