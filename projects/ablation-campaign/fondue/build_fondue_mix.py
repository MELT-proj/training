#!/usr/bin/env python3
"""Render Fondue's training configs: every source of the pool, tagged, filtered, weighted by language and corpus.

Fondue trains on the whole pool for its languages (PI, 2026-10-29 to 11-01):

* **ASR**: the 24 official EU languages, Ukrainian and Russian.
* **ST X->en**: YODAS-Granary ``ast`` (``custom.translation_en``), CoVoST2, Europarl-ST.
* **ST en->X**: CoVoST2, Europarl-ST, and SPITE (CV16 and GigaSpeech English, machine
  translated) for its EU targets and Russian: de, es, fr, it, nl, pt, ru.
* **ST X->Y** between two non-English languages (Europarl-ST): only with ``--x-to-y``.

Two phases, one config each, because the mixture changes at the annealing step and the
loader's mux weights cannot: ``Fondue.yaml`` (warmup and constant LR, ``run.stop_after_step``
at the annealing step) and ``Fondue-anneal.yaml`` (resumed there, same schedule).

Quality filters (PI, 11-01), the policy constants below:

* ASR, Parakeet/Seamless WER: <= 0.3 constant, <= 0.1 annealing; tighter on the
  high-quality corpora, looser for the languages where the recognizer itself fails on
  human-transcribed speech (CV22 + FLEURS keep < 85% at 0.3).
* SPITE: BLASER 2.0 (audio vs English transcript) >= 4.5, then xCOMET-XL >= 0.85
  constant, >= 0.98 annealing.
* ST with no per-cut score (Granary ast, CoVoST2, Europarl): constant phase only.

Weights, per phase: the two-tier policy of ``infra/compute_mix_weights.py`` (Canary,
arXiv:2509.14128 §3.3.1) inside each task, alpha 0.2 within an entry and beta across the
task's entries (0.5 constant, 0.7 annealing, which cuts the tails' passes), on the hours
that SURVIVE the phase's filters. ASR and ST then split the audio in proportion to those
surviving hours. Written as cut probabilities. The surviving share of each leaf comes from
the quality histograms in ``quality/`` (per corpus and language, measured 2026-10-30).

``total_hours`` is the constant phase's surviving pool, ASR and ST ("train on all the
data", PI 11-01); the last fifth of the steps anneals.

    # 1. the sources, unweighted, into Fondue-sources.yaml (audits text fields first)
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py sources \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --scores-root   /mnt/scratch-nyx/giuseppe/melt/melt-data/shar-asr-scores

    # 2. measure (cached in infra/.mix_weights_hours.json; resumes), filter, weight, write both configs
    python3 infra/compute_mix_weights.py --config projects/ablation-campaign/fondue/Fondue-sources.yaml \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar --jobs 32
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py render \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "sft"))

import compute_mix_weights as cmw  # noqa: E402
from build_sft_mix import audit_source  # noqa: E402


TEMPLATE = HERE / "Fondue-template.yaml"
SOURCES = HERE / "Fondue-sources.yaml"
RENDERED = {"constant": HERE / "Fondue.yaml", "anneal": HERE / "Fondue-anneal.yaml"}
QUALITY = HERE / "quality"

ALPHA = 0.2
BETA = {"constant": 0.5, "anneal": 0.7}  # PI, 11-01: fewer passes over the tails while annealing
DECAY_FRACTION = 5  # the last 1/5 of the steps anneals

# --- quality policy (PI, 11-01) -------------------------------------------------------
# WER bands as (constant, annealing) upper bounds. The rule keeps WER < bound (max is
# bound - 0.0001), matching the 0.05-wide histograms the surviving hours come from.
WER_DEFAULT = (0.30, 0.10)
# The high-quality corpora are filtered harder.
WER_BY_CORPUS = {
    "librispeech": (0.20, 0.05),
    "cv22_sidon": (0.20, 0.05),
    "mls_sidon": (0.15, 0.05),
    "voxpopuli": (0.15, 0.05),
    "gigaspeech": (0.15, 0.05),
}
# Languages where the recognizer, not the data, is the problem: on human-transcribed
# speech (CV22 + FLEURS) it keeps < 75% (first tier) or < 85% (second) of the hours at
# WER 0.3, against >= 90% for the big languages. A language band beats a corpus band.
WER_BY_LANG = {
    **dict.fromkeys(("ga", "sl", "et", "lv", "el"), (0.5, 0.2)),
    **dict.fromkeys(("mt", "lt", "sv", "sk"), (0.4, 0.15)),
}
BLASER_MIN = 4.5  # audio vs English transcript, both phases
XCOMET_MIN = {"constant": 0.85, "anneal": 0.98}

EU24 = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr",
    "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
)  # fmt: skip
ASR_LANGS = EU24 + ("uk", "ru")
SPITE_TARGETS = ("de", "es", "fr", "it", "nl", "pt", "ru")
SPITE_CORPORA = ("spite_cv16_euro9b", "spite_gigaspeech_euro9b")

# Directory spellings per corpus. A source is only real if its directory exists.
CV22_DIRS = {c: c for c in ASR_LANGS if c != "hr"} | {"ga": "ga-IE", "sv": "sv-SE"}
MLS_DIRS = {
    "de": "german", "en": "english", "es": "spanish", "fr": "french",
    "it": "italian", "nl": "dutch", "pl": "polish", "pt": "portuguese",
}  # fmt: skip
VOXPOPULI_LANGS = ("cs", "de", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl")
YODAS3_LANGS = (
    "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hu",
    "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
)  # fmt: skip
GRANARY_DIRS = {
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "hr": "Croatian", "hu": "Hungarian", "it": "Italian", "lt": "Lithuanian",
    "lv": "Latvian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "sk": "Slovak", "sv": "Swedish", "uk": "Ukrainian", "ru": "Russian",
}  # fmt: skip
FLEURS_DIRS = {
    "bg": "bg_bg", "cs": "cs_cz", "da": "da_dk", "de": "de_de", "el": "el_gr",
    "en": "en_us", "es": "es_419", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr",
    "ga": "ga_ie", "hr": "hr_hr", "hu": "hu_hu", "it": "it_it", "lt": "lt_lt",
    "lv": "lv_lv", "mt": "mt_mt", "nl": "nl_nl", "pl": "pl_pl", "pt": "pt_br",
    "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "sv": "sv_se", "uk": "uk_ua", "ru": "ru_ru",
}  # fmt: skip
# FLEURS leaves with no custom.pnc_text: they train on the (unpunctuated) supervision.
FLEURS_WITHOUT_PNC = ("ga",)
ENGLISH_EXTRA = (
    "gigaspeech/train",
    "peoples_speech/clean/train",
    "peoples_speech/clean_sa/train",
    "librispeech/clean/train.100",
    "librispeech/clean/train.360",
    "librispeech/other/train.500",
)
COVOST_X_TO_EN = {
    "de": "de_en", "es": "es_en", "fr": "fr_en", "it": "it_en", "nl": "nl_en",
    "pt": "pt_en", "et": "et_en", "lv": "lv_en", "sl": "sl_en", "sv": "sv-SE_en", "ru": "ru_en",
}  # fmt: skip
COVOST_EN_TO_X = {"de": "en_de", "et": "en_et", "lv": "en_lv", "sl": "en_sl", "sv": "en_sv-SE"}
EUROPARL_LANGS = ("de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro")

PNC = "custom.pnc_text"
TRANSLATION_EN = "custom.translation_en"

# Left out, and why. Printed on every run.
EXCLUDED = {
    "voxpopuli/hr/train": "47% of its cuts have no transcript",
    "yodas3/hr/train": "13 cuts (5 minutes), none with pnc_text",
    "voxpopuli_mosel/bg": "no WER sidecar, generic conversion (board, 10-29)",
    "regional languages (ca, eu, gl, ...)": "only official EU languages, plus uk and ru (PI, 10-30)",
    "SPITE ko/zh": "only SPITE's EU targets and ru (PI, 10-30 and 11-01)",
    "instructs2s200k": "not ASR or ST",
}


def leaf(rel: str, tags: dict) -> dict:
    """One flat input_cfg entry."""
    return {"rel": rel, "tags": tags}


def asr_leaves(lang: str, scored: set[str]) -> list[dict]:
    """Every ASR leaf of a language, tagged with its text field and whether it has a WER array."""
    found: list[tuple[str, str | None]] = []
    if lang in CV22_DIRS:
        found.append((f"cv22_sidon/{CV22_DIRS[lang]}/train", "custom.metadata.sentence"))
    if lang in MLS_DIRS:
        found.append((f"mls_sidon/{MLS_DIRS[lang]}/train", PNC))
    if lang in VOXPOPULI_LANGS:
        found.append((f"voxpopuli/{lang}/train", PNC))
    if lang in YODAS3_LANGS:
        found.append((f"yodas3/{lang}/train", PNC))
    if lang in GRANARY_DIRS:
        found.append((f"yodas-granary/{GRANARY_DIRS[lang]}/asr_only", None))
        if lang != "en":  # English has no ast
            found.append((f"yodas-granary/{GRANARY_DIRS[lang]}/ast", None))
    if lang in FLEURS_DIRS:
        found.append((f"fleurs/{FLEURS_DIRS[lang]}/train", None if lang in FLEURS_WITHOUT_PNC else PNC))
    if lang == "en":
        found += [(rel, PNC) for rel in ENGLISH_EXTRA]
    out = []
    for rel, text_field in found:
        tags = {"task": "asr", "lang": lang}
        if text_field:
            tags["text_field"] = text_field
        tags["scored"] = "wer" if rel in scored else "none"
        if rel in scored:
            tags["wer_band"] = band_name(wer_band(rel, lang))
        out.append(leaf(rel, tags))
    return out


def wer_band(rel: str, lang: str) -> tuple[float, float]:
    """(constant, annealing) WER bounds of an ASR leaf: its language's, else its corpus's, else the default."""
    return WER_BY_LANG.get(lang) or WER_BY_CORPUS.get(rel.split("/")[0]) or WER_DEFAULT


def band_name(band: tuple[float, float]) -> str:
    return f"c{round(band[0] * 100):02d}a{round(band[1] * 100):02d}"


def st_tags(src: str, tgt: str, text_field: str | None = None, scored: str = "none") -> dict:
    tags = {"task": "st", "src_lang": src, "tgt_lang": tgt}
    if text_field:
        tags["text_field"] = text_field
    tags["scored"] = scored
    return tags


def st_leaves(x_to_y: bool) -> list[dict]:
    """X->en, then en->X, then (optionally) X->Y."""
    out: list[dict] = []
    others = [lang for lang in ASR_LANGS if lang != "en"]
    for src in others:
        if src in GRANARY_DIRS:
            out.append(leaf(f"yodas-granary/{GRANARY_DIRS[src]}/ast", st_tags(src, "en", TRANSLATION_EN)))
        if src in COVOST_X_TO_EN:
            out.append(leaf(f"covost2/{COVOST_X_TO_EN[src]}/train", st_tags(src, "en")))
        if src in EUROPARL_LANGS:
            out.append(leaf(f"europarl_st/{src}_en/train", st_tags(src, "en")))
    for tgt in others:
        if tgt in COVOST_EN_TO_X:
            out.append(leaf(f"covost2/{COVOST_EN_TO_X[tgt]}/train", st_tags("en", tgt)))
        if tgt in EUROPARL_LANGS:
            out.append(leaf(f"europarl_st/en_{tgt}/train", st_tags("en", tgt)))
        if tgt in SPITE_TARGETS:
            for corpus in SPITE_CORPORA:
                out.append(leaf(f"{corpus}/train", st_tags("en", tgt, f"custom.translation_{tgt}", "spite")))
    if x_to_y:
        pairs = [(s, t) for s in EUROPARL_LANGS for t in EUROPARL_LANGS if "en" not in (s, t) and s != t]
        out += [leaf(f"europarl_st/{s}_{t}/train", st_tags(s, t)) for s, t in pairs]
    return out


def scored_leaves(scores_root: Path) -> set[str]:
    """Leaves (paths below the datasets root) that have a WER array."""
    return {str(p.parent.relative_to(scores_root)) for p in scores_root.rglob("wer.u16.npy")}


def entry_key(tags: dict) -> str:
    return f"asr:{tags['lang']}" if tags["task"] == "asr" else f"st:{tags['src_lang']}-{tags['tgt_lang']}"


def leaf_yaml(entry: dict, indent: int) -> list[str]:
    pad = " " * indent
    tags = ", ".join(f"{k}: {v}" for k, v in entry["tags"].items())
    return [
        f"{pad}- type: lhotse_shar",
        f"{pad}  shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{entry['rel']}",
        f"{pad}  tags: {{{tags}}}",
    ]


def cut_filter_rules(phase: str, bands: list[tuple[float, float]]) -> list[str]:
    """The rules of one phase: a WER rule per band in use, the BLASER gate, an xCOMET rule per SPITE target."""
    k = 0 if phase == "constant" else 1
    lines = []
    for band in sorted(set(bands), key=lambda b: (b[0], b[1])):
        lines += [
            f"        asr_wer_{band_name(band)}: # WER < {band[k]:.2f}",
            f"          match: {{task: asr, scored: wer, wer_band: {band_name(band)}}}",
            "          array: wer",
            "          missing: keep # languages neither recognizer covers",
            f"          max: {band[k] - 0.0001:.4f}",
        ]
    lines += [
        "        st_align: # audio and English transcript aligned before xCOMET says anything",
        "          match: {task: st, scored: spite}",
        "          field: custom.spite.blaser2_src",
        f"          min: {BLASER_MIN}",
    ]
    for tgt in SPITE_TARGETS:
        lines += [
            f"        st_xcomet_{tgt}:",
            f"          match: {{task: st, scored: spite, tgt_lang: {tgt}}}",
            f"          field: custom.spite.scores.{tgt}.xcomet_xl",
            f"          min: {XCOMET_MIN[phase]}",
        ]
    return lines


def validation_leaves() -> list[dict]:
    """One named set per ASR language (CV22 validation, else FLEURS), then per ST direction with a dev split."""
    out = []
    for lang in ASR_LANGS:
        if lang in CV22_DIRS:
            rel, tf = f"cv22_sidon/{CV22_DIRS[lang]}/validation", "custom.metadata.sentence"
        else:
            rel = f"fleurs/{FLEURS_DIRS[lang]}/validation"
            tf = None if lang in FLEURS_WITHOUT_PNC else PNC
        tags = {"task": "asr", "lang": lang} | ({"text_field": tf} if tf else {})
        out.append({"name": f"asr_{lang}", "rel": rel, "tags": tags})
    for src, d in COVOST_X_TO_EN.items():
        out.append({"name": f"st_{src}_en_covost2", "rel": f"covost2/{d}/dev", "tags": st_tags(src, "en")})
    for tgt, d in COVOST_EN_TO_X.items():
        out.append({"name": f"st_en_{tgt}_covost2", "rel": f"covost2/{d}/dev", "tags": st_tags("en", tgt)})
    for tgt in EUROPARL_LANGS:
        if tgt != "en":
            out.append(
                {
                    "name": f"st_en_{tgt}_europarl",
                    "rel": f"europarl_st/en_{tgt}/validation",
                    "tags": st_tags("en", tgt),
                }
            )
    for e in out:
        e["tags"].pop("scored", None)
    return out


def validation_yaml(sets: list[dict], indent: int) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}input_cfg:"]
    for s in sets:
        tags = ", ".join(f"{k}: {v}" for k, v in s["tags"].items())
        lines += [
            f"{pad}  - type: lhotse_shar",
            f"{pad}    name: {s['name']}",
            f"{pad}    shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{s['rel']}",
            f"{pad}    tags: {{{tags}}}",
        ]
    return lines


def replace_block(lines: list[str], section: str, block: list[str]) -> list[str]:
    """Replace ``input_cfg:`` under ``section:`` (and anything indented under it) with ``block``."""
    start, end, _ = cmw._find_block(lines, section, "input_cfg")
    return lines[:start] + block + lines[end:]


# Audio a micro-batch really carries over its batch_duration: lhotse charges a batch
# num_cuts x longest cut, and a cut longer than batch_duration is a batch of its own.
# Measured 26.3 s at batch_duration 30 on this mix (step profile, 2026-10-31, 2.23 cuts
# per micro-batch; 20.6 s at 15). Re-measure when batch_duration or the mix changes.
FILL = 26.3 / 30


def schedule(
    total_hours: float, batch_duration: float, grad_accum: int, nodes: int, gpus_per_node: int, fill: float = FILL
) -> tuple[int, int]:
    """``(max_steps, anneal_step)``: the steps that draw ``total_hours`` at the measured fill, and where the decay starts."""
    per_step = batch_duration * fill * grad_accum * nodes * gpus_per_node
    max_steps = int(total_hours * 3600 / per_step)
    return max_steps, max_steps - max_steps // DECAY_FRACTION


def template_scalar(lines: list[str], key: str) -> float:
    for line in lines:
        m = re.match(rf"^\s*{key}:\s*([0-9.]+)", line)
        if m:
            return float(m.group(1))
    raise ValueError(f"{key} not found in the template")


def cmd_sources(args: argparse.Namespace) -> None:
    root = Path(args.datasets_root)
    scored = scored_leaves(Path(args.scores_root))
    train = [e for lang in ASR_LANGS for e in asr_leaves(lang, scored)] + st_leaves(args.x_to_y)
    missing = sorted({e["rel"] for e in train if not (root / e["rel"]).is_dir()})
    if missing:
        sys.exit("Sources not under the datasets root:\n  " + "\n  ".join(missing))
    val = validation_leaves()
    missing = [v["rel"] for v in val if not (root / v["rel"]).is_dir()]
    if missing:
        sys.exit("Validation sets not under the datasets root:\n  " + "\n  ".join(missing))

    print("Excluded:")
    for what, why in EXCLUDED.items():
        print(f"  {what}: {why}")

    if not args.no_audit:
        targets = sorted({(e["rel"], e["tags"].get("text_field"), e["tags"]["task"]) for e in train}, key=str)
        print(f"\nAuditing {len(targets)} (source, text field) pairs...", flush=True)
        bad = []
        for rel, tf, task in targets:
            a = audit_source(root, rel, tf, task, max_blank=0.10 if rel.startswith("voxpopuli/") else 0.001)
            flag = "; ".join(a.problems())
            print(
                f"  {rel:42s} {str(tf):30s} {a.cuts:7d} cuts  blank {a.rate(a.blank):6.2%}  "
                f"mismatch {a.rate(a.mismatch):6.2%}  {flag or 'ok'}",
                flush=True,
            )
            if flag:
                bad.append(f"{rel} ({tf}): {flag}")
        if bad:
            sys.exit("Refusing to render; the audit failed:\n  " + "\n  ".join(bad))

    lines = TEMPLATE.read_text().splitlines()
    header = [
        "# Fondue-sources.yaml: generated by build_fondue_mix.py `sources`. Do not hand-edit.",
        "# The pool, unweighted and unfiltered; build_fondue_mix.py `render` writes the two phase configs.",
    ]
    lines = "\n".join(lines).replace("# @@HEADER@@", "\n".join(header)).splitlines()
    _, _, indent = cmw._find_block(lines, "train_ds", "input_cfg")
    block = [" " * indent + "input_cfg:"]
    for e in train:
        block += leaf_yaml(e, indent + 2)
    lines = replace_block(lines, "train_ds", block)
    _, _, indent = cmw._find_block(lines, "validation_ds", "input_cfg")
    lines = replace_block(lines, "validation_ds", validation_yaml(val, indent))
    SOURCES.write_text("\n".join(lines) + "\n")

    entries = defaultdict(list)
    for e in train:
        entries[entry_key(e["tags"])].append(e["rel"])
    n_asr = sum(k.startswith("asr:") for k in entries)
    print(
        f"\nWrote {SOURCES.relative_to(REPO_ROOT)}: {len(train)} leaves, {len(entries)} entries "
        f"({n_asr} ASR, {len(entries) - n_asr} ST), {len(val)} validation sets."
    )
    unscored = [e["rel"] for e in train if e["tags"]["task"] == "asr" and e["tags"]["scored"] == "none"]
    print(f"ASR leaves without a WER array (no WER rule applies to them): {len(unscored)}")
    for rel in unscored:
        print(f"  {rel}")


# --- surviving hours --------------------------------------------------------------------


class Survival:
    """The share of a leaf's hours each phase's filters keep, from the quality histograms.

    ASR: the WER histogram (0.05-wide bins, hours) of the leaf's (corpus, language); the
    two Granary subsets are told apart because ``ast`` was scored as its own corpus.
    SPITE: the BLASER x xCOMET histogram (hours) of the leaf's (corpus, target).
    A leaf with no histogram keeps everything, and is reported.
    """

    def __init__(self, quality: Path) -> None:
        asr = json.loads((quality / "asr_wer_dist.json").read_text())
        self.step = asr["step"]
        self.wer: dict[tuple[str, str], list[float]] = {}
        for g in asr["groups"]:
            if g.get("wer_h"):
                h = self.wer.setdefault((g["ds"], g["lang"]), [0.0] * len(g["wer_h"]))
                for i, v in enumerate(g["wer_h"]):
                    h[i] += v
        st = json.loads((quality / "spite_dist.json").read_text())
        self.b_edges, self.x_edges = st["b_edges"], st["x_edges"]
        self.spite = {(g["ds"], g["tgt"]): g["src_h"] for g in st["groups"]}
        self.unknown: set[str] = set()

    @staticmethod
    def asr_key(rel: str, lang: str) -> tuple[str, str]:
        corpus = rel.split("/")[0]
        if corpus == "yodas-granary" and rel.endswith("/ast"):
            corpus = "yodas-granary-ast"
        return corpus, lang

    def share(self, s: dict, phase: str) -> float:
        tags, rel = s["tags"], s["rel"]
        k = 0 if phase == "constant" else 1
        if tags["task"] == "asr":
            if tags.get("scored") != "wer":
                return 1.0
            h = self.wer.get(self.asr_key(rel, tags["lang"]))
            if h is None:
                self.unknown.add(rel)
                return 1.0
            bound = wer_band(rel, tags["lang"])[k]
            return sum(h[: round(bound / self.step)]) / sum(h)
        if tags.get("scored") == "spite":
            corpus = "cv16" if "_cv16_" in rel else "gigaspeech"
            h = self.spite.get((corpus, tags["tgt_lang"]))
            if h is None:
                self.unknown.add(rel)
                return 1.0
            bi = self.b_edges.index(BLASER_MIN) + 1
            xi = self.x_edges.index(XCOMET_MIN[phase]) + 1
            kept = sum(v for b, row in enumerate(h) for x, v in enumerate(row) if b >= bi and x >= xi)
            return kept / sum(map(sum, h))
        # ST with no per-cut score: all of it while the LR is constant, none while it anneals.
        return 1.0 if phase == "constant" else 0.0


def phase_weights(sources: list[dict], phase: str) -> dict[str, float]:
    """The policy inside each task on the surviving hours, then the tasks in proportion to them.

    Returns the surviving hours per task.
    """
    held = {t: sum(s["hours"] for s in sources if s["task"] == t) for t in ("asr", "st")}
    total = sum(held.values())
    for task in ("asr", "st"):
        part = [s for s in sources if s["task"] == task]
        cmw.compute_weights(part, ALPHA, BETA[phase])
        for s in part:
            s["p_l"] *= held[task] / total
            s["p_cl"] *= held[task] / total
    return held


def cmd_render(args: argparse.Namespace) -> None:
    os.environ.setdefault("LOCAL_ASR_SCORES_DIR", "/unused")  # only the cut filter reads it
    pool = cmw.load_sources(SOURCES, args.datasets_root)
    cache = json.loads(Path(args.cache).read_text()) if Path(args.cache).exists() else {}
    todo = sorted(
        {
            s["path"]
            for s in pool
            if s["path"] not in cache
            or cache[s["path"]].get("sample") is not None
            or cache[s["path"]].get("cuts") is None
        }
    )
    if todo:
        sys.exit(
            f"{len(todo)} sources are not measured (run infra/compute_mix_weights.py first):\n  " + "\n  ".join(todo)
        )
    for s in pool:
        s["pool_hours"], s["cuts"] = cache[s["path"]]["hours"], cache[s["path"]]["cuts"]
        s["rel"] = s["raw_path"].replace("${oc.env:LOCAL_DATASETS_DIR}/", "", 1)
    survival = Survival(Path(args.quality))

    phases = {}
    for phase in ("constant", "anneal"):
        sources = []
        for s in copy.deepcopy(pool):
            s["survives"] = survival.share(s, phase)
            if s["survives"] <= 0:
                continue
            # Survivors only leave the leaf, so its weight is a share of surviving hours;
            # the same cuts in fewer hours keeps the mean cut length the leaf measured.
            s["hours"] = s["pool_hours"] * s["survives"]
            s["cuts"] = max(1, round(s["cuts"] * s["survives"]))
            sources.append(s)
        held = phase_weights(sources, phase)
        cmw.to_cut_probabilities(sources)
        phases[phase] = (sources, held)

    template = SOURCES.read_text().splitlines()
    total_hours = sum(phases["constant"][1].values())
    max_steps, anneal_step = schedule(
        total_hours,
        template_scalar(template, "batch_duration"),
        int(template_scalar(template, "gradient_accumulation_steps")),
        args.nodes,
        args.gpus_per_node,
    )
    drawn = {
        "constant": total_hours * anneal_step / max_steps,
        "anneal": total_hours * (max_steps - anneal_step) / max_steps,
    }
    bands = [wer_band(s["rel"], s["tags"]["lang"]) for s in pool if s["tags"].get("wer_band")]

    for phase, (sources, held) in phases.items():
        out = RENDERED[phase]
        cmw.emit_training_config(sources, SOURCES, out, ALPHA, BETA[phase], exp_name="Fondue", weights="cuts")
        text = out.read_text()
        text = re.sub(r"(?m)^(\s*total_hours:)\s*[0-9.]+", rf"\1 {total_hours:.1f}", text, count=1)
        text = text.replace("        # @@CUT_FILTER_RULES@@", "\n".join(cut_filter_rules(phase, bands)))
        # Pinned here, not by the launcher's formula, which assumes a fill of 1.0.
        text = text.replace("  max_steps: -1", f"  max_steps: {max_steps}").replace(
            "    decay_type: cosine", f"    decay_type: cosine\n    num_decay_steps: {max_steps - anneal_step}"
        )
        st_share = held["st"] / sum(held.values())
        if phase == "constant":
            stop = f"stop_after_step: {anneal_step} # the annealing phase resumes here with Fondue-anneal.yaml"
            what = "Phase 1 of 2: warmup and constant LR, up to the annealing step."
        else:
            stop = "stop_after_step: null"
            what = (
                f"Phase 2 of 2: resume from Fondue.yaml's checkpoint-{anneal_step} and anneal.\n"
                "# The data stream restarts (the mixture changed): the resume must not restore the\n"
                "# sampler state, and HF must not skip batches (trainer.ignore_data_skip)."
            )
            text = text.replace(
                "  resume_from_checkpoint: null",
                "  resume_from_checkpoint: null # the launcher sets it\n  ignore_data_skip: true",
            )
        text = text.replace("stop_after_step: null # @@STOP_AFTER_STEP@@", stop)
        text = text.replace(
            "# Fondue-sources.yaml: generated by build_fondue_mix.py `sources`. Do not hand-edit.\n"
            "# The pool, unweighted and unfiltered; build_fondue_mix.py `render` writes the two phase configs.",
            f"# {out.name}: generated by build_fondue_mix.py `render`. Do not hand-edit.\n# {what}\n"
            f"# Weights on the hours surviving this phase's filters: ST {st_share:.1%} of them, then alpha {ALPHA} /\n"
            f"# beta {BETA[phase]} inside each task. total_hours ({total_hours:,.0f}) is the constant phase's surviving\n"
            f"# pool; max_steps {max_steps} draws it at {args.nodes} nodes x {args.gpus_per_node} GPUs and a fill of {FILL:.3f}, and\n"
            f"# the last fifth, from step {anneal_step}, anneals.",
        )
        out.write_text(text)

        print(f"\n=== {phase}: {out.relative_to(REPO_ROOT)} ===")
        print(
            f"Surviving hours: ASR {held['asr']:,.0f}, ST {held['st']:,.0f} (ST {st_share:.1%}); "
            f"draws {drawn[phase]:,.0f} h; beta {BETA[phase]}."
        )
        by_entry: dict[str, list[dict]] = defaultdict(list)
        for s in sources:
            by_entry[s["lang_key"]].append(s)
        print(f"{'entry':<10} {'pool h':>10} {'kept h':>10} {'share':>7} {'drawn h':>9} {'passes':>7}")
        rows = []
        for key, ss in by_entry.items():
            share = ss[0]["p_l_hours"]
            kept = sum(s["hours"] for s in ss)
            rows.append((key, sum(s["pool_hours"] for s in ss), kept, share, share * drawn[phase]))
        for key, pool_h, kept, share, d in sorted(rows, key=lambda r: -r[3]):
            print(f"{key:<10} {pool_h:>10,.1f} {kept:>10,.1f} {share:>7.2%} {d:>9,.0f} {d / kept:>7.2f}")

    print(f"\ntotal_hours {total_hours:,.0f}; max_steps {max_steps}; annealing from step {anneal_step}.")
    if survival.unknown:
        print("Leaves with no quality histogram (kept whole):\n  " + "\n  ".join(sorted(survival.unknown)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources", help="write Fondue-sources.yaml")
    s.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR"), required=False)
    s.add_argument("--scores-root", default=os.environ.get("LOCAL_ASR_SCORES_DIR"))
    s.add_argument("--x-to-y", action="store_true", help="add Europarl-ST between two non-English languages")
    s.add_argument("--no-audit", action="store_true")
    r = sub.add_parser("render", help="filter and weight Fondue-sources.yaml into the two phase configs")
    r.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR"))
    r.add_argument("--quality", default=str(QUALITY), help="the quality histograms")
    r.add_argument("--nodes", type=int, default=16)
    r.add_argument("--gpus-per-node", type=int, default=4)
    r.add_argument("--cache", default=str(REPO_ROOT / "infra" / ".mix_weights_hours.json"))
    args = p.parse_args()
    if not args.datasets_root or (args.cmd == "sources" and not args.scores_root):
        p.error("--datasets-root (and, for sources, --scores-root) or their environment variables are required")
    {"sources": cmd_sources, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
