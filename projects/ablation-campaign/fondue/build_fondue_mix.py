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
arXiv:2509.14128 §3.3.1) inside each task, alpha 0.5 within an entry and beta 0.5 across the
task's entries (Canary's values, both phases), on the hours
that SURVIVE the phase's filters. ASR and ST then split the audio in proportion to those
surviving hours. Written as cut probabilities. The surviving share of each leaf comes from
the quality histograms in ``quality/`` (per corpus and language, measured 2026-10-30;
VoxPopuli-Mosel on MN5, 2026-11-04).

``total_hours`` is the constant phase's surviving pool, ASR and ST ("train on all the
data", PI 11-01), and ``total_cuts`` its cuts (the trainer's start-up log needs one); the last fifth of the steps anneals.

MN5 is what Fondue reads; nyx is where this runs. ``mn5_mirror.py`` lays out a datasets
root and a scores root on nyx that match MN5 (nyx's copies, Yodas3 cut to MN5's languages,
VoxPopuli-Mosel from ``mosel_stats.py`` run on MN5 and its score arrays copied over):

    # 0. on MN5 (a gp node): Mosel's hours, cut counts and WER histograms
    python projects/ablation-campaign/fondue/quality/mosel_stats.py <MN5 shar-indexed> <MN5 shar-asr-scores> mosel_stats.json
    #    -> trim into quality/mosel_wer_dist.json; rsync its wer.u16.npy + meta.json to <scores>
    python3 projects/ablation-campaign/fondue/mn5_mirror.py --out <mirror> --mosel-scores <scores>

    # 1. the sources, unweighted, into Fondue-sources.yaml (Mosel is audited by mosel_stats.py;
    #    the nyx corpora by the full audit of 10-30, so --no-audit against the mirror)
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py sources \\
        --datasets-root <mirror>/shar --scores-root <mirror>/shar-asr-scores --no-audit

    # 2. filter, weight, write both configs (hours from the mirror's cache)
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py render \\
        --datasets-root <mirror>/shar --cache <mirror>/mix_weights_hours.json

    # 3. SpeechBlocks-Synth-v1 into the annealing config (reads the rendered Fondue-anneal.yaml; no cache, no data;
    #    quality/speechblocks_synth_v1.json comes from data-utils/synth_st/make_mix_spec.py in MELT-proj/preprocessing)
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py synth
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

# Canary's values (PI, 11-05). At alpha 0.2 the corpora a filter all but empties were
# drawn thousands of times; 0.7 while annealing is dropped with it.
ALPHA = 0.5
BETA = {"constant": 0.5, "anneal": 0.5}
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
# Pseudo-labelled corpora: a high WER is the label's fault, not the recognizer's, so the
# language bands do not apply (PI, 11-04).
WER_PSEUDO_LABELLED = {"voxpopuli_mosel": WER_DEFAULT}
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
# The languages left on MN5 (PI, 11-03; ro and nl moved back 11-05, Mosel has neither);
# the rest moved off to make room for Mosel.
YODAS3_LANGS = ("bg", "cs", "hu", "it", "nl", "pl", "pt", "ro")
# VoxPopuli-Mosel: pseudo-labelled plenary speech, scored by Parakeet on MN5 (PI, 11-03).
MOSEL_LANGS = ("bg", "cs", "da", "el", "et", "fi", "hr", "hu", "lt", "lv", "mt", "sk", "sl", "sv")
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
    # People's Speech: clean_sa is not used (PI, 11-04). clean's tars carry PAX headers (member names
    # over 100 bytes) that break lhotse's indexed tar reader, which the cut filter needs, so the leaf
    # is the re-shared copy: same cuts, tars with short member names. Its score arrays are a copy of
    # clean/train's under the same relative path (the filter looks them up by the shar path).
    "peoples_speech/clean_repacked/train",
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
    "yodas3 (da de el es et fi fr lt lv sk sl sv)": "moved off MN5 for VoxPopuli-Mosel (PI, 11-03)",
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
    if lang in MOSEL_LANGS:
        found.append((f"voxpopuli_mosel/{lang}/train", None))
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
    """(constant, annealing) WER bounds of an ASR leaf: a pseudo-labelled corpus's, else its
    language's, else its corpus's, else the default."""
    corpus = rel.split("/")[0]
    return WER_PSEUDO_LABELLED.get(corpus) or WER_BY_LANG.get(lang) or WER_BY_CORPUS.get(corpus) or WER_DEFAULT


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
    # os.walk, not rglob: the MN5 mirror (mn5_mirror.py) is a tree of symlinks.
    return {
        str(Path(d).relative_to(scores_root))
        for d, _, files in os.walk(scores_root, followlinks=True)
        if "wer.u16.npy" in files
    }


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
        # VoxPopuli-Mosel was scored on MN5 only; its histograms come from mosel_stats.py there.
        for g in json.loads((quality / "mosel_wer_dist.json").read_text()):
            self.wer[("voxpopuli_mosel", g["lang"])] = g["wer_h"]
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
    # The trainer logs "Num examples" from total_cuts; unset, it is None and the log line raises.
    total_cuts = sum(s["cuts"] for s in phases["constant"][0])
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
        text = re.sub(
            r"(?m)^(\s*)total_hours:\s*[0-9.]+",
            rf"\1total_hours: {total_hours:.1f}\n\1total_cuts: {total_cuts}",
            text,
            count=1,
        )
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
            stop = (
                "stop_after_step: null\n"
                "  restart_data_stream: true # a new mixture: keep the model, optimizer and LR, not the stream"
            )
            what = (
                f"Phase 2 of 2: resume from Fondue.yaml's checkpoint-{anneal_step} and anneal.\n"
                "# The data stream restarts (the mixture changed; run.restart_data_stream), with a\n"
                "# new data seed so it does not replay phase 1's draw order. Launch it with a\n"
                "# MELT_SEED other than phase 1's: the launcher sets shard_seed from it."
            )
            text = text.replace("  resume_from_checkpoint: null", "  resume_from_checkpoint: null # the launcher sets it")
            text = text.replace("    seed: 1337\n    shard_seed: 1337", "    seed: 1338\n    shard_seed: 1338", 1)
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


# --- SpeechBlocks-Synth-v1 ------------------------------------------------------------------
#
# The annealing config plus the synthetic collection (MELT-proj/preprocessing, data-utils/synth_st), as a pass over the
# rendered text: the deployed mixture is not re-derived, so it cannot drift. Every existing group keeps its share of
# the audio relative to the others (all of them scaled down together); the collection's ST leaves enter as new groups,
# weighted by the same alpha/beta over their own hours; its noisy copies of the native tail swap into the language group
# they copy, which keeps its audio share (its weight counts cuts and the copies' cuts are shorter, so the weight grows).
# Needs no hours cache: it reads the numbers the renderer printed into the config.

SYNTH_SPEC = QUALITY / "speechblocks_synth_v1.json"
SYNTH_ROOT = "/gpfs/scratch/epor48/synth/shar-indexed"  # MN5 scratch; /gpfs resolves as is inside the container
SYNTH_OUT = HERE / "Fondue-anneal-synth.yaml"
# A snapshot walks the whole source graph once per buffered cut, so 1,065 leaves instead of 160 make each one ~6x dearer
# (nyx, one worker, world size 64: 6.5 s a snapshotted batch at 905 leaves against 1.0 s at 150; 34 batches per window).
# Every 512 batches that is ~4 minutes of stall in ~4.5; every 8192, ~6% of the run. A resume replays up to this many batches.
SYNTH_SNAPSHOT_EVERY = 8192
# What each `synth_stream` tag stands for, printed into the config (the tags and rule names keep the letters).
SYNTH_STREAMS = {
    "A": "English speech into 18 languages (ST)",
    "B": "FLEURS read speech of the 18 low-resource languages into en/de/fr/es/it (ST)",
    "C": "native speech of the low-resource languages into other languages (ST)",
    "D": "existing ST data (Granary X->en, CoVoST2 en->et/lv/sl/sv), now MetricX-scored (ST)",
    "E": "synthetic (TTS) English speech into 18 languages (ST)",
    "F": "noisy copies of A, C and E (ST, tagged with their source's letter) and of the native tail (ASR, tagged F)",
}

GROUP_NOTE = re.compile(
    r"^(?P<pad> *)# (?P<key>\S+): (?P<hours>[\d,]+\.\d) h across (?P<n>\d+) corpora -- "
    r"(?P<cuts>[\d.]+)% of the cuts, (?P<audio>[\d.]+)% of the audio$"
)
LEAF_NOTE = re.compile(
    r"^(?P<pad> *)# (?P<hours>[\d,]+\.\d) h, mean cut (?P<mean>[\d.]+) s, (?P<audio>[\d.]+)% of the audio$"
)
WEIGHT_LINE = re.compile(r"^(?P<pad> *)weight: (?P<w>[0-9.]+)$")


def parse_mix(lines: list[str], start: int, end: int) -> list[dict]:
    """The language groups of ``train_ds.input_cfg``, as ``compute_mix_weights.build_group_block`` writes them.

    Anything else is an error, so a change of the renderer cannot be mis-read into a wrong edit.
    """
    heads = [i for i in range(start + 1, end) if GROUP_NOTE.match(lines[i])]
    if not heads:
        raise ValueError("no language group found in train_ds.input_cfg")
    groups = []
    for k, head in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else end
        note = GROUP_NOTE.match(lines[head])
        pad = len(note["pad"])
        space = " " * pad
        if lines[head + 1] != f"{space}- type: group":
            raise ValueError(f"line {head + 2}: expected '- type: group' under the group note")
        weight = WEIGHT_LINE.match(lines[head + 2])
        if not weight or len(weight["pad"]) != pad + 2:
            raise ValueError(f"line {head + 3}: expected the group's weight")
        tags: dict[str, str] = {}
        i = head + 3
        if lines[i].strip() == "tags:":
            i += 1
            while len(lines[i]) - len(lines[i].lstrip()) > pad + 2:
                key, value = lines[i].strip().split(": ", 1)
                tags[key] = value
                i += 1
        if lines[i] != f"{space}  input_cfg:":
            raise ValueError(f"line {i + 1}: expected the group's input_cfg")
        leaf_space = " " * (pad + 4)
        firsts = [j for j in range(i + 1, stop) if lines[j] == f"{leaf_space}- type: lhotse_shar"]
        leaves = []
        for n, first in enumerate(firsts):
            last = firsts[n + 1] if n + 1 < len(firsts) else stop
            leaf: dict = {"start": first}
            for j in range(first, last):
                w = WEIGHT_LINE.match(lines[j])
                if w and len(w["pad"]) == pad + 6 and "weight_line" not in leaf:
                    leaf["weight_line"], leaf["w"] = j, float(w["w"])
                m = LEAF_NOTE.match(lines[j])
                if m and len(m["pad"]) == pad + 6:
                    leaf["note_line"] = j
                    leaf["hours"], leaf["mean"] = float(m["hours"].replace(",", "")), float(m["mean"])
            if "weight_line" not in leaf or "note_line" not in leaf:
                raise ValueError(f"line {first + 1}: a leaf without a weight or a hours note")
            leaves.append(leaf)
        if not leaves:
            raise ValueError(f"line {head + 1}: group {note['key']} has no leaves")
        lang_key = tags["lang"] if tags.get("task") == "asr" else f"{tags.get('src_lang')}-{tags.get('tgt_lang')}"
        groups.append(
            {
                "head": head, "stop": stop, "pad": pad, "key": lang_key, "tags": tags, "leaves": leaves,
                "hours": float(note["hours"].replace(",", "")), "n": int(note["n"]),
                "weight_line": head + 2, "w": float(weight["w"]),
            }
        )  # fmt: skip
    return groups


def synth_sources(spec: dict, root: str) -> list[dict]:
    """The spec's ST leaves as the sources ``compute_mix_weights`` weighs: kept hours and cuts, tagged like SPITE's."""
    out = []
    for leaf in spec["st_leaves"]:
        src, tgt = leaf["src"], leaf["tgt"]
        path = f"{root}/{leaf['name']}/train"
        tags = {
            "task": "st", "src_lang": src, "tgt_lang": tgt, "text_field": f"custom.translation_{tgt}",
            "scored": "synth", "synth_stream": leaf["stream"], "src_region_code": src, "tgt_region_code": tgt,
        }  # fmt: skip
        out.append(
            {
                "path": path, "raw_path": path, "task": "st", "lang_key": f"{src}-{tgt}", "tags": tags,
                "hours": leaf["kept_s"] / 3600, "cuts": leaf["kept_pairs"], "group": leaf["group"],
            }
        )  # fmt: skip
    return out


def synth_rules(spec: dict) -> list[str]:
    """One ``field`` rule per (stream, src, tgt): MetricX at most the direction's bound. A cut with no score is dropped."""
    lines = []
    for r in spec["rules"]:
        name = f"synth_qe_{r['stream']}_{r['src']}_{r['tgt']}"
        lines += [
            f"        {name}: # MetricX <= {r['bound']}: the best {r['keep']:.0%} of {r['src']} -> {r['tgt']}, stream {r['stream']}",
            f"          match: {{task: st, scored: synth, synth_stream: {r['stream']}, src_lang: {r['src']}, tgt_lang: {r['tgt']}}}",
            f"          field: custom.synth.metricx_xxl.{r['tgt']}",
            f"          max: {r['bound']}",
        ]  # fmt: skip
    return lines


def asr_leaf_lines(pad: int, leaf: dict, root: str, weight: float, note: str) -> list[str]:
    """A noisy native-tail ASR leaf, as ``build_group_block`` writes a leaf (no WER tags: no array exists for a copy)."""
    space = " " * pad
    tags = {"task": "asr", "lang": leaf["lang"]}
    if "text_field" in leaf["tags"]:
        tags["text_field"] = leaf["tags"]["text_field"]
    tags |= {"scored": "none", "region_code": leaf["tags"]["region_code"], "synth_stream": "F"}
    return [
        f"{space}- type: lhotse_shar",
        f"{space}  shar_path: {root}/{leaf['name']}/train",
        f"{space}  weight: {weight:.10f}",
        f"{space}  tags:",
        *[f"{space}    {k}: {v}" for k, v in tags.items()],
        f"{space}  {note}",
    ]


def swap_in_tail(group: dict, copies: list[dict], mean_exist: float) -> dict:
    """Add noisy copies to one language group, keeping the group's share of the audio.

    Inside a group the policy gives each corpus audio in proportion to hours**alpha. The copies join as corpora, so
    they take their policy share of the group's audio and the old corpora scale down together; cut weights follow from
    each corpus's mean cut length. The copies' cuts are shorter than the group's, so at the same weight (which counts
    cuts) the group would now draw less audio: ``stretch`` is the factor on the group's weight that keeps its audio.
    Returns the numbers the report and the line edits need.
    """
    old = group["leaves"]
    mean_old = sum(x["w"] * x["mean"] for x in old)  # seconds of audio per cut drawn from the group
    ref = max(old, key=lambda x: x["hours"])
    # sum of hours**alpha over the old corpora, from the largest one: its audio share is known to 0.3%
    sum_old = ref["hours"] ** ALPHA / (ref["w"] * ref["mean"] / mean_old)
    for c in copies:
        c["hours"] = c["s"] / 3600
        c["mean"] = c["s"] / c["cuts"]
    sum_new = sum(c["hours"] ** ALPHA for c in copies)
    audio_new = sum_new / (sum_old + sum_new)  # the copies' share of the group's audio
    u = [c["hours"] ** ALPHA / c["mean"] for c in copies]
    u = [x / sum(u) for x in u]  # their weights among themselves
    mean_new = sum(x * c["mean"] for x, c in zip(u, copies))
    cut_new = audio_new * mean_old / (audio_new * mean_old + (1 - audio_new) * mean_new)
    mean_group = (1 - cut_new) * mean_old + cut_new * mean_new  # seconds of audio per cut drawn, copies in
    group_audio = group["w"] * mean_old / mean_exist  # the group's share of the existing mix's audio
    return {
        "audio_new": audio_new, "cut_new": cut_new, "u": u, "group_audio": group_audio,
        "mean_group": mean_group, "stretch": mean_old / mean_group,
        "leaf_audio": [audio_new * c["hours"] ** ALPHA / sum_new for c in copies],
    }  # fmt: skip


def apply_synth(
    base: str,
    spec: dict,
    root: str = SYNTH_ROOT,
    share: float | None = None,
    snapshot_every: int | None = SYNTH_SNAPSHOT_EVERY,
) -> tuple[str, dict]:
    """``base`` (a rendered annealing config) with the collection added. Returns the text and a report.

    ``share`` is the share of the audio drawn that the collection's ST leaves get; by default their hours' share of
    all surviving hours, which is what the policy gives ST data (PI, 11-01). The weights are cut probabilities, so
    the share is converted with the mean cut length of the existing mixture and of the collection.
    """
    if "synth_stream" in base:
        raise ValueError("the base config already carries SpeechBlocks-Synth-v1: start from Fondue-anneal.yaml")
    lines = base.splitlines()
    start, end, indent = cmw._find_block(lines, "train_ds", "input_cfg")
    groups = parse_mix(lines, start, end)
    if abs(sum(g["w"] for g in groups) - 1) > 1e-6:
        raise ValueError("the group weights of the base config do not sum to 1")
    mean_exist = sum(g["w"] * sum(x["w"] * x["mean"] for x in g["leaves"]) for g in groups)
    hours_exist = sum(g["hours"] for g in groups)

    # The noisy copies first. Their cuts are shorter than their group's, so a group that takes them in needs a heavier
    # weight (weights count cuts) to keep its audio: every group keeps the audio share it had among the existing groups.
    by_lang = {g["tags"]["lang"]: g for g in groups if g["tags"].get("task") == "asr"}
    copies_of: dict[str, list[dict]] = defaultdict(list)
    for c in spec["asr_leaves"]:
        copies_of[c["lang"]].append(dict(c))
    swaps = {lang: swap_in_tail(by_lang[lang], copies, mean_exist) for lang, copies in sorted(copies_of.items())}
    for g in groups:
        g["stretch"] = 1.0
    for lang, sw in swaps.items():
        by_lang[lang]["stretch"] = sw["stretch"]
    z = sum(g["w"] * g["stretch"] for g in groups)
    # sum(w * stretch * mean cut now) is still sum(w * mean cut before): the same audio over z times the cuts
    mean_now = mean_exist / z

    sources = synth_sources(spec, root)
    hours_synth = sum(s["hours"] for s in sources)
    audio = share if share is not None else hours_synth / (hours_exist + hours_synth)
    cmw.compute_weights(sources, ALPHA, BETA["anneal"])
    cmw.to_cut_probabilities(sources)
    mean_synth = sum(s["p_cl"] * s["mean_cut"] for s in sources)
    cuts = (audio / mean_synth) / (audio / mean_synth + (1 - audio) / mean_now)  # the collection's share of the cuts
    for s in sources:
        s["p_l"] *= cuts
        s["p_l_hours"] *= audio
        s["p_cl_hours"] *= audio
    keep_cuts, keep_audio = 1 - cuts, 1 - audio

    replace: dict[int, str] = {}
    insert: dict[int, list[str]] = defaultdict(list)
    for g in groups:
        g["w_new"] = g["w"] * g["stretch"] / z * keep_cuts
        pad = " " * (g["pad"] + 2)
        replace[g["weight_line"]] = f"{pad}weight: {g['w_new']:.10f}"
        m = GROUP_NOTE.match(lines[g["head"]])
        g["note"] = dict(m.groupdict())
        in_tail = g["tags"].get("task") == "asr" and g["tags"]["lang"] in swaps
        # the old corpora's share of the group's audio
        old_part = 1 - swaps[g["tags"]["lang"]]["audio_new"] if in_tail else 1.0
        for x in g["leaves"]:
            m = LEAF_NOTE.match(lines[x["note_line"]])
            replace[x["note_line"]] = (
                f"{m['pad']}# {m['hours']} h, mean cut {m['mean']} s, "
                f"{float(m['audio']) * keep_audio * old_part:.1f}% of the audio"
            )

    tail = []
    for lang, copies in sorted(copies_of.items()):
        g = by_lang[lang]
        sw = swaps[lang]
        for x in g["leaves"]:
            old = float(WEIGHT_LINE.match(lines[x["weight_line"]])["w"])
            replace[x["weight_line"]] = f"{' ' * (g['pad'] + 6)}weight: {old * (1 - sw['cut_new']):.10f}"
        new_lines = []
        for c, u, a in zip(copies, sw["u"], sw["leaf_audio"]):
            pct = sw["group_audio"] * keep_audio * a
            note = f"# {c['hours']:,.1f} h, mean cut {c['mean']:.1f} s, {pct:.1%} of the audio"
            new_lines += asr_leaf_lines(g["pad"] + 4, c, root, sw["cut_new"] * u, note)
        insert[g["stop"]] += new_lines
        m = g["note"]
        hours = float(m["hours"].replace(",", "")) + sum(c["hours"] for c in copies)
        replace[g["head"]] = (
            f"{m['pad']}# {m['key']}: {hours:,.1f} h across {int(m['n']) + len(copies)} corpora -- "
            f"{g['w_new'] * 100:.1f}% of the cuts, {float(m['audio']) * keep_audio:.1f}% of the audio"
        )
        tail.append(
            {
                "lang": lang,
                "copies": len(copies),
                "hours": sum(c["hours"] for c in copies),
                "audio_new": sw["audio_new"],
                "stretch": sw["stretch"],
            }
        )
    for g in groups:
        if g["head"] not in replace:
            m = g["note"]
            replace[g["head"]] = (
                f"{m['pad']}# {m['key']}: {m['hours']} h across {m['n']} corpora -- "
                f"{g['w_new'] * 100:.1f}% of the cuts, {float(m['audio']) * keep_audio:.1f}% of the audio"
            )

    block = cmw.build_group_block(sources, indent=indent, weights="cuts")[1:]  # without its own "input_cfg:"
    insert[end] += block
    r_start, r_end, _ = cmw._find_block(lines, "cut_filter", "rules")
    insert[r_end] += synth_rules(spec)
    # the renderer's header comment sits just above `input_cfg:`
    note_at = next(i for i in range(start, 0, -1) if re.match(r"^\s+# \d+ corpora, \d+ language entries", lines[i]))
    n_dir = len({s["lang_key"] for s in sources})
    pad = " " * indent
    insert[note_at + 1] += [
        f"{pad}#",
        f"{pad}# SpeechBlocks-Synth-v1 (build_fondue_mix.py synth, spec of {spec['built']}): {audio:.1%} of the audio drawn",
        f"{pad}# ({cuts:.1%} of the cuts) comes from its {len(sources)} ST leaves in {n_dir} direction groups, the last groups",
        f"{pad}# below, weighted by the same alpha/beta over their own kept hours. The groups above are the deployed",
        f"{pad}# mixture with every audio share scaled by {keep_audio:.6f} (the weights, which count cuts, by {keep_cuts / z:.6f}),",
        f"{pad}# except that {len(spec['asr_leaves'])} noisy copies of the native tail sit inside the group of their language. That group",
        f"{pad}# keeps its audio share, and since the copies' cuts are shorter its weight is raised, by {min(s['stretch'] for s in swaps.values()):.2f} to "
        f"{max(s['stretch'] for s in swaps.values()):.2f}x before the scaling.",
        f"{pad}#",
        f"{pad}# Streams (tag synth_stream):",
        *[f"{pad}#   {k}: {v}" for k, v in SYNTH_STREAMS.items()],
    ]
    replace[0] = (
        "# Fondue-anneal-synth.yaml: generated by build_fondue_mix.py `synth` from Fondue-anneal.yaml. Do not hand-edit."
    )
    insert[1] += [
        f"# Fondue-anneal.yaml plus SpeechBlocks-Synth-v1 ({spec['built']}): {audio:.1%} of the audio drawn ({cuts:.1%} of the cuts)",
        f"# comes from its {len(sources)} ST leaves in {n_dir} direction groups. The notes below describe the deployed mixture.",
    ]
    if snapshot_every is not None:
        i = next(i for i, line in enumerate(lines) if line.strip().startswith("snapshot_every_n_steps:"))
        key_pad = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        replace[i] = f"{key_pad}snapshot_every_n_steps: {snapshot_every}"
        insert[i] += [
            f"{key_pad}# SpeechBlocks-Synth-v1 adds {len(sources) + len(spec['asr_leaves'])} leaves to the {sum(len(g['leaves']) for g in groups)} above, and a snapshot",
            f"{key_pad}# walks the whole graph per buffered cut: measured on nyx (one worker, world size 64), 6.5 s a snapshotted batch",
            f"{key_pad}# at 905 leaves against 1.0 s at 150, in the 34 batches of every window. Every 512 that stalls the GPUs ~4 min",
            f"{key_pad}# in ~4.5; every {snapshot_every}, ~{34 * 7.0 / (snapshot_every * 0.52):.0%} of the run. A resume replays up to {snapshot_every} batches.",
        ]

    out: list[str] = []
    for i, line in enumerate(lines):
        out += insert.get(i, [])
        out.append(replace.get(i, line))
    out += insert.get(len(lines), [])
    text = "\n".join(out) + "\n"
    total_hours = float(re.search(r"(?m)^\s*total_hours:\s*([0-9.]+)", base)[1])
    steps = int(re.search(r"(?m)^\s*max_steps:\s*(\d+)", base)[1])
    decay = int(re.search(r"(?m)^\s*num_decay_steps:\s*(\d+)", base)[1])
    report = {
        "audio_share": audio, "cut_share": cuts, "hours_synth": hours_synth, "hours_existing": hours_exist,
        "mean_cut_existing": mean_now, "cut_scale": keep_cuts / z,
        "drawn_hours": total_hours * decay / steps, "sources": sources, "tail": tail, "n_rules": len(spec["rules"]),
        "n_leaves": len(sources) + len(spec["asr_leaves"]),
    }  # fmt: skip
    return text, report


def cmd_synth(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.spec).read_text())
    base = Path(args.base).read_text()
    share = None if args.share == "auto" else float(args.share)
    text, rep = apply_synth(base, spec, args.root, share, args.snapshot_every or None)
    out = Path(args.out)
    out.write_text(text)
    drawn = rep["drawn_hours"]
    print(f"{spec['name']} ({spec['built']}) -> {out}")
    print(
        f"{rep['audio_share']:.2%} of the audio drawn ({rep['cut_share']:.2%} of the cuts): {rep['hours_synth']:,.0f} kept h "
        f"against {rep['hours_existing']:,.0f} h of the deployed mix; the annealing phase draws {drawn:,.0f} h."
    )
    by_group: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    entries: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for s in rep["sources"]:
        a = by_group[s["group"]]
        a[0] += 1
        a[1] += s["hours"]
        a[2] += s["p_cl_hours"]
        entries[s["lang_key"]][0] += s["hours"]
        entries[s["lang_key"]][1] += s["p_cl_hours"]
    print(f"\n{'stream':<7} {'leaves':>6} {'kept h':>9} {'share':>7} {'drawn h':>9} {'passes':>7}")
    for k, (n, h, p) in sorted(by_group.items()):
        print(f"{k:<7} {int(n):>6} {h:>9,.0f} {p:>7.2%} {p * drawn:>9,.0f} {p * drawn / h:>7.2f}")
    print(f"\n{'entry':<8} {'kept h':>9} {'share':>7} {'drawn h':>9} {'passes':>7}   ({len(entries)} directions)")
    rows = sorted(entries.items(), key=lambda kv: -kv[1][1])
    for key, (h, p) in rows[:12] + rows[-5:]:
        print(f"{key:<8} {h:>9,.1f} {p:>7.3%} {p * drawn:>9,.0f} {p * drawn / h:>7.2f}")
    print(f"\n{'tail':<6} {'copies':>6} {'hours':>8} {'audio of its group':>20} {'group weight x':>15}")
    for t in rep["tail"]:
        print(f"{t['lang']:<6} {t['copies']:>6} {t['hours']:>8.1f} {t['audio_new']:>20.1%} {t['stretch']:>15.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("synth", help="add SpeechBlocks-Synth-v1 to the annealing config (reads the rendered text)")
    y.add_argument(
        "--spec",
        default=str(SYNTH_SPEC),
        help="the leaf spec made by data-utils/synth_st/make_mix_spec.py in MELT-proj/preprocessing",
    )
    y.add_argument("--base", default=str(RENDERED["anneal"]))
    y.add_argument("--out", default=str(SYNTH_OUT))
    y.add_argument("--root", default=SYNTH_ROOT, help="where the collection's Shar dirs are on the training host")
    y.add_argument(
        "--share", default="auto", help="audio share of the ST leaves: 'auto' (their hours' share) or e.g. 0.1"
    )
    y.add_argument(
        "--snapshot-every",
        type=int,
        default=SYNTH_SNAPSHOT_EVERY,
        help="data.train_ds.snapshot_every_n_steps for the ~1,000-leaf graph (0 keeps the base value)",
    )
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
    if args.cmd != "synth" and (not args.datasets_root or (args.cmd == "sources" and not args.scores_root)):
        p.error("--datasets-root (and, for sources, --scores-root) or their environment variables are required")
    {"sources": cmd_sources, "render": cmd_render, "synth": cmd_synth}[args.cmd](args)


if __name__ == "__main__":
    main()
