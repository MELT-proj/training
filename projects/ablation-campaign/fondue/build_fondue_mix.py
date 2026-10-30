#!/usr/bin/env python3
"""Render Fondue's training config: every source of the pool, tagged, weighted by language and corpus.

Fondue trains on the whole pool for its languages (PI, 2026-10-29 and 10-30):

* **ASR**: the 24 official EU languages, Ukrainian and Russian (PI, 2026-10-30: no
  regional languages).
* **ST X->en**: YODAS-Granary ``ast`` (``custom.translation_en``), CoVoST2, Europarl-ST.
* **ST en->X**: CoVoST2, Europarl-ST, and SPITE (CV16 and GigaSpeech English, machine
  translated) for its EU targets only: de, es, fr, it, nl, pt.
* **ST X->Y** between two non-English languages (Europarl-ST): only with ``--x-to-y``.

Weights: first a fixed split of the audio between the two tasks, ``--st-share`` of it to ST
(default ST_SHARE). Inside each task, the two-tier policy of ``infra/compute_mix_weights.py``
(Canary, arXiv:2509.14128 §3.3.1): alpha 0.2 within an entry, beta 0.5 across the task's
entries. Every ASR language and every ST direction is one entry. Without the split, beta
runs over ASR and ST entries together and the ST share falls out of how many directions
there are: SPITE alone, six directions over the same English audio, took ~19% of the run.
Written as cut probabilities.

Tags the cut filter matches on, set here so the rules in the template find their sources:

* ``scored: wer`` on an ASR leaf with a WER array under ``--scores-root``; ``scored: none``
  on one without (the ``asr_wer`` rule would fail at startup on a missing array);
* ``scored: spite`` on SPITE, ``scored: none`` on every other ST leaf.

Two steps, because measuring the pool is a full pass over its manifests:

    # 1. the sources, unweighted, into Fondue-sources.yaml (audits text fields first)
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py sources \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --scores-root   /mnt/scratch-nyx/giuseppe/melt/melt-data/shar-asr-scores

    # 2. measure (cached in infra/.mix_weights_hours.json; resumes), weight, write Fondue.yaml
    python3 infra/compute_mix_weights.py --config projects/ablation-campaign/fondue/Fondue-sources.yaml \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar --jobs 32
    python3 projects/ablation-campaign/fondue/build_fondue_mix.py render \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar

``render`` keeps the template's ``total_hours`` (the hours Fondue draws), not the pool's
size, and prints, per entry, its share, the hours it draws and how many passes over its
data that is. The alpha/beta shares are computed on the hours BEFORE the cut filter.
"""

from __future__ import annotations

import argparse
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
RENDERED = HERE / "Fondue.yaml"

ALPHA = 0.2
BETA = 0.5
# Share of the drawn AUDIO that is ST. ASR is the larger task and the base the
# instruction-following rests on; a quarter for ST still gives it ~80K h at the 330K h
# budget, more than the whole X->en pool. The annealing phase drops the unscored ST
# sources, so ST's real share falls there (to SPITE alone).
ST_SHARE = 0.25

EU24 = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr",
    "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
)  # fmt: skip
# Regional and minority languages of member states with training data on disk.
ASR_LANGS = EU24 + ("uk", "ru")
SPITE_TARGETS = ("de", "es", "fr", "it", "nl", "pt")
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
    "SPITE ko/ru/zh": "only SPITE's EU targets (PI, 10-30)",
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
        out.append(leaf(rel, tags))
    return out


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


def xcomet_rules(c: float, a: float) -> list[str]:
    """One xCOMET rule per SPITE target: a looser band while the LR is constant, a tighter one while it anneals."""
    lines = []
    for tgt in SPITE_TARGETS:
        lines += [
            f"        st_xcomet_{tgt}:",
            f"          match: {{task: st, scored: spite, tgt_lang: {tgt}}}",
            f"          field: custom.spite.scores.{tgt}.xcomet_xl",
            "          phases:",
            f"            - {{from_step: 0, min: {c}}}",
            f"            - {{from_step: 0, min: {a}}} # ANNEAL_STEP",
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


def anneal_step(lines: list[str], nodes: int, gpus_per_node: int) -> int:
    """Where the LR decay starts: max_steps - max_steps / 5, with max_steps from the template's budget and batch."""

    def scalar(key: str) -> float:
        for line in lines:
            m = re.match(rf"^\s*{key}:\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
        raise ValueError(f"{key} not found in the template")

    per_step = scalar("batch_duration") * scalar("gradient_accumulation_steps") * nodes * gpus_per_node
    max_steps = int(scalar("total_hours") * 3600 / per_step)
    return max_steps - max_steps // 5


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
    step = anneal_step(lines, args.nodes, args.gpus_per_node)
    text = "\n".join(lines).replace(
        "@@XCOMET_RULES@@", "\n".join(xcomet_rules(args.xcomet_constant, args.xcomet_anneal))
    )
    text = re.sub(r"\{from_step: 0, (\S+: [^}]+)\} # ANNEAL_STEP", rf"{{from_step: {step}, \1}} # ANNEAL_STEP", text)
    header = [
        "# Fondue-sources.yaml: generated by build_fondue_mix.py `sources`. Do not hand-edit.",
        "# The pool, unweighted; build_fondue_mix.py `render` writes the weighted Fondue.yaml.",
        f"# ANNEAL_STEP {step}: max_steps - max_steps/5 at {args.nodes} nodes x {args.gpus_per_node} GPUs,",
        "# from the template's total_hours, batch_duration and grad_accum; it moves with any of them.",
    ]
    text = text.replace("# @@HEADER@@", "\n".join(header))
    lines = text.splitlines()
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
        f"({n_asr} ASR, {len(entries) - n_asr} ST), {len(val)} validation sets; ANNEAL_STEP {step}."
    )
    unscored = [e["rel"] for e in train if e["tags"]["task"] == "asr" and e["tags"]["scored"] == "none"]
    print(f"ASR leaves without a WER array (the asr_wer rule skips them): {len(unscored)}")
    for rel in unscored:
        print(f"  {rel}")


def split_weights(sources: list[dict], st_share: float) -> None:
    """The policy inside each task, then the task split: p_l and p_cl scaled by the task's share of the audio."""
    if not 0 < st_share < 1:
        raise ValueError(f"--st-share must be in (0, 1), got {st_share}")
    for task, share in (("asr", 1 - st_share), ("st", st_share)):
        part = [s for s in sources if s["task"] == task]
        cmw.compute_weights(part, ALPHA, BETA)
        for s in part:
            s["p_l"] *= share
            s["p_cl"] *= share


def cmd_render(args: argparse.Namespace) -> None:
    os.environ.setdefault("LOCAL_ASR_SCORES_DIR", "/unused")  # only the cut filter reads it
    sources = cmw.load_sources(SOURCES, args.datasets_root)
    cache = json.loads(Path(args.cache).read_text()) if Path(args.cache).exists() else {}
    todo = sorted(
        {
            s["path"]
            for s in sources
            if s["path"] not in cache
            or cache[s["path"]].get("sample") is not None
            or cache[s["path"]].get("cuts") is None
        }
    )
    if todo:
        sys.exit(
            f"{len(todo)} sources are not measured (run infra/compute_mix_weights.py first):\n  " + "\n  ".join(todo)
        )
    for s in sources:
        s["hours"], s["cuts"] = cache[s["path"]]["hours"], cache[s["path"]]["cuts"]
    split_weights(sources, args.st_share)
    cmw.to_cut_probabilities(sources)

    lines = SOURCES.read_text().splitlines()
    budget = None
    for line in lines:
        m = re.match(r"^\s*total_hours:\s*([0-9.]+)", line)
        if m:
            budget = float(m.group(1))
            break
    cmw.emit_training_config(sources, SOURCES, RENDERED, ALPHA, BETA, exp_name="Fondue", weights="cuts")
    # emit_training_config writes the pool's size into total_hours; Fondue draws the budget.
    text = RENDERED.read_text()
    text = re.sub(r"(?m)^(\s*total_hours:)\s*[0-9.]+", rf"\1 {budget:.1f}", text, count=1)
    text = text.replace(
        "# Fondue-sources.yaml: generated by build_fondue_mix.py `sources`. Do not hand-edit.",
        "# Fondue.yaml: generated by build_fondue_mix.py `render`. Do not hand-edit.",
    ).replace(
        "# The pool, unweighted; build_fondue_mix.py `render` writes the weighted Fondue.yaml.",
        f"# The pool weighted: {args.st_share:.0%} of the audio to ST, then alpha {ALPHA} / beta {BETA} inside\n"
        "# each task (the two-tier comment below runs beta inside ASR and inside ST, not across both).\n"
        "# total_hours is what Fondue draws, not the pool.",
    )
    RENDERED.write_text(text)

    by_entry: dict[str, list[dict]] = defaultdict(list)
    for s in sources:
        by_entry[s["lang_key"]].append(s)
    pool = sum({s["path"]: s["hours"] for s in sources}.values())
    print(
        f"\nPool: {pool:,.0f} h of distinct leaves; Fondue draws {budget:,.0f} h. "
        f"ST {args.st_share:.0%} of the audio; alpha {ALPHA}, beta {BETA} inside each task; {len(by_entry)} entries.\n"
    )
    print(f"{'entry':<10} {'pool h':>10} {'share':>7} {'drawn h':>10} {'passes':>7}  largest corpus")
    rows = []
    for key, ss in by_entry.items():
        held = sum(s["hours"] for s in ss)
        share = ss[0]["p_l_hours"]
        top = max(ss, key=lambda s: s["p_c_hours"])
        rows.append(
            (
                key,
                held,
                share,
                share * budget,
                share * budget / held,
                f"{Path(top['raw_path']).relative_to('${oc.env:LOCAL_DATASETS_DIR}')} {top['p_c_hours']:.0%}",
            )
        )
    for key, held, share, drawn, passes, top in sorted(rows, key=lambda r: -r[2]):
        print(f"{key:<10} {held:>10,.1f} {share:>7.2%} {drawn:>10,.0f} {passes:>7.2f}  {top}")
    ast = sum(r[3] for r in rows if "-" in r[0])
    print(f"\nST entries draw {ast:,.0f} h ({ast / budget:.1%}); ASR {budget - ast:,.0f} h.")
    print(f"Wrote {RENDERED.relative_to(REPO_ROOT)}")
    if args.json:
        # Shares inside a task do not depend on the split, so a page can re-split them.
        task_share = {"asr": 1 - args.st_share, "st": args.st_share}
        out = {
            "alpha": ALPHA,
            "beta": BETA,
            "st_share": args.st_share,
            "budget_hours": budget,
            "entries": [
                {
                    "key": key,
                    "task": by_entry[key][0]["task"],
                    "pool_h": round(held, 1),
                    "within": share / task_share[by_entry[key][0]["task"]],
                    "top": top,
                }
                for key, held, share, _, _, top in sorted(rows, key=lambda r: -r[2])
            ],
        }
        args.json.write_text(json.dumps(out, indent=1) + "\n")
        print(f"Wrote {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources", help="write Fondue-sources.yaml")
    s.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR"), required=False)
    s.add_argument("--scores-root", default=os.environ.get("LOCAL_ASR_SCORES_DIR"))
    s.add_argument("--x-to-y", action="store_true", help="add Europarl-ST between two non-English languages")
    s.add_argument("--no-audit", action="store_true")
    s.add_argument("--nodes", type=int, default=16)
    s.add_argument("--gpus-per-node", type=int, default=4)
    s.add_argument("--xcomet-constant", type=float, default=0.9)
    s.add_argument("--xcomet-anneal", type=float, default=0.97)
    r = sub.add_parser("render", help="weight Fondue-sources.yaml into Fondue.yaml")
    r.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR"))
    r.add_argument("--st-share", type=float, default=ST_SHARE, help="share of the drawn audio that is ST")
    r.add_argument("--json", type=Path, default=None, help="also write the per-entry table (for the thresholds page)")
    r.add_argument("--cache", default=str(REPO_ROOT / "infra" / ".mix_weights_hours.json"))
    args = p.parse_args()
    if not args.datasets_root or (args.cmd == "sources" and not args.scores_root):
        p.error("--datasets-root (and, for sources, --scores-root) or their environment variables are required")
    {"sources": cmd_sources, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
