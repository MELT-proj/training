#!/usr/bin/env python3
"""Build a training config for the ablation campaign: equal hours per language,
matched domain mix, and one task composition.

The campaign's central claim is that a cross-language comparison measures
*language*, not corpus. That only holds if every language is trained on the same
mix of domains as well as the same number of hours, so this does not use the
alpha/beta policy in ``compute_mix_weights.py``. Instead:

* every training language gets an **equal share** of the mix (``1/N``), and
* within a language, corpora are drawn in the **proportions of a reference
  language** (Italian by default, because it is the scarcest and therefore the
  one that binds).

Italian's natural mix is roughly 36% CommonVoice, 35% MLS, 17% YODAS, 11%
VoxPopuli, 1% FLEURS, and every other training language holds enough in *every*
one of those corpora to match it at 709 h. That number is not a coincidence —
709 h is exactly all of Italian, so it is the ceiling for a matched design.

Hours are **not** enforced by subsetting manifests. A Shar source pairs
``cuts.NNNNNN.jsonl`` positionally with ``recording.NNNNNN.tar``, so a filtered
manifest no longer lines up with its audio, and rewriting the tars would mean
copying audio the disk budget cannot hold. Hours are therefore enforced the way
the trainer already works: sampling weights plus a step budget.

For that to hold, the sources have to be measurable in the first place — the
length filters read ``custom.num_tokens``, which many sources do not carry (see
``infra/audit_num_tokens.py``). Audit the collection before relying on a mixture
built here.

Usage::

    python3 infra/build_campaign_config.py \\
        --template      config/train/MA-v1.2.yaml \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --budget-hours  709 \\
        --tasks         both \\
        --cache         campaign_hours.json \\
        --out           config/train/ABL-MA-709.yaml

Run it where the data is; it reads manifests, not audio.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_mix_weights import (  # noqa: E402
    measure_shard,
    plan_source,
    write_cache,
)


# ---------------------------------------------------------------------------
# The campaign's language and corpus design
# ---------------------------------------------------------------------------

# Training languages. These five are the only ones sharing all five corpora:
# Dutch is MLS-only and Russian is YODAS-only, so including either would make a
# cross-language gap partly a corpus-domain gap. Both are held out instead.
TRAIN_LANGS = ["en", "de", "fr", "es", "it"]

# Held out entirely, for zero-shot probes. nl and pt give
# unseen-language-within-a-seen-family (Germanic, Romance); pl and uk give the
# unseen-family contrast, since no Slavic language is trained on.
HELD_OUT_LANGS = ["nl", "pt", "pl", "uk"]

# The reference language whose corpus proportions every language copies.
REFERENCE_LANG = "it"

# Per-corpus directory naming, which differs by corpus.
ASR_SOURCES: dict[str, dict[str, str]] = {
    "cv22_sidon": {
        "en": "cv22_sidon/en/train",
        "de": "cv22_sidon/de/train",
        "fr": "cv22_sidon/fr/train",
        "es": "cv22_sidon/es/train",
        "it": "cv22_sidon/it/train",
    },
    "mls_sidon": {
        "en": "mls_sidon/english/train",
        "de": "mls_sidon/german/train",
        "fr": "mls_sidon/french/train",
        "es": "mls_sidon/spanish/train",
        "it": "mls_sidon/italian/train",
    },
    "yodas-granary": {
        "en": "yodas-granary/English/asr_only",
        "de": "yodas-granary/German/asr_only",
        "fr": "yodas-granary/French/asr_only",
        "es": "yodas-granary/Spanish/asr_only",
        "it": "yodas-granary/Italian/asr_only",
    },
    "voxpopuli": {
        "en": "voxpopuli/en/train",
        "de": "voxpopuli/de/train",
        "fr": "voxpopuli/fr/train",
        "es": "voxpopuli/es/train",
        "it": "voxpopuli/it/train",
    },
    "fleurs": {
        "en": "fleurs/en_us/train",
        "de": "fleurs/de_de/train",
        "fr": "fleurs/fr_fr/train",
        "es": "fleurs/es_419/train",
        "it": "fleurs/it_it/train",
    },
}

# CommonVoice keeps its transcript outside the supervision.
TEXT_FIELD_OVERRIDES = {"cv22_sidon": "custom.metadata.sentence"}

# Speech translation. Genuine en->X barely exists in this collection (CoVoST2
# en_de at 430 h is the only direction among the training languages), whereas
# X->en is available at 6,000-25,000 h per language from YODAS Granary, whose
# `ast` sets hold the English translation at custom.translation_en. The en->de
# entry is kept as a small probe that the model can translate *out* of English
# at all.
ST_SOURCES: dict[str, dict] = {
    "de-en": {"path": "yodas-granary/German/ast", "src": "de", "tgt": "en"},
    "fr-en": {"path": "yodas-granary/French/ast", "src": "fr", "tgt": "en"},
    "es-en": {"path": "yodas-granary/Spanish/ast", "src": "es", "tgt": "en"},
    "it-en": {"path": "yodas-granary/Italian/ast", "src": "it", "tgt": "en"},
}
ST_PROBE = {"en-de": {"path": "covost2/en_de/train", "src": "en", "tgt": "de"}}

TRANSLATION_FIELD = "custom.translation_en"


def measure(paths: list[str], root: Path, cache: dict, cache_path: Path | None,
            sample: int | None, jobs: int) -> dict[str, float]:
    """Return hours per source path, reading manifests and reusing the cache."""
    todo = [p for p in paths if p not in cache]
    if todo:
        tasks: list[tuple[str, str]] = []
        totals: dict[str, int] = {}
        for rel in todo:
            shards, total = plan_source(str(root / rel), sample)
            if not shards:
                print(f"  WARNING: no manifests under {rel}", file=sys.stderr)
                cache[rel] = 0.0
                continue
            totals[rel] = total
            tasks.extend((rel, s) for s in shards)

        seconds: dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for rel, secs in pool.map(measure_shard, tasks, chunksize=8):
                seconds[rel] = seconds.get(rel, 0.0) + secs

        for rel, secs in seconds.items():
            read = min(totals[rel], sample) if sample else totals[rel]
            scale = totals[rel] / read if read else 1.0
            cache[rel] = secs / 3600.0 * scale
            if cache_path:
                write_cache(cache, cache_path)

    return {p: cache.get(p, 0.0) for p in paths}


def build_template(hours: dict[str, float], reference: str) -> dict[str, float]:
    """Corpus shares of the reference language, normalised to 1."""
    per_corpus = {
        corpus: hours.get(paths[reference], 0.0)
        for corpus, paths in ASR_SOURCES.items()
    }
    total = sum(per_corpus.values())
    if total <= 0:
        raise SystemExit(
            f"Reference language '{reference}' measured 0 h across all corpora — "
            "check --datasets-root."
        )
    return {corpus: h / total for corpus, h in per_corpus.items()}


def check_feasible(hours: dict[str, float], template: dict[str, float],
                   budget: float) -> list[str]:
    """Report any (language, corpus) that cannot supply its template share."""
    problems = []
    for lang in TRAIN_LANGS:
        for corpus, share in template.items():
            need = budget * share
            have = hours.get(ASR_SOURCES[corpus][lang], 0.0)
            if have < need:
                problems.append(
                    f"  {lang}/{corpus}: needs {need:,.1f} h, has {have:,.1f} h "
                    f"(short by {need - have:,.1f} h)"
                )
    return problems


def yaml_block(template: dict[str, float], hours: dict[str, float],
               budget: float, tasks: str) -> tuple[list[str], float]:
    """Render the input_cfg block, and return it with the total hours it implies."""
    lines: list[str] = ["    input_cfg:"]

    groups: list[tuple[str, float]] = []
    if tasks in ("asr", "both"):
        groups.extend((f"asr:{lang}", 1.0) for lang in TRAIN_LANGS)
    if tasks in ("st", "both"):
        groups.extend((f"st:{pair}", 1.0) for pair in ST_SOURCES)
        groups.append(("st:en-de", 1.0))

    group_weight = 1.0 / len(groups)
    total_hours = 0.0

    for name, _ in groups:
        kind, key = name.split(":", 1)

        if kind == "asr":
            lang = key
            lines.append(f"      # ASR {lang}: {budget:,.1f} h, reference-matched mix")
            lines.append("      - type: group")
            lines.append(f"        weight: {group_weight:.8f}")
            lines.append("        tags:")
            lines.append("          task: asr")
            lines.append(f"          lang: {lang}")
            lines.append("        input_cfg:")
            for corpus, share in sorted(template.items(), key=lambda kv: -kv[1]):
                rel = ASR_SOURCES[corpus][lang]
                lines.append("          - type: lhotse_shar")
                lines.append(
                    f"            shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{rel}"
                )
                lines.append(f"            weight: {share:.8f}")
                lines.append("            tags:")
                lines.append("              task: asr")
                lines.append(f"              lang: {lang}")
                if corpus in TEXT_FIELD_OVERRIDES:
                    lines.append(
                        f"              text_field: {TEXT_FIELD_OVERRIDES[corpus]}"
                    )
                lines.append(f"            # {share * budget:,.1f} h of {corpus}")
            total_hours += budget

        else:
            spec = ST_SOURCES.get(key) or ST_PROBE[key]
            rel = spec["path"]
            available = hours.get(rel, 0.0)
            drawn = min(budget, available)
            lines.append(
                f"      # ST {spec['src']}->{spec['tgt']}: {drawn:,.1f} h "
                f"({available:,.1f} h available)"
            )
            lines.append("      - type: group")
            lines.append(f"        weight: {group_weight:.8f}")
            lines.append("        tags:")
            lines.append("          task: st")
            lines.append(f"          src_lang: {spec['src']}")
            lines.append(f"          tgt_lang: {spec['tgt']}")
            lines.append("        input_cfg:")
            lines.append("          - type: lhotse_shar")
            lines.append(
                f"            shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{rel}"
            )
            lines.append("            weight: 1.00000000")
            lines.append("            tags:")
            lines.append("              task: st")
            lines.append(f"              src_lang: {spec['src']}")
            lines.append(f"              tgt_lang: {spec['tgt']}")
            # YODAS `ast` keeps the transcript in the supervision and the English
            # translation in custom; CoVoST2 puts the translation in the
            # supervision. Getting this wrong trains ASR under an ST prompt.
            if "yodas-granary" in rel:
                lines.append(f"              text_field: {TRANSLATION_FIELD}")
            total_hours += drawn

    return lines, total_hours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path,
                        help="Config to copy everything-but-input_cfg from.")
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--budget-hours", type=float, default=709.0,
                        help="Hours per language (ASR) and per direction (ST).")
    parser.add_argument("--tasks", choices=("asr", "st", "both"), default="both")
    parser.add_argument("--reference-lang", default=REFERENCE_LANG)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--sample-shards", type=int, default=None,
                        help="Read only N shards per source and extrapolate. For "
                             "a quick look only — do not ship a config built this way.")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import json

    cache = {}
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text())

    paths = [p for corpus in ASR_SOURCES.values() for p in corpus.values()]
    paths += [s["path"] for s in ST_SOURCES.values()]
    paths += [s["path"] for s in ST_PROBE.values()]

    print(f"Measuring {len(paths)} sources under {args.datasets_root}")
    hours = measure(paths, args.datasets_root, cache, args.cache,
                    args.sample_shards, args.jobs)

    template = build_template(hours, args.reference_lang)
    print(f"\nDomain template from '{args.reference_lang}':")
    for corpus, share in sorted(template.items(), key=lambda kv: -kv[1]):
        print(f"  {corpus:<16} {share * 100:5.1f}%  "
              f"{share * args.budget_hours:8,.1f} h at the requested budget")

    problems = check_feasible(hours, template, args.budget_hours)
    if problems:
        print(f"\nBudget of {args.budget_hours:,.1f} h/language is not feasible:")
        print("\n".join(problems))
        ceiling = min(
            hours.get(ASR_SOURCES[c][lang], 0.0) / share
            for lang in TRAIN_LANGS
            for c, share in template.items()
            if share > 0
        )
        print(f"\nThe matched ceiling is {ceiling:,.1f} h/language.")
        return 1

    lines, total = yaml_block(template, hours, args.budget_hours, args.tasks)

    text = args.template.read_text().splitlines(keepends=True)
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(text):
        line = text[i]
        if line.startswith("    input_cfg:") and not replaced:
            out.extend(s + "\n" for s in lines)
            i += 1
            # Skip the template's own block.
            while i < len(text) and (
                not text[i].strip() or text[i].startswith("      ")
                or text[i].startswith("        ")
            ):
                i += 1
            replaced = True
            continue
        if line.strip().startswith("total_hours:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}total_hours: {total:.2f}\n")
            i += 1
            continue
        out.append(line)
        i += 1

    if not replaced:
        print("error: no '    input_cfg:' block found in the template",
              file=sys.stderr)
        return 2

    args.out.write_text("".join(out))

    if args.sample_shards:
        print(
            f"\n*** WARNING: built from {args.sample_shards} shards per source. "
            "Extrapolation error on the large corpora runs to tens of percent, "
            "which moves the domain template itself — the whole point of this "
            "config. Do NOT train on this. Re-run without --sample-shards. ***"
        )

    print(f"\nWrote {args.out}")
    print(f"  tasks           : {args.tasks}")
    print(f"  hours/language  : {args.budget_hours:,.1f}")
    print(f"  total hours     : {total:,.1f}")
    print(f"  held out        : {', '.join(HELD_OUT_LANGS)}")
    print(
        "\nHours are enforced by weights plus a step budget, not by subsetting.\n"
        "Set trainer.max_steps so the run consumes the intended audio:\n"
        "  max_steps = total_hours * 3600 / (batch_duration * world_size * grad_accum)\n"
        "and check `infra/audit_num_tokens.py` first — the length filters are\n"
        "inert on any source lacking custom.num_tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
