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
VoxPopuli, and every other training language holds enough in *every* one of
those corpora to match it at up to 700.3 h — that number is not a
coincidence, it is exactly all of Italian across those four corpora, so it
is the ceiling for a matched design.

FLEURS was a fifth corpus here (~1% of Italian's mix) but cannot supply even
that share in every language at any budget near 709 h — en and es fall
0.2-1.5 h short — so the campaign runs with ``--exclude-corpus fleurs``.
Nothing in this script excludes it automatically: omit the flag and a
budget above ~590 h fails feasibility the same way it did during design.
FLEURS is still measured and cached like any other source, just not
trained on, so it stays comparable if this decision is revisited.

Hours are **not** enforced by subsetting manifests. A Shar source pairs
``cuts.NNNNNN.jsonl`` positionally with ``recording.NNNNNN.tar``, so a filtered
manifest no longer lines up with its audio, and rewriting the tars would mean
copying audio the disk budget cannot hold. Hours are therefore enforced the way
the trainer already works: sampling weights plus a step budget.

For that to hold, the sources have to be measurable in the first place — the
length filters read ``custom.num_tokens``, which many sources do not carry.
Check the collection with ``verification/check_shar_content.py`` in the
MELT-proj/preprocessing repo before relying on a mixture built here.

``validation_ds`` is rebuilt the same way: one flat, unweighted entry per
training language for every corpus that carries a real held-out split
(``cv22_sidon``, ``mls_sidon``, ``voxpopuli``). ``yodas-granary``'s ``asr_only``
and ``ast`` directories are train-only monolithic scrapes with no held-out
split anywhere on disk, so it is never part of validation, budget or not.
FLEURS is excluded from validation for the same reason it is excluded from
training here (see above) -- it stays comparable, not part of the mix. Entries
are flat ``type: lhotse_shar`` (no ``type: group``, no ``weight:``): eval
concatenates every source's *full* manifest rather than muxing/subsampling it
(``materialize_cuts_for_eval`` does not support ``type: group`` at all), which
is also why validation is not subject to ``--budget-hours``.  Every entry also
carries a ``name`` (``asr_<lang>`` or ``st_<src>_<tgt>``): the trainer
evaluates each name separately and reports ``eval_<name>_loss`` per set.

``--tasks`` picks the task composition, which is what Phase D of the campaign
compares. ``asr`` and ``both`` are *nested*: the ASR-only config holds exactly
the sources of the ASR+ST one minus its ST groups, at the same per-language
budget and the same reference-matched corpus mix, so a difference between the
two runs is the ST data and nothing else. What changes is only the group
weights, which are always proportional to the hours each group contributes
(see ``yaml_block``): dropping the ST groups renormalises the five ASR
languages from 1/9 each to 1/5 each.

Both renders are modality-alignment configs, so ``data.apply_chat_template`` is
pinned to ``false`` whatever the template said. MA for an *instruct* backbone
still runs the chat template with an empty task instruction, but that is a
command-line override on the run (``--data.apply_chat_template true
--data.prompt_template_selection custom --data.prompt_template '{audio_token}'``),
deliberately kept out of the config so the two arms differ only in the shell
history.

Usage::

    # ASR+ST (the mixed arm)
    python3 infra/build_campaign_config.py \\
        --template       config/train/MA-v1.2.yaml \\
        --datasets-root  /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --budget-hours   700.3 \\
        --exclude-corpus fleurs \\
        --tasks          both \\
        --cache          projects/ablation-campaign/campaign_hours.json \\
        --out            config/train/ABL-MA-700.yaml

    # ASR only (the modality-alignment arm), same flags but --tasks asr
    python3 infra/build_campaign_config.py \\
        --template       config/train/MA-v1.2.yaml \\
        --datasets-root  /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --budget-hours   700.3 \\
        --exclude-corpus fleurs \\
        --tasks          asr \\
        --cache          projects/ablation-campaign/campaign_hours.json \\
        --out            config/train/ABL-MA-700-asr.yaml

Run it where the data is; it reads manifests, not audio.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_mix_weights import (  # noqa: E402
    _find_block,
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

# Where a corpus keeps the text the campaign should train on.
#
# The mix has to be consistent in *casing and punctuation* as well as in hours,
# so every corpus trains on cased, punctuated text. YODAS Granary and CoVoST2
# already carry that in the supervision, and CommonVoice carries it outside the
# supervision (which is empty). MLS and FLEURS keep a lowercase, unpunctuated
# supervision and hold the restored text in ``custom.pnc_text``.
#
# This also settles what ``custom.num_tokens`` means: it is the Qwen3-1.7B count
# of whichever field is named here, so ``max_tokens``/``max_tps`` filter on the
# same string the model is trained on. On MLS the two differ by 10.7%.
#
# With ``data.strict_text_field`` a source lacking the named field fails
# loudly at startup rather than silently falling back to the supervision —
# which is the point. The one cut it lets through quietly is the one with no
# text anywhere: nothing to fall back to means nothing to mislabel, so the
# loader skips it as it always did. See ``docs/pnc_coverage.md`` in the
# MELT-proj/preprocessing repo for exactly which sources have ``pnc_text``
# today; do not add a corpus here ahead of its backfill landing.
TEXT_FIELD_OVERRIDES = {
    "cv22_sidon": "custom.metadata.sentence",
    "mls_sidon": "custom.pnc_text",
    "fleurs": "custom.pnc_text",
    "voxpopuli": "custom.pnc_text",
}

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

# ---------------------------------------------------------------------------
# validation_ds: full, unweighted per-language sets from every corpus that
# actually has a held-out split.
# ---------------------------------------------------------------------------

# Corpus -> its held-out split's directory name. Not spelled the same
# everywhere: cv22_sidon and voxpopuli use "validation", MLS uses "dev".
# yodas-granary and fleurs are deliberately absent: yodas-granary's `asr_only`
# and `ast` directories are train-only monolithic scrapes with no held-out
# split on disk at all, and fleurs is excluded from the campaign entirely (see
# the module docstring), training and validation alike.
VALIDATION_SPLIT: dict[str, str] = {
    "cv22_sidon": "validation",
    "mls_sidon": "dev",
    "voxpopuli": "validation",
}

# The en->de ST probe's held-out split (CoVoST2 spells it "dev"). None of the
# X->en directions get one: yodas-granary's `ast` sets are train-only, same as
# their `asr_only` counterparts.
ST_PROBE_VALIDATION_SPLIT = "dev"


def validation_path(train_path: str, split: str) -> str:
    """Swap a source's trailing ``.../train`` segment for its held-out split."""
    prefix, _, leaf = train_path.rpartition("/")
    if leaf != "train":
        raise ValueError(
            f"expected a path ending in '/train' to derive a '{split}' "
            f"validation path from, got {train_path!r}"
        )
    return f"{prefix}/{split}"


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


def build_template(hours: dict[str, float], reference: str,
                    sources: dict[str, dict[str, str]]) -> dict[str, float]:
    """Corpus shares of the reference language, normalised to 1."""
    per_corpus = {
        corpus: hours.get(paths[reference], 0.0)
        for corpus, paths in sources.items()
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


def group_hours(hours: dict[str, float], budget: float,
                tasks: str) -> list[tuple[str, float]]:
    """The top-level groups this task composition trains, and the hours each draws.

    An ASR language always draws the full budget -- feasibility was checked
    first. An ST direction draws whatever it has, capped at the budget: every
    X->en direction clears it several times over, but the en->de probe holds
    only ~430 h and is *meant* to stay a probe rather than be cycled up to a
    full language's worth (see ``ST_SOURCES``' note and the campaign design).
    """
    groups: list[tuple[str, float]] = []
    if tasks in ("asr", "both"):
        groups.extend((f"asr:{lang}", budget) for lang in TRAIN_LANGS)
    if tasks in ("st", "both"):
        for pair, spec in list(ST_SOURCES.items()) + list(ST_PROBE.items()):
            groups.append((f"st:{pair}", min(budget, hours.get(spec["path"], 0.0))))
    return groups


def yaml_block(template: dict[str, float], hours: dict[str, float],
               budget: float, tasks: str) -> tuple[list[str], float]:
    """Render the input_cfg block, and return it with the total hours it implies.

    Group weights are the group's share of the mixture's *hours*, not a flat
    ``1/len(groups)``: a group that draws fewer hours than the budget has to be
    sampled proportionally less often, or the mixture the trainer actually
    draws stops matching the ``total_hours`` written beside it. This only ever
    bites the en->de probe (~430 h against a 700 h budget); with ``--tasks
    asr`` every group draws the same budget and the weights collapse back to
    ``1/len(TRAIN_LANGS)`` exactly.
    """
    lines: list[str] = ["    input_cfg:"]

    groups = group_hours(hours, budget, tasks)
    mixture_hours = sum(h for _, h in groups)
    if mixture_hours <= 0:
        raise SystemExit(f"No trainable hours for --tasks {tasks}.")
    total_hours = 0.0

    for name, drawn in groups:
        kind, key = name.split(":", 1)
        group_weight = drawn / mixture_hours

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
            total_hours += drawn

        else:
            spec = ST_SOURCES.get(key) or ST_PROBE[key]
            rel = spec["path"]
            available = hours.get(rel, 0.0)
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


def validation_yaml_block(asr_sources: dict[str, dict[str, str]], hours: dict[str, float],
                          tasks: str) -> tuple[list[str], float]:
    """Render validation_ds's ``input_cfg`` block: full, unweighted, per-language.

    Flat ``type: lhotse_shar`` entries only (see the module docstring) --
    every language in ``TRAIN_LANGS`` gets one entry per corpus in
    ``asr_sources`` that also has a held-out split (``VALIDATION_SPLIT``), plus
    the en->de ST probe's split if *tasks* trains it. No ``weight:``: eval
    reads each source's full manifest, which is the point (no subsampling).

    Every entry carries a ``name`` (per language for ASR, per direction for
    ST).  The trainer splits ``validation_ds`` on those names and reports each
    set separately, so a run logs ``eval_asr_<lang>_loss`` /
    ``eval_st_<src>_<tgt>_loss`` plus per-set WER/CER instead of one flat
    ``eval_loss`` over everything.
    """
    lines: list[str] = ["    input_cfg:"]
    total_hours = 0.0

    if tasks in ("asr", "both"):
        val_corpora = [c for c in VALIDATION_SPLIT if c in asr_sources]
        for lang in TRAIN_LANGS:
            lines.append(f"      # ASR {lang} validation: full {'/'.join(val_corpora)} sets")
            for corpus in val_corpora:
                split = VALIDATION_SPLIT[corpus]
                rel = validation_path(ASR_SOURCES[corpus][lang], split)
                h = hours.get(rel, 0.0)
                lines.append("      - type: lhotse_shar")
                lines.append(f"        name: asr_{lang}")
                lines.append(
                    f"        shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{rel}"
                )
                lines.append("        tags:")
                lines.append("          task: asr")
                lines.append(f"          lang: {lang}")
                if corpus in TEXT_FIELD_OVERRIDES:
                    lines.append(
                        f"          text_field: {TEXT_FIELD_OVERRIDES[corpus]}"
                    )
                lines.append(f"        # {h:,.1f} h of {corpus} ({split} split)")
                total_hours += h

    if tasks in ("st", "both"):
        spec = ST_PROBE["en-de"]
        rel = validation_path(spec["path"], ST_PROBE_VALIDATION_SPLIT)
        h = hours.get(rel, 0.0)
        lines.append(
            f"      # ST {spec['src']}->{spec['tgt']} validation "
            f"({ST_PROBE_VALIDATION_SPLIT} split): {h:,.1f} h. No X->en probe: "
            "yodas-granary's `ast` sets have no held-out split."
        )
        lines.append("      - type: lhotse_shar")
        lines.append(f"        name: st_{spec['src']}_{spec['tgt']}")
        lines.append(f"        shar_path: ${{oc.env:LOCAL_DATASETS_DIR}}/{rel}")
        lines.append("        tags:")
        lines.append("          task: st")
        lines.append(f"          src_lang: {spec['src']}")
        lines.append(f"          tgt_lang: {spec['tgt']}")
        total_hours += h

    return lines, total_hours


def _section_span(lines: list[str], section: str) -> tuple[int, int, int]:
    """Index range ``[start, end)`` of a top-level ``section:`` block, and its indent."""
    sec = next((i for i, l in enumerate(lines)
                if re.match(rf"^\s*{section}:\s*$", l)), None)
    if sec is None:
        raise ValueError(f"Could not find a '{section}:' section in the template")
    sec_indent = len(lines[sec]) - len(lines[sec].lstrip())
    end = len(lines)
    for i in range(sec + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= sec_indent:
            end = i
            break
    return sec, end, sec_indent


def _replace_scalar_in_section(lines: list[str], section: str, key: str, value: str) -> None:
    """Rewrite the first ``key: ...`` scalar found within ``section:``'s own block.

    Unlike ``compute_mix_weights._replace_scalar`` (an unbounded forward scan),
    this stays inside *section* -- a config with two ``total_hours:`` scalars
    (train_ds and validation_ds) would otherwise have the wrong one overwritten
    whenever the search started above both of them.
    """
    start, end, _ = _section_span(lines, section)
    pattern = re.compile(rf"^(\s*){key}:\s*\S.*$")
    for i in range(start, end):
        m = pattern.match(lines[i])
        if m:
            lines[i] = f"{m.group(1)}{key}: {value}"
            return
    raise ValueError(f"Could not find '{key}:' scalar under '{section}:' in the template")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path,
                        help="Config to copy everything-but-input_cfg from.")
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--budget-hours", type=float, default=709.0,
                        help="Hours per language (ASR) and per direction (ST).")
    parser.add_argument("--tasks", choices=("asr", "st", "both"), default="both")
    parser.add_argument("--reference-lang", default=REFERENCE_LANG)
    parser.add_argument("--exclude-corpus", nargs="*", default=[],
                        choices=sorted(ASR_SOURCES),
                        help="Drop these corpora from the ASR domain template "
                             "entirely (e.g. one too small to hit the budget "
                             "in every language). Still measured and cached, "
                             "just not trained on.")
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

    asr_sources = {c: p for c, p in ASR_SOURCES.items()
                   if c not in args.exclude_corpus}
    val_corpora = [c for c in VALIDATION_SPLIT if c in asr_sources]

    # Every ASR corpus is measured even when excluded from the mix, so the
    # cache stays comparable if an exclusion is revisited. The ST sources are
    # not: with --tasks asr they contribute nothing to either block, and
    # yodas-granary's `ast` sets are the largest sources in the collection
    # (es->en alone is 25,833 h) — measuring them would dominate a cold run's
    # I/O to produce numbers the config never uses.
    paths = [p for corpus in ASR_SOURCES.values() for p in corpus.values()]
    if args.tasks in ("asr", "both"):
        paths += [
            validation_path(ASR_SOURCES[corpus][lang], VALIDATION_SPLIT[corpus])
            for lang in TRAIN_LANGS
            for corpus in val_corpora
        ]
    if args.tasks in ("st", "both"):
        paths += [s["path"] for s in ST_SOURCES.values()]
        paths += [s["path"] for s in ST_PROBE.values()]
        paths.append(validation_path(ST_PROBE["en-de"]["path"], ST_PROBE_VALIDATION_SPLIT))

    print(f"Measuring {len(paths)} sources under {args.datasets_root}")
    hours = measure(paths, args.datasets_root, cache, args.cache,
                    args.sample_shards, args.jobs)

    template = build_template(hours, args.reference_lang, asr_sources)
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

    groups = group_hours(hours, args.budget_hours, args.tasks)
    mixture_hours = sum(h for _, h in groups)
    print(f"\nTop-level groups for --tasks {args.tasks} "
          f"(weight = share of the mixture's hours):")
    for name, drawn in groups:
        print(f"  {name:<12} {drawn / mixture_hours:.6f}  {drawn:8,.1f} h")

    lines, total = yaml_block(template, hours, args.budget_hours, args.tasks)
    val_lines, val_total = validation_yaml_block(asr_sources, hours, args.tasks)

    out_lines = args.template.read_text(encoding="utf-8").splitlines()
    try:
        for section, block, section_total in (
            ("train_ds", lines, total),
            ("validation_ds", val_lines, val_total),
        ):
            start, end, _ = _find_block(out_lines, section, "input_cfg")
            out_lines[start:end] = block
            _replace_scalar_in_section(out_lines, section, "total_hours", f"{section_total:.2f}")
        # Both renders are MA-stage configs, so pin this rather than inherit it:
        # rendering from an SFT template would otherwise quietly train modality
        # alignment under a chat template. The instruct arms turn it back on per
        # run from the command line (see the module docstring).
        _replace_scalar_in_section(out_lines, "data", "apply_chat_template", "false")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if args.sample_shards:
        print(
            f"\n*** WARNING: built from {args.sample_shards} shards per source. "
            "Extrapolation error on the large corpora runs to tens of percent, "
            "which moves the domain template itself — the whole point of this "
            "config. Do NOT train on this. Re-run without --sample-shards. ***"
        )

    print(f"\nWrote {args.out}")
    print(f"  tasks               : {args.tasks}")
    print(f"  hours/language      : {args.budget_hours:,.1f}")
    print(f"  total train hours   : {total:,.1f}")
    print(f"  total val hours     : {val_total:,.1f} "
          f"(full sets, not subsampled, from: {', '.join(val_corpora) or 'none'})")
    print(f"  held out            : {', '.join(HELD_OUT_LANGS)}")
    print(
        "\nHours are enforced by weights plus a step budget, not by subsetting.\n"
        "Set trainer.max_steps so the run consumes the intended audio:\n"
        "  max_steps = total_hours * 3600 / (batch_duration * world_size * grad_accum)\n"
        "and run preprocessing's `verification/check_shar_content.py` first —\n"
        "the length filters are inert on any source lacking custom.num_tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
