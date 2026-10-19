#!/usr/bin/env python3
"""Render the two SFT data conditions of the self-distillation study, HR5 and EU24.

Both conditions fine-tune the same speech LLM for the same number of steps, so both
draw the SAME TOTAL HOURS of audio; they differ in how that budget is spread.

* An *entry* is one ASR language or one speech-translation direction. HR5 has 13:
  ASR en/de/fr/es/it, and X->en and en->X for de/fr/es/it. EU24 has one entry per
  ASR language of the 24 EU languages, per X->en direction with data, and per en->X
  direction with data.
* HR5: every entry draws ``min(600 h, what it holds)``.
* EU24: every entry draws ``min(c, what it holds)``, with ONE common cap ``c`` solved
  by bisection so that the total equals HR5's. Entries below ``c`` use all their data.
* Inside an entry the hours are split across its corpora in proportion to
  ``(corpus hours) ** 0.5`` -- the alpha of ``infra/compute_mix_weights.py`` -- and
  water-filled so that no corpus is asked for more than it holds. That makes the
  draw one pass over the data with no repetition.
* The mux ``weight:`` values are CUT probabilities, not shares of hours. lhotse's
  multiplexer draws one cut per pick, so a source supplies ``weight x its mean cut
  length`` of audio; ``compute_mix_weights.cut_mux_weights`` turns the hours each
  source must supply into the probabilities that deliver them, and
  ``check_hour_shares`` verifies the numbers as they are written.

yodas-granary ``ast`` leaves carry the transcript in the supervision and the English
translation in ``custom.translation_en``, so one leaf feeds an ASR entry (reading the
supervision) and an X->en entry (reading the translation). Both entries walk the same
shard order (one ``shard_seed``), so the smaller draw lies inside the larger one:
distinct audio is less than the sum of the hours.

Everything about a source that could silently train the wrong thing is explicit:

* ``text_field`` per corpus is copied from config/train/SFT-v1.4.0.yaml and
  projects/ablation-campaign/ABL-IFT-700.yaml;
* before rendering, every included source is AUDITED: a sample of its cuts is read and
  the configured field is resolved the way ``get_text_from_cut(strict=True)`` would.
  The render is refused if a field is empty where the loader would raise, or if a
  source holds more textless cuts than it is allowed (``Source.max_blank``);
* anything left out is listed in ``EXCLUDED`` with the reason, and printed.

``validation_ds`` is identical in both conditions: 20 clips per named set, one set per
language (ASR) or direction (ST), from splits no training source uses.

Hours and cut counts come from the measurement caches of ``infra/check_training_config.py``
(``infra/.config_check_cache.json``, which also holds the duration histograms the bucket
bins are measured from) and ``infra/compute_mix_weights.py`` (``infra/.mix_weights_hours.json``).
A source missing from them, or whose manifests changed, is measured here with ``--jobs``
workers by the same code those tools use. Run it where the data is.

Usage::

    python3 projects/self-distill/sft/build_sft_mix.py \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar --jobs 32

    # audit every candidate source, excluded ones included, print the table and stop
    python3 projects/self-distill/sft/build_sft_mix.py --audit-only

Writes ``SFT-HR5.yaml`` and ``SFT-EU24.yaml`` next to this file, from ``SFT-template.yaml``.
Both are rendered together because EU24's cap depends on HR5's total.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import bucket_bins  # noqa: E402
import check_training_config as ctc  # noqa: E402
import compute_mix_weights as cmw  # noqa: E402
from build_campaign_config import _replace_scalar_in_section, draw_bins  # noqa: E402


# ---------------------------------------------------------------------------
# The two conditions
# ---------------------------------------------------------------------------

HR5 = ("en", "de", "fr", "es", "it")
# Languages with X->en and en->X entries in HR5 (the ASR entries cover all of HR5).
HR5_ST = ("de", "fr", "es", "it")
EU24 = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr",
    "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
)  # fmt: skip

HR5_CAP_HOURS = 600.0
# alpha of infra/compute_mix_weights.py: the corpus split inside an entry.
ALPHA = 0.5

# ---------------------------------------------------------------------------
# Where each corpus keeps what, by language code. Directory spellings differ per
# corpus; a source is only real if its directory exists (checked when measuring).
# ---------------------------------------------------------------------------

CV22_DIRS = {code: code for code in EU24 if code != "hr"} | {"ga": "ga-IE", "sv": "sv-SE"}
MLS_DIRS = {
    "de": "german", "en": "english", "es": "spanish", "fr": "french",
    "it": "italian", "nl": "dutch", "pl": "polish", "pt": "portuguese",
}  # fmt: skip
VOXPOPULI_LANGS = ("cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl")
GRANARY_DIRS = {
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "hr": "Croatian", "hu": "Hungarian", "it": "Italian", "lt": "Lithuanian",
    "lv": "Latvian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "sk": "Slovak", "sv": "Swedish",
}  # fmt: skip
FLEURS_DIRS = {
    "bg": "bg_bg", "cs": "cs_cz", "da": "da_dk", "de": "de_de", "el": "el_gr",
    "en": "en_us", "es": "es_419", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr",
    "ga": "ga_ie", "hr": "hr_hr", "hu": "hu_hu", "it": "it_it", "lt": "lt_lt",
    "lv": "lv_lv", "mt": "mt_mt", "nl": "nl_nl", "pl": "pl_pl", "pt": "pt_br",
    "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "sv": "sv_se",
}  # fmt: skip
# CoVoST2 spells Swedish with its Common Voice locale, sv-SE.
COVOST_X_TO_EN = {
    "de": "de_en", "es": "es_en", "fr": "fr_en", "it": "it_en", "nl": "nl_en",
    "pt": "pt_en", "et": "et_en", "lv": "lv_en", "sl": "sl_en", "sv": "sv-SE_en",
}  # fmt: skip
COVOST_EN_TO_X = {"de": "en_de", "et": "en_et", "lv": "en_lv", "sl": "en_sl", "sv": "en_sv-SE"}
EUROPARL_EN_TO_X = ("de", "es", "fr", "it", "nl", "pl", "pt", "ro")

# The text field each corpus keeps its training text in, copied from SFT-v1.4.0 and
# ABL-IFT-700. A corpus absent here trains on the supervision text. yodas-granary
# `ast` is also read as ST, where the target is the English translation instead.
TEXT_FIELDS = {
    "cv22_sidon": "custom.metadata.sentence",  # the supervision is empty by design
    "mls_sidon": "custom.pnc_text",  # the supervision is lowercase and unpunctuated
    "voxpopuli": "custom.pnc_text",
    "fleurs": "custom.pnc_text",
}
TRANSLATION_FIELD = "custom.translation_en"

# VoxPopuli outside de/en/es/fr/it went through a generic converter: positional ids
# (chunk{N}_item{M}, colliding across languages), a custom block only on cuts that
# have a transcript, and no filter on empty transcripts. Measured over the whole
# leaves, 0.2-4.8% of the cuts of these ten have no transcript. The loader skips such
# a cut (no text_field value, no supervision text: nothing to mislabel) but its batch
# slot is wasted, so a source may hold up to GENERIC_VOXPOPULI_MAX_BLANK of them.
GENERIC_VOXPOPULI = ("cs", "et", "fi", "hu", "lt", "nl", "pl", "ro", "sk", "sl")
GENERIC_VOXPOPULI_MAX_BLANK = 0.10
GENERIC_VOXPOPULI_NOTE = "generic converter: 0.2-4.8% of its cuts have no transcript and are skipped by the loader"

# FLEURS leaves whose cuts carry no custom.pnc_text (no custom block at all), so
# the corpus-wide text_field would raise on every cut under strict_text_field. They
# train on the supervision text, which is lowercase and unpunctuated, instead of
# being dropped: for Irish this leaf is 12 h against 0.6 h of CV22. The loader
# lowercases every target anyway, so what differs is the punctuation. A PNC pass over
# the leaf would remove the exception.
FLEURS_WITHOUT_PNC = ("ga",)
FLEURS_WITHOUT_PNC_NOTE = "no pnc_text on this leaf: trains on the unpunctuated supervision text"

# Sources left out, and why. Keyed by path below the datasets root; printed on every run.
EXCLUDED = {
    "voxpopuli/hr/train": (
        "47.2% of its cuts (17.0 of 34.8 h) have no transcript, so about half of every draw would be "
        "skipped by the loader; its validation and test splits are the same (38.6% and 49.8%)."
    ),
}

# Held-out clips for languages whose CV22 split does not exist (CV22 has no Croatian).
# VoxPopuli is the obvious second choice, but 38.6% of its hr validation cuts have no
# transcript and max_samples is applied BEFORE they are dropped, so 20 clips would score
# about 12. FLEURS hr_hr validation is complete (377 cuts, all with pnc_text).
ASR_VALIDATION_FALLBACK = {"hr": ("fleurs/hr_hr/validation", "custom.pnc_text")}

# What the audit tolerates in an included source (share of the sampled cuts).
AUDIT_MAX_BLANK = 0.001
AUDIT_MAX_MISMATCH = 0.001
AUDIT_MAX_SAME_AS_SOURCE = 0.20
AUDIT_MANIFESTS = 3

# The train loader's duration window (data.train_ds in the template), the number of
# duration buckets, and the decimals the weights are written with.
MIN_DURATION = 0.5
MAX_DURATION = 60.0
NUM_BUCKETS = 30
WEIGHT_DIGITS = 10


@dataclass(frozen=True)
class Source:
    """One Shar leaf an entry can draw from."""

    rel: str  # path below the datasets root, e.g. "cv22_sidon/de/train"
    corpus: str  # top-level directory
    text_field: str | None  # tags.text_field; None trains on the supervision text
    region: str | None = None  # the locale spelling in the directory, when it is not the language code
    max_blank: float = AUDIT_MAX_BLANK  # share of cuts allowed to carry no transcript
    note: str | None = None

    @property
    def label(self) -> str:
        """Corpus name for tables, with the yodas-granary variant (asr_only or ast) that tells two leaves apart."""
        return f"{self.corpus}/{self.rel.split('/')[-1]}" if self.corpus == "yodas-granary" else self.corpus


@dataclass(frozen=True)
class Entry:
    """One ASR language or one speech-translation direction."""

    task: str  # "asr" or "st"
    src: str  # the language spoken
    tgt: str | None  # the language written, for st
    sources: tuple[Source, ...]

    @property
    def key(self) -> str:
        """Short unique name: ``asr:de`` or ``st:de-en``."""
        return f"asr:{self.src}" if self.task == "asr" else f"st:{self.src}-{self.tgt}"

    @property
    def label(self) -> str:
        """Human name for comments and tables: ``ASR de`` or ``ST de->en``."""
        return f"ASR {self.src}" if self.task == "asr" else f"ST {self.src}->{self.tgt}"

    def group_tags(self) -> dict[str, str]:
        """Tags a group carries: what is constant across the entry."""
        if self.task == "asr":
            return {"task": "asr", "lang": self.src}
        return {"task": "st", "src_lang": self.src, "tgt_lang": str(self.tgt)}

    def leaf_tags(self, source: Source) -> dict[str, str]:
        """Tags of one leaf: the group's, then text_field, then the locale spelling if it differs."""
        tags = self.group_tags()
        if source.text_field:
            tags["text_field"] = source.text_field
        if source.region:
            if self.task == "asr":
                tags["region_code"] = source.region
            else:
                tags["src_region_code" if self.tgt == "en" else "tgt_region_code"] = source.region
        return tags


def _region(directory: str, code: str) -> str | None:
    """The locale spelling of a directory, unless it is the language code itself."""
    return directory if directory.lower() != code.lower() else None


def asr_sources(lang: str, skip: frozenset[str] = frozenset(EXCLUDED)) -> tuple[Source, ...]:
    """Every ASR source for a language: the leaves that exist for it, minus ``skip``."""
    found: list[Source] = []
    if lang in CV22_DIRS:
        d = CV22_DIRS[lang]
        found.append(Source(f"cv22_sidon/{d}/train", "cv22_sidon", TEXT_FIELDS["cv22_sidon"], _region(d, lang)))
    if lang in MLS_DIRS:
        found.append(Source(f"mls_sidon/{MLS_DIRS[lang]}/train", "mls_sidon", TEXT_FIELDS["mls_sidon"]))
    if lang in VOXPOPULI_LANGS:
        generic = lang in GENERIC_VOXPOPULI or lang == "hr"
        found.append(
            Source(
                f"voxpopuli/{lang}/train",
                "voxpopuli",
                TEXT_FIELDS["voxpopuli"],
                max_blank=GENERIC_VOXPOPULI_MAX_BLANK if generic else AUDIT_MAX_BLANK,
                note=GENERIC_VOXPOPULI_NOTE if generic else None,
            )
        )
    if lang in GRANARY_DIRS:
        found.append(Source(f"yodas-granary/{GRANARY_DIRS[lang]}/asr_only", "yodas-granary", None))
        if lang != "en":  # English has no ast
            found.append(Source(f"yodas-granary/{GRANARY_DIRS[lang]}/ast", "yodas-granary", None))
    if lang in FLEURS_DIRS:
        d = FLEURS_DIRS[lang]
        no_pnc = lang in FLEURS_WITHOUT_PNC
        found.append(
            Source(
                f"fleurs/{d}/train",
                "fleurs",
                None if no_pnc else TEXT_FIELDS["fleurs"],
                _region(d, lang),
                note=FLEURS_WITHOUT_PNC_NOTE if no_pnc else None,
            )
        )
    return tuple(s for s in found if s.rel not in skip)


def x_to_en_sources(src: str, skip: frozenset[str] = frozenset(EXCLUDED)) -> tuple[Source, ...]:
    """Every X->en source: yodas-granary ast (translation_en) and CoVoST2 (the supervision is the English)."""
    found: list[Source] = []
    if src in GRANARY_DIRS and src != "en":
        found.append(Source(f"yodas-granary/{GRANARY_DIRS[src]}/ast", "yodas-granary", TRANSLATION_FIELD))
    if src in COVOST_X_TO_EN:
        d = COVOST_X_TO_EN[src]
        found.append(Source(f"covost2/{d}/train", "covost2", None, _region(d.split("_")[0], src)))
    return tuple(s for s in found if s.rel not in skip)


def en_to_x_sources(tgt: str, skip: frozenset[str] = frozenset(EXCLUDED)) -> tuple[Source, ...]:
    """Every en->X source: CoVoST2 and Europarl-ST (the supervision is the translation)."""
    found: list[Source] = []
    if tgt in COVOST_EN_TO_X:
        d = COVOST_EN_TO_X[tgt]
        found.append(Source(f"covost2/{d}/train", "covost2", None, _region(d.split("_")[1], tgt)))
    if tgt in EUROPARL_EN_TO_X:
        found.append(Source(f"europarl_st/en_{tgt}/train", "europarl_st", None))
    return tuple(s for s in found if s.rel not in skip)


def build_entries(
    asr_langs: tuple[str, ...],
    x_to_en: tuple[str, ...] | None,
    en_to_x: tuple[str, ...] | None,
    skip: frozenset[str] = frozenset(EXCLUDED),
) -> list[Entry]:
    """Entries for a condition, ASR first, then X->en, then en->X. ``None`` means every direction with data."""
    others = [lang for lang in EU24 if lang != "en"]
    sources_in = [lang for lang in others if x_to_en_sources(lang, skip)] if x_to_en is None else list(x_to_en)
    targets = [lang for lang in others if en_to_x_sources(lang, skip)] if en_to_x is None else list(en_to_x)
    entries = [Entry("asr", lang, None, asr_sources(lang, skip)) for lang in asr_langs]
    entries += [Entry("st", lang, "en", x_to_en_sources(lang, skip)) for lang in sources_in]
    entries += [Entry("st", "en", lang, en_to_x_sources(lang, skip)) for lang in targets]
    empty = [e.key for e in entries if not e.sources]
    if empty:
        raise ValueError(f"entries without a single source: {empty}")
    return entries


def hr5_entries(skip: frozenset[str] = frozenset(EXCLUDED)) -> list[Entry]:
    """The 13 HR5 entries."""
    return build_entries(HR5, HR5_ST, HR5_ST, skip)


def eu24_entries(skip: frozenset[str] = frozenset(EXCLUDED)) -> list[Entry]:
    """The EU24 entries: 24 ASR languages, and every X->en and en->X direction that has data."""
    return build_entries(EU24, None, None, skip)


# ---------------------------------------------------------------------------
# Validation: identical in both conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSet:
    """One named validation set: a single held-out leaf, scored on its own."""

    name: str
    rel: str
    task: str
    src: str
    tgt: str | None
    text_field: str | None

    def tags(self) -> dict[str, str]:
        """Tags of the leaf (validation leaves are flat: there are no groups)."""
        if self.task == "asr":
            tags = {"task": "asr", "lang": self.src}
        else:
            tags = {"task": "st", "src_lang": self.src, "tgt_lang": str(self.tgt)}
        if self.text_field:
            tags["text_field"] = self.text_field
        return tags


def validation_sets() -> list[EvalSet]:
    """ASR per EU language (CV22 validation), then ST per direction (CoVoST2 dev, Europarl-ST validation)."""
    sets: list[EvalSet] = []
    for lang in EU24:
        if lang in CV22_DIRS:
            rel, text_field = f"cv22_sidon/{CV22_DIRS[lang]}/validation", TEXT_FIELDS["cv22_sidon"]
        else:
            rel, text_field = ASR_VALIDATION_FALLBACK[lang]
        sets.append(EvalSet(f"asr_{lang}", rel, "asr", lang, None, text_field))
    for lang, d in COVOST_X_TO_EN.items():
        sets.append(EvalSet(f"st_{lang}_en_covost2", f"covost2/{d}/dev", "st", lang, "en", None))
    for lang, d in COVOST_EN_TO_X.items():
        sets.append(EvalSet(f"st_en_{lang}_covost2", f"covost2/{d}/dev", "st", "en", lang, None))
    for lang in EUROPARL_EN_TO_X:
        sets.append(EvalSet(f"st_en_{lang}_europarl", f"europarl_st/en_{lang}/validation", "st", "en", lang, None))
    return sets


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def water_fill(budget: float, capacity: list[float], alpha: float = ALPHA) -> list[float]:
    """Split ``budget`` across items in proportion to ``capacity ** alpha``, never giving one more than it holds.

    An item whose proportional share exceeds its capacity gives its capacity and the
    surplus is shared among the rest, until no share exceeds its capacity. When the
    budget covers everything, every item gives all it has.
    """
    if budget < 0:
        raise ValueError(f"negative budget: {budget}")
    if any(c < 0 for c in capacity):
        raise ValueError(f"negative capacity in {capacity}")
    if budget >= sum(capacity):
        return list(capacity)
    given = [0.0] * len(capacity)
    active = [i for i, c in enumerate(capacity) if c > 0]
    remaining = budget
    while active:
        weight = {i: capacity[i] ** alpha for i in active}
        total = sum(weight.values())
        share = {i: remaining * weight[i] / total for i in active}
        over = [i for i in active if share[i] > capacity[i]]
        if not over:
            for i in active:
                given[i] = share[i]
            break
        for i in over:
            given[i] = capacity[i]
            remaining -= capacity[i]
        active = [i for i in active if i not in over]
    return given


def solve_common_cap(available: list[float], target: float, tol: float = 1e-9) -> float:
    """The cap ``c`` for which ``sum(min(c, a) for a in available) == target``, by bisection.

    Entries holding less than ``c`` give everything they have. Raises ValueError when
    ``target`` exceeds what all the entries hold together: no cap can reach it.
    """
    holds = sum(available)
    if target > holds + tol:
        raise ValueError(f"the target of {target:,.3f} h exceeds the {holds:,.3f} h the entries hold together")
    if target >= holds:
        return max(available)
    lo, hi = 0.0, max(available)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if sum(min(mid, a) for a in available) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Measurement: hours, cuts and duration histograms per source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measure:
    """What a source holds."""

    hours: float
    cuts: int
    window_hours: float  # inside [MIN_DURATION, MAX_DURATION]: the cuts the train loader keeps

    @property
    def mean_cut(self) -> float:
        """Mean cut length in seconds."""
        return cmw.mean_cut_seconds(self.hours, self.cuts, "source")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _window_hours(entry: dict) -> float:
    """Hours of a cached source whose cuts fall inside the train loader's duration window."""
    res = float(entry.get("res", ctc.HIST_RESOLUTION))
    lo, hi = round(MIN_DURATION / res), round(MAX_DURATION / res)
    return sum(int(k) * res * int(v) for k, v in entry["hist"].items() if lo <= int(k) <= hi) / 3600.0


def measure_all(
    root: Path, rels: list[str], hist_cache: Path, hours_cache: Path, jobs: int, measure: bool = True
) -> tuple[dict[str, Measure], dict]:
    """Hours and cut counts per source, measuring (and caching) whatever is not cached or has changed.

    Uses the measurement of ``infra/check_training_config.py``, which records hours, cuts
    and a duration histogram in one pass, and mirrors hours and cuts into the cache of
    ``infra/compute_mix_weights.py``. Returns the measures and the histogram cache.
    """
    cache = _load_json(hist_cache)
    todo: list[str] = []
    for rel in dict.fromkeys(rels):
        path = str(root / rel)
        entry = cache.get(path)
        if not ctc.cache_is_complete(entry) or float(entry.get("res", ctc.HIST_RESOLUTION)) != ctc.HIST_RESOLUTION:
            todo.append(path)
        elif entry.get("mtime") is not None:
            shards = cmw.shar_manifest_files(Path(path))
            if shards and ctc.manifest_mtime(shards) > entry["mtime"] + 1:
                print(f"  manifests changed since last measurement: {path}")
                todo.append(path)
    missing = [p for p in todo if not Path(p).is_dir()]
    if missing:
        raise SystemExit("These sources do not exist under the datasets root:\n  " + "\n  ".join(missing))
    if todo:
        if not measure:
            raise SystemExit(
                f"{len(todo)} sources are not measured, or changed, and --no-measure was given:\n  "
                + "\n  ".join(todo)
            )
        n_shards = sum(len(cmw.shar_manifest_files(Path(p))) for p in todo)
        print(f"Measuring {len(todo)} sources ({n_shards} shards) with {jobs} workers...", flush=True)
        ctc.measure_sources(todo, cache, hist_cache, [(MIN_DURATION, MAX_DURATION)], jobs)
        mirror = _load_json(hours_cache)
        for path in todo:
            e = cache[path]
            mirror[path] = {"hours": e["hours"], "cuts": e["cuts"], "shards": e["shards"], "sample": None}
        cmw.write_cache(mirror, hours_cache)

    measures: dict[str, Measure] = {}
    for rel in dict.fromkeys(rels):
        entry = cache[str(root / rel)]
        if not entry["cuts"] or entry["hours"] <= 0:
            raise SystemExit(f"{rel} measures as empty ({entry['cuts']} cuts, {entry['hours']} h)")
        measures[rel] = Measure(entry["hours"], int(entry["cuts"]), _window_hours(entry))
    return measures, cache


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    """The share of one entry's hours that one source supplies."""

    source: Source
    available: float  # hours the source holds
    hours: float  # hours this entry draws from it
    mean_cut: float  # seconds


@dataclass(frozen=True)
class Plan:
    """One entry's budget and how it is spread over its sources."""

    entry: Entry
    available: float  # hours the entry holds across its sources
    hours: float  # hours it draws: min(limit, available)
    draws: tuple[Draw, ...]

    @property
    def at_limit(self) -> bool:
        """Whether the cap held the entry back (as opposed to it using all of its data)."""
        return self.hours < self.available - 1e-9


def plan_entry(entry: Entry, limit: float, measures: dict[str, Measure], alpha: float = ALPHA) -> Plan:
    """Draw ``min(limit, available)`` hours for an entry, split over its corpora by ``hours ** alpha``."""
    held = [measures[s.rel].hours for s in entry.sources]
    available = sum(held)
    budget = min(limit, available)
    given = water_fill(budget, held, alpha)
    draws = [Draw(s, h, g, measures[s.rel].mean_cut) for s, h, g in zip(entry.sources, held, given) if g > 0]
    draws.sort(key=lambda d: (-d.hours, d.source.rel))
    return Plan(entry, available, budget, tuple(draws))


def plan_condition(entries: list[Entry], limit: float, measures: dict[str, Measure]) -> list[Plan]:
    """Plan every entry of a condition under one per-entry limit."""
    return [plan_entry(e, limit, measures) for e in entries]


def entry_availability(entries: list[Entry], measures: dict[str, Measure]) -> list[float]:
    """Hours each entry holds in total."""
    return [sum(measures[s.rel].hours for s in e.sources) for e in entries]


def mux_weights(plans: list[Plan]) -> tuple[list[float], list[list[float]]]:
    """Group weights and per-group source weights, as cut probabilities that deliver each source's hours.

    Verified as they will be written (``WEIGHT_DIGITS`` decimals): a ValueError means the
    rounding alone would move a source's share of the audio.
    """
    targets = [[d.hours for d in p.draws] for p in plans]
    mean_cut = [[d.mean_cut for d in p.draws] for p in plans]
    group_w, leaf_w = cmw.cut_mux_weights(targets, mean_cut)
    cmw.check_hour_shares(targets, group_w, leaf_w, mean_cut, ndigits=WEIGHT_DIGITS)
    return group_w, leaf_w


# ---------------------------------------------------------------------------
# Audit: read a sample of cuts and resolve the text field as the loader would
# ---------------------------------------------------------------------------


def _fallback_text(record: dict) -> str | None:
    """Supervision text, else custom.text: the fallback of ``get_text_from_cut``."""
    texts = [s.get("text") for s in (record.get("supervisions") or []) if s.get("text")]
    text = " ".join(texts) if texts else None
    if text is None:
        text = (record.get("custom") or {}).get("text")
    if text is not None:
        text = text.strip()
    return text or None


def resolve_text(record: dict, text_field: str | None) -> tuple[str, str | None]:
    """Classify a raw cut record the way ``get_text_from_cut(strict=True)`` treats it.

    Returns ``("ok", text)`` when the configured field holds text, ``("blank", None)``
    when the cut has no text anywhere (the loader skips it silently), and
    ``("mismatch", None)`` when the field is empty but a fallback exists (strict mode
    raises, or skips with ``skip_text_field_mismatch``).
    """
    if not text_field or text_field == "text":
        text = _fallback_text(record)
        return ("ok", text) if text else ("blank", None)
    value: object = record
    for part in text_field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        value = (record.get("custom") or {}).get(text_field)
    if value is not None and str(value).strip():
        return "ok", str(value).strip()
    return ("blank", None) if _fallback_text(record) is None else ("mismatch", None)


@dataclass
class Audit:
    """What a sample of a source's cuts says about its text field."""

    rel: str
    text_field: str | None
    task: str
    max_blank: float = AUDIT_MAX_BLANK
    manifests: str = ""
    cuts: int = 0
    ok: int = 0
    blank: int = 0
    mismatch: int = 0
    compared: int = 0  # ST only: cuts with both a target and a source-language text
    same: int = 0  # ST only: of those, the target equals the source text
    first_text: str = ""

    def rate(self, count: int) -> float:
        """A count as a share of the sampled cuts."""
        return count / self.cuts if self.cuts else 0.0

    def problems(self) -> list[str]:
        """Why this source may not be rendered; empty when it may."""
        out = []
        if not self.cuts:
            out.append("no cuts could be read")
        if self.rate(self.blank) > self.max_blank:
            out.append(
                f"{self.rate(self.blank):.1%} of the sampled cuts have no text anywhere (allowed {self.max_blank:.1%})"
            )
        if self.rate(self.mismatch) > AUDIT_MAX_MISMATCH:
            out.append(
                f"{self.rate(self.mismatch):.1%} of the sampled cuts have nothing at {self.text_field!r} "
                f"but do have other text (allowed {AUDIT_MAX_MISMATCH:.1%})"
            )
        if self.task == "st" and self.compared and self.same / self.compared > AUDIT_MAX_SAME_AS_SOURCE:
            out.append(f"{self.same / self.compared:.0%} of the translations equal their source text")
        return out


def _read_records(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def audit_source(root: Path, rel: str, text_field: str | None, task: str, max_blank: float = AUDIT_MAX_BLANK) -> Audit:
    """Read whole manifests of a source: all of them if it has at most six, else the first, middle and last."""
    files = cmw.shar_manifest_files(root / rel)
    chosen = files if len(files) <= 2 * AUDIT_MANIFESTS else [files[0], files[len(files) // 2], files[-1]]
    result = Audit(rel, text_field, task, max_blank, manifests=f"{len(chosen)}/{len(files)}")
    for path in chosen:
        for record in _read_records(path):
            result.cuts += 1
            bucket, text = resolve_text(record, text_field)
            if bucket == "ok":
                result.ok += 1
                result.first_text = result.first_text or (text or "")[:70]
                if task == "st":
                    custom = record.get("custom") or {}
                    source_text = _fallback_text(record) if text_field == TRANSLATION_FIELD else custom.get("sentence")
                    if source_text:
                        result.compared += 1
                        result.same += source_text.strip().lower() == (text or "").strip().lower()
            elif bucket == "blank":
                result.blank += 1
            else:
                result.mismatch += 1
    return result


def audit_targets(entries: list[Entry], eval_sets: list[EvalSet]) -> list[tuple[str, str | None, str, float]]:
    """Every distinct (source, text field, task, tolerated blank share) the conditions read."""
    targets: dict[tuple[str, str | None, str], float] = {}
    for entry in entries:
        for s in entry.sources:
            key = (s.rel, s.text_field, entry.task)
            targets[key] = min(targets.get(key, 1.0), s.max_blank)
    for ev in eval_sets:
        targets.setdefault((ev.rel, ev.text_field, ev.task), AUDIT_MAX_BLANK)
    return [(rel, tf, task, blank) for (rel, tf, task), blank in targets.items()]


def print_audit(results: list[Audit]) -> None:
    """The audit table: one line per source and text field."""
    print(f"\n{'source':<44} {'text_field':<26} {'cuts':>7} {'ok':>7} {'blank':>7} {'mismatch':>8}  first text")
    for a in sorted(results, key=lambda r: (r.rel, str(r.text_field), r.task)):
        problems = a.problems()
        flag = "  <-- " + "; ".join(problems) if problems else ""
        print(
            f"{a.rel:<44} {a.text_field or 'text':<26} {a.cuts:>7} {a.rate(a.ok):>7.1%} {a.rate(a.blank):>7.1%} "
            f"{a.rate(a.mismatch):>8.2%}  {a.first_text[:40]!r}{flag}"
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SHAR_ROOT_VAR = "${oc.env:LOCAL_DATASETS_DIR}"


def entry_order(plans: list[Plan]) -> list[int]:
    """Indices of the plans as they are written: ASR, then X->en, then en->X; the entries drawing most first."""

    def key(i: int) -> tuple[int, float, str]:
        e = plans[i].entry
        kind = 0 if e.task == "asr" else (1 if e.tgt == "en" else 2)
        return kind, -plans[i].hours, e.key

    return sorted(range(len(plans)), key=key)


def render_train_input_cfg(
    plans: list[Plan], group_w: list[float], leaf_w: list[list[float]], cap_note: str
) -> list[str]:
    """The ``train_ds.input_cfg`` block: one group per entry, its sources as children."""
    total = sum(p.hours for p in plans)
    lines = [
        "    input_cfg:",
        "      # `weight` is a per-CUT draw probability: lhotse's mux picks one cut per draw, so a source's",
        "      # share of the audio is weight x its mean cut length. Each weight is the share of the cuts",
        "      # that make up the hours listed beside it (hours / mean cut), so those hours are what a run",
        "      # draws (build_sft_mix.py, compute_mix_weights.cut_mux_weights). A group's weight is its",
        "      # entry's share of the cuts, a child's weight its share within the entry; the two levels are",
        "      # muxed separately, so do NOT pre-multiply them.",
        f"      # {len(plans)} entries, {total:,.1f} h. {cap_note}",
    ]
    for i in entry_order(plans):
        plan, e = plans[i], plans[i].entry
        held = f"capped, of {plan.available:,.1f} h" if plan.at_limit else "all of its data"
        lines.append(
            f"      # {e.label}: {plan.hours:,.1f} h ({held}); {group_w[i]:.1%} of the cuts, {plan.hours / total:.1%} of the audio"
        )
        lines += ["      - type: group", f"        weight: {group_w[i]:.{WEIGHT_DIGITS}f}", "        tags:"]
        lines += [f"          {k}: {v}" for k, v in e.group_tags().items()]
        lines.append("        input_cfg:")
        for d, weight in zip(plan.draws, leaf_w[i]):
            s = d.source
            lines += [
                "          - type: lhotse_shar",
                f"            shar_path: {SHAR_ROOT_VAR}/{s.rel}",
                f"            weight: {weight:.{WEIGHT_DIGITS}f}",
                "            tags:",
            ]
            lines += [f"              {k}: {v}" for k, v in e.leaf_tags(s).items()]
            note = f"; {s.note}" if s.note else ""
            lines.append(
                f"            # {d.hours:,.1f} h of {d.available:,.1f} h in {s.corpus}, mean cut {d.mean_cut:.1f} s{note}"
            )
    return lines


def render_validation_input_cfg(eval_sets: list[EvalSet], measures: dict[str, Measure]) -> list[str]:
    """The ``validation_ds.input_cfg`` block: flat, unweighted, one named set per held-out leaf."""
    lines = ["    input_cfg:"]
    for ev in eval_sets:
        m = measures[ev.rel]
        lines += [
            "      - type: lhotse_shar",
            f"        name: {ev.name}",
            f"        shar_path: {SHAR_ROOT_VAR}/{ev.rel}",
            "        tags:",
        ]
        lines += [f"          {k}: {v}" for k, v in ev.tags().items()]
        lines.append(f"        # {m.hours:,.1f} h, {m.cuts:,} cuts held out")
    return lines


def total_cuts_drawn(plans: list[Plan]) -> int:
    """Cuts the mix draws: the hours each source supplies over its mean cut length."""
    return round(sum(d.hours * 3600.0 / d.mean_cut for p in plans for d in p.draws))


def replace_block(lines: list[str], section: str, block: list[str]) -> None:
    """Replace ``input_cfg:`` under ``section:`` with ``block``: a template splice that keeps its comments."""
    start, end, _ = cmw._find_block(lines, section, "input_cfg")
    lines[start:end] = block


def _wrapped(prefix: str, text: str) -> list[str]:
    """A comment paragraph, wrapped."""
    return textwrap.wrap(text, width=110, initial_indent=prefix, subsequent_indent="#   ")


def header_lines(
    name: str,
    intro: list[str],
    plans: list[Plan],
    wanted: set[str],
    title: list[str] | None = None,
    provenance: list[str] | None = None,
    validation: list[str] | None = None,
) -> list[str]:
    """The comment block at the top of a rendered config, carrying the total hours.

    ``wanted`` is every source the condition would read were nothing excluded; the sources
    left out and the caveats listed are the ones that concern this condition. ``title``,
    ``provenance`` and ``validation`` replace the SFT wording of those paragraphs (the
    self-distillation condition says something else); the defaults are the SFT text.
    """
    total = sum(p.hours for p in plans)
    used = {d.source.rel for p in plans for d in p.draws}
    lines = (
        list(title)
        if title is not None
        else [
            f"# {name} -- supervised fine-tuning of Whisper-large-v3 + MLP adapter (stack_factor 5) + Qwen3.5-2B,",
            "# all three components trainable.",
        ]
    )
    lines += ["#"]
    lines += [f"# {line}" if line else "#" for line in intro]
    lines += [
        "#",
        f"# TOTAL HOURS: {total:,.1f} h over {len(plans)} entries (data.train_ds.total_hours). The launcher derives",
    ]
    lines += (
        list(provenance)
        if provenance is not None
        else [
            "# max_steps from it; HR5 and EU24 draw the same total, so they get the same compute.",
            "#",
            "# Rendered by projects/self-distill/sft/build_sft_mix.py from SFT-template.yaml. Both conditions are",
            "# rendered together, because EU24's cap depends on HR5's total. Do not hand-edit: change the template",
            "# or the script and render again. The script has read a sample of cuts from every source below and",
            "# resolved each text_field the way the loader does; it refuses to render if one does not resolve.",
        ]
    )
    left_out = [rel for rel in EXCLUDED if rel in wanted]
    if left_out:
        lines += ["#", "# Left out of the mix (build_sft_mix.EXCLUDED):"]
        for rel in left_out:
            lines += _wrapped(f"#   {rel}: ", EXCLUDED[rel])
    generic = [lang for lang in GENERIC_VOXPOPULI if f"voxpopuli/{lang}/train" in used]
    if "fleurs/ga_ie/train" in used or generic:
        lines += ["#", "# In the mix with a caveat (the reason is repeated beside each source):"]
    if "fleurs/ga_ie/train" in used:
        lines += _wrapped(
            "#   fleurs/ga_ie/train: ",
            "no pnc_text (no custom block at all), so it trains on the unpunctuated supervision text; it is 12 h "
            "of Irish against 0.6 h of CV22. Under strict_text_field the corpus-wide custom.pnc_text would "
            "raise on every cut.",
        )
    if generic:
        lines += _wrapped(
            "#   voxpopuli/{" + ",".join(generic) + "}/train: ",
            "went through a generic converter (positional chunk{N}_item{M} ids that collide across languages, "
            "no filter on empty transcripts). 0.2-4.8% of their cuts have no transcript; the loader skips those "
            "(there is no text to mislabel) and their batch slots are wasted.",
        )
    lines += (
        list(validation)
        if validation is not None
        else [
            "#",
            "# Validation is the same in both conditions: 20 clips per named set from splits no training source",
            "# reads. asr_hr is scored on FLEURS because CV22 has no Croatian and 38.6% of VoxPopuli hr",
            "# validation cuts have no transcript (max_samples is applied before they are dropped).",
        ]
    )
    lines += [
        "#",
        "# Hours are the unfiltered durations of the manifests, as everywhere else in this repo; the loader",
        "# then drops cuts outside [min_duration, max_duration].",
    ]
    return lines


def hr5_intro() -> list[str]:
    """What HR5 is, for its header."""
    return [
        "Data condition 1 of 2, HR5: five high-resource languages. ASR en/de/fr/es/it, and X->en and en->X for",
        "de/fr/es/it (13 entries). Every entry draws min(600 h, what it holds). Within an entry the hours are",
        "split across its corpora by (corpus hours)**0.5, water-filled so that no corpus gives more than it",
        "holds: one pass, no repetition.",
    ]


def eu24_intro(cap: float, hr5_total: float) -> list[str]:
    """What EU24 is, for its header."""
    return [
        f"Data condition 2 of 2, EU24: all 24 EU languages at HR5's total ({hr5_total:,.1f} h). Every entry draws",
        f"min(c, what it holds) with the common cap c = {cap:,.3f} h, solved by bisection so that the total",
        "equals HR5's; entries below c use all their data. One ASR entry per language, one X->en entry per",
        "language with data (yodas-granary ast, CoVoST2), one en->X entry per target with data (CoVoST2,",
        "Europarl-ST). Same corpus split inside an entry as HR5.",
    ]


def render_config(
    template: str,
    name: str,
    intro: list[str],
    plans: list[Plan],
    cap_note: str,
    bins: str,
    eval_sets: list[EvalSet],
    measures: dict[str, Measure],
    wanted: set[str] | None = None,
    title: list[str] | None = None,
    provenance: list[str] | None = None,
    validation: list[str] | None = None,
) -> str:
    """Splice one condition into the template.

    ``wanted`` is every source the condition would read were nothing excluded (default: the
    sources it does read), which decides which exclusions its header mentions.
    """
    group_w, leaf_w = mux_weights(plans)
    total = sum(p.hours for p in plans)
    lines = template.splitlines()
    marker = lines.index("# @@HEADER@@")
    wanted = wanted if wanted is not None else {d.source.rel for p in plans for d in p.draws}
    lines[marker : marker + 1] = header_lines(name, intro, plans, wanted, title, provenance, validation)
    replace_block(lines, "train_ds", render_train_input_cfg(plans, group_w, leaf_w, cap_note))
    replace_block(lines, "validation_ds", render_validation_input_cfg(eval_sets, measures))
    _replace_scalar_in_section(lines, "run", "exp_name", name)
    _replace_scalar_in_section(lines, "train_ds", "total_hours", f"{total:.2f}")
    _replace_scalar_in_section(lines, "train_ds", "total_cuts", str(total_cuts_drawn(plans)))
    _replace_scalar_in_section(lines, "train_ds", "bucket_duration_bins", bins)
    _replace_scalar_in_section(
        lines, "validation_ds", "total_hours", f"{sum(measures[e.rel].hours for e in eval_sets):.2f}"
    )
    _replace_scalar_in_section(lines, "validation_ds", "total_cuts", str(sum(measures[e.rel].cuts for e in eval_sets)))
    return "\n".join(lines) + "\n"


def measure_bins(plans: list[Plan], hist_cache: dict, root: Path) -> str:
    """``bucket_duration_bins`` measured on the audio this render draws, as a YAML list."""
    group_w, leaf_w = mux_weights(plans)
    draw = [
        (d.source.rel, round(gw, WEIGHT_DIGITS) * round(lw, WEIGHT_DIGITS))
        for p, gw, lws in zip(plans, group_w, leaf_w)
        for d, lw in zip(p.draws, lws)
    ]
    bins = draw_bins(draw, hist_cache, root, MIN_DURATION, MAX_DURATION, NUM_BUCKETS)
    return bucket_bins.format_bins_for_yaml(bins)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def in_window_hours(plan: Plan, measures: dict[str, Measure]) -> float:
    """Hours of a plan the loader's duration window keeps: each draw scaled by its source's kept share."""
    return sum(d.hours * measures[d.source.rel].window_hours / measures[d.source.rel].hours for d in plan.draws)


def print_table(title: str, plans: list[Plan], measures: dict[str, Measure]) -> None:
    """Hours per entry: what it holds, what it draws, and what the loader's duration window keeps of that."""
    total = sum(p.hours for p in plans)
    print(f"\n{title}: {len(plans)} entries, {total:,.1f} h")
    print(f"  {'entry':<11} {'holds h':>11} {'draws h':>9} {'share':>6} {'in-window h':>12}  sources (h drawn)")
    for i in entry_order(plans):
        p = plans[i]
        names = ", ".join(f"{d.source.label}:{d.hours:,.1f}" for d in p.draws)
        print(
            f"  {p.entry.label:<11} {p.available:>11,.1f} {p.hours:>9,.1f} {p.hours / total:>6.1%} "
            f"{in_window_hours(p, measures):>12,.1f}  {names}{'' if p.at_limit else '  [all data]'}"
        )
    in_window = sum(in_window_hours(p, measures) for p in plans)
    print(f"  {'total':<11} {sum(p.available for p in plans):>11,.1f} {total:>9,.1f} {'':>6} {in_window:>12,.1f}")


def distinct_hours(plans: list[Plan]) -> float:
    """Audio drawn when each leaf counts once: the largest draw from it.

    A leaf can feed two entries (yodas-granary ast: ASR and X->en). Both walk the same shard
    order, so the smaller draw lies inside the larger one and the leaf contributes the larger.
    """
    per_leaf: dict[str, float] = {}
    for p in plans:
        for d in p.draws:
            per_leaf[d.source.rel] = max(per_leaf.get(d.source.rel, 0.0), d.hours)
    return sum(per_leaf.values())


def sd24_intro(cap: float, total: float) -> list[str]:
    """What the self-distillation condition is, for its header."""
    return [
        f"Self-distillation data: the 24 ASR entries of EU24 and nothing else ({total:,.1f} h). The student's audio",
        "is paired with the frozen decoder's own reply to a repeat or a translate instruction over the transcript,",
        "so no translation data is needed and the mix carries no ST entries. Every entry draws min(c, what it",
        f"holds) with the common cap c = {cap:,.3f} h, solved by bisection so that the total is the requested",
        "one; entries below c use all their data. Same corpus split inside an entry as HR5 and EU24.",
    ]


def sd24_title(name: str) -> list[str]:
    """First lines of the self-distillation header."""
    return [
        f"# {name} -- Whisper-large-v3 + MLP adapter (stack_factor 5) + Qwen3.5-2B, adapter-only self-distillation:",
        "# the encoder and the decoder (the teacher) stay frozen.",
    ]


SD24_PROVENANCE = [
    "# max_steps from it (one pass over the mix).",
    "#",
    "# Rendered by projects/self-distill/sft/build_sft_mix.py --sd24-hours from SD-template.yaml. Do not hand-edit:",
    "# change the template or the script and render again. The script has read a sample of cuts from every",
    "# source below and resolved each text_field the way the loader does; it refuses to render if one does not.",
]

SD24_VALIDATION = [
    "#",
    "# Validation: ASR only, 20 clips per language from splits no training source reads (the asr_<lang> sets of",
    "# the SFT conditions). The distillation trainer builds its teacher prompts from the transcript, so a",
    "# translation set would put the translation where the transcript belongs. asr_hr is scored on FLEURS",
    "# because CV22 has no Croatian and 38.6% of VoxPopuli hr validation cuts have no transcript.",
]


def render_sd24(args: argparse.Namespace, root: Path, hours: float) -> int:
    """Render SD-EU24-<hours>h.yaml: the 24 ASR entries at *hours* in total, for the adapter-only run."""
    entries = [e for e in eu24_entries() if e.task == "asr"]
    eval_sets = [ev for ev in validation_sets() if ev.task == "asr"]
    rels = sorted({s.rel for e in entries for s in e.sources} | {ev.rel for ev in eval_sets})
    measures, hist_cache = measure_all(root, rels, args.hist_cache, args.cache, args.jobs, not args.no_measure)

    cap = solve_common_cap(entry_availability(entries, measures), hours)
    plans = plan_condition(entries, cap, measures)
    total = sum(p.hours for p in plans)
    name = f"SD-EU24-{hours:g}h"
    print_table(f"{name} (ASR only, common cap c = {cap:,.4f} h)", plans, measures)
    print(f"\n{name} total {total:,.3f} h (requested {hours:g} h), common cap c = {cap:,.4f} h")
    print(f"Distinct audio, each leaf counted once at its largest draw: {distinct_hours(plans):,.1f} h")

    if not args.skip_audit:
        print("\nAuditing the text fields (a sample of cuts per source)...", flush=True)
        results = [
            audit_source(root, rel, tf, task, blank) for rel, tf, task, blank in audit_targets(entries, eval_sets)
        ]
        print_audit(results)
        failed = [a for a in results if a.problems()]
        if failed:
            print(f"\n{len(failed)} source(s) failed the audit; nothing was written.", file=sys.stderr)
            return 1

    template_path = args.template or Path(__file__).with_name("SD-template.yaml")
    wanted = {s.rel for e in eu24_entries(frozenset()) if e.task == "asr" for s in e.sources}
    bins = measure_bins(plans, hist_cache, root)
    text = render_config(
        template_path.read_text(encoding="utf-8"),
        name,
        sd24_intro(cap, total),
        plans,
        f"Common cap c = {cap:,.4f} h.",
        bins,
        eval_sets,
        measures,
        wanted,
        sd24_title(name),
        SD24_PROVENANCE,
        SD24_VALIDATION,
    )
    print(f"\n{name} bucket_duration_bins: {bins}")
    if not args.dry_run:
        (args.out_dir / f"{name}.yaml").write_text(text, encoding="utf-8")
        print(f"Wrote {args.out_dir / (name + '.yaml')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets-root", type=Path, default=None, help="LOCAL_DATASETS_DIR (default: from the environment)."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Default: SFT-template.yaml next to this file, or SD-template.yaml with --sd24-hours.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--hist-cache",
        type=Path,
        default=REPO_ROOT / "infra" / ".config_check_cache.json",
        help="Per-source hours, cuts and duration histograms (check_training_config.py's cache).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "infra" / ".mix_weights_hours.json",
        help="compute_mix_weights.py's hours cache; measured hours and cuts are mirrored into it.",
    )
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 4))
    parser.add_argument("--no-measure", action="store_true", help="Fail instead of measuring an uncached source.")
    parser.add_argument("--skip-audit", action="store_true", help="Do not read cuts to audit the text fields.")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit every candidate source, excluded ones included, print the table and stop.",
    )
    parser.add_argument(
        "--sd24-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="Render SD-EU24-<HOURS>h.yaml ALONE: the 24 ASR entries at this total (common cap solved), for the "
        "adapter-only self-distillation run. HR5 and EU24 are not rendered.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the tables, write nothing.")
    args = parser.parse_args()

    datasets_root = args.datasets_root or (
        Path(os.environ["LOCAL_DATASETS_DIR"]) if os.environ.get("LOCAL_DATASETS_DIR") else None
    )
    if datasets_root is None or not datasets_root.is_dir():
        print("error: pass --datasets-root or set LOCAL_DATASETS_DIR to the Shar collection", file=sys.stderr)
        return 2
    root = Path(os.path.normpath(datasets_root))
    os.environ["LOCAL_DATASETS_DIR"] = str(root)

    if args.audit_only:
        candidates = audit_targets(eu24_entries(frozenset()) + hr5_entries(frozenset()), validation_sets())
        # What the corpus-wide field would do on the two leaves that cannot use it, and the rejected hr set.
        candidates.append(("fleurs/ga_ie/train", TEXT_FIELDS["fleurs"], "asr", AUDIT_MAX_BLANK))
        candidates.append(("voxpopuli/hr/validation", TEXT_FIELDS["voxpopuli"], "asr", GENERIC_VOXPOPULI_MAX_BLANK))
        print_audit([audit_source(root, rel, tf, task, blank) for rel, tf, task, blank in candidates])
        return 0

    if args.sd24_hours is not None:
        return render_sd24(args, root, args.sd24_hours)

    hr5, eu24, eval_sets = hr5_entries(), eu24_entries(), validation_sets()
    rels = sorted({s.rel for e in hr5 + eu24 for s in e.sources} | {ev.rel for ev in eval_sets})
    measures, hist_cache = measure_all(root, rels, args.hist_cache, args.cache, args.jobs, not args.no_measure)

    hr5_plans = plan_condition(hr5, HR5_CAP_HOURS, measures)
    hr5_total = sum(p.hours for p in hr5_plans)
    cap = solve_common_cap(entry_availability(eu24, measures), hr5_total)
    eu24_plans = plan_condition(eu24, cap, measures)
    eu24_total = sum(p.hours for p in eu24_plans)

    print_table(f"HR5 (cap {HR5_CAP_HOURS:g} h per entry)", hr5_plans, measures)
    print_table(f"EU24 (common cap c = {cap:,.4f} h)", eu24_plans, measures)
    print(f"\nHR5 total  {hr5_total:,.3f} h")
    print(f"EU24 total {eu24_total:,.3f} h (difference {eu24_total - hr5_total:+.1e} h), common cap c = {cap:,.4f} h")
    print(
        "Distinct audio, each leaf counted once at its largest draw (yodas-granary ast feeds ASR and X->en): "
        f"HR5 {distinct_hours(hr5_plans):,.1f} h, EU24 {distinct_hours(eu24_plans):,.1f} h"
    )
    print("\nExcluded sources:")
    for rel, why in EXCLUDED.items():
        print(f"  {rel}: {why}")

    if not args.skip_audit:
        print("\nAuditing the text fields (a sample of cuts per source)...", flush=True)
        results = [
            audit_source(root, rel, tf, task, blank) for rel, tf, task, blank in audit_targets(hr5 + eu24, eval_sets)
        ]
        print_audit(results)
        failed = [a for a in results if a.problems()]
        if failed:
            print(
                f"\n{len(failed)} source(s) failed the audit; nothing was written. Fix the data, or add the "
                "source to EXCLUDED with the reason.",
                file=sys.stderr,
            )
            return 1

    template = (args.template or Path(__file__).with_name("SFT-template.yaml")).read_text(encoding="utf-8")
    everything = frozenset()  # no exclusions: what each condition would read without EXCLUDED
    conditions = {
        "SFT-HR5": (hr5_plans, hr5_intro(), f"Cap {HR5_CAP_HOURS:g} h per entry.", hr5_entries(everything)),
        "SFT-EU24": (
            eu24_plans,
            eu24_intro(cap, hr5_total),
            f"Common cap c = {cap:,.4f} h.",
            eu24_entries(everything),
        ),
    }
    for name, (plans, intro, cap_note, unskipped) in conditions.items():
        bins = measure_bins(plans, hist_cache, root)
        wanted = {s.rel for e in unskipped for s in e.sources}
        text = render_config(template, name, intro, plans, cap_note, bins, eval_sets, measures, wanted)
        print(f"\n{name} bucket_duration_bins: {bins}")
        if not args.dry_run:
            (args.out_dir / f"{name}.yaml").write_text(text, encoding="utf-8")
            print(f"Wrote {args.out_dir / (name + '.yaml')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
