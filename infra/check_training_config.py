#!/usr/bin/env python3
"""Verify that a training config's data section is internally consistent.

Several numbers in a training config are derived offline and pasted in, so they
go stale silently.  Nothing in the training stack re-checks them:

* ``total_hours`` drives ``estimate_steps_per_epoch`` -> ``max_steps`` -> the LR
  schedule.  A stale value rescales warmup and decay without any error.
* ``bucket_duration_bins`` is handed straight to lhotse.  Length, monotonicity
  and range are never validated.
* A ``validation_ds`` leaf without ``name`` is accepted: when *no* leaf is named,
  ``split_eval_config_by_name`` returns None and every validation source is
  pooled into one ``eval_loss``, losing per-language reporting.
* Mixture ``weight`` values are only checked for all-or-none within a level.
  Whether they are the *right* numbers is never checked.

This script checks all of that and prints, for anything that does not check out,
the command to run and the config key to change.  It only reports; it never
edits a config.  Mixture changes belong on the documented path -- edit the
template, re-run ``compute_mix_weights.py``, re-emit -- see docs/mixture_weights.md.

Dependencies are PyYAML and numpy only.  In particular ``estimate_bucket_bins``
is deliberately NOT imported: it needs joblib and omegaconf, which are missing
from the lhotse 2 venv, so importing it would make this script unrunnable in the
one environment that has the data mounted.  The bin arithmetic it shares lives in
``bucket_bins.py``.

Usage:
    # everything, measuring whatever the cache is missing
    python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml

    # structure only: no data access, no datasets root needed
    python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml --offline

    # allow a full measuring pass (~22k shards, ~20 min at 16 jobs)
    python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml --measure

Exit status: 0 all clear, 1 at least one FAIL, 2 could not run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - matches compute_mix_weights' behaviour
    sys.exit("PyYAML is required: pip install pyyaml")

# Siblings in infra/, imported the way build_campaign_config.py does it: running
# `python3 infra/check_training_config.py` puts infra/ on sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bucket_bins  # noqa: E402
import compute_mix_weights as cmw  # noqa: E402

FAIL, WARN, INFO, PASS, SKIP = "FAIL", "WARN", "INFO", "PASS", "SKIP"

# Every check this script knows how to run, in report order.  The section is
# what --only filters on; the text is what the report prints beside the verdict.
CHECKS: dict[str, tuple[str, str]] = {
    "W1": ("weights", "weight is set on all entries of a level or none of them"),
    "W2": ("weights", "each level's weights sum to 1"),
    "W3": ("weights", "weights match the two-tier policy over measured hours"),
    "W4": ("weights", "each group holds exactly one language entry, with matching tags"),
    "N1": ("names", "every validation source declares a name"),
    "N2": ("names", "validation naming is not partial"),
    "H1": ("stats", "train_ds.total_hours matches the measured mixture"),
    "H2": ("stats", "train_ds.total_cuts matches the measured mixture"),
    "H3": ("stats", "validation_ds.total_hours / total_cuts are consistent"),
    "B1": ("bins", "dynamic_bucketing declares num_buckets"),
    "B2": ("bins", "bucket_duration_bins is well formed"),
    "B3": ("bins", "bucket_duration_bins matches this mixture's durations"),
    "T1": ("tags", "no group tag overrides a differing leaf tag"),
    "T2": ("tags", "ASR sources tag lang; ST sources tag src_lang and tgt_lang"),
    "T3": ("tags", "locale codes are folded by LOCALE_ALIASES"),
    "T4": ("tags", "text_field resolves on a real cut, and ST targets are translations"),
    "T5": ("tags", "leaves of one corpus agree on text_field"),
    "D1": ("disk", "every shar_path exists and holds cut manifests"),
    "D2": ("disk", "shar index sidecars are complete and not stale"),
    "D3": ("disk", "no source is shared between train and validation"),
    "E1": ("runtime", "force_estimate can actually measure this input_cfg"),
    "E2": ("runtime", "total_cuts is usable by the batch_size code path"),
    "E3": ("runtime", "steps per epoch is derivable"),
    "C1": ("runtime", "strict_text_field reaches the eval path"),
    "C2": ("runtime", "per_device_eval_batch_size is valid for eval"),
    "C3": ("runtime", "cuts above the encoder's audio window are chunked, not truncated"),
    "C4": ("bins", "the top bucket covers the longest cuts"),
    "C5": ("bins", "bins were measured for this mixture, not copied"),
    "C6": ("runtime", "generation-based eval has a bounded validation set"),
}

# The requested four are hard failures; everything else this script noticed is
# advisory, so a config that is merely unusual still exits 0.  --strict promotes
# warnings to failures.
FAIL_CHECKS = {"W1", "W2", "W3", "W4", "N1", "N2", "H1", "H2", "H3", "B1", "B2", "B3"}

# Which text field a corpus keeps its transcript in is deliberately NOT hardcoded
# here.  An earlier draft of this script carried such a table, taken from the
# 2026-08-11 collection audit, and it was already wrong: voxpopuli and MLS
# Polish have since been backfilled with custom.pnc_text, so the table produced
# false positives on five sources.  The manifests are the authority, so T4 reads
# one real cut per source instead.  What the data cannot tell us -- whether two
# leaves of the same corpus should agree, and whether an ST source is pointed at
# a translation -- is checked structurally in T5 and T4 respectively.

# w2v-bert-2.0 emits one frame per 20 ms, so max_audio_seq_len frames is a
# hard ceiling in seconds on what the encoder can see.
ENCODER_FRAME_SECONDS = 0.02

# Encoders whose input frame rate differs from that default, keyed by a substring of
# ``model.encoder.name``. This deliberately duplicates the ``frame_seconds`` field of
# ``melt/modeling/encoder_specs.py`` rather than importing it: this script is
# constrained to PyYAML and numpy so it stays runnable in the lhotse 2 venv, which is
# the only environment with the data mounted. Keep the two in step.
ENCODER_FRAME_SECONDS_BY_NAME = {
    "whisper": 0.01,  # log-mel at hop_length 160 / 16 kHz, no stride-stacking
}


def encoder_frame_seconds(encoder_name: str | None) -> float:
    """Seconds of audio per encoder *input* frame, for the named encoder."""
    if encoder_name:
        lowered = str(encoder_name).lower()
        for key, seconds in ENCODER_FRAME_SECONDS_BY_NAME.items():
            if key in lowered:
                return seconds
    return ENCODER_FRAME_SECONDS

HIST_RESOLUTION = 0.01


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """One thing that is wrong, with what to do about it."""

    check: str
    severity: str
    where: str
    message: str
    line: int | None = None
    detail: list[str] = field(default_factory=list)
    fix: list[str] = field(default_factory=list)


class Report:
    """Findings plus which checks actually ran, so PASS can be distinguished from SKIP."""

    def __init__(self, config_path: Path, strict: bool = False) -> None:
        self.config_path = config_path
        self.strict = strict
        self.findings: list[Finding] = []
        self.ran: set[str] = set()
        self.skipped: dict[str, str] = {}
        self.notes: list[str] = []

    def ran_check(self, *checks: str) -> None:
        for check in checks:
            self.ran.add(check)
            self.skipped.pop(check, None)

    def skip(self, check: str, why: str) -> None:
        # First reason wins: the earliest caller is the most specific about why
        # the check never got a chance to run (e.g. --offline beats "no measured
        # hours", which is only a consequence of it).
        if check not in self.ran:
            self.skipped.setdefault(check, why)

    def add(
        self,
        check: str,
        where: str,
        message: str,
        line: int | None = None,
        detail: list[str] | None = None,
        fix: list[str] | None = None,
        severity: str | None = None,
    ) -> None:
        if severity is None:
            severity = FAIL if check in FAIL_CHECKS else WARN
        self.ran_check(check)
        self.findings.append(
            Finding(check, severity, where, message, line, list(detail or []), list(fix or []))
        )

    def note(self, text: str) -> None:
        self.notes.append(text)

    def status(self, check: str) -> str:
        severities = {f.severity for f in self.findings if f.check == check}
        if FAIL in severities:
            return FAIL
        if WARN in severities:
            return FAIL if self.strict else WARN
        if INFO in severities or check in self.ran:
            return PASS
        return SKIP

    def failed(self, sections: list[str] | None = None) -> bool:
        """Whether any reported check failed.

        Scoped to ``sections`` so that ``--only bins`` exits on bin problems
        alone: an exit code that reflects checks the caller asked not to see
        would make the flag useless in a pipeline.
        """
        return any(
            self.status(check) == FAIL
            for check, (section, _) in CHECKS.items()
            if sections is None or section in sections
        )

    def location(self, line: int | None) -> str:
        if line is None:
            return str(self.config_path)
        return f"{self.config_path}:{line}"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def build_line_index(text: str) -> dict[str, int]:
    """Map a dotted YAML path to its 1-based line, so findings can be clicked.

    Built from ``yaml.compose`` over the *unexpanded* text: env expansion is a
    within-line regex substitution, so the structure and the line numbering are
    the same either way, and composing the raw text keeps this independent of
    whether the environment resolves.
    """
    index: dict[str, int] = {}

    def walk(node, path: tuple[str, ...]) -> None:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                child = path + (str(key.value),)
                index[".".join(child)] = key.start_mark.line + 1
                walk(value, child)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                child = path + (str(i),)
                index[".".join(child)] = item.start_mark.line + 1
                walk(item, child)

    root = yaml.compose(text)
    if root is not None:
        walk(root, ())
    return index


def load_config(config_path: Path, datasets_root: str | None) -> tuple[dict, dict[str, int]]:
    """Parse the config with ``${oc.env:...}`` resolved, plus its line index."""
    text = config_path.read_text(encoding="utf-8")
    line_index = build_line_index(text)

    if datasets_root is not None:
        os.environ["LOCAL_DATASETS_DIR"] = datasets_root
    # Placeholders for vars this script does not care about, so expanding the
    # rest of the file cannot fail on an unrelated key.  Same trick as
    # compute_mix_weights.load_sources.
    for var in ("OUTPUT_DIR", "HF_HOME", "WANDB_PROJECT", "VENV_PATH"):
        os.environ.setdefault(var, "/unused")

    cfg = yaml.safe_load(cmw._expand_env(text))
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: top level is not a mapping")
    return cfg, line_index


# ---------------------------------------------------------------------------
# Flattening input_cfg
# ---------------------------------------------------------------------------

@dataclass
class Leaf:
    """One ``lhotse_shar`` source, with its position and effective tags."""

    where: str
    yaml_path: str
    raw_path: str
    path: str
    leaf_tags: dict
    eff_tags: dict
    weight: float | None
    has_weight: bool
    name: str | None
    line: int | None
    group_yaml_path: str | None
    # The datasets root, kept so messages can show a path relative to it.
    root: str | None = None

    @property
    def display(self) -> str:
        """Short name for messages: the path relative to the datasets root.

        Env expansion happens on the config text before parsing, so ``raw_path``
        is already absolute and the placeholder is gone by the time we see it.
        Stripping the root back off keeps messages readable and, under
        ``--offline``, keeps the sentinel root out of them.
        """
        if self.root:
            root = self.root.rstrip("/") + "/"
            if self.path.startswith(root):
                return self.path[len(root):]
        return self.raw_path

    @property
    def task(self) -> str:
        return str(self.eff_tags.get("task", "asr")).lower()

    @property
    def lang_key(self) -> str | None:
        """The language entry this leaf belongs to, per the weighting policy."""
        if self.task == "st":
            src, tgt = self.eff_tags.get("src_lang"), self.eff_tags.get("tgt_lang")
            if not src or not tgt:
                return None
            return f"{cmw.canonical_lang(src)}-{cmw.canonical_lang(tgt)}"
        lang = self.eff_tags.get("lang")
        return cmw.canonical_lang(lang) if lang else None

    @property
    def corpus(self) -> str:
        """The corpus directory, e.g. ``cv22_sidon``: the first path component
        below the datasets root."""
        parts = [p for p in self.display.replace("\\", "/").split("/") if p]
        return parts[0] if parts else self.display

    @property
    def text_field(self) -> str | None:
        value = self.eff_tags.get("text_field")
        return str(value) if value else None

    def suggested_name(self) -> str:
        """A ``name:`` for this validation leaf, from its effective tags."""
        if self.task == "st":
            src = cmw.canonical_lang(self.eff_tags.get("src_lang", "xx"))
            tgt = cmw.canonical_lang(self.eff_tags.get("tgt_lang", "xx"))
            return f"st_{src}_{tgt}".replace("-", "_").lower()
        lang = cmw.canonical_lang(self.eff_tags.get("lang", "xx"))
        return f"{self.task}_{lang}".replace("-", "_").lower()


@dataclass
class Level:
    """One muxed level of ``input_cfg``: the root list, or a group's children."""

    where: str
    yaml_path: str
    line: int | None
    kind: str  # "root" or "group"
    weights: list[float | None]
    entry_lines: list[int | None]
    group_tags: dict
    leaves: list[Leaf]
    # For a group level, the weight on the group's own entry one level up -- p_l
    # in the two-tier policy. None for the root level, which has no parent.
    own_weight: float | None = None


def flatten(
    entries: list,
    where: str,
    line_index: dict[str, int],
    yaml_prefix: str,
    root: str | None = None,
) -> tuple[list[Leaf], list[Level]]:
    """Walk ``input_cfg`` into leaves and levels, mirroring ``_combine_entries``.

    Tag precedence follows the loader exactly: a group applies its tags *after*
    its children have applied theirs, so on a shared key the outermost group
    wins and a child-only key such as ``text_field`` survives.  This is what
    makes a group with a stale ``src_lang``/``tgt_lang`` silently override
    correct leaf tags, so the ordering here has to match rather than approximate.
    """
    leaves: list[Leaf] = []
    levels: list[Level] = []

    def walk(items: list, prefix: str, ancestors: list[dict], group_path: str | None,
             kind: str, group_tags: dict, level_line: int | None,
             own_weight: float | None) -> None:
        level = Level(
            where=where,
            yaml_path=prefix,
            line=level_line,
            kind=kind,
            weights=[],
            entry_lines=[],
            group_tags=group_tags,
            leaves=[],
            own_weight=own_weight,
        )
        levels.append(level)

        for i, entry in enumerate(items):
            entry = entry if isinstance(entry, dict) else {}
            entry_path = f"{prefix}.{i}"
            entry_line = line_index.get(entry_path)
            has_weight = "weight" in entry
            level.weights.append(float(entry["weight"]) if has_weight else None)
            level.entry_lines.append(entry_line)

            tags = dict(entry.get("tags") or {})
            source_type = str(entry.get("type", "lhotse_shar"))

            if source_type == "group":
                children = entry.get("input_cfg") or []
                walk(
                    children,
                    f"{entry_path}.input_cfg",
                    ancestors + [tags],
                    entry_path,
                    "group",
                    tags,
                    entry_line,
                    float(entry["weight"]) if has_weight else None,
                )
                # The group's leaves belong to this level for weight-sum purposes
                # via its own entry; its descendants are recorded on the nested
                # level.  Collect them here too so W4 can inspect membership.
                level.leaves.extend(
                    leaf for leaf in leaves if leaf.group_yaml_path == entry_path
                )
                continue

            raw_path = str(entry.get("shar_path", ""))
            # Outermost-last so the outermost group wins, matching the loader.
            eff: dict = dict(tags)
            for ancestor in reversed(ancestors):
                eff.update(ancestor)

            leaf = Leaf(
                where=where,
                yaml_path=entry_path,
                raw_path=raw_path,
                path=os.path.expandvars(raw_path),
                leaf_tags=tags,
                eff_tags=eff,
                weight=float(entry["weight"]) if has_weight else None,
                has_weight=has_weight,
                name=entry.get("name"),
                line=entry_line,
                group_yaml_path=group_path,
                root=root,
            )
            leaves.append(leaf)
            level.leaves.append(leaf)

    walk(entries, yaml_prefix, [], None, "root", {}, line_index.get(yaml_prefix), None)
    return leaves, levels


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _filter_key(min_duration: float, max_duration: float) -> str:
    return f"{float(min_duration)}:{float(max_duration)}"


def measure_shard_detailed(task: tuple[str, str, float]) -> tuple[str, float, int, dict[int, int]]:
    """Return (source, seconds, cuts, duration histogram) for one shard manifest.

    Only the top-level ``duration`` is read.  The unit of work is a shard, not a
    source, because shard counts span four orders of magnitude here (1 to 2373),
    and mapping over sources would leave one worker reading the largest corpus
    alone while the rest idle.

    The histogram is unfiltered and quantised, which is what lets one pass
    answer bin questions for any duration filter and any bucket count later.
    """
    source, shard, resolution = task
    seconds = 0.0
    cuts = 0
    hist: dict[int, int] = {}
    opener = gzip.open if str(shard).endswith(".gz") else open
    try:
        with opener(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    duration = float(json.loads(line)["duration"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                seconds += duration
                cuts += 1
                key = int(round(duration / resolution))
                hist[key] = hist.get(key, 0) + 1
    except OSError:
        # A vanished or unreadable shard is reported by D1/D2 from its own stat
        # pass; counting it as empty here keeps the measurement resumable.
        return source, 0.0, 0, {}
    return source, seconds, cuts, hist


def manifest_mtime(shards: list[Path]) -> float:
    """Newest manifest mtime in a source, used to invalidate a cached measurement."""
    newest = 0.0
    for shard in shards:
        try:
            newest = max(newest, shard.stat().st_mtime)
        except OSError:
            continue
    return round(newest, 3)


def seed_cache_from_mix_weights(cache: dict, seed_path: Path) -> int:
    """Fold compute_mix_weights' hours cache into ours as hours-only entries.

    Both tools sum the same unfiltered top-level ``duration`` over the same
    manifests, so the hours transfer exactly (verified to float precision when
    the seeded cache was built -- see history/18).  Cut counts and the histogram
    are not in that file, so a seeded entry can answer the hours and weight
    checks but not total_cuts or the bins, and is marked accordingly.
    """
    if not seed_path.exists():
        return 0
    try:
        seed = json.loads(seed_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    added = 0
    for path, entry in seed.items():
        if path in cache or not isinstance(entry, dict) or entry.get("hours") is None:
            continue
        if entry.get("sample"):
            # An extrapolated measurement is not something to check a config
            # against; history/16 records a --sample-shards run putting YODAS at
            # 24.5% against a true 16.9%.
            continue
        cache[path] = {
            "hours": entry["hours"],
            "cuts": None,
            "shards": entry.get("shards"),
            "hist": None,
            "filtered": {},
            "mtime": None,
            "source": str(seed_path),
        }
        added += 1
    return added


def cache_is_complete(entry: dict | None) -> bool:
    """Whether a cache entry can answer the cut-count and bins checks."""
    return bool(entry and entry.get("cuts") is not None and entry.get("hist"))


def measure_sources(
    paths: list[str],
    cache: dict,
    cache_path: Path,
    filters: list[tuple[float, float]],
    jobs: int,
    resolution: float = HIST_RESOLUTION,
) -> None:
    """Measure the given sources in full and write each into the cache as it lands."""
    plans: dict[str, list[Path]] = {}
    tasks: list[tuple[str, str, float]] = []
    for path in paths:
        shards = cmw.shar_manifest_files(Path(path))
        plans[path] = shards
        tasks.extend((path, str(shard), resolution) for shard in shards)

    for path, shards in plans.items():
        if not shards:
            cache[path] = {
                "hours": 0.0, "cuts": 0, "shards": 0, "hist": {},
                "filtered": {}, "mtime": None,
                "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    if not tasks:
        cmw.write_cache(cache, cache_path)
        return

    print(
        f"Measuring {len(plans)} sources / {len(tasks)} shards with {jobs} workers…",
        flush=True,
    )
    remaining = {path: len(shards) for path, shards in plans.items() if shards}
    seconds: dict[str, float] = defaultdict(float)
    cuts: dict[str, int] = defaultdict(int)
    hists: dict[str, dict[int, int]] = defaultdict(dict)
    done = 0
    started = time.time()

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        # pool.map preserves input order and tasks are grouped by source, so a
        # source is finished as soon as its last shard comes back.  Writing then
        # rather than at the end makes an interrupted pass resumable, which
        # matters when a full run is tens of thousands of manifests.
        for path, secs, n_cuts, hist in pool.map(measure_shard_detailed, tasks, chunksize=4):
            seconds[path] += secs
            cuts[path] += n_cuts
            target = hists[path]
            for key, count in hist.items():
                target[key] = target.get(key, 0) + count
            remaining[path] -= 1
            done += 1
            if remaining[path]:
                continue

            hist_all = hists.pop(path)
            entry = {
                "hours": seconds[path] / 3600.0,
                "cuts": cuts[path],
                "shards": len(plans[path]),
                "res": resolution,
                "hist": {str(k): v for k, v in sorted(hist_all.items())},
                "filtered": {},
                "mtime": manifest_mtime(plans[path]),
                "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            for min_duration, max_duration in filters:
                lo = int(round(min_duration / resolution))
                hi = int(round(max_duration / resolution))
                f_cuts = sum(v for k, v in hist_all.items() if lo <= k <= hi)
                f_secs = sum(k * resolution * v for k, v in hist_all.items() if lo <= k <= hi)
                entry["filtered"][_filter_key(min_duration, max_duration)] = {
                    "hours": f_secs / 3600.0,
                    "cuts": f_cuts,
                }
            cache[path] = entry
            elapsed = time.time() - started
            print(
                f"  [{done}/{len(tasks)} shards, {elapsed:6.1f}s] "
                f"{entry['hours']:10.1f} h {entry['cuts']:10d} cuts  {path}",
                flush=True,
            )
            cmw.write_cache(cache, cache_path)

    cmw.write_cache(cache, cache_path)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def summed_histogram(
    leaves: list[Leaf], cache: dict, min_duration: float, max_duration: float,
    resolution: float = HIST_RESOLUTION,
) -> tuple[dict[int, int], int]:
    """Duration histogram over a split, filtered, counting duplicate leaves twice.

    A leaf listed twice contributes twice, matching both the sampler (which
    draws from it under each task) and estimate_bucket_bins.py (which
    concatenates durations per input_cfg entry).
    """
    total: dict[int, int] = {}
    missing = 0
    for leaf in leaves:
        entry = cache.get(leaf.path)
        if not cache_is_complete(entry) or entry.get("res", resolution) != resolution:
            # A histogram binned at a different resolution cannot be summed with
            # the others; treat it as unmeasured rather than mixing scales.
            missing += 1
            continue
        lo = int(round(min_duration / resolution))
        hi = int(round(max_duration / resolution))
        for key, count in entry["hist"].items():
            key = int(key)
            if lo <= key <= hi:
                total[key] = total.get(key, 0) + int(count)
    return total, missing


def split_totals(
    leaves: list[Leaf], cache: dict, min_duration: float, max_duration: float,
) -> dict:
    """Unfiltered and filtered hours/cuts for a split, with and without duplicates."""
    key = _filter_key(min_duration, max_duration)
    out = {
        "hours": 0.0, "cuts": 0, "hours_distinct": 0.0, "cuts_distinct": 0,
        "hours_filtered": 0.0, "cuts_filtered": 0,
        "missing_hours": [], "missing_cuts": [], "missing_filtered": [],
        "approx_filtered": False,
    }
    seen: set[str] = set()
    for leaf in leaves:
        entry = cache.get(leaf.path)
        if not entry or entry.get("hours") is None:
            # Unmeasured means unmeasured for every total, not just hours: a
            # missing source must not be silently counted as zero cuts.
            out["missing_hours"].append(leaf.path)
            out["missing_cuts"].append(leaf.path)
            out["missing_filtered"].append(leaf.path)
            continue
        out["hours"] += entry["hours"]
        if leaf.path not in seen:
            out["hours_distinct"] += entry["hours"]

        if entry.get("cuts") is None:
            out["missing_cuts"].append(leaf.path)
        else:
            out["cuts"] += entry["cuts"]
            if leaf.path not in seen:
                out["cuts_distinct"] += entry["cuts"]

        filtered = (entry.get("filtered") or {}).get(key)
        if filtered is None and entry.get("hist"):
            # Derivable from the histogram, but the filter boundary is quantised,
            # so flag the result as approximate rather than presenting it as
            # measured at that filter.
            resolution = entry.get("res", HIST_RESOLUTION)
            lo = int(round(min_duration / resolution))
            hi = int(round(max_duration / resolution))
            f_cuts = sum(int(v) for k, v in entry["hist"].items() if lo <= int(k) <= hi)
            f_secs = sum(int(k) * resolution * int(v) for k, v in entry["hist"].items()
                         if lo <= int(k) <= hi)
            filtered = {"hours": f_secs / 3600.0, "cuts": f_cuts}
            out["approx_filtered"] = True
        if filtered:
            out["hours_filtered"] += filtered["hours"]
            out["cuts_filtered"] += filtered["cuts"]
        else:
            out["missing_filtered"].append(leaf.path)
        seen.add(leaf.path)
    return out


def sibling_configs(config_path: Path) -> list[Path]:
    """Other training configs, used to recognise copy-pasted bins."""
    roots = [config_path.parent, config_path.parent.parent / "train"]
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(p for p in root.glob("*.yaml") if p != config_path))
    return list(dict.fromkeys(out))


@lru_cache(maxsize=None)
def _sibling_bins(config_path: Path) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Every ``bucket_duration_bins`` in the sibling configs, as (location, bins).

    Cached because both splits ask, and these configs run to 55 KB apiece.
    """
    found: list[tuple[str, tuple[float, ...]]] = []
    for other in sibling_configs(config_path):
        try:
            text = other.read_text(encoding="utf-8")
            data = yaml.safe_load(cmw._expand_env(text))
            index = build_line_index(text)
        except Exception:
            # A sibling that does not parse is not this config's problem.
            continue
        if not isinstance(data, dict):
            continue
        for split in ("train_ds", "validation_ds"):
            bins = ((data.get("data") or {}).get(split) or {}).get("bucket_duration_bins")
            if not bins:
                continue
            line = index.get(f"data.{split}.bucket_duration_bins")
            location = f"{other}:{line}" if line else str(other)
            found.append((location, tuple(round(float(b), 4) for b in bins)))
    return tuple(found)


def find_bins_source(bins: list, config_path: Path) -> list[str]:
    """Configs carrying byte-identical bins, as ``path:line`` strings."""
    target = tuple(round(float(b), 4) for b in bins)
    return [location for location, other in _sibling_bins(config_path) if other == target]


# ---------------------------------------------------------------------------
# Checks: structure, tags, disk
# ---------------------------------------------------------------------------

def check_tags(report: Report, leaves: list[Leaf], where: str) -> None:
    """T1-T3: tag presence, group overrides, and locale folding."""
    report.ran_check("T1", "T2", "T3")

    for leaf in leaves:
        clashes = [
            (key, leaf.leaf_tags[key], leaf.eff_tags.get(key))
            for key in ("task", "lang", "src_lang", "tgt_lang", "text_field")
            if key in leaf.leaf_tags and leaf.leaf_tags[key] != leaf.eff_tags.get(key)
        ]
        if clashes:
            report.add(
                "T1", where,
                f"{leaf.display}: a group tag overrides this leaf's own tag",
                line=leaf.line,
                detail=[f"{key}: leaf says {have!r} -> group wins with {want!r}"
                        for key, have, want in clashes],
                fix=[
                    "The loader applies a group's tags after its children's, so the group wins",
                    "(melt/training/data/audio/lhotse/dataloader.py:606-613). Fix the group's",
                    f"tags at {report.location(group_line_of(leaf))} or drop the key from the",
                    "group so the leaf value survives.",
                ],
            )

        if leaf.task == "st":
            if not leaf.eff_tags.get("src_lang") or not leaf.eff_tags.get("tgt_lang"):
                report.add(
                    "T2", where,
                    f"{leaf.display}: ST source is missing src_lang and/or tgt_lang",
                    line=leaf.line,
                    fix=["Add both under tags:. compute_mix_weights.py refuses to run without them."],
                )
        elif not leaf.eff_tags.get("lang"):
            report.add(
                "T2", where,
                f"{leaf.display}: ASR source has no lang tag",
                line=leaf.line,
                fix=["Add tags.lang. compute_mix_weights.py refuses to run without it."],
            )

        for key in ("lang", "src_lang", "tgt_lang"):
            code = leaf.eff_tags.get(key)
            if not code:
                continue
            code = str(code)
            looks_locale = ("-" in code or "_" in code)
            if looks_locale and cmw.canonical_lang(code) == code:
                report.add(
                    "T3", where,
                    f"{leaf.display}: {key}={code!r} looks like a locale but is not in LOCALE_ALIASES",
                    line=leaf.line,
                    fix=[
                        f"Add a row to LOCALE_ALIASES in infra/compute_mix_weights.py: {code.lower()!r}: '<language>',",
                        "then re-run compute_mix_weights.py. Left unfolded, this code becomes its own",
                        "language entry and collects a full language-level share under beta.",
                    ],
                )


def group_line_of(leaf: Leaf) -> int | None:
    """Line of the group entry that owns a leaf, attached during flattening."""
    return getattr(leaf, "_group_line", None)


@dataclass
class Probe:
    """What one real cut from a source says about its text."""

    text_field_value: str | None
    supervision_text: str | None
    pnc_text: str | None
    translation_fields: list[str]


def read_first_cut(path: str) -> dict | None:
    """The first cut record in a source, or None if it cannot be read."""
    shards = cmw.shar_manifest_files(Path(path))
    if not shards:
        return None
    opener = gzip.open if str(shards[0]).endswith(".gz") else open
    try:
        with opener(shards[0], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return json.loads(line)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def probe_source(leaf: Leaf, record: dict | None) -> Probe | None:
    """Resolve, for one leaf, the text it would actually train on.

    Takes the record rather than reading it so that callers can share one read
    across leaves pointing at the same directory -- but the resolution itself is
    per leaf, because two leaves over one path can carry different text_field
    tags. The yodas ast sources are exactly that: the same directory appears
    once as ST reading custom.translation_en and once as ASR reading the
    supervision, and caching a resolved probe per path would report the ST
    source's field as missing whenever the ASR twin was read first.
    """
    if record is None:
        return None

    supervisions = record.get("supervisions") or []
    supervision_text = None
    if supervisions and isinstance(supervisions[0], dict):
        supervision_text = str(supervisions[0].get("text") or "") or None

    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    pnc_text = str(custom.get("pnc_text") or "") or None
    translations = sorted(k for k in custom if k.startswith("translation"))

    value = None
    field_name = leaf.text_field
    if field_name:
        cursor = record
        for part in field_name.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                cursor = None
                break
        value = cursor
        if value is None:
            # get_text_from_cut also tries the whole dotted name as a single key
            # under custom, so mirror that before declaring it unresolved.
            value = custom.get(field_name)

    return Probe(
        text_field_value=str(value) if value is not None and str(value).strip() else None,
        supervision_text=supervision_text,
        pnc_text=pnc_text,
        translation_fields=translations,
    )


def check_text_fields(report: Report, leaves: list[Leaf], where: str, probe: bool,
                      strict_text_field: bool) -> None:
    """T4: what text each source would actually train on, read from the manifests."""
    if not probe:
        report.skip("T4", "--no-probe")
        return
    report.ran_check("T4")

    # One manifest read per distinct path, but resolution per leaf: see
    # probe_source on why a resolved probe cannot be shared across leaves.
    records: dict[str, dict | None] = {}
    for leaf in leaves:
        if leaf.path not in records:
            records[leaf.path] = read_first_cut(leaf.path)

    for leaf in leaves:
        found = probe_source(leaf, records.get(leaf.path))
        if found is None:
            continue
        field_name = leaf.text_field

        if field_name and not found.text_field_value:
            fix = [f"The first cut has no usable value at {field_name!r}."]
            if found.supervision_text and not strict_text_field:
                fix.append(
                    "data.strict_text_field is not set, so the loader silently falls back to "
                    "the supervision text -- different content than the tag promises. This is "
                    "the failure mode behind training#66."
                )
            elif found.supervision_text:
                fix.append(
                    "data.strict_text_field is true, so this raises at training time instead "
                    "of falling back quietly."
                )
            else:
                fix.append("There is no supervision text either, so these cuts are skipped.")
            fix.append("Either correct tags.text_field or backfill the field in the manifests.")
            report.add(
                "T4", where,
                f"{leaf.display}: text_field {field_name!r} does not resolve on the first cut",
                line=leaf.line, fix=fix,
            )
            continue

        if not field_name:
            if not found.supervision_text:
                report.add(
                    "T4", where,
                    f"{leaf.display}: no text_field is set and the supervision text is empty",
                    line=leaf.line,
                    detail=[f"custom holds: {', '.join(sorted(set(found.translation_fields) | ({'pnc_text'} if found.pnc_text else set()))) or '(nothing text-like)'}"],
                    fix=["The loader has nothing to train on and skips these cuts.",
                         "Set tags.text_field to wherever this corpus keeps its transcript."],
                )
            elif leaf.task == "st" and found.translation_fields:
                report.add(
                    "T4", where,
                    f"{leaf.display}: tagged task: st with no text_field, so it trains on the "
                    f"supervision text rather than a translation",
                    line=leaf.line,
                    detail=[f"a translation is available at custom.{f}"
                            for f in found.translation_fields],
                    fix=["The supervision holds the source-language transcript here, so this",
                         "trains ASR text against an ST prompt.",
                         f"Set tags.text_field: custom.{found.translation_fields[0]}"],
                )
            elif found.pnc_text and found.pnc_text != found.supervision_text:
                report.add(
                    "T4", where,
                    f"{leaf.display}: no text_field is set, so it trains on the unpunctuated "
                    f"supervision text although custom.pnc_text exists",
                    line=leaf.line, severity=INFO,
                    detail=[f"supervision: {found.supervision_text[:60]!r}",
                            f"pnc_text:    {found.pnc_text[:60]!r}"],
                    fix=["Deliberate for some corpora, but it means this source is cased and",
                         "punctuated differently from the ones that do set the field.",
                         "Set tags.text_field: custom.pnc_text to match them."],
                )


def check_text_field_consistency(report: Report, leaves: list[Leaf], where: str) -> None:
    """T5: two leaves of the same corpus and task should read the same field.

    Structural, not data-driven: it catches a single source left on the old
    convention when a corpus was migrated, which no per-cut probe can see
    because each leaf is individually valid.
    """
    report.ran_check("T5")
    by_corpus: dict[tuple[str, str], list[Leaf]] = defaultdict(list)
    for leaf in leaves:
        by_corpus[(leaf.corpus, leaf.task)].append(leaf)

    for (corpus, task), members in sorted(by_corpus.items()):
        fields = {leaf.text_field for leaf in members}
        if len(fields) < 2:
            continue
        counts = defaultdict(list)
        for leaf in members:
            counts[leaf.text_field].append(leaf)
        majority = max(counts, key=lambda f: len(counts[f]))
        outliers = [leaf for field, group in counts.items() if field != majority
                    for leaf in group]
        report.add(
            "T5", where,
            f"{corpus} ({task}): {len(members)} sources disagree on text_field",
            line=outliers[0].line,
            detail=[f"{field!r}: {len(group)} sources" for field, group in counts.items()]
                   + [f"outlier: {report.location(leaf.line)} {leaf.display}"
                      for leaf in outliers[:5]],
            fix=[f"Most of this corpus uses {majority!r}. Unless the split genuinely differs,",
                 "align the outliers, or the same corpus trains with two text conventions."],
        )


def check_disk(report: Report, splits: dict[str, list[Leaf]]) -> None:
    """D1-D3: paths, manifests, index sidecars, and train/validation overlap."""
    report.ran_check("D1", "D2", "D3")

    for where, leaves in splits.items():
        # One report per distinct path: a leaf listed twice is one directory on
        # disk, and reporting a broken path twice adds nothing.
        for leaf in {l.path: l for l in leaves}.values():
            path = Path(leaf.path)
            if not path.exists():
                report.add(
                    "D1", where, f"{leaf.display}: shar_path does not exist",
                    line=leaf.line,
                    fix=[f"Resolved to {leaf.path}",
                         "Fix the path, or point --datasets-root at the right collection.",
                         "The loader raises FileNotFoundError on this at startup."],
                )
                continue
            shards = cmw.shar_manifest_files(path)
            if not shards:
                report.add(
                    "D1", where,
                    f"{leaf.display}: no cuts.*.jsonl[.gz] manifests found",
                    line=leaf.line,
                    fix=["This source measures as 0 h, which skews every mixture weight",
                         "without failing. Check the path or re-sync the dataset."],
                )
                continue

            plain = [s for s in shards if not str(s).endswith(".gz")]
            idx = {s.with_suffix(s.suffix + ".idx") for s in shards}
            present = {p for p in idx if p.exists()}
            if present and len(present) != len(shards):
                report.add(
                    "D2", where,
                    f"{leaf.display}: {len(present)}/{len(shards)} shards have an .idx sidecar",
                    line=leaf.line,
                    fix=["Indexing is per-source, so a partial index makes the source's",
                         "partitioning depend on which shard is read.",
                         f"python infra/index_shar.py --root {path.parent}"],
                )
            stale = []
            for shard in shards:
                sidecar = shard.with_suffix(shard.suffix + ".idx")
                if not sidecar.exists():
                    continue
                try:
                    if sidecar.stat().st_mtime + 1 < shard.stat().st_mtime:
                        stale.append(shard.name)
                except OSError:
                    continue
            if stale:
                report.add(
                    "D2", where,
                    f"{leaf.display}: {len(stale)} .idx sidecars are older than their manifest",
                    line=leaf.line,
                    detail=[", ".join(stale[:5]) + (" …" if len(stale) > 5 else "")],
                    fix=["A rewritten manifest invalidates the byte offsets beside it, and",
                         "nothing warns at read time -- you get wrong data or a late crash.",
                         f"python infra/index_shar.py --root {path.parent}"],
                )
            if plain and not present:
                report.add(
                    "D2", where,
                    f"{leaf.display}: manifests are uncompressed but carry no .idx sidecars",
                    line=leaf.line, severity=INFO,
                    fix=["Plain .jsonl without .idx is the shape an interrupted indexing run",
                         "leaves behind. Harmless, but the source streams instead of partitioning."],
                )

    train = {l.path: l for l in splits.get("train_ds", [])}
    shared = [l for l in splits.get("validation_ds", []) if l.path in train]
    for leaf in shared:
        report.add(
            "D3", "validation_ds",
            f"{leaf.display}: also used in train_ds",
            line=leaf.line,
            fix=["Validation loss over data the model trains on is not a held-out measure.",
                 f"train_ds uses it at {report.location(train[leaf.path].line)}"],
        )


# ---------------------------------------------------------------------------
# Checks: weights
# ---------------------------------------------------------------------------

def check_weights(
    report: Report, leaves: list[Leaf], levels: list[Level], cache: dict,
    alpha: float, beta: float, tol: float, template_hint: str,
) -> None:
    """W1-W4: the mixture weights."""
    report.ran_check("W1")

    # W1: all-or-none within each level (mirrors _resolve_weights).
    any_explicit = False
    for level in levels:
        if len(level.weights) <= 1:
            continue
        n_explicit = sum(w is not None for w in level.weights)
        if n_explicit:
            any_explicit = True
        if n_explicit and n_explicit != len(level.weights):
            missing = [i for i, w in enumerate(level.weights) if w is None]
            report.add(
                "W1", level.where,
                f"{level.yaml_path}: {n_explicit} of {len(level.weights)} entries set 'weight'",
                line=level.line,
                detail=[f"entries missing it: {missing}"],
                fix=["Set weight on all of them or none. The loader raises here",
                     "(dataloader.py:675-682): an explicit weight is a share of the level while",
                     "an automatic one is a raw cut count, and muxing normalises both onto one",
                     "scale, starving the explicitly weighted sources."],
            )

    if not any_explicit:
        report.note(
            "No entry sets 'weight': the loader will auto-weight by cut count, so every cut "
            "is drawn at the same rate regardless of language. That is a valid policy, but it "
            "is not the two-tier balancing in docs/mixture_weights.md, so W2-W4 do not apply."
        )
        for check in ("W2", "W3", "W4"):
            report.skip(check, "the mixture sets no explicit weights")
        return

    # W2: every level is a distribution.
    report.ran_check("W2")
    for level in levels:
        weights = [w for w in level.weights if w is not None]
        if len(weights) <= 1:
            continue
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            report.add(
                "W2", level.where,
                f"{level.yaml_path}: weights sum to {total:.8f}, not 1",
                line=level.line,
                fix=[f"Re-emit the config: {template_hint}"],
            )
        if any(w <= 0 for w in weights):
            report.add(
                "W2", level.where,
                f"{level.yaml_path}: has a non-positive weight",
                line=level.line,
                detail=[f"weights: {weights}"],
                fix=["A zero weight removes the source from the mixture silently."],
            )

    # W4: the group structure has to be the language partition the policy assumes.
    report.ran_check("W4")
    grouped = defaultdict(list)
    for leaf in leaves:
        grouped[leaf.group_yaml_path].append(leaf)

    if any(key is None for key in grouped) and len(grouped) > 1:
        report.add(
            "W4", leaves[0].where if leaves else "train_ds",
            "some sources sit at the root while others are inside groups",
            fix=["The two-tier policy expects one group per language entry.",
                 f"Re-emit the config: {template_hint}"],
        )

    for group_path, members in grouped.items():
        if group_path is None:
            continue
        keys = {m.lang_key for m in members}
        if len(keys) > 1:
            report.add(
                "W4", members[0].where,
                f"{group_path}: group mixes {len(keys)} language entries: {sorted(str(k) for k in keys)}",
                line=members[0].line,
                fix=["A group is one language entry: its weight is that language's share p_l.",
                     "Mixing entries means the beta-level balancing is not what the config says.",
                     f"Re-emit the config: {template_hint}"],
            )
        group_tags = members[0].eff_tags
        expected = {k: v for k, v in group_tags.items() if k in cmw.GROUP_TAG_KEYS}
        for member in members:
            declared = {k: v for k, v in member.leaf_tags.items() if k in cmw.GROUP_TAG_KEYS}
            mismatch = {k: (v, expected.get(k)) for k, v in declared.items()
                        if k in expected and v != expected[k]}
            if mismatch:
                report.add(
                    "W4", member.where,
                    f"{member.display}: leaf tags disagree with its group's",
                    line=member.line,
                    detail=[f"{k}: leaf {have!r} vs group {want!r}" for k, (have, want) in mismatch.items()],
                    fix=["The group wins at load time, so the group's value is what trains.",
                         f"Re-emit the config: {template_hint}"],
                )

    # W3: the numbers themselves.
    usable = [leaf for leaf in leaves if (cache.get(leaf.path) or {}).get("hours") is not None]
    if len(usable) != len(leaves):
        report.skip(
            "W3",
            f"{len(leaves) - len(usable)} of {len(leaves)} sources have no measured hours",
        )
        return

    report.ran_check("W3")
    sources = [
        {"path": leaf.path, "lang_key": leaf.lang_key or "?",
         "hours": cache[leaf.path]["hours"], "leaf": leaf}
        for leaf in leaves
    ]
    cmw.compute_weights(sources, alpha, beta)

    worst_pc = worst_pl = 0.0
    offenders: list[str] = []
    for source in sources:
        leaf: Leaf = source["leaf"]
        if leaf.weight is not None:
            delta = abs(source["p_c"] - leaf.weight)
            worst_pc = max(worst_pc, delta)
            if delta > tol:
                offenders.append(
                    f"{leaf.display}: weight {leaf.weight:.8f}, expected p_c {source['p_c']:.8f} "
                    f"(delta {delta:.2e})"
                )

    # p_l is a property of the language entry, so every member of a group agrees
    # on it; reading it off any one member is safe.
    group_expected: dict[str, float] = {}
    for source in sources:
        leaf = source["leaf"]
        if leaf.group_yaml_path:
            group_expected[leaf.group_yaml_path] = source["p_l"]
    for level in levels:
        # A group level's own weight was recorded from its entry in the parent
        # list while flattening, so no search is needed here.
        if level.kind != "group" or level.own_weight is None:
            continue
        expected = group_expected.get(level.yaml_path.rsplit(".input_cfg", 1)[0])
        if expected is None:
            continue
        delta = abs(expected - level.own_weight)
        worst_pl = max(worst_pl, delta)
        if delta > tol:
            offenders.append(
                f"{level.yaml_path}: group weight {level.own_weight:.8f}, expected p_l "
                f"{expected:.8f} (delta {delta:.2e})"
            )

    if offenders:
        report.add(
            "W3", leaves[0].where,
            f"{len(offenders)} weights do not match the two-tier policy at "
            f"alpha={alpha}, beta={beta}",
            detail=offenders[:20] + ([f"… and {len(offenders) - 20} more"] if len(offenders) > 20 else []),
            fix=[f"Re-emit the config: {template_hint}"],
        )
    else:
        report.note(
            f"Weights reproduce the policy at alpha={alpha}, beta={beta}: "
            f"worst |delta p_c| = {worst_pc:.2e}, worst |delta p_l| = {worst_pl:.2e}."
        )


# ---------------------------------------------------------------------------
# Checks: validation names
# ---------------------------------------------------------------------------

def check_names(report: Report, leaves: list[Leaf]) -> None:
    """N1-N2: per-eval-set naming."""
    if not leaves:
        report.skip("N1", "no validation_ds.input_cfg")
        report.skip("N2", "no validation_ds.input_cfg")
        return
    report.ran_check("N1", "N2")

    named = [leaf for leaf in leaves if leaf.name]
    if not named:
        suggestions = []
        for leaf in leaves:
            location = f"{report.config_path}:{leaf.line}" if leaf.line else leaf.display
            suggestions.append(f"{location}   name: {leaf.suggested_name()}")
        curves = sorted({leaf.suggested_name() for leaf in leaves})
        report.add(
            "N1", "validation_ds",
            f"none of the {len(leaves)} validation sources declares a name, so every one of "
            "them is pooled into a single eval_loss",
            line=leaves[0].line,
            detail=suggestions,
            fix=[
                "split_eval_config_by_name returns None when no source is named "
                "(dataloader.py:1343), so the trainer builds one flat eval set and the "
                "per-language signal is lost.",
                "Add a `name:` key beside `type:`/`shar_path:` on each validation entry -- "
                "sources sharing a name are evaluated together and reported as one curve.",
                "Suggested names are listed above; they would give these curves:",
                "  " + ", ".join(f"eval_{n}_loss" for n in curves),
            ],
        )
        return

    if len(named) != len(leaves):
        unnamed = [leaf for leaf in leaves if not leaf.name]
        report.add(
            "N2", "validation_ds",
            f"{len(named)} of {len(leaves)} validation sources declare a name",
            line=unnamed[0].line,
            detail=[f"{report.location(leaf.line)}   {leaf.display}   -> name: {leaf.suggested_name()}"
                    for leaf in unnamed],
            fix=["The loader raises on this at startup (dataloader.py:1351): naming is "
                 "all-or-none. Name every source or none of them."],
        )
        return

    groups = defaultdict(list)
    for leaf in leaves:
        groups[str(leaf.name)].append(leaf)
    report.note(
        f"{len(leaves)} validation sources in {len(groups)} named eval sets: "
        + ", ".join(f"eval_{name}_loss ({len(members)})" for name, members in groups.items())
    )


# ---------------------------------------------------------------------------
# Checks: totals and bins
# ---------------------------------------------------------------------------

def check_totals(
    report: Report, where: str, split: dict, leaves: list[Leaf], cache: dict,
    line_index: dict[str, int], checks: tuple[str, str],
) -> None:
    """H1-H3: total_hours and total_cuts against the measured mixture."""
    hours_check, cuts_check = checks
    min_duration = float(split.get("min_duration") or 0.0)
    max_duration = float(split.get("max_duration") or float("inf"))
    totals = split_totals(leaves, cache, min_duration, max_duration)

    declared_hours = split.get("total_hours")
    declared_cuts = split.get("total_cuts")
    force_estimate = bool(split.get("force_estimate"))
    hours_line = line_index.get(f"data.{where}.total_hours")
    cuts_line = line_index.get(f"data.{where}.total_cuts")

    duplicate_note = ""
    if len(leaves) != len({l.path for l in leaves}):
        n_dup = len(leaves) - len({l.path for l in leaves})
        distinct = f"{totals['hours_distinct']:,.1f} h"
        if not totals["missing_cuts"]:
            distinct += f" / {totals['cuts_distinct']:,} cuts"
        duplicate_note = (
            f" ({n_dup} sources are listed twice and counted twice, on purpose: "
            f"distinct audio is {distinct})"
        )

    if totals["missing_hours"]:
        report.skip(
            hours_check,
            f"{len(totals['missing_hours'])} of {len(leaves)} {where} sources have no measured hours",
        )
    else:
        report.ran_check(hours_check)
        measured = totals["hours"]
        if totals["missing_filtered"]:
            filtered_note = (
                f" (the figure after the [{min_duration}, {max_duration}] s filter needs "
                "duration histograms, which --measure collects)"
            )
        else:
            # Derived figures carry a "~": they come from histograms binned at
            # 0.01 s, so they can land a hair above the exact unfiltered sum,
            # which looks like nonsense unless it is marked as an estimate.
            approx = "~" if totals["approx_filtered"] else ""
            filtered_note = (
                f", {approx}{totals['hours_filtered']:,.1f} h within "
                f"[{min_duration}, {max_duration}] s"
            )
        report.note(
            f"{where}: measured {measured:,.1f} h unfiltered{filtered_note}{duplicate_note}"
        )
        if declared_hours is None:
            if not force_estimate:
                pass  # E3 owns this case.
            else:
                report.add(
                    hours_check, where,
                    f"total_hours is null with force_estimate: true, so the loader re-measures "
                    f"every source at startup ({measured:,.1f} h across {len(leaves)} sources)",
                    line=hours_line, severity=INFO,
                    fix=[f"Pin it to skip that: data.{where}.total_hours: {measured:.1f}"],
                )
        else:
            declared = float(declared_hours)
            drift = abs(declared - measured)
            if drift > max(0.1, 0.0005 * measured):
                report.add(
                    hours_check, where,
                    f"total_hours is {declared:,.1f} but the mixture measures {measured:,.1f} h "
                    f"(off by {drift:,.1f} h, {drift / measured * 100:.2f}%)",
                    line=hours_line,
                    fix=[f"Set data.{where}.total_hours: {measured:.1f}",
                         "This feeds estimate_steps_per_epoch -> max_steps -> the LR schedule,",
                         "so a stale value silently rescales warmup and decay."],
                )

    if totals["missing_cuts"]:
        report.skip(
            cuts_check,
            f"{len(totals['missing_cuts'])} of {len(leaves)} {where} sources have no measured "
            "cut count (the seeded hours cache does not record one)",
        )
    else:
        report.ran_check(cuts_check)
        measured_cuts = totals["cuts"]
        if declared_cuts is None:
            if force_estimate:
                report.add(
                    cuts_check, where,
                    f"total_cuts is null with force_estimate: true; the loader will measure "
                    f"{measured_cuts:,} cuts at startup",
                    line=cuts_line, severity=INFO,
                    fix=[f"Pin it to skip that: data.{where}.total_cuts: {measured_cuts}"],
                )
        else:
            declared = int(declared_cuts)
            drift = abs(declared - measured_cuts)
            if drift > max(1, int(0.0001 * measured_cuts)):
                report.add(
                    cuts_check, where,
                    f"total_cuts is {declared:,} but the mixture measures {measured_cuts:,} "
                    f"(off by {drift:,})",
                    line=cuts_line,
                    fix=[f"Set data.{where}.total_cuts: {measured_cuts}",
                         "compute_mix_weights.py updates total_hours but not total_cuts",
                         "(docs/mixture_weights.md), so this one is maintained by hand."],
                )
            if totals["approx_filtered"]:
                report.note(
                    f"{where}: filtered totals were derived from cached histograms, so they are "
                    "accurate to one 0.01 s bin at the filter boundary."
                )


def check_bins_structure(
    report: Report, where: str, split: dict, line_index: dict[str, int],
) -> int | None:
    """B1-B2: the bucketing configuration's shape, without touching the data.

    Returns ``num_buckets`` when the split is bucketed and worth measuring, else
    None.
    """
    sampler = split.get("lhotse_sampler_type")
    bins = split.get("bucket_duration_bins")
    num_buckets = split.get("num_buckets")
    min_duration = float(split.get("min_duration") or 0.0)
    max_duration = float(split.get("max_duration") or float("inf"))
    bins_line = line_index.get(f"data.{where}.bucket_duration_bins")

    if sampler != "dynamic_bucketing":
        for check in ("B1", "B2", "B3", "C4", "C5"):
            report.skip(check, f"{where} uses lhotse_sampler_type: {sampler!r}")
        return None

    report.ran_check("B1")
    if num_buckets is None:
        report.add(
            "B1", where,
            "lhotse_sampler_type is dynamic_bucketing but num_buckets is not set",
            line=line_index.get(f"data.{where}"),
            fix=["The loader raises on this at startup (dataloader.py:864).",
                 f"Set data.{where}.num_buckets: 30"],
        )
        return None
    num_buckets = int(num_buckets)

    if bins:
        report.ran_check("B2")
        values = [float(b) for b in bins]
        if len(values) != num_buckets - 1:
            report.add(
                "B2", where,
                f"bucket_duration_bins has {len(values)} values but num_buckets is "
                f"{num_buckets}, which needs {num_buckets - 1}",
                line=bins_line,
                fix=["Bins are bucket boundaries, so there is one fewer than buckets.",
                     f"Either fix the list or set num_buckets: {len(values) + 1}"],
            )
        if any(b >= c for b, c in zip(values, values[1:])):
            bad = [i for i, (b, c) in enumerate(zip(values, values[1:])) if b >= c]
            report.add(
                "B2", where,
                "bucket_duration_bins is not strictly increasing",
                line=bins_line,
                detail=[f"at index {i}: {values[i]} >= {values[i + 1]}" for i in bad[:5]],
                fix=["Bucket boundaries must ascend."],
            )
        outside = [b for b in values if b < min_duration or b > max_duration]
        if outside:
            report.add(
                "B2", where,
                f"{len(outside)} bins fall outside the duration filter "
                f"[{min_duration}, {max_duration}] s",
                line=bins_line,
                detail=[f"{outside[:8]}"],
                fix=["A bin outside the filter can never receive a cut, so that bucket stays empty."],
            )
    elif split.get("batch_duration") or split.get("batch_size"):
        report.add(
            "B2", where,
            "bucket_duration_bins is not set, so lhotse estimates the bins at runtime "
            "from the first buffer_size cuts",
            line=line_index.get(f"data.{where}.bucket_duration_bins")
                 or line_index.get(f"data.{where}.num_buckets"),
            severity=INFO,
            fix=["That estimate varies with shuffling, so batch composition is not reproducible",
                 "across runs. B3 prints the measured bins to pin here."],
        )
    return num_buckets


def check_bins_values(
    report: Report, where: str, split: dict, leaves: list[Leaf], cache: dict,
    line_index: dict[str, int], num_buckets: int,
    tol_abs: float, tol_rel: float, precision: int,
) -> None:
    """B3, C4, C5: the bin values against the mixture's real duration distribution."""
    bins = split.get("bucket_duration_bins")
    min_duration = float(split.get("min_duration") or 0.0)
    max_duration = float(split.get("max_duration") or float("inf"))
    bins_line = line_index.get(f"data.{where}.bucket_duration_bins")

    hist, missing = summed_histogram(leaves, cache, min_duration, max_duration)
    if missing or not hist:
        why = (f"{missing} of {len(leaves)} {where} sources have no cached duration histogram"
               if missing else f"no durations measured for {where}")
        for check in ("B3", "C4"):
            report.skip(check, why)
        report.skip("C5", why)
        return

    report.ran_check("C4", "C5")
    expected = bucket_bins.estimate_bins_from_histogram(hist, HIST_RESOLUTION, num_buckets)
    expected_rounded = [round(b, precision) for b in expected]
    formatted = bucket_bins.format_bins_for_yaml(expected, precision)
    values = [float(b) for b in bins] if bins else []

    if not bins:
        report.ran_check("B3")
        report.add(
            "B3", where,
            f"bucket_duration_bins is unset; the measured distribution gives "
            f"{num_buckets - 1} bins",
            line=line_index.get(f"data.{where}.num_buckets"),
            severity=INFO,
            fix=[f"data.{where}.bucket_duration_bins: {formatted}"],
        )
    elif len(values) != len(expected):
        # Comparing element-wise across different lengths would compare
        # unrelated boundaries, and B2 already reports the length itself, so
        # defer to it rather than inventing a verdict here.
        report.skip(
            "B3",
            f"{len(values)} bins where num_buckets implies {len(expected)} (see B2)",
        )
    else:
        report.ran_check("B3")
        deltas = [abs(a - b) for a, b in zip(values, expected)]
        worst = max(deltas)
        worst_rel = max((abs(a - b) / b if b else 0.0) for a, b in zip(values, expected))
        # Both tolerances must be exceeded: bins drift a little with any
        # remeasurement, and the target here is bins that were never measured
        # for this mixture at all, which drift much further than that.
        if worst > tol_abs and worst_rel > tol_rel:
            index = deltas.index(worst)
            report.add(
                "B3", where,
                f"bucket_duration_bins does not match this mixture: worst bin is off by "
                f"{worst:.2f} s ({worst_rel * 100:.1f}%) at index {index}",
                line=bins_line,
                detail=[f"config:   {bucket_bins.format_bins_for_yaml(values, precision)}",
                        f"measured: {formatted}"],
                fix=[f"data.{where}.bucket_duration_bins: {formatted}",
                     "",
                     "Or recompute them with the dedicated tool (which needs joblib and",
                     "omegaconf, absent from the lhotse2 venv):",
                     f"  python infra/estimate_bucket_bins.py --config {report.config_path} "
                     f"--{'train' if where == 'train_ds' else 'val'}-buckets {num_buckets}"],
            )
        else:
            report.note(
                f"{where}: bins match the measured distribution, worst bin off by {worst:.3f} s."
            )

    # C5: bins copied from another config are a common source of drift. Only
    # raised once B3 has actually rejected them -- bins shared with another
    # config that still describe this mixture are simply two configs over
    # similar data, which is not worth a warning.
    if values and report.status("B3") == FAIL:
        matches = find_bins_source(values, report.config_path)
        if matches and expected_rounded != [round(v, precision) for v in values]:
            report.add(
                "C5", where,
                "these bins are byte-identical to another config's, so they were copied "
                "rather than measured for this mixture",
                line=bins_line,
                detail=[f"also in {m}" for m in matches[:5]],
                fix=[f"data.{where}.bucket_duration_bins: {formatted}"],
            )

    # C4: does the top bucket actually reach the long cuts?
    keys = sorted(hist)
    total_cuts = sum(hist.values())
    cumulative = 0
    p99 = keys[-1] * HIST_RESOLUTION
    for key in keys:
        cumulative += hist[key]
        if cumulative >= 0.99 * total_cuts:
            p99 = key * HIST_RESOLUTION
            break
    longest = keys[-1] * HIST_RESOLUTION
    if bins:
        top = max(float(b) for b in bins)
        if top < p99:
            report.add(
                "C4", where,
                f"the highest bin is {top:.2f} s but 1% of cuts are longer than {p99:.2f} s "
                f"(longest {longest:.2f} s)",
                line=bins_line,
                fix=["Everything above the top bin lands in one bucket, so that bucket mixes",
                     "durations and pads heavily.",
                     f"data.{where}.bucket_duration_bins: {formatted}"],
            )


# ---------------------------------------------------------------------------
# Checks: runtime traps
# ---------------------------------------------------------------------------

def check_runtime(
    report: Report, cfg: dict, splits: dict[str, list[Leaf]],
    grouped: dict[str, bool], line_index: dict[str, int],
) -> None:
    """E1-E3, C1-C3: config shapes that fail or mislead at training time."""
    data = cfg.get("data") or {}
    report.ran_check("E1", "E2", "E3", "C1", "C2", "C3", "C6")

    for where in ("train_ds", "validation_ds"):
        split = data.get(where) or {}
        if not split:
            continue
        total_hours = split.get("total_hours")
        total_cuts = split.get("total_cuts")
        force_estimate = bool(split.get("force_estimate"))
        batch_size = split.get("batch_size")
        batch_duration = split.get("batch_duration")

        if total_hours is None and force_estimate and grouped.get(where):
            report.add(
                "E1", where,
                "force_estimate: true, but input_cfg uses type: group and the estimator "
                "does not recurse into groups",
                line=line_index.get(f"data.{where}.force_estimate"),
                fix=["compute_dataset_duration iterates input_cfg flatly (dataloader.py:277),",
                     "so every group entry is skipped and it returns 0 h / 0 cuts. The trainer",
                     "then raises 'Could not estimate optimization steps per epoch'.",
                     f"Set data.{where}.total_hours and total_cuts explicitly instead."],
            )

        # Only a trap when total_hours is set: with total_hours null and
        # force_estimate on, compute_dataset_duration fills total_cuts in before
        # the division, so the None never reaches it.
        if (total_hours is not None and total_cuts is None
                and isinstance(batch_size, int) and batch_size > 0):
            report.add(
                "E2", where,
                f"total_cuts is null while total_hours is set and batch_size is {batch_size}",
                line=line_index.get(f"data.{where}.total_cuts"),
                fix=["With total_hours set, the estimator takes the batch_size path and computes",
                     "ceil(total_cuts / batch_size) (dataloader.py:370), which raises TypeError",
                     "on None.",
                     f"Pin data.{where}.total_cuts, or clear total_hours and set force_estimate."],
            )

        if total_hours is None and not force_estimate:
            report.add(
                "E3", where,
                "total_hours is null and force_estimate is not set, so steps per epoch "
                "cannot be estimated",
                line=line_index.get(f"data.{where}.total_hours"),
                fix=["estimate_steps_per_epoch returns -1 (dataloader.py:362) and the trainer",
                     "raises 'Could not estimate optimization steps per epoch'.",
                     f"Set data.{where}.total_hours, or force_estimate: true."],
            )

        if not batch_size and not batch_duration:
            report.add(
                "E3", where,
                "neither batch_size nor batch_duration is set",
                line=line_index.get(f"data.{where}"),
                fix=["Steps per epoch cannot be derived without one of them",
                     "(dataloader.py:373-375)."],
            )

    # C1: strict_text_field is declared at data. but eval only inherits the
    # four formatting keys, so eval silently runs non-strict.
    validation = data.get("validation_ds") or {}
    if data.get("strict_text_field") and "strict_text_field" not in validation:
        uses_text_field = any(leaf.text_field for leaf in splits.get("validation_ds", []))
        if uses_text_field:
            report.add(
                "C1", "validation_ds",
                "data.strict_text_field is true but eval does not inherit it, so validation "
                "silently falls back to the supervision text where a text_field is empty",
                line=line_index.get("data.strict_text_field"),
                fix=["resolve_eval_data_config only inherits apply_chat_template, prompt_template,",
                     "prompt_template_selection and chat_template_config (dataloader.py:1226-1231).",
                     "Add strict_text_field: true inside data.validation_ds to match training."],
            )

    trainer = cfg.get("trainer") or {}
    eval_enabled = bool(trainer.get("do_eval")) and trainer.get("eval_strategy") not in (None, "no")
    if eval_enabled and trainer.get("per_device_eval_batch_size") == -1:
        report.add(
            "C2", "trainer",
            "per_device_eval_batch_size is -1 while eval is enabled",
            line=line_index.get("trainer.per_device_eval_batch_size"),
            fix=["Eval does not go through Lhotse; it uses a plain DataLoader and -1 crashes",
                 "the first eval. Set a real batch size (4 is known good on 64 GB H100s)."],
        )

    # Generation-based eval decodes token by token: roughly N sequential decoder
    # steps per sample instead of one forward. The full validation set was
    # already the dominant cost under teacher forcing; under generation it does
    # not finish. max_samples is what bounds it.
    validation = data.get("validation_ds") or {}
    if (
        eval_enabled
        and trainer.get("predict_with_generate") is not False
        and validation.get("input_cfg")
        and validation.get("max_samples") in (None, 0)
    ):
        report.add(
            "C6", "validation_ds",
            "eval decodes with generate() but validation_ds has no max_samples",
            line=line_index.get("data.validation_ds"),
            fix=["Generation costs ~one decoder step per output token per sample, against",
                 "a single forward before -- ~2.2 s/utterance at generation_max_length 64",
                 "on an H100, scaling with the budget. Set data.validation_ds.max_samples;",
                 "note it applies PER NAMED eval set, so 200 across five named sets scores",
                 "1000 utterances. materialize_cuts_for_eval applies it with a seeded",
                 "shuffle, so every run scores the same subset."],
        )

    encoder = ((cfg.get("model") or {}).get("encoder") or {})
    max_frames = encoder.get("max_audio_seq_len")
    frame_seconds = encoder_frame_seconds(encoder.get("name"))
    if max_frames:
        window = float(max_frames) * frame_seconds
        for where in ("train_ds", "validation_ds"):
            split = data.get(where) or {}
            max_duration = split.get("max_duration")
            if max_duration and float(max_duration) > window:
                num_chunks = math.ceil(float(max_duration) / window)
                report.add(
                    "C3", where,
                    f"max_duration is {float(max_duration):g} s, above the encoder's "
                    f"{window:g} s window ({max_frames} frames x "
                    f"{frame_seconds * 1000:g} ms) -- expected, not a truncation risk",
                    line=line_index.get(f"data.{where}.max_duration"),
                    fix=[f"MELTAudioStack.encoder chunks any sequence longer than {max_frames} "
                         "frames instead of truncating it (modeling_melt.py:509-555): a "
                         f"{float(max_duration):g} s cut is split into "
                         f"ceil({float(max_duration):g} / {window:g}) = {num_chunks} chunks of "
                         f"up to {window:g} s, the last one zero-padded, stacked into the batch "
                         "dimension, and encoded independently before the adapter reassembles "
                         "them. No config change is implied by this message."],
                )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

MARK = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}


def print_report(report: Report, sections: list[str]) -> None:
    print()
    print("=" * 78)
    print(f"Config consistency: {report.config_path}")
    print("=" * 78)

    by_section: dict[str, list[str]] = defaultdict(list)
    for check, (section, _) in CHECKS.items():
        by_section[section].append(check)

    for section in sections:
        checks = by_section.get(section, [])
        if not checks:
            continue
        print(f"\n[{section}]")
        for check in checks:
            status = report.status(check)
            _, description = CHECKS[check]
            suffix = ""
            if status == SKIP:
                suffix = f"  ({report.skipped.get(check, 'not run')})"
            print(f"  {MARK[status]}  {check}  {description}{suffix}")
            for finding in [f for f in report.findings if f.check == check]:
                tag = "INFO" if finding.severity == INFO else finding.severity
                print(f"          {tag} {finding.where}: {finding.message}")
                if finding.line:
                    print(f"               at {report.location(finding.line)}")
                for line in finding.detail:
                    print(f"               {line}")

    if report.notes:
        print("\n[measured]")
        for note in report.notes:
            print(f"  - {note}")

    # Ordered so the hard failures come first, but informational suggestions
    # (pin this value, name these sets) still get their command printed.
    rank = {FAIL: 0, WARN: 1, INFO: 2}
    actionable = sorted(
        (f for f in report.findings if f.fix),
        key=lambda f: (rank.get(f.severity, 3), f.check),
    )
    if actionable:
        print("\n" + "=" * 78)
        print("WHAT TO DO")
        print("=" * 78)
        for i, finding in enumerate(actionable, 1):
            print(f"\n{i}. [{finding.check} {finding.severity}] {finding.where}: {finding.message}")
            if finding.line:
                print(f"   {report.location(finding.line)}")
            for line in finding.fix:
                print(f"   {line}" if line else "")

    counts = defaultdict(int)
    for check in CHECKS:
        counts[report.status(check)] += 1
    print()
    print("-" * 78)
    print(
        f"{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, "
        f"{counts[SKIP]} skipped"
    )
    if counts[SKIP]:
        print("Skipped checks need measurements; re-run with --measure to fill the cache.")


def report_to_json(report: Report) -> dict:
    return {
        "config": str(report.config_path),
        "failed": report.failed(),
        "checks": {
            check: {
                "section": CHECKS[check][0],
                "description": CHECKS[check][1],
                "status": report.status(check),
                "skipped_because": report.skipped.get(check),
                "findings": [
                    {
                        "severity": f.severity, "where": f.where, "message": f.message,
                        "line": f.line, "detail": f.detail, "fix": f.fix,
                    }
                    for f in report.findings if f.check == check
                ],
            }
            for check in CHECKS
        },
        "notes": report.notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def guess_template(config_path: Path) -> Path | None:
    """The hand-edited template an emitted config came from, by convention."""
    candidate = config_path.with_name(f"{config_path.stem}-template{config_path.suffix}")
    return candidate if candidate.exists() else None


def template_command(config_path: Path, template: Path | None, datasets_root: str | None,
                     cache: Path, alpha: float, beta: float) -> str:
    """The compute_mix_weights.py invocation that would regenerate this config."""
    source = template or config_path
    exp_name = config_path.stem
    parts = [
        "python3 infra/compute_mix_weights.py",
        f"--config {source}",
        f"--datasets-root {datasets_root or '$LOCAL_DATASETS_DIR'}",
        f"--cache {cache}",
        f"--emit-config {config_path}",
        f"--exp-name {exp_name}",
    ]
    if (alpha, beta) != (0.5, 0.5):
        parts.insert(3, f"--alpha {alpha} --beta {beta}")
    hint = " \\\n      ".join(parts)
    if template is None:
        hint += (
            "\n      # NOTE: no <stem>-template.yaml found, so this reads the emitted config"
            "\n      # itself. Pass --template if the hand-edited source lives elsewhere;"
            "\n      # edits inside train_ds.input_cfg are lost on re-emit."
        )
    return hint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR") or None,
                        help="Value for LOCAL_DATASETS_DIR (default: from the environment).")
    parser.add_argument("--template", type=Path, default=None,
                        help="The hand-edited template this config was emitted from "
                             "(default: <stem>-template.yaml beside it).")
    parser.add_argument("--cache", type=Path, default=Path("infra/.config_check_cache.json"),
                        help="Where per-source measurements are cached.")
    parser.add_argument("--seed-cache", type=Path, default=Path("infra/.mix_weights_hours.json"),
                        help="compute_mix_weights.py's hours cache, folded in as a starting point.")
    parser.add_argument("--offline", action="store_true",
                        help="Structure only: never touch the data. No datasets root needed.")
    parser.add_argument("--measure", action="store_true",
                        help="Allow a full measuring pass for anything the cache is missing.")
    parser.add_argument("--force-measure", action="store_true",
                        help="Re-measure every source, ignoring the cache.")
    parser.add_argument("--max-auto-shards", type=int, default=200,
                        help="Without --measure, fill cache gaps only up to this many shards "
                             "(default: 200). Deliberately small: this script is meant to run "
                             "in seconds before a launch, and cold manifest reads over a "
                             "network mount manage only a few shards per second, so a larger "
                             "budget turns every invocation into a coffee break. A validation "
                             "split fits comfortably; a training mixture needs --measure.")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip the one-cut text_field probe.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Corpus-level subsampling factor the weights should match.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Language-level upsampling factor the weights should match.")
    parser.add_argument("--weight-tol", type=float, default=1e-6)
    parser.add_argument("--bin-tol-abs", type=float, default=0.05,
                        help="Absolute bin tolerance in seconds (default: 0.05).")
    parser.add_argument("--bin-tol-rel", type=float, default=0.02,
                        help="Relative bin tolerance (default: 0.02).")
    parser.add_argument("--precision", type=int, default=2,
                        help="Decimal places for emitted bin values (default: 2).")
    parser.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("--only", action="append", default=None,
                        choices=sorted({s for s, _ in CHECKS.values()}),
                        help="Restrict the report to these sections (repeatable).")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the full report as JSON.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"error: no such config: {args.config}", file=sys.stderr)
        return 2

    datasets_root = args.datasets_root
    if not datasets_root:
        if not args.offline:
            print(
                "error: LOCAL_DATASETS_DIR is not set and --datasets-root was not given.\n"
                "Either export it:\n"
                "    export LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar\n"
                "or run the checks that do not need the data:\n"
                f"    python3 infra/check_training_config.py --config {args.config} --offline",
                file=sys.stderr,
            )
            return 2
        # A sentinel keeps expansion working so the structural checks can run.
        datasets_root = "/nonexistent-datasets-root"

    try:
        cfg, line_index = load_config(args.config, datasets_root)
    except Exception as exc:
        print(f"error: could not parse {args.config}: {exc}", file=sys.stderr)
        return 2

    data = cfg.get("data")
    if not isinstance(data, dict):
        print(f"error: {args.config} has no 'data' section", file=sys.stderr)
        return 2

    report = Report(args.config, strict=args.strict)
    splits: dict[str, list[Leaf]] = {}
    levels_by_split: dict[str, list[Level]] = {}
    grouped: dict[str, bool] = {}

    for where in ("train_ds", "validation_ds"):
        split = data.get(where)
        if not isinstance(split, dict):
            report.add("W1", where, f"data.{where} is missing", severity=WARN)
            continue
        entries = split.get("input_cfg") or []
        if not entries:
            report.add("W1", where, f"data.{where}.input_cfg is empty", severity=WARN)
            continue
        leaves, levels = flatten(entries, where, line_index, f"data.{where}.input_cfg",
                                 root=datasets_root)
        splits[where] = leaves
        levels_by_split[where] = levels
        grouped[where] = any(level.kind == "group" for level in levels)
        # Attach the owning group's line so T1 can point at it.
        group_lines = {level.yaml_path: level.line for level in levels}
        for leaf in leaves:
            leaf._group_line = group_lines.get(f"{leaf.group_yaml_path}.input_cfg") \
                if leaf.group_yaml_path else None

    if not splits:
        print(f"error: {args.config} declares no usable input_cfg", file=sys.stderr)
        return 2

    all_leaves = [leaf for leaves in splits.values() for leaf in leaves]
    print(
        f"Parsed {args.config}: "
        + ", ".join(
            f"{where} {len(leaves)} sources ({len({l.path for l in leaves})} distinct)"
            for where, leaves in splits.items()
        )
    )

    # ---- tags and structure (no data needed) -----------------------------
    for where, leaves in splits.items():
        check_tags(report, leaves, where)
        check_text_field_consistency(report, leaves, where)

    # ---- measurement ------------------------------------------------------
    cache: dict = {}
    if args.offline:
        for check in ("W3", "H1", "H2", "H3", "B3", "C4", "C5", "D1", "D2", "D3", "T4"):
            report.skip(check, "--offline")
    else:
        if args.cache.exists() and not args.force_measure:
            try:
                cache = json.loads(args.cache.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: ignoring unreadable cache {args.cache}: {exc}")
        seeded = 0
        if not args.force_measure:
            seeded = seed_cache_from_mix_weights(cache, args.seed_cache)
            if seeded:
                print(
                    f"Seeded {seeded} sources' hours from {args.seed_cache} "
                    "(hours only: cut counts and duration histograms are not recorded there)."
                )

        filters = []
        for where in splits:
            split = data.get(where) or {}
            filters.append((float(split.get("min_duration") or 0.0),
                            float(split.get("max_duration") or float("inf"))))
        filters = list(dict.fromkeys(filters))

        todo: list[str] = []
        for path in dict.fromkeys(leaf.path for leaf in all_leaves):
            entry = cache.get(path)
            if args.force_measure or not cache_is_complete(entry):
                todo.append(path)
            elif entry.get("mtime") is not None:
                shards = cmw.shar_manifest_files(Path(path))
                if shards and manifest_mtime(shards) > entry["mtime"] + 1:
                    print(f"  manifests changed since last measurement: {path}")
                    todo.append(path)

        if todo:
            shard_counts = {p: len(cmw.shar_manifest_files(Path(p))) for p in todo}
            n_shards = sum(shard_counts.values())
            if args.measure or args.force_measure:
                measure_sources(todo, cache, args.cache, filters, args.jobs)
            else:
                # Cheapest first, up to the budget, rather than all-or-nothing:
                # otherwise one huge corpus in the train mix suppresses the
                # measurement of a validation split worth half a second.
                budget = args.max_auto_shards
                affordable: list[str] = []
                spent = 0
                for path in sorted(todo, key=lambda p: shard_counts[p]):
                    if spent + shard_counts[path] > budget:
                        continue
                    affordable.append(path)
                    spent += shard_counts[path]
                if affordable:
                    measure_sources(affordable, cache, args.cache, filters, args.jobs)
                remaining = [p for p in todo if p not in set(affordable)]
                if remaining:
                    print(
                        f"\n{len(remaining)} of {len(todo)} unmeasured sources "
                        f"({n_shards - spent} of {n_shards} shards) are above the "
                        f"--max-auto-shards {args.max_auto_shards} budget, so checks needing "
                        "cut counts or duration histograms are skipped for them.\n"
                        "To measure them (resumable -- the cache is written per source):\n"
                        f"    python3 infra/check_training_config.py --config {args.config} "
                        f"--datasets-root {datasets_root} --measure --jobs {args.jobs}\n"
                    )

        check_disk(report, splits)
        for where, leaves in splits.items():
            check_text_fields(
                report, leaves, where, probe=not args.no_probe,
                strict_text_field=bool(data.get("strict_text_field")),
            )

    # ---- weights ----------------------------------------------------------
    template = args.template or guess_template(args.config)
    hint = template_command(args.config, template, args.datasets_root, args.seed_cache,
                            args.alpha, args.beta)
    if "train_ds" in splits:
        check_weights(
            report, splits["train_ds"], levels_by_split["train_ds"], cache,
            args.alpha, args.beta, args.weight_tol, hint,
        )
    if template is not None:
        compare_to_template(report, args.config, template, splits, args.datasets_root)

    # ---- names ------------------------------------------------------------
    check_names(report, splits.get("validation_ds", []))

    # ---- totals and bins --------------------------------------------------
    # The bin *shape* checks need no data, so they run even under --offline.
    bucket_counts = {
        where: check_bins_structure(report, where, data[where], line_index)
        for where in splits
    }
    if not args.offline:
        if "train_ds" in splits:
            check_totals(report, "train_ds", data["train_ds"], splits["train_ds"], cache,
                         line_index, ("H1", "H2"))
        if "validation_ds" in splits:
            check_totals(report, "validation_ds", data["validation_ds"],
                         splits["validation_ds"], cache, line_index, ("H3", "H3"))
        for where, leaves in splits.items():
            num_buckets = bucket_counts.get(where)
            if num_buckets:
                check_bins_values(report, where, data[where], leaves, cache, line_index,
                                  num_buckets, args.bin_tol_abs, args.bin_tol_rel,
                                  args.precision)

    # ---- runtime traps ----------------------------------------------------
    check_runtime(report, cfg, splits, grouped, line_index)

    sections = args.only or sorted({s for s, _ in CHECKS.values()})
    order = ["weights", "names", "stats", "bins", "tags", "disk", "runtime"]
    sections = [s for s in order if s in sections]
    print_report(report, sections)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_to_json(report), indent=2))
        print(f"\nJSON report written to {args.json}")

    return 1 if report.failed(sections) else 0


def compare_to_template(
    report: Report, config_path: Path, template: Path,
    splits: dict[str, list[Leaf]], datasets_root: str | None,
) -> None:
    """Check the emitted config still describes the template's source set.

    The template is the hand-edited source of truth; the emitted config is
    regenerated wholesale from it.  If they disagree on which leaves exist, the
    emitted config predates a template edit and re-emitting will change the mix.
    """
    try:
        template_cfg, _ = load_config(template, datasets_root)
    except Exception as exc:
        report.note(f"Could not read the template {template}: {exc}")
        return

    for where in ("train_ds", "validation_ds"):
        split = (template_cfg.get("data") or {}).get(where)
        if not isinstance(split, dict) or where not in splits:
            continue
        t_leaves, _ = flatten(split.get("input_cfg") or [], where, {},
                             f"data.{where}.input_cfg", root=datasets_root)
        emitted = sorted(leaf.path for leaf in splits[where])
        templated = sorted(leaf.path for leaf in t_leaves)
        if emitted == templated:
            continue
        only_template = sorted(set(templated) - set(emitted))
        only_emitted = sorted(set(emitted) - set(templated))
        report.add(
            "W4", where,
            f"{where} source set differs from the template {template.name} "
            f"({len(templated)} vs {len(emitted)} sources)",
            detail=([f"only in template: {p}" for p in only_template[:5]]
                    + [f"only in config:   {p}" for p in only_emitted[:5]]),
            fix=[f"The template is the source of truth. Re-emit the config from it, or move",
                 "the edit into the template if it was made in the emitted file."],
        )


if __name__ == "__main__":
    sys.exit(main())
