"""Per-source, step-aware filtering of training cuts on any per-cut value.

``data.train_ds.cut_filter`` holds named rules. A rule applies to the leaves
whose merged tags match its ``match`` (e.g. ``{task: asr}``), and keeps a cut
when its value satisfies the rule's condition; a cut must satisfy every rule
that applies to its leaf. The value comes from one of two places:

- ``field``: a dotted path into the cut's manifest entry (``custom.st_qe``,
  ``supervisions.0.custom.score``, ``duration``). Read from the cut itself, so
  the cut is deserialised before it can be judged.
- ``array``: a compact sidecar next to (not inside) the Shar manifests,
  ``<scores_root>/<shar_path relative to datasets_root>/<array>.u16.npy``, one
  uint16 per cut in Shar order (value = round(10000 * score), 65535 = no
  value). ``infra/build_asr_scores.py`` writes ``wer``/``cer`` arrays from the
  ASR-WER sidecars. The join is positional: the indexed Shar reader tags each
  cut with ``_graph_origin = (global_idx, shar_epoch)``, ``global_idx`` being
  its row in its leaf, so ``scores[global_idx]`` is exact even where cut ids
  repeat (FLEURS), and the row is judged *before* the cut is read.

That tag exists only on the bare leaf, so the filter sits on each leaf right
after ``from_shar`` and before ``.repeat()``/mux: the leaf keeps its mux weight
and survivors are simply drawn more often.

A condition is ``min``/``max`` (inclusive), ``in`` (a list of allowed values,
``field`` rules only), or, for ``array`` rules, ``pct_min``/``pct_max``:
percentiles of that source's own values, which keeps the same share of every
source instead of the same absolute band. A condition can change at optimizer
steps (``phases``); the step is a shared counter the trainer callback writes and
the dataloader workers read. The switch gates admission into the sampler; cuts
already in its bucket buffer leave over a few hundred steps (a transition
period, not a cut-over).
"""

from __future__ import annotations

import bisect
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
from lhotse.lazy import LazyFilter, get_graph_origin

from .....logging_utils import get_logger
from .helpers import _get_nested_value


logger = get_logger(__name__)

SCORE_SCALE = 10_000
NULL_SCORE = 65_535

# Written by the trainer's callback, read by dataloader workers. Created at import
# time so forked workers inherit the same shared memory.
_STEP = multiprocessing.Value("q", 0, lock=False)


def set_step(step: int) -> None:
    _STEP.value = int(step)


def get_step() -> int:
    return int(_STEP.value)


def merged_tags(leaf_tags: dict | None, outer_tags: dict | None) -> dict:
    """Tags a leaf's cuts end up with: group tags are applied after the leaf's, so the outer one wins."""
    return {**(leaf_tags or {}), **(outer_tags or {})}


def rule_matches(match: dict | None, tags: dict) -> bool:
    """Every key of ``match`` equals the tag (case-insensitive), or is one of a list of values."""
    for key, want in (match or {}).items():
        allowed = want if isinstance(want, (list, tuple)) else [want]
        have = str(tags.get(key, "") or "").strip().lower()
        if have not in {str(a).strip().lower() for a in allowed}:
            return False
    return True


def _plain(cfg: Any) -> Any:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


_COND_KEYS = ("min", "max", "pct_min", "pct_max", "in")


def _phases_of(level: dict, where: str) -> list[dict] | None:
    """The phases one config level defines, or None if it defines no condition."""
    if "phases" in level and level["phases"] is not None:
        phases = [dict(p) for p in level["phases"]]
        if not phases:
            raise ValueError(f"{where}.phases is empty")
        for p in phases:
            p.setdefault("from_step", 0)
        phases.sort(key=lambda p: int(p["from_step"]))
        if int(phases[0]["from_step"]) != 0:
            raise ValueError(f"{where}.phases: the first phase must start at from_step 0")
        return phases
    if any(k in level for k in _COND_KEYS):
        return [{"from_step": 0, **{k: level[k] for k in _COND_KEYS if k in level}}]
    return None


def validate_config(cfg: dict) -> dict:
    """Check the global ``cut_filter`` block; returns ``{name: rule}`` with plain values."""
    rules = cfg.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("cut_filter.rules must be a non-empty mapping of rule name -> rule")
    out = {}
    for name, rule in rules.items():
        rule = _plain(rule)
        where = f"cut_filter.rules.{name}"
        if not isinstance(rule, dict):
            raise ValueError(f"{where} must be a mapping")
        if ("array" in rule) == ("field" in rule):
            raise ValueError(f"{where}: set exactly one of `array` (sidecar) or `field` (manifest path)")
        if rule.get("missing", "drop") not in ("drop", "keep"):
            raise ValueError(f"{where}.missing must be 'drop' or 'keep', got {rule['missing']!r}")
        if "array" in rule and not cfg.get("scores_root"):
            raise ValueError(f"{where} reads an array: cut_filter.scores_root must be set")
        out[name] = rule
    return out


def resolve_leaf_rules(rules: dict, overrides: list[Any], tags: dict) -> dict[str, dict]:
    """The rules that apply to one leaf, merged with the per-group / per-source overrides.

    ``overrides`` are the ``cut_filter`` values from the outermost group to the
    leaf; the nearest wins. A level is ``False`` (exempt from every rule), or a
    mapping of rule name -> ``False`` (exempt from that rule) / ``True`` / a
    mapping that replaces the rule's ``missing`` or its whole condition (any
    level that names a condition replaces the phase list). A rule whose
    ``match`` does not hold for ``tags`` never applies, whatever the overrides.
    """
    specs: dict[str, dict | None] = {}
    for name, rule in rules.items():
        if rule_matches(rule.get("match"), tags):
            specs[name] = {
                "name": name,
                "array": rule.get("array"),
                "field": rule.get("field"),
                "missing": rule.get("missing", "drop"),
                "phases": _phases_of(rule, f"cut_filter.rules.{name}"),
            }
    active = dict.fromkeys(specs, True)
    for level in overrides:
        level = _plain(level)
        if level is None or level is True:
            continue
        if level is False:
            active = dict.fromkeys(specs, False)
            continue
        if not isinstance(level, dict):
            raise ValueError(f"cut_filter override must be a mapping or a bool, got {level!r}")
        for name, value in level.items():
            if name not in rules:
                raise ValueError(f"cut_filter override names unknown rule {name!r} (rules: {sorted(rules)})")
            if name not in specs or value is None:
                continue  # the rule does not apply to this leaf's task/tags
            if value is False:
                active[name] = False
                continue
            active[name] = True
            if value is True:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"cut_filter override of {name!r} must be a mapping or a bool, got {value!r}")
            if "missing" in value:
                if value["missing"] not in ("drop", "keep"):
                    raise ValueError(f"cut_filter.{name}.missing must be 'drop' or 'keep'")
                specs[name]["missing"] = value["missing"]
            phases = _phases_of(value, f"cut_filter.{name}")
            if phases is not None:
                specs[name]["phases"] = phases
    out = {}
    for name, spec in specs.items():
        if not active[name]:
            continue
        if spec["phases"] is None:
            raise ValueError(f"cut_filter rule {name!r} defines no condition: set min/max, pct_min/pct_max, in, or phases")
        out[name] = spec
    return out


def scores_file(scores_root: str, datasets_root: str, shar_path: str, array: str) -> Path:
    rel = os.path.relpath(os.path.abspath(shar_path), os.path.abspath(datasets_root))
    if rel.startswith(".."):
        raise ValueError(f"shar_path {shar_path} is not under datasets_root {datasets_root}")
    return Path(scores_root) / rel / f"{array}.u16.npy"


def _quantile_bound(hist: np.ndarray, pct: float) -> int:
    """Value (in u16 units) at a percentile of the scored cuts of one source."""
    scored = hist[:NULL_SCORE]
    total = int(scored.sum())
    if total == 0:
        raise ValueError("source has no scored cuts; cannot take a percentile")
    target = max(1, int(np.ceil(pct / 100.0 * total)))
    return int(np.searchsorted(np.cumsum(scored), target))


def compile_phases(phases: list[dict], scores: np.ndarray) -> list[tuple[int, int, int]]:
    """Array rule: ``[(from_step, lo, hi)]`` in u16 units, inclusive on both ends."""
    hist = None
    out = []
    for p in phases:
        if "in" in p:
            raise ValueError("`in` applies to `field` rules; an array rule takes min/max or pct_min/pct_max")
        if "pct_min" in p or "pct_max" in p:
            if "min" in p or "max" in p:
                raise ValueError("a phase uses min/max or pct_min/pct_max, not both")
            if hist is None:
                hist = np.bincount(scores, minlength=NULL_SCORE + 1)
            lo = _quantile_bound(hist, float(p["pct_min"])) if float(p.get("pct_min", 0)) > 0 else 0
            hi = _quantile_bound(hist, float(p["pct_max"])) if "pct_max" in p else NULL_SCORE - 1
        else:
            lo = int(round(float(p.get("min", 0.0)) * SCORE_SCALE))
            hi = int(round(float(p["max"]) * SCORE_SCALE)) if p.get("max") is not None else NULL_SCORE - 1
        out.append((int(p["from_step"]), lo, hi))
    return out


def compile_field_phases(phases: list[dict]) -> list[tuple[int, float | None, float | None, frozenset | None]]:
    """Field rule: ``[(from_step, lo, hi, allowed)]``; None means unbounded / any value."""
    out = []
    for p in phases:
        if "pct_min" in p or "pct_max" in p:
            raise ValueError(
                "pct_min/pct_max need the source's whole distribution, which a `field` rule only "
                "sees cut by cut; build a sidecar array and use an `array` rule instead"
            )
        lo = float(p["min"]) if p.get("min") is not None else None
        hi = float(p["max"]) if p.get("max") is not None else None
        allowed = frozenset(p["in"]) if p.get("in") is not None else None
        if allowed is not None and (lo is not None or hi is not None):
            raise ValueError("a phase uses min/max or in, not both")
        out.append((int(p["from_step"]), lo, hi, allowed))
    return out


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class _Phased:
    """One rule's phase list plus the once-per-worker phase-entry log."""

    def __init__(self, name: str, phases: list[tuple], missing: str) -> None:
        self.name = name
        self.phases = phases
        self.missing = missing
        self._starts = [p[0] for p in phases]
        self._phase = None

    def current(self, step: int) -> tuple:
        phase = bisect.bisect_right(self._starts, step) - 1
        if phase != self._phase:
            # Log once per worker when it first admits under a new phase (the smoke test reads this).
            self._phase = phase
            logger.info(
                f"cut_filter {self.name}: phase {phase} from step {self.phases[phase][0]} -> "
                f"{self.describe(self.phases[phase])}, first admission at step {step}"
            )
        return self.phases[phase]


class ArrayRule(_Phased):
    def __init__(self, name: str, scores_path: str, phases: list[tuple], missing: str) -> None:
        super().__init__(name, phases, missing)
        self.scores_path = str(scores_path)
        self._scores = None

    @staticmethod
    def describe(phase: tuple) -> str:
        return f"[{phase[1] / SCORE_SCALE:.4f}, {phase[2] / SCORE_SCALE:.4f}]"

    def keep(self, idx: int, step: int) -> bool:
        if self._scores is None:
            self._scores = np.load(self.scores_path, mmap_mode="r")
        value = int(self._scores[idx])
        if value == NULL_SCORE:
            return self.missing == "keep"
        _, lo, hi = self.current(step)
        return lo <= value <= hi

    def __getstate__(self):
        return {**self.__dict__, "_scores": None}


class FieldRule(_Phased):
    def __init__(self, name: str, field: str, phases: list[tuple], missing: str) -> None:
        super().__init__(name, phases, missing)
        self.field = field

    @staticmethod
    def describe(phase: tuple) -> str:
        _, lo, hi, allowed = phase
        return f"in {sorted(allowed, key=str)}" if allowed is not None else f"[{lo}, {hi}]"

    def keep(self, cut, step: int) -> bool:
        value = _get_nested_value(cut, self.field)
        if _is_missing(value):
            return self.missing == "keep"
        _, lo, hi, allowed = self.current(step)
        if allowed is not None:
            return value in allowed
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"cut_filter {self.name}: cut {cut.id} has non-numeric {self.field}={value!r} "
                "under a min/max condition"
            ) from None
        if math.isnan(value):
            return self.missing == "keep"
        return (lo is None or value >= lo) and (hi is None or value <= hi)


class CutFilter(LazyFilter):
    """``LazyFilter`` that keeps cuts satisfying every rule of their leaf under the current phase.

    Differences from the stock node: the predicate reads the shared step; array
    rules judge the row index before the cut is read; and ``__getitem__`` (the
    sampler-checkpoint restore path) does not re-check, so a cut buffered under
    an earlier condition restores instead of raising.
    """

    def __init__(
        self,
        source,
        array_rules: list[ArrayRule],
        field_rules: list[FieldRule],
        step_fn=get_step,
    ) -> None:
        self.array_rules = array_rules
        self.field_rules = field_rules
        self._step_fn = step_fn
        super().__init__(source, self._keep)

    def _keep_index(self, idx: int, step: int) -> bool:
        return all(r.keep(idx, step) for r in self.array_rules)

    def _keep_cut(self, cut, step: int) -> bool:
        return all(r.keep(cut, step) for r in self.field_rules)

    def _keep(self, cut) -> bool:
        step = self._step_fn()
        if self.array_rules:
            token = get_graph_origin(cut)
            if token is None:
                raise RuntimeError(
                    f"cut_filter array rules need indexed Shar sources: cut {cut.id} carries no row index"
                )
            if not self._keep_index(token[0], step):
                return False
        return self._keep_cut(cut, step)

    def __iter__(self):
        """With array rules, decide on the row index *before* reading the cut.

        Reading and deserialising a cut only to drop it is what the stock filter does, and on
        GPFS a source that keeps 7% costs ~13 random reads per cut it yields. Same traversal,
        position bookkeeping and epoch handling as ``LazyIndexedSharIterator.__iter__``.
        """
        if not self.array_rules:
            for cut in self.source:
                if self._keep_cut(cut, self._step_fn()):
                    yield cut
            return
        source = self.source
        for idx in source._iter_state.iterate(source._total_len):
            step = self._step_fn()
            if self._keep_index(idx, step):
                cut = source[idx]
                if self._keep_cut(cut, step):
                    yield cut
        source.epoch += 1

    def __getitem__(self, token):
        from lhotse.lazy import attach_graph_origin, normalize_graph_token

        token = normalize_graph_token(token)
        return attach_graph_origin(self.source[token], token)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["predicate"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.predicate = self._keep


def apply_cut_filter(
    cuts,
    *,
    shar_path: str,
    n_cuts: int,
    specs: dict[str, dict],
    scores_root: str | None,
    datasets_root: str | None,
    where: str,
):
    """Wrap one leaf's CutSet in a :class:`CutFilter`; fail loudly on any mismatch."""
    from lhotse import CutSet

    array_rules, field_rules = [], []
    for name, spec in specs.items():
        if spec["field"] is not None:
            phases = compile_field_phases(spec["phases"])
            for phase in phases:
                logger.info(
                    f"  {where}: cut_filter {name} from step {phase[0]}: {spec['field']} "
                    f"{FieldRule.describe(phase)} (read from the manifest; share kept unknown until read)"
                )
            field_rules.append(FieldRule(name, spec["field"], phases, spec["missing"]))
            continue
        if getattr(cuts.data, "is_indexed", False) is not True:
            raise ValueError(f"{where}: cut_filter rule {name!r} reads an array and needs an indexed Shar source ({shar_path} has no .idx)")
        if not hasattr(cuts.data, "_iter_state"):
            raise ValueError(f"{where}: cut_filter must wrap the bare indexed Shar reader")
        path = scores_file(scores_root, datasets_root, shar_path, spec["array"])
        if not path.exists():
            raise FileNotFoundError(
                f"{where}: no {spec['array']} array at {path} for cut_filter rule {name!r}. Build it "
                f"(infra/build_asr_scores.py for wer/cer), or exempt this source with `cut_filter: {{{name}: false}}`."
            )
        scores = np.load(path, mmap_mode="r")
        if scores.dtype != np.uint16 or scores.ndim != 1:
            raise ValueError(f"{where}: {path} is not a 1-D uint16 array")
        if scores.shape[0] != n_cuts:
            raise ValueError(
                f"{where}: {path} has {scores.shape[0]} rows but {shar_path} has {n_cuts} cuts; "
                "the manifests changed after the array was built"
            )
        values = np.asarray(scores)
        phases = compile_phases(spec["phases"], values)
        scored = values != NULL_SCORE
        for start, lo, hi in phases:
            kept = int(np.count_nonzero(scored & (values >= lo) & (values <= hi)))
            if spec["missing"] == "keep":
                kept += int(np.count_nonzero(~scored))
            logger.info(
                f"  {where}: cut_filter {name} from step {start}: {spec['array']} [{lo / SCORE_SCALE:.4f}, "
                f"{hi / SCORE_SCALE:.4f}] keeps {kept}/{n_cuts} cuts ({100.0 * kept / max(n_cuts, 1):.1f}%)"
            )
            if kept == 0:
                raise ValueError(f"{where}: cut_filter {name} from step {start} keeps no cut of {shar_path}")
        array_rules.append(ArrayRule(name, str(path), phases, spec["missing"]))
    return CutSet(CutFilter(cuts.data, array_rules, field_rules))


def make_step_callback():
    """Trainer callback that publishes ``global_step`` to the dataloader workers."""
    from transformers import TrainerCallback

    class CutFilterStepCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            set_step(state.global_step)  # restored step on resume

        def on_step_end(self, args, state, control, **kwargs):
            set_step(state.global_step)

    return CutFilterStepCallback()
