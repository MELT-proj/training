"""Per-source, step-aware filtering of ASR cuts by a precomputed WER/CER score.

The score of every cut lives in a compact array next to (not inside) the Shar
manifests: ``<scores_root>/<shar_path relative to datasets_root>/<metric>.u16.npy``,
one uint16 per cut in Shar order (value = round(10000 * score), 65535 = no
score). ``infra/build_asr_scores.py`` writes them from the ASR-WER sidecars.

The join is positional. The indexed Shar reader tags each cut with
``_graph_origin = (global_idx, shar_epoch)``, ``global_idx`` being the cut's row in
its leaf, so ``scores[global_idx]`` is exact even where cut ids repeat (FLEURS).
That tag exists only on the bare leaf, so the filter sits on each ASR leaf right
after ``from_shar`` and before ``.repeat()``/mux: the leaf keeps its mux weight,
survivors are simply drawn more often, and ST leaves never get a node at all.

A band is ``min``/``max`` in score units, or ``pct_min``/``pct_max``: percentiles
of that source's own scored cuts, which keeps the same share of every source
instead of the same absolute band. A band can change at optimizer steps
(``phases``); the step is a shared counter the trainer callback writes and the
dataloader workers read. The switch gates admission into the sampler; cuts
already in its bucket buffer leave over a few hundred steps (a transition
period, not a cut-over).
"""

from __future__ import annotations

import bisect
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
from lhotse.lazy import LazyFilter, get_graph_origin

from .....logging_utils import get_logger


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


def effective_task(leaf_tags: dict | None, outer_tags: dict | None) -> str:
    """Task a leaf's cuts end up with: group tags are applied after the leaf's, so the outer one wins."""
    merged = {**(leaf_tags or {}), **(outer_tags or {})}
    return str(merged.get("task", "") or "").strip().lower()


def _plain(cfg: Any) -> Any:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


_BAND_KEYS = ("min", "max", "pct_min", "pct_max")


def _phases_of(level: dict) -> list[dict] | None:
    """The phases one config level defines, or None if it defines no band."""
    if "phases" in level and level["phases"] is not None:
        phases = [dict(p) for p in level["phases"]]
        if not phases:
            raise ValueError("asr_filter.phases is empty")
        for p in phases:
            p.setdefault("from_step", 0)
        phases.sort(key=lambda p: int(p["from_step"]))
        if int(phases[0]["from_step"]) != 0:
            raise ValueError("asr_filter.phases: the first phase must start at from_step 0")
        return phases
    if any(k in level for k in _BAND_KEYS):
        return [{"from_step": 0, **{k: level[k] for k in _BAND_KEYS if k in level}}]
    return None


def resolve_source_filter(global_cfg: dict, overrides: list[Any]) -> dict | None:
    """Merge the global ``asr_filter`` with the per-group / per-source overrides.

    ``overrides`` runs from the outermost group to the leaf; the nearest wins.
    ``False`` exempts the source (returns None). A level that names a band
    replaces the whole phase list; ``missing`` and ``metric`` replace singly.
    """
    spec = {
        "metric": global_cfg.get("metric", "wer"),
        "missing": global_cfg.get("missing", "drop"),
        "phases": _phases_of(global_cfg),
    }
    for level in overrides:
        level = _plain(level)
        if level is None:
            continue
        if level is False:
            return None
        if level is True:
            continue
        if not isinstance(level, dict):
            raise ValueError(f"asr_filter override must be a mapping or a bool, got {level!r}")
        for k in ("metric", "missing"):
            if k in level:
                spec[k] = level[k]
        phases = _phases_of(level)
        if phases is not None:
            spec["phases"] = phases
    if spec["phases"] is None:
        raise ValueError("asr_filter defines no band: set min/max, pct_min/pct_max or phases")
    if spec["missing"] not in ("drop", "keep"):
        raise ValueError(f"asr_filter.missing must be 'drop' or 'keep', got {spec['missing']!r}")
    if spec["metric"] not in ("wer", "cer"):
        raise ValueError(f"asr_filter.metric must be 'wer' or 'cer', got {spec['metric']!r}")
    return spec


def scores_file(scores_root: str, datasets_root: str, shar_path: str, metric: str) -> Path:
    rel = os.path.relpath(os.path.abspath(shar_path), os.path.abspath(datasets_root))
    if rel.startswith(".."):
        raise ValueError(f"shar_path {shar_path} is not under datasets_root {datasets_root}")
    return Path(scores_root) / rel / f"{metric}.u16.npy"


def _quantile_bound(hist: np.ndarray, pct: float) -> int:
    """Score value (in u16 units) at a percentile of the scored cuts of one source."""
    scored = hist[:NULL_SCORE]
    total = int(scored.sum())
    if total == 0:
        raise ValueError("source has no scored cuts; cannot take a percentile")
    target = max(1, int(np.ceil(pct / 100.0 * total)))
    return int(np.searchsorted(np.cumsum(scored), target))


def compile_phases(phases: list[dict], scores: np.ndarray) -> list[tuple[int, int, int]]:
    """``[(from_step, lo, hi)]`` in u16 units, inclusive on both ends."""
    hist = None
    out = []
    for p in phases:
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


class PhasedScoreFilter(LazyFilter):
    """``LazyFilter`` that keeps cuts whose score is in the band of the current phase.

    Differences from the stock node: the predicate reads the shared step, and
    ``__getitem__`` (the sampler-checkpoint restore path) does not re-check it, so
    a cut buffered under an earlier band restores instead of raising.
    """

    def __init__(
        self,
        source,
        scores_path: str,
        phases: list[tuple[int, int, int]],
        missing: str,
        step_fn=get_step,
    ) -> None:
        self.scores_path = str(scores_path)
        self.phases = phases
        self.missing = missing
        self._starts = [p[0] for p in phases]
        self._step_fn = step_fn
        self._scores = None
        super().__init__(source, self._keep)

    def _array(self) -> np.ndarray:
        if self._scores is None:
            self._scores = np.load(self.scores_path, mmap_mode="r")
        return self._scores

    def band(self, step: int | None = None) -> tuple[int, int]:
        step = self._step_fn() if step is None else step
        _, lo, hi = self.phases[bisect.bisect_right(self._starts, step) - 1]
        return lo, hi

    def _keep(self, cut) -> bool:
        token = get_graph_origin(cut)
        if token is None:
            raise RuntimeError(
                f"asr_filter needs indexed Shar sources: cut {cut.id} carries no row index"
            )
        value = int(self._array()[token[0]])
        if value == NULL_SCORE:
            return self.missing == "keep"
        lo, hi = self.band()
        return lo <= value <= hi

    def __getitem__(self, token):
        from lhotse.lazy import attach_graph_origin, normalize_graph_token

        token = normalize_graph_token(token)
        return attach_graph_origin(self.source[token], token)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_scores"] = None
        state["predicate"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.predicate = self._keep


def apply_asr_filter(
    cuts,
    *,
    shar_path: str,
    n_cuts: int,
    spec: dict,
    scores_root: str,
    datasets_root: str,
    where: str,
):
    """Wrap one ASR leaf's CutSet in a :class:`PhasedScoreFilter`; fail loudly on any mismatch."""
    from lhotse import CutSet

    if getattr(cuts.data, "is_indexed", False) is not True:
        raise ValueError(f"{where}: asr_filter needs an indexed Shar source ({shar_path} has no .idx)")
    path = scores_file(scores_root, datasets_root, shar_path, spec["metric"])
    if not path.exists():
        raise FileNotFoundError(
            f"{where}: no {spec['metric']} score file at {path}. Build it with "
            "infra/build_asr_scores.py, or exempt this source with `asr_filter: false`."
        )
    scores = np.load(path, mmap_mode="r")
    if scores.dtype != np.uint16 or scores.ndim != 1:
        raise ValueError(f"{where}: {path} is not a 1-D uint16 array")
    if scores.shape[0] != n_cuts:
        raise ValueError(
            f"{where}: {path} has {scores.shape[0]} rows but {shar_path} has {n_cuts} cuts; "
            "the manifests changed after the scores were built"
        )
    phases = compile_phases(spec["phases"], np.asarray(scores))
    values = np.asarray(scores)
    scored = values != NULL_SCORE
    for start, lo, hi in phases:
        kept = int(np.count_nonzero(scored & (values >= lo) & (values <= hi)))
        if spec["missing"] == "keep":
            kept += int(np.count_nonzero(~scored))
        logger.info(
            f"  {where}: asr_filter from step {start}: [{lo / SCORE_SCALE:.4f}, "
            f"{hi / SCORE_SCALE:.4f}] keeps {kept}/{n_cuts} cuts ({100.0 * kept / max(n_cuts, 1):.1f}%)"
        )
        if kept == 0:
            raise ValueError(f"{where}: the band from step {start} keeps no cut of {shar_path}")
    return CutSet(PhasedScoreFilter(cuts.data, str(path), phases, spec["missing"]))


def make_step_callback():
    """Trainer callback that publishes ``global_step`` to the dataloader workers."""
    from transformers import TrainerCallback

    class AsrFilterStepCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            set_step(state.global_step)  # restored step on resume

        def on_step_end(self, args, state, control, **kwargs):
            set_step(state.global_step)

    return AsrFilterStepCallback()
