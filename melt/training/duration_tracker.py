"""Track cumulative training audio hours seen, broken down by task and language.

Caveats worth knowing before reading numbers off this metric:

1. This counts hours *fed to the model*, not hours in the corpus. Cuts
   dropped for failed audio load or empty text never reach the counter, so
   this number will **not** reconcile with ``utils/get_dataset_stats.py`` —
   a gap between the two is a data bug worth investigating, not a rounding
   error.
2. ``FallbackDataset`` (``data/audio/lhotse/dataset.py``) replays the last
   good batch on a load failure, so replayed durations are counted twice.
   That is intentional under this metric's name ("hours seen" — the model
   really did see those seconds again), but it is a second reason this
   won't match the manifest.
3. Speech-translation cuts are keyed ``st/{src_lang}-{tgt_lang}`` rather
   than by ``lang``, because duration is a property of the *source* audio.
   ``get_tags_from_cut`` collapses ``lang`` to the target language for ST
   tasks (kept for backward compat), so keying on ``lang`` would file
   source-language audio under the wrong language.
"""

from collections import defaultdict

import torch


_UNKNOWN = "unknown"


def _norm(value: str | None) -> str:
    """Normalise a task/language tag: ``None``/empty/whitespace -> ``"unknown"``."""
    if value is None:
        return _UNKNOWN
    value = value.strip()
    return value or _UNKNOWN


class DurationTracker:
    """Accumulates per-(task, language) audio seconds seen during training.

    Keys are discovered at runtime from whatever tasks/languages appear in
    the data — there is no fixed key universe, so a new language or task in
    a future run is picked up automatically.
    """

    def __init__(self) -> None:
        self._seconds: defaultdict[str, float] = defaultdict(float)

    @staticmethod
    def make_key(
        task: str | None,
        lang: str | None,
        src_lang: str | None,
        tgt_lang: str | None,
    ) -> str:
        """Build the metric key for one cut's task/language tags.

        ASR-style tasks are keyed ``{task}/{lang}``. Speech-translation
        tasks (``st``/``translate``) are keyed ``{task}/{src}-{tgt}``,
        since duration is a property of the source audio and ``lang`` holds
        the *target* language for those tasks (see module docstring).
        """
        norm_task = _norm(task)
        norm_lang = _norm(lang)

        if norm_task in ("st", "translate"):
            src = _norm(src_lang)
            tgt = _norm(tgt_lang)
            if src == _UNKNOWN:
                src = norm_lang
            if tgt == _UNKNOWN:
                tgt = norm_lang
            return f"{norm_task}/{src}-{tgt}"

        return f"{norm_task}/{norm_lang}"

    def update(
        self,
        durations: list[float] | None,
        tasks: list[str] | None,
        langs: list[str] | None,
        src_langs: list[str] | None,
        tgt_langs: list[str] | None,
    ) -> None:
        """Accumulate *durations* (seconds) into the per-key totals.

        Any of ``tasks``/``langs``/``src_langs``/``tgt_langs`` may be
        ``None`` or shorter than ``durations`` — missing entries are
        treated as ``"unknown"`` rather than raising, so a mis-tagged cut
        still contributes its duration to the total. ``durations`` being
        ``None`` is a no-op.
        """
        if not durations:
            return

        for i, duration in enumerate(durations):
            task = tasks[i] if tasks is not None and i < len(tasks) else None
            lang = langs[i] if langs is not None and i < len(langs) else None
            src_lang = src_langs[i] if src_langs is not None and i < len(src_langs) else None
            tgt_lang = tgt_langs[i] if tgt_langs is not None and i < len(tgt_langs) else None
            key = self.make_key(task, lang, src_lang, tgt_lang)
            self._seconds[key] += float(duration)

    def state_dict(self) -> dict[str, float]:
        """Return a plain dict snapshot of the rank-local per-key seconds."""
        return dict(self._seconds)

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Replace the current contents with *state* (does not accumulate)."""
        self._seconds = defaultdict(float, state)

    def reduced_hours(self, prefix: str = "train_hours") -> dict[str, float]:
        """Return per-key cumulative hours, reduced across all ranks.

        Must be called unconditionally on every rank when running under
        distributed training: it performs an ``all_gather_object`` collective,
        gathering each rank's local key -> seconds dict rather than
        all-reducing a fixed set of tensors, because ranks may legitimately
        hold different key sets (e.g. a rank that never saw Italian audio)
        and the key set is not known ahead of time.
        """
        local = self.state_dict()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict[str, float] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
            merged: defaultdict[str, float] = defaultdict(float)
            for rank_state in gathered:
                if not rank_state:
                    continue
                for key, seconds in rank_state.items():
                    merged[key] += seconds
        else:
            merged = defaultdict(float, local)

        hours = {f"{prefix}/{key}": seconds / 3600.0 for key, seconds in merged.items()}
        hours[f"{prefix}/total"] = sum(merged.values()) / 3600.0
        return hours
