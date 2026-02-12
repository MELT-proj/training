"""Metrics registry for evaluation.

Provides a simple registry pattern so that new metrics can be added by
implementing the :class:`Metric` protocol and calling :func:`register_metric`.

Each metric receives two parallel lists — hypotheses and references — and
returns a ``dict[str, float]`` of named scalar values (e.g. ``{"wer": 0.12}``).

Example — registering a new metric::

    from src.evaluation.metrics import Metric, register_metric

    class WordErrorRate(Metric):
        name = "wer"

        def compute(self, hypotheses: list[str], references: list[str]) -> dict[str, float]:
            ...
            return {"wer": wer_value}

    register_metric(WordErrorRate)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Metric(ABC):
    """Base class for all evaluation metrics.

    Subclasses must set a unique ``name`` class attribute and implement
    :meth:`compute`.
    """

    name: str  # unique identifier used in YAML configs

    @abstractmethod
    def compute(
        self,
        hypotheses: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute the metric over a corpus.

        Args:
            hypotheses: Model-generated predictions (one per sample).
            references: Ground-truth targets (one per sample).

        Returns:
            Dictionary mapping metric sub-names to scalar values.
            Typically ``{self.name: value}``, but a single metric may
            return multiple related values
            (e.g. ``{"wer": 0.12, "cer": 0.04}``).
        """
        ...


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, type[Metric]] = {}
"""Maps metric names to their implementing classes."""


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Register a :class:`Metric` subclass in the global registry.

    Can be used as a decorator::

        @register_metric
        class MyMetric(Metric):
            name = "my_metric"
            ...

    Args:
        cls: A concrete subclass of :class:`Metric`.

    Returns:
        The same class, unmodified (so it doubles as a decorator).

    Raises:
        ValueError: If a metric with the same name is already registered.
    """
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Metric class {cls.__name__} must define a non-empty 'name' attribute.")
    if cls.name in METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{cls.name}' is already registered "
            f"(existing: {METRIC_REGISTRY[cls.name].__name__}, new: {cls.__name__})."
        )
    METRIC_REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> Metric:
    """Instantiate a registered metric by name.

    Args:
        name: The metric name (as declared in ``Metric.name``).

    Returns:
        An instance of the corresponding :class:`Metric` subclass.

    Raises:
        KeyError: If no metric with that name has been registered.
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")
    return METRIC_REGISTRY[name]()
