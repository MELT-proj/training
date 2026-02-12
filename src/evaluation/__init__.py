"""Evaluation package for MELT models.

This package provides a modular evaluation framework with:
- A metrics registry for extensible metric computation
- Dataset abstractions supporting Lhotse CutSets and (future) HF datasets
- Model backend abstractions supporting MELT generate and (future) HF pipelines
- A main eval script that ties everything together

Usage:
    python -m src.evaluation.eval --config config/eval/asr.yaml
"""

from .metrics import METRIC_REGISTRY, Metric, register_metric

__all__ = [
    "METRIC_REGISTRY",
    "Metric",
    "register_metric",
]
