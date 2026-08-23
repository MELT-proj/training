"""The steps-per-epoch estimate has to account for `quadratic_duration`.

`batch_duration` is handed to lhotse as `max_duration`, which budgets *effective*
seconds: each cut is charged `d + d^2/quadratic_duration`. A batch therefore
fills up before it holds `batch_duration` seconds of audio, and an estimate that
divides total duration by `batch_duration` under-counts the batches in an epoch.

That is not cosmetic. The count feeds `max_steps` when it is derived from
`num_train_epochs`, and from there the LR schedule -- so a "1 epoch" run covered
barely half the data while reporting `epoch: 1.0` (MN5 job 44947472, where the
run's own train_hours counter measured 0.571 h/step against the 1.067 h/step the
uncorrected arithmetic implied).
"""

import math

import pytest
from omegaconf import OmegaConf

from melt.training.data.audio.lhotse.dataloader import (
    _effective_duration_inflation,
    estimate_steps_per_epoch,
)


# The bins shipped in projects/ablation-campaign/ABL-MA-125-asr.yaml.
ABL_BINS = [
    2.6, 4.0, 5.0, 6.0, 7.02, 8.18, 9.54, 10.56, 11.38, 12.16, 12.94, 13.68,
    14.41, 15.14, 15.86, 16.59, 17.32, 18.05, 18.81, 19.58, 21.18, 23.78,
    26.32, 28.9, 33.18, 35.62, 36.98, 38.08, 39.08,
]


def _cfg(**overrides):
    base = {
        "total_hours": 625.0,
        "total_cuts": 1_000_000,
        "batch_size": None,
        "batch_duration": 120,
        "quadratic_duration": 30,
        "bucket_duration_bins": list(ABL_BINS),
        "min_duration": 0.5,
        "max_duration": 90.0,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_no_quadratic_duration_means_no_inflation():
    """Without the penalty a batch really does hold batch_duration of audio."""
    assert _effective_duration_inflation(_cfg(quadratic_duration=None)) == 1.0


def test_the_penalty_inflates_the_batch_count():
    """ABL-MA-125-asr: the uncorrected estimate is optimistic by a wide margin."""
    inflation = _effective_duration_inflation(_cfg())
    assert inflation > 1.5, inflation
    # Measured 1.87 on job 44947472; midpoints under-read, so allow the gap but
    # pin that we are in the right neighbourhood rather than off by 2x.
    assert 1.55 < inflation < 1.75, inflation


def test_a_weaker_penalty_inflates_less():
    """Raising quadratic_duration lets more audio into a batch, so fewer batches.

    This is the direction the campaign moved in (30 -> 35): a *larger*
    quadratic_duration is a *smaller* penalty.
    """
    strong = _effective_duration_inflation(_cfg(quadratic_duration=25))
    current = _effective_duration_inflation(_cfg(quadratic_duration=30))
    weak = _effective_duration_inflation(_cfg(quadratic_duration=35))
    assert strong > current > weak > 1.0, (strong, current, weak)


def test_inflation_is_reciprocal_to_the_fill_fraction():
    """1/inflation is the share of batch_duration that is real audio."""
    q = 35.0
    inflation = _effective_duration_inflation(_cfg(quadratic_duration=q))
    # Reconstruct the weighted mean the helper derived and check the algebra.
    edges = [0.5] + ABL_BINS + [90.0]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    expected = 1.0 + (sum(mids) / len(mids)) / q
    assert inflation == pytest.approx(expected)


def test_missing_bins_warn_rather_than_silently_returning_one(caplog):
    """Silently returning 1.0 is exactly the bug; an uncorrectable config must say so."""
    with caplog.at_level("WARNING"):
        assert _effective_duration_inflation(_cfg(bucket_duration_bins=None)) == 1.0
    assert any("cannot be corrected" in r.getMessage() for r in caplog.records)


def test_the_correction_reaches_the_step_count():
    """The whole point: steps_per_epoch must grow, not just the helper's return."""
    cfg = _cfg()
    corrected, *_ = estimate_steps_per_epoch(cfg, gradient_accumulation_steps=4, world_size=8)
    uncorrected, *_ = estimate_steps_per_epoch(
        _cfg(quadratic_duration=None), gradient_accumulation_steps=4, world_size=8
    )
    assert uncorrected == 586, uncorrected  # the historical, optimistic number
    assert corrected > uncorrected
    inflation = _effective_duration_inflation(cfg)
    assert corrected == pytest.approx(uncorrected * inflation, rel=0.02)


def test_the_batch_size_path_is_untouched():
    """quadratic_duration only applies to duration-based batching."""
    cfg = _cfg(batch_size=16, batch_duration=None, total_cuts=32_000)
    steps, *_ = estimate_steps_per_epoch(cfg, gradient_accumulation_steps=1, world_size=1)
    assert steps == math.ceil(32_000 / 16)
