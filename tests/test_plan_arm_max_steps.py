"""Tests for the max_steps policy field in projects/ablation-campaign/plan_arm.py.

derive_steps (like the trainer's estimate_steps_per_epoch) assumes every batch is full.
The 2,100 h crossing's batches carry ~80% of their nominal audio (1,500-step smoke,
board 2026-10-08: 239.7 s of 300 s), so its rows state their length outright, 1.25x the
estimate, instead of inheriting it from total_hours. Everything that is a function of
the run length -- eval/save cadence, the WSD decay window, the run name -- has to follow
the stated value, and the trainer has to be told it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import campaign  # noqa: E402
import plan_arm  # noqa: E402
from plan_arm import ArmAxes  # noqa: E402


CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-2100-asr.yaml")


def _axes(**overrides) -> ArmAxes:
    """One node, the crossing's batch: 10,500 h / (75 s x 4 ranks) = 126,000 derived steps."""
    defaults = {
        "config": CONFIG,
        "stage": "MA",
        "world_size": 4,
        "batch_duration": "75",
        "grad_accum_steps": "1",
    }
    defaults.update(overrides)
    return ArmAxes(**defaults)


def _value(plan, flag):
    return plan.overrides[plan.overrides.index(flag) + 1]


def test_unset_derives_the_steps_and_emits_no_override():
    plan = plan_arm.plan(_axes())

    assert plan.steps == 126_000
    assert "--trainer.max_steps" not in plan.overrides


def test_unset_leaves_the_exp_name_without_a_step_tag():
    """Byte-for-byte regression: this field must not rename any arm that does not set it."""
    assert not any(tok.startswith("ms") and tok[2:].isdigit() for tok in plan_arm.plan(_axes()).exp_name.split("-"))


def test_stated_steps_replace_the_derived_ones():
    plan = plan_arm.plan(_axes(max_steps=157_500))

    assert plan.steps == 157_500


def test_trainer_is_told_the_length():
    """Without the flag the trainer would stop at its own estimate, 126,000."""
    plan = plan_arm.plan(_axes(max_steps=157_500))

    assert _value(plan, "--trainer.max_steps") == "157500"


def test_eval_and_save_cadence_follow_the_stated_steps():
    plan = plan_arm.plan(_axes(max_steps=157_500))

    assert plan.eval_steps == round(157_500 / 11) == 14_318
    assert plan.save_steps == plan.eval_steps


def test_wsd_decay_window_follows_the_stated_steps():
    plan = plan_arm.plan(_axes(max_steps=157_500, lr_scheduler="warmup_stable_decay"))

    kwargs = json.loads(_value(plan, "--trainer.lr_scheduler_kwargs"))
    assert kwargs["num_decay_steps"] == round(plan_arm.WSD_DECAY_FRACTION * 157_500) == 31_500


def test_tagged_into_exp_name():
    """Unlike eval_rounds, it changes what the run trains on, so the name must say so."""
    baseline = plan_arm.plan(_axes()).exp_name
    stated = plan_arm.plan(_axes(max_steps=157_500)).exp_name

    assert "ms157500" in stated.split("-")
    assert stated.replace("-ms157500", "") == baseline


def test_refused_together_with_epochs():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(max_steps=157_500, epochs="2"))


@pytest.mark.parametrize("bad", [0, -5])
def test_refused_when_not_positive(bad):
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(max_steps=bad))


def test_campaign_accepts_it_as_a_grid_field():
    assert "max_steps" in campaign.POLICY_FIELDS
    assert "max_steps" in campaign.KNOWN_FIELDS


def test_campaign_passes_the_row_value_through():
    defaults, arms = campaign.load_grid()
    row = next(a for a in arms if a["id"] == "MA-2100-crossing-w2vb-mlp")

    axes, _ = campaign.resolve(row, defaults, arms)

    assert axes.max_steps == row["max_steps"]


def test_every_crossing_row_states_1p25x_the_estimate():
    """The recipe this field exists for: the estimate assumes full batches, the smoke measured ~80%.

    The rows are checked against their own derived steps, so changing a row's nodes
    without restating its steps (or the reverse) fails here. Not pinned to a row count.
    """
    defaults, arms = campaign.load_grid()
    crossing = [a for a in arms if a["id"].startswith("MA-2100-crossing-")]

    assert len(crossing) >= 16
    for row in crossing:
        axes, _ = campaign.resolve(row, defaults, arms)
        assert axes.max_steps is not None, row["id"]
        derived = plan_arm.plan(dataclasses.replace(axes, max_steps=None)).steps
        assert axes.max_steps * 4 == derived * 5, row["id"]
