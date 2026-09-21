"""Tests for the lr_scheduler / warmup_ratio axes in plan_arm.py.

Step 0b adopted warmup_stable_decay as the MA default, but it was never a
campaign axis: its `num_decay_steps` is an absolute step count that differs
per arm, so it was hand-passed on every submission. The five-language
screen's first twelve arms therefore inherited `cosine` from
ABL-MA-700-asr.yaml, and its `warmup_steps: 20` on top -- a fixed count that
means three different fractions of training across the screen's own
1200/600/300 s batch levels. Both are derived here instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import campaign  # noqa: E402
import plan_arm  # noqa: E402
from plan_arm import ArmAxes  # noqa: E402

CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-700-asr.yaml")


def _axes(**overrides) -> ArmAxes:
    defaults = dict(config=CONFIG, stage="MA", world_size=8)
    defaults.update(overrides)
    return ArmAxes(**defaults)


def _value(plan, flag):
    return plan.overrides[plan.overrides.index(flag) + 1]


def test_unset_reproduces_the_pre_axis_command():
    plan = plan_arm.plan(_axes())

    assert "--trainer.lr_scheduler_type" not in plan.overrides
    assert "--trainer.lr_scheduler_kwargs" not in plan.overrides
    assert "--trainer.warmup_ratio" not in plan.overrides
    assert "wsd" not in plan.exp_name.split("-")


def test_wsd_derives_its_decay_tail_from_this_arms_own_steps():
    """The reason it was hand-passed, and the reason it got forgotten."""
    plan = plan_arm.plan(_axes(lr_scheduler="warmup_stable_decay"))
    kwargs = json.loads(_value(plan, "--trainer.lr_scheduler_kwargs"))

    assert kwargs["num_decay_steps"] == round(0.2 * plan.steps)
    assert kwargs["min_lr_ratio"] == 0.1
    assert kwargs["decay_type"] == "cosine"


def test_decay_tail_tracks_the_effective_batch():
    """1200 s and 300 s are 10,500 and 42,000 steps; 20% of each differs."""
    big = plan_arm.plan(_axes(lr_scheduler="warmup_stable_decay", grad_accum_steps="1"))
    small = plan_arm.plan(
        _axes(
            lr_scheduler="warmup_stable_decay",
            batch_duration="75",
            grad_accum_steps="1",
            world_size=4,
        )
    )

    assert big.steps == 10_500 and small.steps == 42_000
    assert json.loads(_value(big, "--trainer.lr_scheduler_kwargs"))["num_decay_steps"] == 2100
    assert json.loads(_value(small, "--trainer.lr_scheduler_kwargs"))["num_decay_steps"] == 8400


def test_wsd_is_tagged_because_no_other_tag_carries_the_schedule():
    plan = plan_arm.plan(_axes(lr_scheduler="warmup_stable_decay"))

    assert "wsd" in plan.exp_name.split("-")


def test_cosine_is_not_tagged_and_emits_nothing_when_it_matches_the_config():
    plan = plan_arm.plan(_axes(lr_scheduler="cosine"))

    assert "--trainer.lr_scheduler_type" not in plan.overrides
    assert "wsd" not in plan.exp_name.split("-")


def test_unknown_scheduler_is_refused():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(lr_scheduler="one_cycle"))


def test_warmup_ratio_forces_warmup_steps_to_zero():
    """ABL-MA-700-asr.yaml sets warmup_steps: 20, which silently wins."""
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    assert plan_arm.get(cfg, "trainer.warmup_steps")  # the trap this guards

    plan = plan_arm.plan(_axes(warmup_ratio="0.03"))

    assert _value(plan, "--trainer.warmup_ratio") == "0.03"
    assert _value(plan, "--trainer.warmup_steps") == "0"


def test_warmup_ratio_is_tagged():
    plan = plan_arm.plan(_axes(warmup_ratio="0.03"))

    assert "wu0p03" in plan.exp_name.split("-")


def test_campaign_accepts_both_as_grid_fields():
    assert {"lr_scheduler", "warmup_ratio"} <= campaign.AXIS_FIELDS


def test_every_screen_row_sets_the_schedule():
    """The regression this file exists for: no screen arm may inherit cosine."""
    _, arms = campaign.load_grid()
    screen = [a for a in arms if a["id"].startswith("MA-700-screen-")]

    assert len(screen) == 12
    for arm in screen:
        assert arm.get("lr_scheduler") == "warmup_stable_decay", arm["id"]
        assert str(arm.get("warmup_ratio")) == "0.03", arm["id"]
