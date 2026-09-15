"""Tests for the stack_factor axis in projects/ablation-campaign/plan_arm.py.

plan_arm.py composes EXP_NAME and the CLI override list from an ArmAxes
purely (no side effects, no torch/omegaconf import -- see its own module
docstring), so it is imported directly here rather than driven through the
campaign.py/launch_campaign.sh wrapper.

stack_factor follows the same "tag into EXP_NAME only when overridden" rule
as batch_duration/grad_accum_steps: ABL-MA-700-asr.yaml (used below) predates
the axis and declares no model.adapter.stack_factor key, so leaving it
unset must keep reproducing that config's exact pre-existing EXP_NAME.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import plan_arm  # noqa: E402
from plan_arm import ArmAxes  # noqa: E402

CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-700-asr.yaml")


def _axes(**overrides) -> ArmAxes:
    defaults = dict(config=CONFIG, stage="MA", world_size=8)
    defaults.update(overrides)
    return ArmAxes(**defaults)


def test_unset_stack_factor_emits_no_override_and_no_tag():
    """The config declares no stack_factor key; inheriting it must not touch EXP_NAME."""
    plan = plan_arm.plan(_axes())

    assert not any("model.adapter.stack_factor" in tok for tok in plan.overrides)
    assert not any(seg.startswith("sk") for seg in plan.exp_name.split("-"))


def test_explicit_default_value_still_emits_no_override():
    """Requesting the same value the config would fall back to is not an override."""
    plan = plan_arm.plan(_axes(stack_factor="1"))

    assert not any("model.adapter.stack_factor" in tok for tok in plan.overrides)
    assert not any(seg.startswith("sk") for seg in plan.exp_name.split("-"))


def test_overridden_stack_factor_emits_the_cli_override():
    plan = plan_arm.plan(_axes(stack_factor="4"))

    idx = plan.overrides.index("--model.adapter.stack_factor")
    assert plan.overrides[idx + 1] == "4"


def test_overridden_stack_factor_is_tagged_into_exp_name():
    plan = plan_arm.plan(_axes(stack_factor="4"))
    assert "sk4" in plan.exp_name.split("-")


def test_stack_factor_tag_sits_between_the_adapter_segment_and_lr_tags():
    """Position matters for readability and for not colliding with other tags."""
    baseline = plan_arm.plan(_axes()).exp_name
    stacked = plan_arm.plan(_axes(stack_factor="4")).exp_name

    assert stacked == baseline.replace("-elr", "-sk4-elr", 1)
