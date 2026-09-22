"""Tests for the template_selection axis in projects/ablation-campaign/plan_arm.py.

Language ID in the prompt has to be a factor an arm can pin, not the coin flip
"random" gives on a TASK_TEMPLATES bucket that mixes LID and no-LID templates.
The axis follows the "tag only when overridden" rule, so every arm composed
before it existed must keep its exact EXP_NAME and overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_existing_verbatim_arm_is_unchanged():
    """template_task_override alone: forced random, no new tag."""
    plan = plan_arm.plan(_axes(template_task_override="verbatim", epochs="3"))

    assert _value(plan, "--data.prompt_template_selection") == "random"
    assert "lid" not in plan.exp_name.split("-") and "nolid" not in plan.exp_name.split("-")
    assert "-ep3-ttverbatim-" in plan.exp_name


def test_explicit_random_is_the_same_arm_as_leaving_it_empty():
    a = plan_arm.plan(_axes(template_task_override="verbatim"))
    b = plan_arm.plan(_axes(template_task_override="verbatim", template_selection="random"))

    assert a.exp_name == b.exp_name
    assert a.overrides == b.overrides


@pytest.mark.parametrize(
    "mode,tag", [("with_language", "lid"), ("without_language", "nolid")]
)
def test_language_filter_is_passed_through_and_tagged(mode, tag):
    plan = plan_arm.plan(_axes(template_task_override="verbatim", template_selection=mode))

    assert _value(plan, "--data.prompt_template_selection") == mode
    assert _value(plan, "--data.template_task_override") == "verbatim"
    assert tag in plan.exp_name.split("-")


def test_lid_and_no_lid_arms_get_different_names():
    lid = plan_arm.plan(_axes(template_task_override="verbatim", template_selection="with_language"))
    nolid = plan_arm.plan(_axes(template_task_override="verbatim", template_selection="without_language"))

    assert lid.exp_name != nolid.exp_name


def test_tag_follows_the_task_tag():
    plan = plan_arm.plan(_axes(template_task_override="verbatim", template_selection="with_language"))

    assert "-ttverbatim-lid-" in plan.exp_name


def test_selection_without_a_template_task_is_an_error(capsys):
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(template_selection="with_language"))
    assert "TEMPLATE_TASK_OVERRIDE" in capsys.readouterr().err


def test_unknown_selection_is_an_error(capsys):
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(template_task_override="verbatim", template_selection="best"))
    assert "with_language" in capsys.readouterr().err


def test_campaign_grid_accepts_the_field():
    assert "template_selection" in campaign.AXIS_FIELDS
