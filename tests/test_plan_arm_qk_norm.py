"""Tests for the qk_norm axis in projects/ablation-campaign/plan_arm.py.

model.adapter.qk_norm RMS-normalises the Conformer's and the Q-Former's attention queries
and keys per head (MELTAdapterConfig says why). It changes the function the adapter
computes, so an arm that sets it has to say so to the trainer, say so in its run name, and
be refused on an adapter with no attention to normalise -- while every arm that does not set
it must render exactly as before.
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


CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-2100-asr.yaml")


def _axes(**overrides) -> ArmAxes:
    defaults = {
        "config": CONFIG,
        "stage": "MA",
        "world_size": 4,
        "batch_duration": "75",
        "grad_accum_steps": "1",
        "frame_rate_hz": "10",
        "adapter": "qformer",
    }
    defaults.update(overrides)
    return ArmAxes(**defaults)


@pytest.mark.parametrize("adapter", ["qformer", "conformer"])
def test_unset_emits_no_override_and_tags_nothing(adapter):
    """Byte-for-byte regression: the field must not rename or change any arm that omits it."""
    plan = plan_arm.plan(_axes(adapter=adapter))

    assert "--model.adapter.qk_norm" not in plan.overrides
    assert "qkn" not in plan.exp_name.split("-")


@pytest.mark.parametrize("adapter", ["qformer", "conformer"])
def test_true_tells_the_trainer_and_the_run_name(adapter):
    plan = plan_arm.plan(_axes(adapter=adapter, qk_norm="true"))

    assert plan.overrides[plan.overrides.index("--model.adapter.qk_norm") + 1] == "true"
    assert f"{adapter}T-qkn" in plan.exp_name


def test_the_tag_follows_the_adapter_and_changes_nothing_else_in_the_name():
    plain = plan_arm.plan(_axes()).exp_name
    normed = plan_arm.plan(_axes(qk_norm="true")).exp_name

    assert normed == plain.replace("qformerT-", "qformerT-qkn-", 1)


def test_false_is_the_config_default_so_it_emits_nothing():
    """The base config omits the key, i.e. false: asking for false is not an override."""
    plan = plan_arm.plan(_axes(qk_norm="false"))

    assert "--model.adapter.qk_norm" not in plan.overrides
    assert "qkn" not in plan.exp_name.split("-")


@pytest.mark.parametrize("adapter", ["mlp", "moe"])
def test_an_adapter_without_attention_is_refused(adapter):
    """The arm would be tagged with an architecture it does not have."""
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter=adapter, qk_norm="true"))


def test_a_value_that_is_not_a_bool_is_refused():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(qk_norm="maybe"))


def test_a_grid_row_can_set_it():
    """campaign.py must accept the field, or a typo-guard refuses the row."""
    assert "qk_norm" in campaign.AXIS_FIELDS
    row = {"id": "x", "stage": "MA", "config": "ABL-MA-2100-asr.yaml", "adapter": "qformer",
           "frame_rate_hz": 10, "qk_norm": True}
    axes, _ = campaign.resolve(row, {}, [])

    assert axes.qk_norm == "True"
    assert plan_arm.plan(axes).overrides.count("--model.adapter.qk_norm") == 1


def test_the_rest_of_the_arm_is_unchanged_by_the_option():
    plain = plan_arm.plan(_axes())
    normed = plan_arm.plan(_axes(qk_norm="true"))

    rest = list(normed.overrides)
    i = rest.index("--model.adapter.qk_norm")
    del rest[i : i + 2]
    assert rest == plain.overrides
    assert (normed.steps, normed.eval_steps, normed.save_steps) == (plain.steps, plain.eval_steps, plain.save_steps)
