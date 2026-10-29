"""Tests for the ddp_find_unused_parameters axis in projects/ablation-campaign/plan_arm.py.

The MoE adapter's top-k router can leave an expert with zero tokens routed to it in a
given micro-batch -- small train_cuts_per_batch make this common, not an edge case -- and
DDP's default expects every parameter passed to it to receive a gradient on every step.
Reproduced on real GPU 2026-09-27 (board.md): the crossing-prep MoE smoke crashed at step
1 with torch's "Expected to have finished reduction" RuntimeError, both eval_on_start
rounds having already run clean. The other three adapters route every parameter every
step and should not pay this flag's traversal overhead, so it is a per-arm override
(same character as gradient_checkpointing), never a change to config/accelerate/ddp.yaml
itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_unset_emits_no_override():
    """ABL-MA-700-asr.yaml already declares the key false; inheriting it is a no-op."""
    plan = plan_arm.plan(_axes())

    assert not any("ddp_find_unused_parameters" in tok for tok in plan.overrides)


def test_unset_reproduces_the_pre_axis_exp_name():
    """Byte-for-byte regression: this axis must not rename any arm that has already run."""
    plan = plan_arm.plan(_axes())

    assert plan.exp_name == (
        "MA-700asr-w2vbF-llama1bInsF-mlpT-elr6e6-dlr2e5-lr2e5-s42-8g"
    )


def test_explicit_false_still_emits_no_override():
    """Requesting the same value the config would fall back to is not an override."""
    plan = plan_arm.plan(_axes(ddp_find_unused_parameters="false"))

    assert not any("ddp_find_unused_parameters" in tok for tok in plan.overrides)


def test_explicit_true_emits_the_cli_override():
    plan = plan_arm.plan(_axes(adapter="moe", ddp_find_unused_parameters="true"))

    assert _value(plan, "--trainer.ddp_find_unused_parameters") == "true"


def test_not_tagged_into_exp_name():
    """Mechanical DDP concern, not a scientific axis -- same as gradient_checkpointing."""
    baseline = plan_arm.plan(_axes(adapter="moe")).exp_name
    flagged = plan_arm.plan(_axes(adapter="moe", ddp_find_unused_parameters="true")).exp_name

    assert flagged == baseline


def test_campaign_accepts_it_as_a_grid_field():
    assert "ddp_find_unused_parameters" in campaign.AXIS_FIELDS
