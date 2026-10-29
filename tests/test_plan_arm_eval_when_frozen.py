"""Tests for the eval_when_frozen axis in projects/ablation-campaign/plan_arm.py.

HF's Trainer calls model.train() on the whole model, and MELT never puts a frozen
encoder back into eval mode on its own -- see melt/modeling/modeling_melt.py's
MELTAudioEncoder.train() for the fix and the measured size of the effect. This is the
opt-in axis that turns it on so it can be A/B'd against the historical behaviour before
becoming a default: "" inherits the config's own value (every ABL-*.yaml today omits the
key, i.e. false), and it is tagged into EXP_NAME by EFFECTIVE value, like decoder_lora's
`-lora`, because the two arms of the A/B must not collide on one output directory.
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


def test_unset_emits_no_override_and_no_tag():
    """The config declares no eval_when_frozen key; inheriting it must not touch EXP_NAME."""
    plan = plan_arm.plan(_axes())

    assert not any("eval_when_frozen" in tok for tok in plan.overrides)
    assert "evalfrozen" not in plan.exp_name


def test_unset_reproduces_the_pre_axis_exp_name():
    """Byte-for-byte regression: this axis must not rename any arm that has already run."""
    plan = plan_arm.plan(_axes())

    assert plan.exp_name == (
        "MA-700asr-w2vbF-llama1bInsF-mlpT-elr6e6-dlr2e5-lr2e5-s42-8g"
    )


def test_explicit_false_still_emits_no_override():
    """Requesting the same value the config would fall back to is not an override."""
    plan = plan_arm.plan(_axes(eval_when_frozen="false"))

    assert not any("eval_when_frozen" in tok for tok in plan.overrides)
    assert "evalfrozen" not in plan.exp_name


def test_explicit_true_emits_the_cli_override():
    # ABL-MA-700-asr.yaml already freezes the encoder, so this exercises the tagged,
    # no-warning, intended-use case: an explicit request on an already-frozen encoder.
    plan = plan_arm.plan(_axes(eval_when_frozen="true"))

    assert _value(plan, "--model.encoder.eval_when_frozen") == "true"


def test_explicit_true_is_tagged_by_effective_value():
    plan = plan_arm.plan(_axes(eval_when_frozen="true"))

    assert "evalfrozen" in plan.exp_name


def test_the_tag_sits_right_after_the_encoder_segment():
    """Same slot decoder_lora's `-lora` uses on the decoder segment, one segment over."""
    baseline = plan_arm.plan(_axes()).exp_name
    flagged = plan_arm.plan(_axes(eval_when_frozen="true")).exp_name

    assert flagged == baseline.replace("w2vbF-", "w2vbF-evalfrozen-", 1)


def test_the_two_ab_arms_get_different_names():
    """The whole reason this is tagged by effective value: the two halves of the A/B must
    not collide on one output directory."""
    control = plan_arm.plan(_axes()).exp_name
    treatment = plan_arm.plan(_axes(eval_when_frozen="true")).exp_name

    assert control != treatment


def test_warns_when_the_encoder_is_not_frozen(capsys):
    """The flag is a silent no-op without a fully-frozen encoder (MELTAudioEncoder.train()
    checks requires_grad at call time); catching it here is cheaper than on a wasted run."""
    plan_arm.plan(_axes(eval_when_frozen="true", encoder_freeze="false"))

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "not frozen" in err


def test_no_warning_when_the_encoder_is_frozen(capsys):
    # By ABL-MA-700-asr.yaml's own default -- no explicit ENCODER_FREEZE needed.
    plan_arm.plan(_axes(eval_when_frozen="true"))

    assert "WARNING" not in capsys.readouterr().err


def test_no_warning_without_the_flag(capsys):
    plan_arm.plan(_axes(encoder_freeze="false"))  # not frozen, but the flag isn't set either

    assert "WARNING" not in capsys.readouterr().err


def test_campaign_accepts_it_as_a_grid_field():
    assert "eval_when_frozen" in campaign.AXIS_FIELDS
