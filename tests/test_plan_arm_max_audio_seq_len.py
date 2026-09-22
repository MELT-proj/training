"""Tests for the max_audio_seq_len axis in projects/ablation-campaign/plan_arm.py.

Whisper's encoder raises on anything that is not exactly 3000 mel frames, so a
Whisper arm planned against a w2v-BERT config (max_audio_seq_len 1500) dies at
startup. That is what happened to job 45985946, and the workaround was to
hand-pass the override on every submission -- which keeps the real command out
of arms.tsv and only works while the operator remembers. The axis derives it
instead, and must do so without renaming any arm that has already run.
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
WHISPER = "openai/whisper-large-v3"


def _axes(**overrides) -> ArmAxes:
    defaults = dict(config=CONFIG, stage="MA", world_size=8)
    defaults.update(overrides)
    return ArmAxes(**defaults)


def _value(plan, flag):
    return plan.overrides[plan.overrides.index(flag) + 1]


def test_unset_on_the_configs_own_encoder_emits_nothing():
    """The pre-axis behaviour, byte for byte: no override, no rename."""
    plan = plan_arm.plan(_axes())

    assert "--model.encoder.max_audio_seq_len" not in plan.overrides


def test_whisper_derives_its_window_without_being_asked():
    plan = plan_arm.plan(_axes(encoder=WHISPER))

    assert _value(plan, "--model.encoder.max_audio_seq_len") == "3000"


def test_deriving_does_not_rename_the_arm():
    """The window is a property of the encoder, already in the encoder tag.

    MA-librispeech-w ran before this axis existed; tagging the window would
    orphan its output directory.
    """
    derived = plan_arm.plan(_axes(encoder=WHISPER))
    explicit = plan_arm.plan(_axes(encoder=WHISPER, max_audio_seq_len="3000"))

    assert derived.exp_name == explicit.exp_name
    assert "3000" not in derived.exp_name


def test_explicit_value_for_an_encoder_the_table_does_not_know():
    plan = plan_arm.plan(_axes(max_audio_seq_len="750"))

    assert _value(plan, "--model.encoder.max_audio_seq_len") == "750"


def test_explicit_value_matching_the_config_emits_nothing():
    import yaml

    with open(CONFIG) as fh:
        cfg_value = str(plan_arm.get(yaml.safe_load(fh), "model.encoder.max_audio_seq_len"))
    plan = plan_arm.plan(_axes(max_audio_seq_len=cfg_value))

    assert "--model.encoder.max_audio_seq_len" not in plan.overrides


def test_explicit_value_contradicting_whisper_is_refused():
    """The failure this axis exists to prevent must not be re-expressible."""
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(encoder=WHISPER, max_audio_seq_len="1500"))


def test_non_integer_is_refused():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(max_audio_seq_len="3000 frames"))


def test_campaign_accepts_it_as_a_grid_field():
    assert "max_audio_seq_len" in campaign.AXIS_FIELDS
