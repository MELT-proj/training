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


# ---------------------------------------------------------------------------
# Raw-waveform encoders
#
# For these the value counts SAMPLES, not frames. MELTAudioEncoder refuses the
# base config's 1500 at startup, which is after the queue wait, so it is derived.
# ---------------------------------------------------------------------------

MMS = "facebook/mms-1b"
MHUBERT = "utter-project/mHuBERT-147"
WAVEFORM_ENCODERS = (MMS, MHUBERT)


def _config_with_window(tmp_path, encoder_name=None, window=None) -> str:
    import yaml

    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if encoder_name is not None:
        cfg["model"]["encoder"]["name"] = encoder_name
    if window is not None:
        cfg["model"]["encoder"]["max_audio_seq_len"] = window
    path = tmp_path / "ABL-MA-700-asr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.mark.parametrize("encoder", WAVEFORM_ENCODERS)
def test_waveform_encoders_derive_the_window_in_samples(encoder):
    plan = plan_arm.plan(_axes(encoder=encoder))

    assert _value(plan, "--model.encoder.max_audio_seq_len") == "480000"


@pytest.mark.parametrize("encoder", WAVEFORM_ENCODERS)
def test_the_derived_window_satisfies_the_models_own_guard(encoder):
    """MELTAudioEncoder wants at least one second and a multiple of the conv stride (320)."""
    value = int(_value(plan_arm.plan(_axes(encoder=encoder)), "--model.encoder.max_audio_seq_len"))

    assert value >= 16_000
    assert value % 320 == 0


def test_every_crossing_encoder_sees_the_same_window_in_seconds():
    """The comparison's point: one attention span, whatever unit each encoder counts in."""
    w2v_bert_seconds = 1500 * 0.02  # the base config's own value, in 20 ms frames
    whisper_seconds = int(
        _value(plan_arm.plan(_axes(encoder=WHISPER)), "--model.encoder.max_audio_seq_len")
    ) * 0.01
    waveform_seconds = [
        int(_value(plan_arm.plan(_axes(encoder=e)), "--model.encoder.max_audio_seq_len")) / 16_000
        for e in WAVEFORM_ENCODERS
    ]

    assert {w2v_bert_seconds, whisper_seconds, *waveform_seconds} == {30.0}


def test_the_derivation_follows_the_configs_window(tmp_path):
    """Derived from the base config, not hardcoded: 2000 frames of 20 ms is 640000 samples."""
    plan = plan_arm.plan(_axes(config=_config_with_window(tmp_path, window=2000), encoder=MMS))

    assert _value(plan, "--model.encoder.max_audio_seq_len") == "640000"


def test_a_config_that_already_counts_samples_is_taken_as_it_is(tmp_path):
    config = _config_with_window(tmp_path, encoder_name=MMS, window=480_000)

    plan = plan_arm.plan(_axes(config=config, encoder=MMS))

    assert "--model.encoder.max_audio_seq_len" not in plan.overrides


@pytest.mark.parametrize("encoder", WAVEFORM_ENCODERS)
def test_explicit_value_equal_to_the_derived_one_is_accepted(encoder):
    derived = plan_arm.plan(_axes(encoder=encoder))
    explicit = plan_arm.plan(_axes(encoder=encoder, max_audio_seq_len="480000"))

    assert derived.overrides == explicit.overrides
    assert derived.exp_name == explicit.exp_name


@pytest.mark.parametrize("value", ["1500", "960000"])
def test_explicit_value_contradicting_a_waveform_encoder_is_refused(value):
    """1500 is the frame-sized mistake; 960000 is a defensible window, but the name
    would not record it, so two different arms would share one directory."""
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(encoder=MMS, max_audio_seq_len=value))


@pytest.mark.parametrize("encoder", WAVEFORM_ENCODERS)
def test_deriving_the_window_does_not_rename_the_arm(encoder):
    assert "480000" not in plan_arm.plan(_axes(encoder=encoder)).exp_name


def test_w2vbert_still_inherits_the_configs_window():
    plan = plan_arm.plan(_axes(encoder="facebook/w2v-bert-2.0"))

    assert "--model.encoder.max_audio_seq_len" not in plan.overrides
