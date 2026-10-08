"""Tests for the encoder-derived settings in projects/ablation-campaign/plan_arm.py.

The audio-stack crossing runs four encoders through one recipe, and three things
about an encoder are not the same across them and are not the operator's to remember:
the name tag, whether the checkpoint time-masks by default (facebook/mms-1b and
mHuBERT-147 do, w2v-BERT and Whisper do not), and which attention backend is
affordable. The window an encoder is fed is covered next to Whisper's, in
test_plan_arm_max_audio_seq_len.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import plan_arm  # noqa: E402
from plan_arm import ArmAxes  # noqa: E402

CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-700-asr.yaml")
W2VB = "facebook/w2v-bert-2.0"
WHISPER = "openai/whisper-large-v3"
MMS = "facebook/mms-1b"
MHUBERT = "utter-project/mHuBERT-147"
SHIPS_SPEC_AUGMENT = (MMS, MHUBERT)


def _axes(**overrides) -> ArmAxes:
    defaults = dict(config=CONFIG, stage="MA", world_size=8)
    defaults.update(overrides)
    return ArmAxes(**defaults)


def _value(plan, flag):
    return plan.overrides[plan.overrides.index(flag) + 1]


def _config_without_spec_augment_pin(tmp_path, value=None) -> str:
    """A copy of the base config with `model.encoder.apply_spec_augment` removed or set.

    Named like the original so the EXP_NAME data tag is unchanged.
    """
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    encoder = cfg["model"]["encoder"]
    if value is None:
        encoder.pop("apply_spec_augment", None)
    else:
        encoder["apply_spec_augment"] = value
    path = tmp_path / "ABL-MA-700-asr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_new_encoders_get_descriptive_tags():
    assert plan_arm.encoder_tag(MMS) == "mms1b"
    assert plan_arm.encoder_tag(MHUBERT) == "mhubert147"
    assert "mms1bF" in plan_arm.plan(_axes(encoder=MMS)).exp_name
    assert "mhubert147F" in plan_arm.plan(_axes(encoder=MHUBERT)).exp_name


def test_whisper_keeps_the_name_it_ran_under():
    """The tag was made explicit, so it must not have moved.

    This is `MA-700-screen-whisper-lr1e3-b1200`'s directory, on MN5 and in arms.tsv.
    """
    plan = plan_arm.plan(
        _axes(
            encoder=WHISPER,
            stack_factor="5",
            lr_scheduler="warmup_stable_decay",
            warmup_ratio="0.03",
            adapter_lr="1e-3",
            batch_duration="150",
            grad_accum_steps="1",
        )
    )

    assert plan.exp_name == (
        "MA-700asr-whisperlargeF-llama1bInsF-mlpT-sk5-ga1-wsd-wus0p03-elr6e6-dlr2e5-lr1e3-s42-8g"
    )


def test_w2vbert_tag_is_unchanged():
    assert plan_arm.encoder_tag(W2VB) == "w2vb"


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoder", SHIPS_SPEC_AUGMENT)
def test_the_base_config_pins_spec_augment_off(encoder):
    """The pin lives in the base YAML; both encoders that ship it on render against it."""
    with open(CONFIG) as fh:
        assert yaml.safe_load(fh)["model"]["encoder"]["apply_spec_augment"] is False

    plan = plan_arm.plan(_axes(encoder=encoder))

    assert plan.exp_name


@pytest.mark.parametrize("encoder", SHIPS_SPEC_AUGMENT)
def test_a_config_that_does_not_pin_it_is_refused_for_the_encoders_that_ship_it_on(
    encoder, tmp_path
):
    unpinned = _config_without_spec_augment_pin(tmp_path)

    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(config=unpinned, encoder=encoder))


@pytest.mark.parametrize("encoder", SHIPS_SPEC_AUGMENT)
def test_a_config_that_pins_it_on_is_refused_too(encoder, tmp_path):
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(config=_config_without_spec_augment_pin(tmp_path, True), encoder=encoder))


@pytest.mark.parametrize("encoder", [W2VB, WHISPER])
def test_encoders_that_ship_it_off_do_not_need_the_pin(encoder, tmp_path):
    """Older configs (LibriSpeech, IFT) omit the key, and those arms must keep planning."""
    unpinned = _config_without_spec_augment_pin(tmp_path)

    assert plan_arm.plan(_axes(config=unpinned, encoder=encoder)).exp_name


def test_the_pin_never_reaches_the_command_line():
    """Identical across arms, so it is the base YAML's, never a per-arm override."""
    for encoder in (W2VB, WHISPER, MMS, MHUBERT):
        plan = plan_arm.plan(_axes(encoder=encoder))
        assert not any("apply_spec_augment" in tok for tok in plan.overrides)


# ---------------------------------------------------------------------------
# Attention backend
# ---------------------------------------------------------------------------


GIVEN_FLASH = (MMS, WHISPER)


def test_mms_is_given_flash_attention():
    """1.54x slower under sdpa (measured): at the crossing's length that is 72 h+ per arm."""
    plan = plan_arm.plan(_axes(encoder=MMS))

    assert _value(plan, "--model.encoder.attn_implementation") == "flash_attention_2"


def test_whisper_is_given_flash_attention():
    """Flash-eligible and never checked before (PI, 2026-09-27): its own forward() never
    reads the attention mask at all, so there is no masking complication to trip over."""
    plan = plan_arm.plan(_axes(encoder=WHISPER))

    assert _value(plan, "--model.encoder.attn_implementation") == "flash_attention_2"


@pytest.mark.parametrize("encoder", [W2VB, MHUBERT])
def test_the_other_encoders_keep_the_configs_own_backend(encoder):
    """w2v-BERT has no flash path (relative-position bias); mHuBERT-147 (95M) is not
    expensive enough to matter."""
    plan = plan_arm.plan(_axes(encoder=encoder))

    assert "--model.encoder.attn_implementation" not in plan.overrides


@pytest.mark.parametrize("encoder", GIVEN_FLASH)
def test_a_config_that_already_says_flash_emits_nothing(encoder, tmp_path):
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["model"]["encoder"]["attn_implementation"] = "flash_attention_2"
    path = tmp_path / "ABL-MA-700-asr.yaml"
    path.write_text(yaml.safe_dump(cfg))

    plan = plan_arm.plan(_axes(config=str(path), encoder=encoder))

    assert "--model.encoder.attn_implementation" not in plan.overrides


@pytest.mark.parametrize("encoder", GIVEN_FLASH)
def test_the_backend_is_not_tagged_into_the_name(encoder):
    """A property of the encoder, already in its tag, like the window."""
    plan = plan_arm.plan(_axes(encoder=encoder))

    assert "flash" not in plan.exp_name


def test_every_whisper_family_member_in_the_window_table_also_gets_flash():
    """Flash support is architectural (WhisperPreTrainedModel), not per-checkpoint, so the
    two tables should not silently diverge for a whisper-family encoder."""
    assert set(plan_arm.ENCODER_WINDOW_FRAMES) <= set(plan_arm.ENCODER_ATTN_IMPLEMENTATION)
