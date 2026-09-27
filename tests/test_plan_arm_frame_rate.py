"""Tests for the frame_rate_hz axis in projects/ablation-campaign/plan_arm.py.

Every arm of the audio-stack crossing runs at one decoder frame rate (10 Hz,
plan/03-audio-stack.md §1b), and each adapter gets there differently: the MLP and MoE
stack frames, the Conformer strides, the Q-Former is 10 Hz by construction. Left to the
rows that is a note per row to remember, which is how Whisper's window and the WSD
schedule each cost an arm. So a row states the rate and plan_arm derives the route, and
the rate is tagged into EXP_NAME because the Q-Former's is in no other tag.
"""

from __future__ import annotations

import dataclasses
import re
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
ENCODERS = (
    "facebook/w2v-bert-2.0",
    "facebook/mms-1b",
    "openai/whisper-large-v3",
    "utter-project/mHuBERT-147",
)
ADAPTERS = ("mlp", "conformer", "moe", "qformer")


def _axes(**overrides) -> ArmAxes:
    defaults = dict(config=CONFIG, stage="MA", world_size=4)
    defaults.update(overrides)
    return ArmAxes(**defaults)


def _value(plan, flag):
    return plan.overrides[plan.overrides.index(flag) + 1]


def _tags(plan) -> list[str]:
    return plan.exp_name.split("-")


def _config_with_adapter(tmp_path, **adapter_keys) -> str:
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["model"]["adapter"].update(adapter_keys)
    path = tmp_path / "ABL-MA-700-asr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


# ---------------------------------------------------------------------------
# Undeclared: the pre-axis behaviour, byte for byte
# ---------------------------------------------------------------------------


def test_no_declared_rate_adds_no_tag_and_no_override():
    plan = plan_arm.plan(_axes(adapter="conformer"))

    assert not any(t.startswith("hz") or re.fullmatch(r"cs\d+k\d+", t) for t in _tags(plan))
    assert "--model.adapter.adapter_stride" not in plan.overrides
    assert "--model.adapter.adapter_kernel_size" not in plan.overrides


def test_an_explicit_stack_factor_without_a_rate_is_unchanged():
    """The screen's rows state stack_factor 5 themselves; they must keep planning as before."""
    plan = plan_arm.plan(_axes(stack_factor="5"))

    assert _value(plan, "--model.adapter.stack_factor") == "5"
    assert "sk5" in _tags(plan)
    assert not any(t.startswith("hz") for t in _tags(plan))


# ---------------------------------------------------------------------------
# MLP and MoE: stacking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ["mlp", "moe"])
def test_stacking_adapters_derive_stack_factor_from_the_rate(adapter):
    plan = plan_arm.plan(_axes(adapter=adapter, frame_rate_hz="10"))

    assert _value(plan, "--model.adapter.stack_factor") == "5"
    assert _tags(plan).index("sk5") < _tags(plan).index("hz10")


def test_the_derived_stack_factor_is_the_same_plan_as_typing_it():
    """A row stating the rate reproduces what a row stating the factor would have made,
    apart from the rate tag it adds."""
    derived = plan_arm.plan(_axes(frame_rate_hz="10"))
    typed = plan_arm.plan(_axes(stack_factor="5"))

    assert derived.overrides == typed.overrides
    assert derived.exp_name == typed.exp_name.replace("-sk5-", "-sk5-hz10-", 1)


def test_an_agreeing_explicit_stack_factor_is_accepted():
    plan_arm.plan(_axes(frame_rate_hz="10", stack_factor="5"))


def test_a_contradicting_stack_factor_is_refused():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(frame_rate_hz="10", stack_factor="2"))


def test_a_rate_that_is_not_an_integer_division_is_refused():
    """50 Hz / 7 is no stacking factor."""
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(frame_rate_hz="7"))


def test_a_fractional_rate_that_divides_is_derived_and_tagged_readably():
    plan = plan_arm.plan(_axes(frame_rate_hz="12.5"))

    assert _value(plan, "--model.adapter.stack_factor") == "4"
    assert "hz12p5" in _tags(plan)


@pytest.mark.parametrize("bad", ["fast", "0", "-10", "10 Hz"])
def test_a_malformed_rate_is_refused(bad):
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(frame_rate_hz=bad))


# ---------------------------------------------------------------------------
# Conformer: one strided layer, kernel equal to stride
# ---------------------------------------------------------------------------


def test_the_conformer_derives_stride_and_kernel():
    plan = plan_arm.plan(_axes(adapter="conformer", frame_rate_hz="10"))

    assert _value(plan, "--model.adapter.adapter_stride") == "5"
    assert _value(plan, "--model.adapter.adapter_kernel_size") == "5"
    assert "--model.adapter.stack_factor" not in plan.overrides


def test_the_conformer_tags_its_route_and_its_rate():
    tags = _tags(plan_arm.plan(_axes(adapter="conformer", frame_rate_hz="10")))

    assert tags.index("conformerT") < tags.index("cs5k5") < tags.index("hz10")


def test_one_layer_is_the_default_and_is_not_overridden():
    plan = plan_arm.plan(_axes(adapter="conformer", frame_rate_hz="10"))

    assert "--model.adapter.num_adapter_layers" not in plan.overrides


def test_a_multi_layer_conformer_is_refused_because_its_strides_would_multiply(tmp_path):
    config = _config_with_adapter(tmp_path, num_adapter_layers=2)

    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(config=config, adapter="conformer", frame_rate_hz="10"))


def test_a_stack_factor_on_the_conformer_is_refused():
    """The Conformer never reads it: the tag would claim a rate the arm does not have."""
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="conformer", stack_factor="5"))
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="conformer", stack_factor="5", frame_rate_hz="10"))


# ---------------------------------------------------------------------------
# Q-Former: native, so checked and never stacked
# ---------------------------------------------------------------------------


def test_the_qformer_is_ten_hz_at_its_defaults_and_derives_nothing():
    """Window 15 with 3 queries (15 // 5) is 50 * 3 / 15 = 10 Hz."""
    plan = plan_arm.plan(_axes(adapter="qformer", frame_rate_hz="10"))

    assert "hz10" in _tags(plan)
    assert not any(tok.startswith("--model.adapter.") and "_type" not in tok for tok in plan.overrides)


def test_a_stack_factor_on_the_qformer_is_refused_with_or_without_a_rate():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="qformer", stack_factor="5"))
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="qformer", stack_factor="5", frame_rate_hz="10"))


def test_a_stack_factor_of_one_on_the_qformer_is_a_harmless_no_op():
    plan = plan_arm.plan(_axes(adapter="qformer", stack_factor="1"))

    assert "sk1" not in _tags(plan)


def test_the_qformer_refuses_a_rate_it_does_not_have():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="qformer", frame_rate_hz="25"))


def test_the_qformer_check_reads_the_configs_window(tmp_path):
    """Window 10 with downsample 5 is two queries, still 10 Hz; window 15 / 3 is 16.7 Hz."""
    ok = _config_with_adapter(tmp_path, window_size=10, downsample_rate=5)
    plan_arm.plan(_axes(config=ok, adapter="qformer", frame_rate_hz="10"))

    bad = _config_with_adapter(tmp_path, window_size=15, downsample_rate=3)
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(config=bad, adapter="qformer", frame_rate_hz="10"))


# ---------------------------------------------------------------------------
# Other adapters
# ---------------------------------------------------------------------------


def test_an_adapter_without_a_route_is_refused():
    with pytest.raises(SystemExit):
        plan_arm.plan(_axes(adapter="linear", frame_rate_hz="10"))


# ---------------------------------------------------------------------------
# The crossing: sixteen cells
# ---------------------------------------------------------------------------


def _crossing_plans() -> dict[tuple[str, str], plan_arm.ArmPlan]:
    return {
        (encoder, adapter): plan_arm.plan(
            _axes(
                encoder=encoder,
                adapter=adapter,
                frame_rate_hz="10",
                lr_scheduler="warmup_stable_decay",
                warmup_ratio="0.03",
                adapter_lr="2e-3",
                batch_duration="75",
                grad_accum_steps="1",
                epochs="5",
            )
        )
        for encoder in ENCODERS
        for adapter in ADAPTERS
    }


def test_all_sixteen_cells_render():
    assert len(_crossing_plans()) == 16


def test_every_cell_has_its_own_name_and_says_ten_hz():
    names = [p.exp_name for p in _crossing_plans().values()]

    assert len(set(names)) == 16
    assert all("hz10" in name.split("-") for name in names)


def test_the_cells_differ_only_by_encoder_and_adapter():
    """One recipe for all sixteen: strip encoder, adapter and route from each name and
    what is left is identical."""
    residues = set()
    for (encoder, adapter), plan in _crossing_plans().items():
        residue = [
            t for t in _tags(plan)
            if not (
                t.startswith(plan_arm.encoder_tag(encoder))
                or t.startswith(adapter)
                or re.fullmatch(r"sk\d+|cs\d+k\d+", t)
            )
        ]
        residues.add("-".join(residue))

    assert len(residues) == 1


def test_each_adapter_takes_its_own_route_on_every_encoder():
    for (encoder, adapter), plan in _crossing_plans().items():
        stacked = "--model.adapter.stack_factor" in plan.overrides
        strided = "--model.adapter.adapter_stride" in plan.overrides
        assert stacked == (adapter in ("mlp", "moe")), (encoder, adapter)
        assert strided == (adapter == "conformer"), (encoder, adapter)


def test_the_encoders_settings_come_out_per_encoder_and_the_rest_do_not():
    plans = _crossing_plans()
    windows = {
        encoder: _value(plans[(encoder, "mlp")], "--model.encoder.max_audio_seq_len")
        if "--model.encoder.max_audio_seq_len" in plans[(encoder, "mlp")].overrides
        else None
        for encoder in ENCODERS
    }

    assert windows == {
        "facebook/w2v-bert-2.0": None,  # the config's own 1500 frames
        "facebook/mms-1b": "480000",
        "openai/whisper-large-v3": "3000",
        "utter-project/mHuBERT-147": "480000",
    }
    # 300 s effective batch and five epochs are the same for all sixteen: same step count.
    assert len({p.steps for p in plans.values()}) == 1


def test_campaign_accepts_the_rate_as_a_grid_field():
    assert "frame_rate_hz" in campaign.AXIS_FIELDS
    assert "frame_rate_hz" in {f.name for f in dataclasses.fields(ArmAxes)}
