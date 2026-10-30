"""The EuroLLM 2512 pair (utter-project/EuroLLM-1.7B-2512 and -Instruct-2512) next to the original pair.

The default decoder tag is a 16-character slug of the hub name, which cuts "EuroLLM-1.7B-Instruct-2512" to the original
Instruct's tag. Two arms with the same tag and axes render the same run name, so the second would resume the first's
output directory. The 2512 pair therefore pins its tags, and these tests keep the four run names apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import plan_arm  # noqa: E402
from plan_arm import ArmAxes  # noqa: E402

CONFIG = str(REPO_ROOT / "projects" / "ablation-campaign" / "ABL-MA-700-whisper-asr.yaml")
PAIRS = {
    "original": ("utter-project/EuroLLM-1.7B", "utter-project/EuroLLM-1.7B-Instruct"),
    "2512": ("utter-project/EuroLLM-1.7B-2512", "utter-project/EuroLLM-1.7B-Instruct-2512"),
}


def _plan(decoder: str):
    return plan_arm.plan(
        ArmAxes(
            config=CONFIG,
            stage="MA",
            world_size=4,
            decoder=decoder,
            encoder="openai/whisper-large-v3",
            adapter="moe",
        )
    )


def test_the_four_eurollm_decoders_get_distinct_tags_and_run_names():
    decoders = [name for pair in PAIRS.values() for name in pair]

    tags = {name: plan_arm.decoder_tag(name) for name in decoders}
    names = {name: _plan(name).exp_name for name in decoders}

    assert len(set(tags.values())) == 4, tags
    assert len(set(names.values())) == 4, names


@pytest.mark.parametrize("decoder", PAIRS["2512"])
def test_2512_profiles_match_the_original_pair_on_tokens_and_chat_template(decoder):
    original = plan_arm.DECODER_PROFILES[
        "utter-project/EuroLLM-1.7B-Instruct" if decoder.endswith("Instruct-2512") else "utter-project/EuroLLM-1.7B"
    ]
    profile = plan_arm.DECODER_PROFILES[decoder]

    for key in ("eos_token", "pad_token", "chat_template_config"):
        assert profile[key] == original[key]


def test_2512_base_borrows_the_2512_instruct_template_not_the_original():
    base, instruct = PAIRS["2512"]

    assert plan_arm.DECODER_PROFILES[base]["chat_template_from"] == instruct
    assert "chat_template_from" not in plan_arm.DECODER_PROFILES[instruct]
