"""Regression tests for issue #77.

`OmegaConf.from_dotlist` parses CLI override values as YAML, and under YAML
1.1 the bare tokens `no`/`off`/`false` and `yes`/`on`/`true` are booleans, not
strings. `--trainer.eval_strategy no` therefore used to reach
Seq2SeqTrainingArguments as `False`, which HF's IntervalStrategy rejected
with `ValueError: False is not a valid IntervalStrategy`.
"""

import pytest
from omegaconf import OmegaConf

from melt.training.config import get_default_config, trainer_args_dict
from transformers import Seq2SeqTrainingArguments


@pytest.mark.parametrize(
    "key,cli_value,expected",
    [
        ("eval_strategy", "no", "no"),
        ("save_strategy", "no", "no"),
        ("logging_strategy", "no", "no"),
    ],
)
def test_yaml_bool_token_is_restored_to_string(tmp_path, key, cli_value, expected):
    cfg = get_default_config()
    # bf16 defaults to true, which Seq2SeqTrainingArguments rejects outright
    # on a GPU-less CI runner; disable it since it's unrelated to this test.
    cli_cfg = OmegaConf.from_dotlist(
        [f"trainer.{key}={cli_value}", "trainer.bf16=false", f"trainer.output_dir={tmp_path}"]
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)

    result = trainer_args_dict(cfg)

    assert result[key] == expected
    # Must not just be string-typed in our dict -- HF must accept it too.
    targs = Seq2SeqTrainingArguments(**result)
    assert str(getattr(targs, key)).lower().endswith(expected)


def test_real_bool_fields_are_left_untouched(tmp_path):
    cfg = get_default_config()
    cli_cfg = OmegaConf.from_dotlist(["trainer.bf16=true", "trainer.do_train=false", f"trainer.output_dir={tmp_path}"])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    result = trainer_args_dict(cfg)

    assert result["bf16"] is True
    assert result["do_train"] is False


def test_resume_from_checkpoint_true_is_not_turned_into_yes(tmp_path):
    # campaign.py --resume passes `True`; train.py scans output_dir for it. Turned into the string "yes" it listed a
    # directory of that name and the resume died (backbone Llama arms, 2026-10-16).
    cfg = get_default_config()
    cli_cfg = OmegaConf.from_dotlist(["trainer.resume_from_checkpoint=True", f"trainer.output_dir={tmp_path}"])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    assert trainer_args_dict(cfg)["resume_from_checkpoint"] is True


def test_train_py_only_keys_stay_out_of_training_arguments(tmp_path):
    # overwrite_output_dir and ddp_gradient_as_bucket_view are read by train.py
    # off the raw YAML; neither is a Seq2SeqTrainingArguments field.
    cfg = get_default_config()
    cli_cfg = OmegaConf.from_dotlist(
        ["trainer.ddp_gradient_as_bucket_view=true", "trainer.bf16=false", f"trainer.output_dir={tmp_path}"]
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)

    result = trainer_args_dict(cfg)

    assert "ddp_gradient_as_bucket_view" not in result
    assert "overwrite_output_dir" not in result
    assert cfg.trainer.ddp_gradient_as_bucket_view is True
    Seq2SeqTrainingArguments(**result)
