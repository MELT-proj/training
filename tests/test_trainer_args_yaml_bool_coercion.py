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
