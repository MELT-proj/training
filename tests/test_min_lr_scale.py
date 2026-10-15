"""`optimization.min_lr_scale` is the LR every decaying schedule ends at."""

import pytest
import torch
from omegaconf import OmegaConf

from melt.training.config import get_default_config, trainer_args_dict
from transformers import Seq2SeqTrainingArguments, get_scheduler


def _args(tmp_path, *dotlist):
    cfg = get_default_config()
    cli = OmegaConf.from_dotlist(["trainer.bf16=false", f"trainer.output_dir={tmp_path}", *dotlist])
    return trainer_args_dict(OmegaConf.merge(cfg, cli))


def _final_lrs(result, peaks=(1e-3, 2e-5), steps=100, warmup=3):
    targs = Seq2SeqTrainingArguments(**result)
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in peaks]
    opt = torch.optim.SGD([{"params": [p], "lr": lr} for p, lr in zip(params, peaks)])
    kwargs = dict(targs.lr_scheduler_kwargs)
    if targs.lr_scheduler_type.value == "warmup_stable_decay":
        kwargs.setdefault("num_decay_steps", 20)
    sched = get_scheduler(
        targs.lr_scheduler_type,
        opt,
        num_warmup_steps=warmup,
        num_training_steps=steps,
        scheduler_specific_kwargs=kwargs,
    )
    for _ in range(steps):
        opt.step()
        sched.step()
    return [g["lr"] for g in opt.param_groups]


@pytest.mark.parametrize("sched", ["cosine", "cosine_with_min_lr", "warmup_stable_decay"])
def test_every_group_ends_at_scale_times_its_peak(tmp_path, sched):
    result = _args(tmp_path, f"trainer.lr_scheduler_type={sched}")

    final = _final_lrs(result)

    assert final == pytest.approx([1e-4, 2e-6])


def test_cosine_runs_as_cosine_with_min_lr(tmp_path):
    result = _args(tmp_path, "trainer.lr_scheduler_type=cosine")

    assert result["lr_scheduler_type"] == "cosine_with_min_lr"
    assert result["lr_scheduler_kwargs"]["min_lr_rate"] == 0.1


def test_scale_zero_keeps_cosine_decaying_to_zero(tmp_path):
    result = _args(tmp_path, "optimization.min_lr_scale=0")

    assert _final_lrs(result) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_wsd_keeps_the_other_kwargs(tmp_path):
    result = _args(
        tmp_path,
        "trainer.lr_scheduler_type=warmup_stable_decay",
        'trainer.lr_scheduler_kwargs={"num_decay_steps": 7, "decay_type": "cosine"}',
    )

    assert result["lr_scheduler_kwargs"] == {
        "num_decay_steps": 7,
        "decay_type": "cosine",
        "min_lr_ratio": 0.1,
    }


def test_agreeing_hand_passed_floor_is_accepted(tmp_path):
    result = _args(
        tmp_path,
        "trainer.lr_scheduler_type=warmup_stable_decay",
        'trainer.lr_scheduler_kwargs={"num_decay_steps": 7, "min_lr_ratio": 0.1}',
    )

    assert result["lr_scheduler_kwargs"]["min_lr_ratio"] == 0.1


def test_disagreeing_hand_passed_floor_fails(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        _args(
            tmp_path,
            "trainer.lr_scheduler_type=warmup_stable_decay",
            'trainer.lr_scheduler_kwargs={"num_decay_steps": 7, "min_lr_ratio": 0.0}',
        )


def test_scheduler_without_a_floor_fails(tmp_path):
    with pytest.raises(ValueError, match="no LR floor"):
        _args(tmp_path, "trainer.lr_scheduler_type=linear")


def test_constant_schedules_are_left_alone(tmp_path):
    result = _args(tmp_path, "trainer.lr_scheduler_type=constant_with_warmup")

    assert result["lr_scheduler_type"] == "constant_with_warmup"
    assert not result.get("lr_scheduler_kwargs")
