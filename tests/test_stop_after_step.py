"""run.stop_after_step: end a run at an optimizer step, the LR schedule of the full run untouched."""

import pytest
import torch
from omegaconf import OmegaConf

from melt.training.callbacks import StopAfterStep, stop_after_step_from_config
from melt.training.config import get_default_config
from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments


def _control_after(step: int, stop_at: int = 5) -> TrainerControl:
    state = TrainerState(global_step=step, max_steps=10)
    return StopAfterStep(stop_at).on_step_end(None, state, TrainerControl())


def test_nothing_is_requested_before_the_step():
    control = _control_after(4)
    assert not control.should_save
    assert not control.should_training_stop


@pytest.mark.parametrize("step", [5, 6])
def test_a_save_and_a_stop_are_requested_from_the_step_on(step):
    """At the step itself, and past it (a resume from a later checkpoint ends after its first step)."""
    control = _control_after(step)
    assert control.should_save
    assert control.should_training_stop


@pytest.mark.parametrize("bad", [0, -1])
def test_the_callback_rejects_a_step_below_one(bad):
    with pytest.raises(ValueError, match="must be >= 1"):
        StopAfterStep(bad)


def test_the_option_is_off_by_default():
    assert get_default_config().run.stop_after_step is None
    assert stop_after_step_from_config(get_default_config().run, 11499) is None
    assert stop_after_step_from_config({}, 11499) is None


def test_a_valid_step_is_returned_as_an_int():
    assert stop_after_step_from_config(OmegaConf.create({"stop_after_step": 2872}), 11499) == 2872
    assert stop_after_step_from_config({"stop_after_step": 11499}, 11499) == 11499, "the last step is allowed"
    assert stop_after_step_from_config({"stop_after_step": 5}, -1) == 5, "an epoch-based run has no max_steps to check"


@pytest.mark.parametrize("bad", [0, -4, "7", 2.5, True])
def test_a_value_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="positive integer"):
        stop_after_step_from_config({"stop_after_step": bad}, 100)


def test_a_step_beyond_max_steps_is_refused():
    """The run would end at max_steps first and the option would quietly do nothing."""
    with pytest.raises(ValueError, match="beyond trainer.max_steps=100"):
        stop_after_step_from_config({"stop_after_step": 101}, 100)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, labels=None):
        out = self.linear(x)
        return {"loss": torch.nn.functional.mse_loss(out, labels), "logits": out}


class _Data(torch.utils.data.Dataset):
    def __init__(self, n: int = 64):
        generator = torch.Generator().manual_seed(0)
        self.x, self.y = torch.randn(n, 2, generator=generator), torch.randn(n, 1, generator=generator)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.y[i]}


def _trainer(out_dir, callbacks=()) -> Trainer:
    """A ten-step run with a linear decay (so the LR differs at every step), saving every four steps."""
    torch.manual_seed(0)
    args = TrainingArguments(
        output_dir=str(out_dir), max_steps=10, per_device_train_batch_size=2, learning_rate=0.1,
        lr_scheduler_type="linear", warmup_steps=0, logging_steps=1, save_steps=4, save_strategy="steps",
        report_to="none", use_cpu=True, disable_tqdm=True, seed=0, data_seed=0, dataloader_num_workers=0,
    )  # fmt: skip
    return Trainer(model=_Toy(), args=args, train_dataset=_Data(), callbacks=list(callbacks))


def _lrs(trainer: Trainer) -> list[float]:
    return [h["learning_rate"] for h in trainer.state.log_history if "learning_rate" in h]


def _checkpoints(out_dir) -> list[str]:
    return sorted(p.name for p in out_dir.glob("checkpoint-*"))


def test_the_run_stops_at_the_step_and_trains_the_first_steps_of_the_full_run(tmp_path):
    full = _trainer(tmp_path / "full")
    full.train()
    stopped = _trainer(tmp_path / "stopped", [StopAfterStep(6)])
    stopped.train()

    assert full.state.global_step == 10
    assert stopped.state.global_step == 6
    # Step 6 is not a multiple of save_steps=4: the callback forced the save, and nothing came after it.
    assert _checkpoints(tmp_path / "stopped") == ["checkpoint-4", "checkpoint-6"]
    assert _checkpoints(tmp_path / "full") == ["checkpoint-10", "checkpoint-4", "checkpoint-8"]
    # The LR of every step trained is the full run's: max_steps still sets the schedule.
    assert _lrs(stopped) == _lrs(full)[:6]
    assert len(set(_lrs(stopped))) == 6
    assert torch.load(tmp_path / "stopped" / "checkpoint-6" / "scheduler.pt")["last_epoch"] == 6


def test_a_stop_on_a_save_step_saves_once(tmp_path):
    stopped = _trainer(tmp_path, [StopAfterStep(8)])
    stopped.train()
    assert stopped.state.global_step == 8
    assert _checkpoints(tmp_path) == ["checkpoint-4", "checkpoint-8"]


def test_resuming_the_stopped_run_follows_the_schedule_of_the_full_run(tmp_path):
    full = _trainer(tmp_path / "full")
    full.train()
    _trainer(tmp_path / "stopped", [StopAfterStep(6)]).train()

    resumed = _trainer(tmp_path / "stopped")
    resumed.train(resume_from_checkpoint=str(tmp_path / "stopped" / "checkpoint-6"))

    assert resumed.state.global_step == 10
    assert _lrs(resumed)[-4:] == _lrs(full)[-4:], "steps 7-10 decay exactly as in the run that was never stopped"
