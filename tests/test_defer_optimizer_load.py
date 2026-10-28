"""run.defer_optimizer_load: a resume that loads the optimizer state at its first optimizer step ends where a plain resume ends."""

from types import SimpleNamespace

import torch

from melt.training.callbacks import StopAfterStep
from melt.training.config import get_default_config
from melt.training.trainer import load_before_first_step
from transformers import Trainer, TrainingArguments


def test_the_option_is_off_by_default():
    assert get_default_config().run.defer_optimizer_load is False


def _optimizer() -> tuple[torch.nn.Parameter, torch.optim.AdamW]:
    p = torch.nn.Parameter(torch.ones(3))
    return p, torch.optim.AdamW([p], lr=0.1)


def _step(p: torch.nn.Parameter, opt: torch.optim.Optimizer) -> None:
    p.grad = torch.ones_like(p)
    opt.step()


def test_the_load_runs_once_at_the_first_step_before_its_update():
    p, opt = _optimizer()
    seen = []
    load_before_first_step(opt, lambda: seen.append(p.detach().clone()))
    assert seen == [], "nothing is loaded when the hook is registered"
    _step(p, opt)
    assert len(seen) == 1
    assert torch.equal(seen[0], torch.ones(3)), "it ran ahead of the update"
    assert not torch.equal(p.detach(), torch.ones(3))
    _step(p, opt)
    assert len(seen) == 1, "later steps do not load again"


def test_the_load_sees_a_wrapped_optimizer():
    """accelerate hands the Trainer a wrapper whose ``optimizer`` attribute is the torch one."""
    p, opt = _optimizer()
    seen = []
    load_before_first_step(SimpleNamespace(optimizer=opt), lambda: seen.append(True))
    _step(p, opt)
    assert seen == [True]


def test_what_the_load_puts_in_the_optimizer_is_what_the_first_step_uses():
    """A learning rate set by the load, as ``Optimizer.load_state_dict`` sets the parameter groups', is the one the step applies."""
    p, opt = _optimizer()

    def load():
        opt.param_groups[0]["lr"] = 0.5

    load_before_first_step(opt, load)
    _step(p, opt)
    # AdamW's first step moves each weight by lr (Adam's bias-corrected update is +-1 on the first step) after decoupled decay.
    assert torch.allclose(p.detach(), torch.full((3,), 1.0 * (1 - 0.5 * 0.01) - 0.5))


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


class _Deferred(Trainer):
    """MELTTrainer's override of ``_load_optimizer_and_scheduler``, on the plain Trainer."""

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            super()._load_optimizer_and_scheduler(checkpoint)
            return
        load_before_first_step(
            self.optimizer, lambda: super(_Deferred, self)._load_optimizer_and_scheduler(checkpoint)
        )


def _trainer(out_dir, cls=Trainer, callbacks=()) -> Trainer:
    """A ten-step AdamW run with a linear decay (the LR differs at every step), saving every four steps."""
    torch.manual_seed(0)
    args = TrainingArguments(
        output_dir=str(out_dir), max_steps=10, per_device_train_batch_size=2, learning_rate=0.1,
        lr_scheduler_type="linear", warmup_steps=0, logging_steps=1, save_steps=4, save_strategy="steps",
        report_to="none", use_cpu=True, disable_tqdm=True, seed=0, data_seed=0, dataloader_num_workers=0,
        gradient_accumulation_steps=2,
    )  # fmt: skip
    return cls(model=_Toy(), args=args, train_dataset=_Data(), callbacks=list(callbacks))


def _lrs(trainer: Trainer) -> list[float]:
    return [h["learning_rate"] for h in trainer.state.log_history if "learning_rate" in h]


def test_a_deferred_resume_ends_where_a_plain_resume_ends(tmp_path):
    """Same checkpoint, resumed both ways: the losses, the learning rates and the final weights are the same, bit for bit."""
    _trainer(tmp_path / "stopped", callbacks=[StopAfterStep(6)]).train()
    checkpoint = str(tmp_path / "stopped" / "checkpoint-6")
    # Two copies of the checkpoint, so the second resume does not start from what the first one wrote around it.
    for name in ("plain", "deferred"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "checkpoint-6").symlink_to(checkpoint)

    plain = _trainer(tmp_path / "plain")
    plain.train(resume_from_checkpoint=str(tmp_path / "plain" / "checkpoint-6"))
    deferred = _trainer(tmp_path / "deferred", _Deferred)
    deferred.train(resume_from_checkpoint=str(tmp_path / "deferred" / "checkpoint-6"))

    assert plain.state.global_step == deferred.state.global_step == 10
    assert _lrs(deferred) == _lrs(plain)
    assert [h["loss"] for h in deferred.state.log_history if "loss" in h] == [
        h["loss"] for h in plain.state.log_history if "loss" in h
    ]
    for (name, a), (_, b) in zip(plain.model.named_parameters(), deferred.model.named_parameters()):
        assert torch.equal(a, b), f"{name} differs"
