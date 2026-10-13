"""``model.master_dtype`` / ``model.master_dtype_scope``: which parameters get fp32 masters (``train.cast_master_weights``)."""

import pytest
import torch

from melt.training.train import cast_master_weights


def _backbone_and_adapter() -> torch.nn.Module:
    """A frozen bf16 'backbone' and a trainable 'adapter', as a loaded adapter-only run has them."""
    model = torch.nn.ModuleDict({"backbone": torch.nn.Linear(4, 4), "adapter": torch.nn.Linear(4, 4)}).to(
        torch.bfloat16
    )
    for param in model["backbone"].parameters():
        param.requires_grad = False
    return model


def test_scope_all_casts_every_parameter():
    model = cast_master_weights(_backbone_and_adapter(), "float32")
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_scope_trainable_casts_only_what_an_optimizer_updates():
    model = cast_master_weights(_backbone_and_adapter(), "float32", "trainable")

    assert {p.dtype for p in model["adapter"].parameters()} == {torch.float32}
    assert {p.dtype for p in model["backbone"].parameters()} == {torch.bfloat16}
    assert all(p.requires_grad for p in model["adapter"].parameters())  # still the same trainable Parameters


def test_the_cast_trainable_parameters_still_train_under_autocast():
    model = cast_master_weights(_backbone_and_adapter(), "float32", "trainable")
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model["adapter"](model["backbone"](x)).float().pow(2).mean()
    loss.backward()
    assert model["adapter"].weight.grad is not None and model["adapter"].weight.grad.dtype == torch.float32
    assert model["backbone"].weight.grad is None


@pytest.mark.parametrize("dtype", ["int8", "not_a_dtype", "bool"])
def test_a_non_floating_point_dtype_is_refused(dtype):
    with pytest.raises(ValueError, match="floating-point"):
        cast_master_weights(_backbone_and_adapter(), dtype)


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="'all' or 'trainable'"):
        cast_master_weights(_backbone_and_adapter(), "float32", "adapter")
