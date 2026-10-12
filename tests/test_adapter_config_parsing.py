import pytest
from omegaconf import OmegaConf

from melt.training.config import get_default_config


def test_adapter_defaults_in_training_config():
    cfg = get_default_config()
    ad = cfg.model.adapter

    assert ad._type == "mlp"
    assert ad.freeze is False
    assert ad.hidden_size == 1024
    assert ad.num_hidden_layers == 2
    assert ad.intermediate_size == 4096
    assert ad.hidden_act == "gelu"
    assert ad.dropout == pytest.approx(0.1)
    assert ad.downsample_rate == 5
    assert ad.window_size == 15
    assert ad.num_adapter_layers == 1
    assert ad.layerdrop == pytest.approx(0.0)
    assert ad.adapter_kernel_size == 3
    assert ad.adapter_stride == 2
    assert ad.mlp_hidden_size is None
    assert ad.stack_factor == 1
    assert ad.qk_norm is False
    assert ad.num_experts == 8
    assert ad.num_experts_per_tok == 2
    assert ad.moe_intermediate_size == 1024
    assert ad.use_shared_expert is False
    assert ad.shared_expert_intermediate_size == 1024
    assert ad.router_aux_loss_coef == pytest.approx(0.01)


def test_qk_norm_is_off_unless_the_config_says_so():
    """QK-norm changes the function an attention adapter computes, so a config written before
    the option existed (every earlier checkpoint) must come back with it off."""
    from melt.modeling import MELTAdapterConfig

    assert MELTAdapterConfig().qk_norm is False
    old = MELTAdapterConfig(_type="conformer").to_dict()
    old.pop("qk_norm")
    assert MELTAdapterConfig.from_dict(old).qk_norm is False


def test_qk_norm_is_saved_with_the_adapter_config():
    from melt.modeling import MELTAdapterConfig

    saved = MELTAdapterConfig(_type="qformer", qk_norm=True).to_dict()

    assert saved["qk_norm"] is True
    assert MELTAdapterConfig.from_dict(saved).qk_norm is True


@pytest.mark.parametrize("value", ["true", 1, None])
def test_qk_norm_must_be_a_bool(value):
    """A typo such as a quoted "true" must not silently build the plain adapter."""
    from melt.modeling import MELTAdapterConfig

    with pytest.raises(ValueError, match="qk_norm"):
        MELTAdapterConfig(qk_norm=value)


def test_qk_norm_only_counts_when_it_is_exactly_true():
    from unittest.mock import MagicMock

    from melt.modeling.modeling_melt import _qk_norm_enabled

    assert _qk_norm_enabled(None) is False
    assert _qk_norm_enabled(MagicMock()) is False  # a mock config has every attribute
    assert _qk_norm_enabled(MagicMock(qk_norm=True)) is True
    assert _qk_norm_enabled(MagicMock(qk_norm=False)) is False


def test_config_merge_parses_adapter_fields():
    """Test that OmegaConf merge works correctly for adapter fields."""
    base = get_default_config()
    override = OmegaConf.create(
        {
            "model": {
                "adapter": {
                    "_type": "qformer",
                    "freeze": True,
                    "hidden_size": 512,
                    "num_hidden_layers": 4,
                    "intermediate_size": 2048,
                    "hidden_act": "relu",
                    "dropout": 0.2,
                    "downsample_rate": 7,
                    "window_size": 21,
                    "num_adapter_layers": 2,
                    "layerdrop": 0.15,
                    "adapter_kernel_size": 5,
                    "adapter_stride": 3,
                    "mlp_hidden_size": 256,
                    "stack_factor": 4,
                    "qk_norm": True,
                    "num_experts": 16,
                    "num_experts_per_tok": 4,
                    "moe_intermediate_size": 2048,
                    "use_shared_expert": True,
                    "shared_expert_intermediate_size": 4096,
                    "router_aux_loss_coef": 0.02,
                }
            }
        }
    )

    cfg = OmegaConf.merge(base, override)
    ad = cfg.model.adapter

    assert ad._type == "qformer"
    assert ad.freeze is True
    assert ad.hidden_size == 512
    assert ad.num_hidden_layers == 4
    assert ad.intermediate_size == 2048
    assert ad.hidden_act == "relu"
    assert ad.dropout == pytest.approx(0.2)
    assert ad.downsample_rate == 7
    assert ad.window_size == 21
    assert ad.num_adapter_layers == 2
    assert ad.layerdrop == pytest.approx(0.15)
    assert ad.adapter_kernel_size == 5
    assert ad.adapter_stride == 3
    assert ad.mlp_hidden_size == 256
    assert ad.stack_factor == 4
    assert ad.qk_norm is True
    assert ad.num_experts == 16
    assert ad.num_experts_per_tok == 4
    assert ad.moe_intermediate_size == 2048
    assert ad.use_shared_expert is True
    assert ad.shared_expert_intermediate_size == 4096
    assert ad.router_aux_loss_coef == pytest.approx(0.02)
