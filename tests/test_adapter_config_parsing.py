import pytest
from omegaconf import OmegaConf

from src.training.config import get_default_config


def test_adapter_defaults_in_training_config():
    cfg = get_default_config()
    ad = cfg.model.adapter

    assert ad.type == "mlp"
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


def test_config_merge_parses_adapter_fields():
    """Test that OmegaConf merge works correctly for adapter fields."""
    base = get_default_config()
    override = OmegaConf.create(
        {
            "model": {
                "adapter": {
                    "type": "qformer",
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
                }
            }
        }
    )

    cfg = OmegaConf.merge(base, override)
    ad = cfg.model.adapter

    assert ad.type == "qformer"
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
