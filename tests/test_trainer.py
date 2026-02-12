import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


# Ensure top-level `ddp` import used by src.trainer resolves during tests
sys.modules["ddp"] = importlib.import_module("src.ddp")

from src.modeling import MELTConfig, MELTForCausalLM
from src.training.trainer import MELTTrainer


def _make_minimal_model():
    config = MELTConfig(
        audio_encoder="facebook/wav2vec2-base",
        text_decoder="gpt2",
        adapter_config={"_type": "mlp"},
    )
    config.audio_encoder_config.num_hidden_layers = 1
    config.text_decoder_config.n_layer = 1
    config.audio_bos_token_id = 100

    model = MELTForCausalLM(config)
    return model


def test_create_optimizer_freeze_flags():
    model = _make_minimal_model()

    # Build a fake args object with the flags and lr values
    args = SimpleNamespace(
        adapter_lr=1e-4,
        encoder_lr=1e-5,
        decoder_lr=1e-3,
        freeze_adapter=True,
        freeze_encoder=False,
        freeze_decoder=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=1e-5,
    )

    # Instantiate trainer without running Trainer.__init__ (avoid heavy initialization)
    from unittest.mock import patch

    with patch.object(MELTTrainer, "__init__", lambda self, **kwargs: None):
        trainer = MELTTrainer.__new__(MELTTrainer)
        trainer.model = model
        trainer.args = args
        trainer.config = SimpleNamespace(
            optimization=SimpleNamespace(
                adam_beta1=0.9,
                adam_beta2=0.999,
            )
        )
        trainer._global_rank = 0
        trainer._world_size = 1

        trainer.create_optimizer()

        # Optimizer exists
        assert hasattr(trainer, "optimizer")

        # Collect param id sets for each component
        adapter_param_ids = {id(p) for p in model.audio_stack.adapter.parameters()}
        decoder_param_ids = {id(p) for p in model.text_decoder.parameters()}
        enc_param_ids = {id(p) for p in model.audio_stack.encoder.parameters()}

        # Collect all param ids in optimizer groups
        optim_param_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}

        # Frozen components (adapter, decoder) should NOT be in optimizer groups
        assert not (adapter_param_ids & optim_param_ids), "Adapter params should be excluded when freeze_adapter=True"
        assert not (decoder_param_ids & optim_param_ids), "Decoder params should be excluded when freeze_decoder=True"

        # Encoder params (not frozen) should be in optimizer groups
        assert enc_param_ids & optim_param_ids, "Encoder params should be in optimizer when freeze_encoder=False"
