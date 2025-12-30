import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from transformers import AutoConfig


# Ensure top-level `ddp` import used by src.trainer resolves during tests
sys.modules["ddp"] = importlib.import_module("src.ddp")

from src.melt import MELTConfig, MELTForConditionalGeneration
from src.trainer import MELTTrainer


def _make_minimal_model():
    audio_config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
    text_config = AutoConfig.from_pretrained("gpt2")
    audio_config.num_hidden_layers = 1
    text_config.n_layer = 1

    config = MELTConfig(
        audio_encoder_config=audio_config,
        text_decoder_config=text_config,
        adapter_type="mlp",
    )
    config.audio_bos_token_id = 100

    model = MELTForConditionalGeneration(config)
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
        trainer.config = None
        trainer._global_rank = 0
        trainer._world_size = 1

        trainer.create_optimizer()

        # Check adapter params are frozen
        adapter = model.audio_stack.adapter
        assert adapter is not None
        assert all(not p.requires_grad for p in adapter.parameters())

        # Check decoder params are frozen via freeze_decoder
        assert all(not p.requires_grad for p in model.text_decoder.parameters())

        # Optimizer exists and param groups correspond to encoder params only
        assert hasattr(trainer, "optimizer")
        # collect ids of encoder params
        enc_params = {id(p) for p in model.audio_stack.encoder.parameters()}
        grouped_params = [id(p) for g in trainer.optimizer.param_groups for p in g["params"]]
        assert any(p_id in enc_params for p_id in grouped_params)
