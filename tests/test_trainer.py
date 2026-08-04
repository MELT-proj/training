from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from melt.modeling import MELTConfig, MELTForCausalLM
from melt.training.trainer import MELTTrainer


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

    # Freeze/unfreeze components by toggling requires_grad (matching train.py's _freeze).
    # create_optimizer filters by requires_grad, not by freeze flags.
    for p in model.audio_stack.adapter.parameters():
        p.requires_grad = False  # freeze_adapter=True
    for p in model.text_decoder.parameters():
        p.requires_grad = False  # freeze_decoder=True
    for p in model.audio_stack.encoder.parameters():
        p.requires_grad = True   # freeze_encoder=False

    # Build a fake args object with lr values (freeze flags are no longer read
    # by create_optimizer itself — freezing happens via requires_grad above).
    args = SimpleNamespace(
        adapter_lr=1e-4,
        encoder_lr=1e-5,
        decoder_lr=1e-3,
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
        assert not (adapter_param_ids & optim_param_ids), "Adapter params should be excluded when frozen"
        assert not (decoder_param_ids & optim_param_ids), "Decoder params should be excluded when frozen"

        # Encoder params (not frozen) should be in optimizer groups
        assert enc_param_ids & optim_param_ids, "Encoder params should be in optimizer when not frozen"


# ---------------------------------------------------------------------------
# get_eval_dataloader
# ---------------------------------------------------------------------------


class _TinyEvalDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {"x": idx}


def _make_eval_trainer(**arg_overrides):
    """Build a MELTTrainer shell with only what get_eval_dataloader touches."""
    args = SimpleNamespace(
        per_device_eval_batch_size=4,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)

    with patch.object(MELTTrainer, "__init__", lambda self, **kwargs: None):
        trainer = MELTTrainer.__new__(MELTTrainer)
    trainer.args = args
    trainer.eval_dataset = _TinyEvalDataset()
    trainer._eval_collator = lambda batch: batch
    trainer._prepared_eval_dataloaders = {}
    # prepare_data_loader is identity here: we assert on the DataLoader we built,
    # not on accelerate's distributed wrapping.
    trainer.accelerator = SimpleNamespace(prepare_data_loader=lambda dl: dl)
    return trainer


def test_eval_dataloader_normalizes_batch_size_sentinel():
    """-1 means "batching handled elsewhere"; DataLoader rejects it outright."""
    dl = _make_eval_trainer(per_device_eval_batch_size=-1).get_eval_dataloader()
    assert dl.batch_size == 1


def test_eval_dataloader_zero_workers_has_no_prefetch_factor():
    """prefetch_factor must be None with num_workers=0 or DataLoader raises."""
    dl = _make_eval_trainer(dataloader_num_workers=0).get_eval_dataloader()
    assert dl.num_workers == 0
    assert dl.prefetch_factor is None


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_eval_dataloader_honors_worker_count(workers):
    """Workers used to be clamped to 1, silently ignoring the configured value."""
    dl = _make_eval_trainer(dataloader_num_workers=workers).get_eval_dataloader()
    assert dl.num_workers == workers
    assert dl.prefetch_factor == 8  # default when unset


def test_eval_dataloader_prefetch_factor_override():
    dl = _make_eval_trainer(
        dataloader_num_workers=2, dataloader_prefetch_factor=3
    ).get_eval_dataloader()
    assert dl.prefetch_factor == 3


def test_eval_dataloader_persistent_workers_reuses_loader():
    """evaluate() rebuilds the loader each call; persistence must survive that."""
    trainer = _make_eval_trainer(
        dataloader_num_workers=2, dataloader_persistent_workers=True
    )
    first = trainer.get_eval_dataloader()
    assert first.persistent_workers is True
    assert trainer.get_eval_dataloader() is first


def test_eval_dataloader_not_cached_without_persistent_workers():
    trainer = _make_eval_trainer(dataloader_num_workers=2)
    assert trainer.get_eval_dataloader() is not trainer.get_eval_dataloader()
