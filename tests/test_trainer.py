from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from transformers.trainer_utils import IntervalStrategy

from melt.modeling import MELTConfig, MELTForCausalLM
from melt.training.trainer import MELTTrainer, _validate_eval_batch_size


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


# ---------------------------------------------------------------------------
# eval batch size validation
# ---------------------------------------------------------------------------


def _eval_bs_args(**overrides):
    args = SimpleNamespace(
        per_device_eval_batch_size=-1,
        do_eval=True,
        eval_on_start=False,
        eval_strategy="steps",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_eval_batch_size_sentinel_rejected_when_evaluating():
    """The train-side -1 sentinel crashes HF's evaluation_loop; catch it early.

    Left alone it surfaces as "Trying to create tensor with negative dimension
    -1" from ``losses.repeat(batch_size)``, at the first eval rather than at
    startup.
    """
    with pytest.raises(ValueError, match="per_device_eval_batch_size"):
        _validate_eval_batch_size(_eval_bs_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"do_eval": True},
        {"do_eval": False, "eval_on_start": True, "eval_strategy": "no"},
        {"do_eval": False, "eval_strategy": "epoch"},
    ],
)
def test_eval_batch_size_sentinel_rejected_however_eval_is_enabled(overrides):
    with pytest.raises(ValueError):
        _validate_eval_batch_size(_eval_bs_args(**overrides))


def test_eval_batch_size_sentinel_allowed_when_eval_is_off():
    """No eval, no eval batch to size — don't block a train-only run."""
    _validate_eval_batch_size(
        _eval_bs_args(do_eval=False, eval_on_start=False, eval_strategy="no")
    )


def test_eval_batch_size_enum_strategy_is_understood():
    """`eval_strategy` reaches us as an IntervalStrategy, not a bare string."""
    with pytest.raises(ValueError):
        _validate_eval_batch_size(
            _eval_bs_args(do_eval=False, eval_strategy=IntervalStrategy.STEPS)
        )
    _validate_eval_batch_size(
        _eval_bs_args(
            do_eval=False, eval_on_start=False, eval_strategy=IntervalStrategy.NO
        )
    )


def test_positive_eval_batch_size_accepted():
    _validate_eval_batch_size(_eval_bs_args(per_device_eval_batch_size=4))


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


# ---------------------------------------------------------------------------
# Reproducible model initialisation
# ---------------------------------------------------------------------------


def _adapter_state(model):
    return {k: v.clone() for k, v in model.audio_stack.adapter.state_dict().items()}


def test_model_init_is_reproducible_under_set_seed():
    """Two models built from the same config must be identical.

    The adapter is randomly initialised. HF only calls set_seed inside
    Trainer.__init__, which runs long after train.py builds the model, so
    reproducible init depends on train.py seeding first.
    """
    from transformers import set_seed

    set_seed(42)
    first = _adapter_state(_make_minimal_model())

    set_seed(42)
    second = _adapter_state(_make_minimal_model())

    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), f"adapter param {key} differs"

        
def test_model_init_differs_without_reseeding():
    """Guards the test above against silently passing on a constant init.

    If the adapter were initialised deterministically for some other reason,
    the reproducibility test would pass while proving nothing.
    """
    from transformers import set_seed

    set_seed(42)
    first = _adapter_state(_make_minimal_model())
    second = _adapter_state(_make_minimal_model())  # no reseed in between

    assert any(
        not torch.equal(first[key], second[key]) for key in first
    ), "adapter init appears constant; the reproducibility test proves nothing"
    
    
def test_default_eval_workers_is_nonzero():
    """The packaged default must actually use workers.

    dataloader_num_workers is read only by the eval path, so a zero here makes
    every shipped config evaluate single-process. Kept as a test because the
    value is easy to reset while tuning and hard to notice afterwards.
    """
    from melt.training.config import get_default_config

    assert get_default_config().trainer.dataloader_num_workers >= 1


# ---------------------------------------------------------------------------
# Dataloader teardown (issue #63)
# ---------------------------------------------------------------------------


class _Countdown(torch.utils.data.IterableDataset):
    """Endless trivial source — stands in for the repeating Lhotse train stream."""

    def __iter__(self):
        i = 0
        while True:
            yield i
            i += 1


def _make_teardown_trainer(train_loader=None):
    with patch.object(MELTTrainer, "__init__", lambda self, **kwargs: None):
        trainer = MELTTrainer.__new__(MELTTrainer)
    trainer._global_rank = 0
    trainer._lhotse_resume_from = None
    trainer._train_dataloader_ref = train_loader
    trainer._prepared_eval_dataloaders = {}
    return trainer


def test_shutdown_dataloaders_stops_stateful_workers():
    """A finished run must leave no dataloader worker or pin-memory thread alive.

    Leaving the iterator for __del__ is what makes the interpreter abort at exit
    with "cannot join current thread" and turns a completed run into a FAILED
    SLURM job (issue #63).
    """
    from torchdata.stateful_dataloader import StatefulDataLoader

    loader = StatefulDataLoader(
        _Countdown(),
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
    )
    it = iter(loader)
    next(it)
    del it

    assert loader._iterator is not None, "loader should be holding a live iterator"
    workers = list(loader._iterator._workers)
    assert any(w.is_alive() for w in workers)

    trainer = _make_teardown_trainer(loader)
    trainer._shutdown_dataloaders()

    assert loader._iterator is None
    assert trainer._train_dataloader_ref is None
    for w in workers:
        w.join(timeout=10)
        assert not w.is_alive(), "worker survived the explicit shutdown"


def test_shutdown_dataloaders_is_safe_on_idle_and_repeated_calls():
    """Must not resurrect workers or raise when there is nothing to shut down."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    loader = StatefulDataLoader(_Countdown(), batch_size=None, num_workers=0)
    trainer = _make_teardown_trainer(loader)

    trainer._shutdown_dataloaders()  # never iterated
    assert loader._iterator is None

    trainer._train_dataloader_ref = loader
    trainer._shutdown_dataloaders()  # again, still fine
    assert trainer._train_dataloader_ref is None


def test_shutdown_dataloaders_clears_cached_eval_loaders():
    """Cached eval loaders pin memory too, so they can strand a pin thread."""
    trainer = _make_teardown_trainer()
    eval_loader = torch.utils.data.DataLoader(_TinyEvalDataset(), batch_size=2)
    trainer._prepared_eval_dataloaders = {"eval": (object(), eval_loader)}

    trainer._shutdown_dataloaders()

    assert trainer._prepared_eval_dataloaders == {}


def test_train_shuts_dataloaders_down_even_when_training_raises():
    """The teardown is in a finally: a crashing run must not also strand workers."""
    from transformers import Trainer

    trainer = _make_teardown_trainer(train_loader=None)
    calls = []
    trainer._shutdown_dataloaders = lambda: calls.append(True)

    with patch.object(Trainer, "train", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trainer.train()

    assert calls == [True], "train() must shut the dataloaders down on the error path"
