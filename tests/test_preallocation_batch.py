"""Regression tests for the memory-preallocation warmup batch.

`run.memory_preallocation` builds a synthetic worst-case batch and runs one
forward+backward before the first training step, so an OOM shows up in the
first minute instead of thousands of steps in.

The pass had never run under DDP. `MELTTrainer.training_step` is handed
whatever the training loop wraps the model in, and under DDP that is a
`DistributedDataParallel`, which proxies `forward()` but not attribute access.
Reading `model.config` off it raised `AttributeError` and killed the run on
every rank (MN5 job 45024395, exit 1:0 at step 0). These tests pin the unwrap.
"""

from types import SimpleNamespace

import pytest
import torch

from melt.training.trainer import MELTTrainer


class _FakeDDP(torch.nn.Module):
    """Stands in for DistributedDataParallel.

    The important property is inherited, not written here: an `nn.Module` that
    holds a submodule does NOT forward attribute lookups to it, so `.config`
    raises `AttributeError` exactly as the real wrapper does.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module


def _stub_trainer(max_tokens=400, batch_duration=180.0, max_duration=60.0):
    """The minimum surface `_build_max_length_batch` reads off `self`."""
    return SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(
                train_ds=SimpleNamespace(
                    max_duration=max_duration,
                    batch_duration=batch_duration,
                    batch_size=None,
                    max_tokens=max_tokens,
                )
            )
        ),
        processor=SimpleNamespace(
            feature_extractor=SimpleNamespace(
                hop_length=160, sampling_rate=16_000, stride=2, feature_size=80
            )
        ),
        args=SimpleNamespace(device=torch.device("cpu"), bf16=False, fp16=False),
        accelerator=SimpleNamespace(unwrap_model=lambda m: getattr(m, "module", m)),
        _global_rank=0,
    )


def _build(stub, model, **kw):
    # Unbound call: constructing a real MELTTrainer would need a model, a
    # dataset and an accelerator, none of which this behaviour depends on.
    return MELTTrainer._build_max_length_batch(stub, model, **kw)


@pytest.fixture
def bare_model():
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(audio_token_id=128256)
    return model


def test_wrapped_model_does_not_break_the_warmup_batch(bare_model):
    """The DDP case: this is the exact failure that killed job 45024395."""
    batch = _build(_stub_trainer(), _FakeDDP(bare_model), duration_per_utt=60.0)
    assert (batch["input_ids"][:, 1] == 128256).all()


def test_unwrapped_model_still_works(bare_model):
    """Single-process training passes the bare model; it must keep working."""
    batch = _build(_stub_trainer(), bare_model, duration_per_utt=60.0)
    assert (batch["input_ids"][:, 1] == 128256).all()


def test_the_fake_wrapper_really_hides_config(bare_model):
    """Guard the guard: if this stops raising, the test above proves nothing."""
    with pytest.raises(AttributeError):
        _FakeDDP(bare_model).config  # noqa: B018


def test_batch_shape_follows_duration_and_batch_duration(bare_model):
    """3 utterances of 60 s fill a 180 s batch, at 20 ms output frames."""
    batch = _build(_stub_trainer(), _FakeDDP(bare_model), duration_per_utt=60.0)
    # hop 160 / 16000 * stride 2 = 20 ms -> 60 s = 3000 frames; 80 * 2 = 160 dim.
    assert batch["input_features"].shape == (3, 3000, 160)
    assert batch["features_attention_mask"].shape == (3, 3000)


def test_short_utterances_give_many_items(bare_model):
    """The min_duration pass is the decoder's worst case: most items per batch."""
    batch = _build(_stub_trainer(), _FakeDDP(bare_model), duration_per_utt=0.5)
    assert batch["input_features"].shape[0] == 360  # 180 s / 0.5 s


def test_text_width_follows_max_tokens(bare_model):
    """max_tokens bounds the transcript, so the warmup should reflect it."""
    batch = _build(_stub_trainer(max_tokens=400), _FakeDDP(bare_model),
                   duration_per_utt=60.0)
    assert batch["input_ids"].shape[1] == 432  # 400 + 32 of slack

    unset = _build(_stub_trainer(max_tokens=None), _FakeDDP(bare_model),
                   duration_per_utt=60.0)
    assert unset["input_ids"].shape[1] == 1024


def test_labels_produce_a_finite_loss(bare_model):
    """All -100 would give NaN; exactly one supervised position avoids that."""
    labels = _build(_stub_trainer(), _FakeDDP(bare_model), duration_per_utt=60.0)["labels"]
    assert (labels[:, 0] == 0).all()
    assert (labels[:, 1:] == -100).all()
