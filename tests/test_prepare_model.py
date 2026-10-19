"""``model.master_dtype`` in ``melt.training.train.prepare_model``.

transformers 5 loads ``model.ckpt`` with ``dtype="auto"``, which applies the checkpoint's
top-level ``config.dtype`` to the whole model.  A stage-1 checkpoint from a DDP run has a
bf16 decoder (straight off the Hub) and fp32 encoder and adapter, and ``save_pretrained``
records the *first* parameter's dtype -- the decoder's -- so its config says bfloat16.  A
stage-2 run started from it came up all bf16, and DDP, unlike FSDP, never upcasts: AdamW
then updates bf16 weights in place and every update below half a bf16 ulp of its weight is
lost.

These tests build such a checkpoint in-process (no Hub) and push it through the real
``prepare_model``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, Wav2Vec2BertConfig

from melt.modeling import MELTConfig, MELTForCausalLM
from melt.training import train
from melt.training.config import get_default_config, load_config


def _melt_config() -> MELTConfig:
    """A tiny MELTConfig built in-process, so no Hub access (same recipe as test_modeling_melt)."""
    return MELTConfig(
        audio_encoder_config=Wav2Vec2BertConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, feature_projection_input_dim=16, output_hidden_size=32,
        ),
        text_decoder_config=LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            tie_word_embeddings=True,  # as in Llama-3.2-1B: lm_head shares the embedding matrix
        ),
        adapter_config={"_type": "mlp"},
    )


def _bf16_decoder_model(config: MELTConfig | None = None) -> MELTForCausalLM:
    """What a DDP stage-1 run holds: the decoder in the Hub's bf16, encoder and adapter fp32."""
    model = MELTForCausalLM(config or _melt_config(), load_backbones=False)
    model.text_decoder.to(torch.bfloat16)
    return model


def _processor() -> MagicMock:
    """Answers only what prepare_model asks of a processor: a vocab size and a pad id."""
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = 64  # == the decoder's vocab_size, so no embedding resize
    tokenizer.convert_tokens_to_ids.return_value = [0]
    tokenizer.pad_token = "<pad>"
    processor = MagicMock()
    processor.tokenizer = tokenizer
    return processor


def _cfg(ckpt, master_dtype, scope=None):
    cfg = OmegaConf.create(
        {
            "model": {
                "encoder": {"freeze": True},
                "decoder": {"freeze": False},
                "adapter": {"freeze": True},
                "ckpt": None if ckpt is None else str(ckpt),
                "master_dtype": master_dtype,
            },
            "trainer": {},
        }
    )
    if scope is not None:
        cfg.model.master_dtype_scope = scope
    return cfg


def _adapter_only_cfg(ckpt, scope):
    """The adapter-only recipe (self-distillation): the encoder and the decoder are frozen, the adapter trains."""
    cfg = _cfg(ckpt, "float32", scope)
    cfg.model.encoder.freeze = True
    cfg.model.decoder.freeze = True
    cfg.model.adapter.freeze = False
    return cfg


def _targs(tmp_path):
    return SimpleNamespace(resume_from_checkpoint=None, output_dir=str(tmp_path / "out"), do_train=True)


def _dtypes(model) -> set[torch.dtype]:
    return {p.dtype for p in model.parameters()}


@pytest.fixture
def stage1(tmp_path):
    """A checkpoint shaped like a DDP stage-1 run's output, and the state it holds."""
    model = _bf16_decoder_model()
    ckpt = tmp_path / "stage1"
    model.save_pretrained(ckpt)
    return ckpt, {k: v.detach().clone() for k, v in model.state_dict().items()}


def _prepare_from_ckpt(ckpt, master_dtype, tmp_path, monkeypatch):
    processor = _processor()
    monkeypatch.setattr(train, "MELTProcessor", SimpleNamespace(from_pretrained=lambda _path: processor))
    model, *_ = train.prepare_model(_cfg(ckpt, master_dtype), _targs(tmp_path), processor)
    return model


def test_a_ddp_stage1_checkpoint_records_the_decoders_dtype_for_the_whole_model(stage1):
    ckpt, saved = stage1

    assert MELTConfig.from_pretrained(ckpt).dtype == torch.bfloat16
    assert {v.dtype for v in saved.values()} == {torch.bfloat16, torch.float32}


def test_without_master_dtype_the_checkpoint_comes_up_all_bf16(stage1, tmp_path, monkeypatch):
    """Why the key exists.  If transformers stops applying one dtype to the whole model,
    this fails, and the key can be reconsidered."""
    ckpt, _ = stage1

    model = _prepare_from_ckpt(ckpt, None, tmp_path, monkeypatch)

    assert _dtypes(model) == {torch.bfloat16}


def test_master_dtype_float32_loads_the_checkpoint_exactly(stage1, tmp_path, monkeypatch):
    """fp32 everywhere, and the fp32 encoder and adapter are the saved bits, not bf16-rounded ones."""
    ckpt, saved = stage1

    model = _prepare_from_ckpt(ckpt, "float32", tmp_path, monkeypatch)

    assert _dtypes(model) == {torch.float32}
    state = model.state_dict()
    assert state.keys() == saved.keys()
    for name, value in state.items():
        assert torch.equal(value, saved[name].to(torch.float32)), name


def test_from_scratch_master_dtype_widens_the_hub_dtype(tmp_path, monkeypatch):
    processor = _processor()
    monkeypatch.setattr(train, "prepare_melt_config", lambda _cfg, _processor: _melt_config())
    monkeypatch.setattr(train, "MELTForCausalLM", lambda config, load_backbones: _bf16_decoder_model(config))

    kept, *_ = train.prepare_model(_cfg(None, None), _targs(tmp_path), processor)
    widened, *_ = train.prepare_model(_cfg(None, "float32"), _targs(tmp_path), processor)

    assert _dtypes(kept) == {torch.bfloat16, torch.float32}
    assert _dtypes(widened) == {torch.float32}


@pytest.mark.parametrize("bad", ["int8", "bool", "not_a_dtype", "Tensor"])
def test_a_bad_master_dtype_fails_before_the_checkpoint_is_touched(bad, tmp_path):
    # The checkpoint does not exist: reaching it would raise something else.
    with pytest.raises(ValueError, match="master_dtype"):
        train.prepare_model(_cfg(tmp_path / "missing", bad), _targs(tmp_path), _processor())


def test_master_dtype_is_off_by_default_and_settable_from_the_cli():
    assert get_default_config().model.master_dtype is None
    assert load_config(cli_args=["model.master_dtype=float32"]).model.master_dtype == "float32"


# --- model.master_dtype_scope: "trainable" widens only what the optimizer updates -------------------------------------


def test_master_dtype_scope_defaults_to_all_and_is_settable_from_the_cli():
    assert get_default_config().model.master_dtype_scope == "all"
    assert load_config(cli_args=["model.master_dtype_scope=trainable"]).model.master_dtype_scope == "trainable"


def test_scope_all_widens_the_frozen_backbone_too(stage1, tmp_path, monkeypatch):
    ckpt, _ = stage1
    processor = _processor()
    monkeypatch.setattr(train, "MELTProcessor", SimpleNamespace(from_pretrained=lambda _path: processor))

    model, *_ = train.prepare_model(_adapter_only_cfg(ckpt, None), _targs(tmp_path), processor)

    assert _dtypes(model) == {torch.float32}


def test_scope_trainable_widens_only_the_adapter_of_an_adapter_only_run(stage1, tmp_path, monkeypatch):
    """A frozen backbone gains nothing from fp32 masters, and pays a cast of its weights on every use under autocast."""
    ckpt, _ = stage1
    processor = _processor()
    monkeypatch.setattr(train, "MELTProcessor", SimpleNamespace(from_pretrained=lambda _path: processor))

    model, *_ = train.prepare_model(_adapter_only_cfg(ckpt, "trainable"), _targs(tmp_path), processor)

    assert {p.dtype for p in model.audio_stack.adapter.parameters()} == {torch.float32}
    assert {p.dtype for p in model.audio_stack.encoder.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in model.text_decoder.parameters()} == {torch.bfloat16}


def test_from_scratch_scope_trainable_leaves_the_frozen_decoder_in_the_hub_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "prepare_melt_config", lambda _cfg, _processor: _melt_config())
    monkeypatch.setattr(train, "MELTForCausalLM", lambda config, load_backbones: _bf16_decoder_model(config))

    model, *_ = train.prepare_model(_adapter_only_cfg(None, "trainable"), _targs(tmp_path), _processor())

    assert {p.dtype for p in model.audio_stack.adapter.parameters()} == {torch.float32}
    assert {p.dtype for p in model.text_decoder.parameters()} == {torch.bfloat16}


def test_a_bad_master_dtype_scope_fails_before_the_checkpoint_is_touched(tmp_path):
    with pytest.raises(ValueError, match="master_dtype_scope"):
        train.prepare_model(_cfg(tmp_path / "missing", "float32", "adapter"), _targs(tmp_path), _processor())
