"""Tests for projects/ablation-campaign/text_prior_export.py.

The exporter turns a MELT checkpoint (``text_decoder.*`` plus
``audio_stack.*``) into the plain causal LM ``melteval text-prior`` can load.
A silent mistake here would not raise anywhere: the eval would happily score
a truncated or renamed decoder. So the test checks the mapping tensor by
tensor, not just the key count.

CPU only, synthetic 4x4 tensors: no checkpoint, no hub, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import text_prior_export as tpe  # noqa: E402


def _tensor(seed: int, shape: tuple[int, int] = (4, 4)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator)


def _write_checkpoint(
    directory: Path, with_lm_head: bool = False, tokenizer_in_parent: bool = False
) -> dict[str, torch.Tensor]:
    """Write a minimal MELT checkpoint and return its text_decoder tensors."""
    directory.mkdir(parents=True, exist_ok=True)
    decoder = {
        "model.embed_tokens.weight": _tensor(0, (12, 4)),
        "model.layers.0.self_attn.q_proj.weight": _tensor(1),
        "model.norm.weight": _tensor(2),
    }
    if with_lm_head:
        decoder["lm_head.weight"] = _tensor(3, (12, 4))
    state = {f"text_decoder.{key}": value for key, value in decoder.items()}
    state["audio_stack.encoder.conv1.weight"] = _tensor(4)
    state["audio_stack.adapter.experts.0.down_proj.weight"] = _tensor(5)
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})

    config = {
        "model_type": "melt",
        "text_decoder_config": {"model_type": "llama", "vocab_size": 12, "hidden_size": 4},
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tokenizer_dir = directory.parent if tokenizer_in_parent else directory
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (tokenizer_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    return decoder


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def test_export_strips_prefix_and_drops_audio_stack(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    expected = _write_checkpoint(checkpoint)

    out = tmp_path / "export"
    summary = tpe.export(checkpoint, out)

    with safe_open(out / "model.safetensors", framework="pt") as handle:
        written = set(handle.keys())
    assert written == set(expected)
    assert not any(key.startswith(("text_decoder.", "audio_stack.")) for key in written)
    assert summary["keys_kept"] == len(expected)
    assert summary["keys_dropped"] == 2


def test_exported_tensors_are_byte_identical(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    expected = _write_checkpoint(checkpoint)
    out = tmp_path / "export"
    tpe.export(checkpoint, out)

    with safe_open(out / "model.safetensors", framework="pt") as handle:
        for name, tensor in expected.items():
            assert handle.get_tensor(name).shape == tensor.shape
            assert _hash(handle.get_tensor(name)) == _hash(tensor)


def test_vocab_size_comes_from_the_embedding_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    _write_checkpoint(checkpoint)
    # The config says 12 but the embedding says otherwise; rows win.
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_decoder_config"]["vocab_size"] = 999
    config_path.write_text(json.dumps(config), encoding="utf-8")

    out = tmp_path / "export"
    summary = tpe.export(checkpoint, out)

    written = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert written["vocab_size"] == 12
    assert summary["vocab_size"] == 12


def test_no_lm_head_means_tied_embeddings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    _write_checkpoint(checkpoint, with_lm_head=False)
    out = tmp_path / "export"
    summary = tpe.export(checkpoint, out)

    assert summary["tie_word_embeddings"] is True
    assert json.loads((out / "config.json").read_text())["tie_word_embeddings"] is True


def test_lm_head_means_untied_embeddings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    _write_checkpoint(checkpoint, with_lm_head=True)
    out = tmp_path / "export"
    summary = tpe.export(checkpoint, out)

    assert summary["tie_word_embeddings"] is False
    assert json.loads((out / "config.json").read_text())["tie_word_embeddings"] is False
    with safe_open(out / "model.safetensors", framework="pt") as handle:
        assert "lm_head.weight" in set(handle.keys())


def test_tokenizer_falls_back_to_the_run_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    _write_checkpoint(checkpoint, tokenizer_in_parent=True)
    out = tmp_path / "export"
    summary = tpe.export(checkpoint, out)

    assert summary["tokenizer_from"] == str(checkpoint.parent)
    assert (out / "tokenizer.json").exists()
    assert (out / "chat_template.jinja").exists()
    # processor_config.json is deliberately not needed by a text-only export.
    assert not (out / "processor_config.json").exists()


def test_missing_tokenizer_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-10"
    _write_checkpoint(checkpoint)
    (checkpoint / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        tpe.export(checkpoint, tmp_path / "export")
