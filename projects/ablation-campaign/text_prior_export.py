"""Export the text decoder of a MELT checkpoint as a plain causal LM.

``melteval text-prior`` (melt-eval, ``02-backbones.md`` §3) scores a reference
string under a bare ``AutoModelForCausalLM``. A MELT checkpoint is not one: its
``state_dict`` is ``text_decoder.*`` (the decoder) plus ``audio_stack.*`` (the
encoder and adapter), and its ``config.json`` declares ``model_type: "melt"``,
which transformers does not know. This script bridges the two so the *same*
pre-IFT measurement can be repeated on an IFT checkpoint and differenced
against the pre-training numbers.

What it writes into ``OUT_DIR``:

- ``model.safetensors``: every ``text_decoder.*`` tensor, prefix stripped, the
  ``audio_stack.*`` tensors dropped. One file, whatever the source layout.
- ``config.json``: the decoder's own config. Taken from the MELT config's
  nested ``text_decoder_config`` when present, else from ``--base-model`` in
  the local HF cache (MELT hosts are offline). ``vocab_size`` always comes from
  the embedding rows, and ``tie_word_embeddings`` from whether an
  ``lm_head.weight`` survived the strip: MELT's Llama and Qwen decoders share
  the embedding matrix and save no ``lm_head``, EuroLLM's does not.
- the tokenizer files, copied verbatim. The vocabulary is *not* trimmed to the
  base model's: MELT appends audio special tokens, and the ids have to keep
  matching the embedding rows.

The checkpoint directory itself (``.../checkpoint-6310``) usually holds only
weights and ``config.json``; the tokenizer lives one level up, in the run
directory. Both locations are searched, the checkpoint first.

Usage::

    python text_prior_export.py CHECKPOINT_DIR OUT_DIR [--base-model ID]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


TEXT_DECODER_PREFIX = "text_decoder."
EMBEDDING_SUFFIXES = ("model.embed_tokens.weight", "transformer.wte.weight")
LM_HEAD_KEY = "lm_head.weight"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)


def _source_shards(checkpoint_dir: Path) -> dict[str, Path]:
    """Map every tensor name in the checkpoint to the safetensors file holding it.

    Handles both layouts HF Trainer writes: a single ``model.safetensors`` and
    a sharded pair with ``model.safetensors.index.json``.

    Args:
        checkpoint_dir: MELT checkpoint directory.

    Returns:
        ``{tensor_name: shard_path}`` covering the whole checkpoint.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return {key: checkpoint_dir / shard for key, shard in weight_map.items()}

    single = checkpoint_dir / "model.safetensors"
    if not single.exists():
        found = sorted(p.name for p in checkpoint_dir.glob("*.safetensors"))
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {checkpoint_dir}; found {found}"
        )
    with safe_open(single, framework="pt", device="cpu") as handle:
        return dict.fromkeys(handle.keys(), single)


def _decoder_config(checkpoint_dir: Path, base_model: str | None, vocab_size: int, tied: bool) -> dict:
    """Build the config of the exported decoder.

    Args:
        checkpoint_dir: MELT checkpoint directory (its ``config.json`` carries
            the decoder config under ``text_decoder_config``).
        base_model: HF id to read the decoder config from when the MELT config
            has none. Resolved from the local cache only.
        vocab_size: Rows in the exported embedding matrix.
        tied: Whether the exported decoder has no separate ``lm_head``.

    Returns:
        A config dict transformers' ``AutoConfig`` accepts.
    """
    melt_config_path = checkpoint_dir / "config.json"
    config: dict | None = None
    if melt_config_path.exists():
        config = json.loads(melt_config_path.read_text(encoding="utf-8")).get("text_decoder_config")
    if not config:
        if base_model is None:
            raise ValueError(
                f"{melt_config_path} has no text_decoder_config; pass --base-model to read it from the local cache"
            )
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base_model, local_files_only=True).to_dict()

    config = dict(config)
    config["vocab_size"] = vocab_size
    config["tie_word_embeddings"] = tied
    return config


def _tokenizer_source(checkpoint_dir: Path) -> Path:
    """Directory to copy the tokenizer files from: the checkpoint, else its run dir.

    Args:
        checkpoint_dir: MELT checkpoint directory.

    Returns:
        The first of the checkpoint and its parent that holds ``tokenizer.json``.

    Raises:
        FileNotFoundError: If neither does.
    """
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        if (candidate / "tokenizer.json").exists():
            return candidate
    raise FileNotFoundError(
        f"No tokenizer.json in {checkpoint_dir} or {checkpoint_dir.parent} "
        "(the exporter needs the tokenizer the decoder was trained with)"
    )


def export(checkpoint_dir: Path, out_dir: Path, base_model: str | None = None) -> dict:
    """Write a plain causal-LM copy of *checkpoint_dir*'s text decoder.

    Args:
        checkpoint_dir: MELT checkpoint directory.
        out_dir: Destination directory (created if missing).
        base_model: Fallback source of the decoder config.

    Returns:
        A summary dict: kept/dropped key counts, vocab size, parameter count and
        the tie decision.
    """
    shards = _source_shards(checkpoint_dir)
    kept = {key: shard for key, shard in shards.items() if key.startswith(TEXT_DECODER_PREFIX)}
    if not kept:
        raise ValueError(f"No {TEXT_DECODER_PREFIX}* tensors in {checkpoint_dir}")

    tensors: dict = {}
    for key, shard in kept.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors[key[len(TEXT_DECODER_PREFIX) :]] = handle.get_tensor(key)

    embedding = next((tensor for name, tensor in tensors.items() if name.endswith(EMBEDDING_SUFFIXES)), None)
    if embedding is None:
        raise ValueError(f"No embedding matrix among the exported keys of {checkpoint_dir}")
    vocab_size = int(embedding.shape[0])
    tied = LM_HEAD_KEY not in tensors

    config = _decoder_config(checkpoint_dir, base_model, vocab_size, tied)
    tokenizer_dir = _tokenizer_source(checkpoint_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_dir / "model.safetensors", metadata={"format": "pt"})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    copied = []
    for name in TOKENIZER_FILES:
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
            copied.append(name)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "out_dir": str(out_dir),
        "keys_kept": len(tensors),
        "keys_dropped": len(shards) - len(kept),
        "vocab_size": vocab_size,
        "num_parameters": sum(tensor.numel() for tensor in tensors.values()),
        "tie_word_embeddings": tied,
        "tokenizer_files": copied,
        "tokenizer_from": str(tokenizer_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, export, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_dir", type=Path, help="MELT checkpoint directory.")
    parser.add_argument("out_dir", type=Path, help="Destination directory.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF id to read the decoder config from when the MELT config lacks text_decoder_config.",
    )
    args = parser.parse_args(argv)

    summary = export(args.checkpoint_dir, args.out_dir, args.base_model)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
