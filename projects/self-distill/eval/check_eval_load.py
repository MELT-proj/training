"""Load a checkpoint the way melt-eval's ``melt`` provider does and report what was loaded.

SFT checkpoints (``checkpoint-N``) keep fp32 master weights; melt-eval's provider loads
with ``dtype="bfloat16"`` and ``attn_implementation="flash_attention_2"`` by default. This
builds the model through the provider's own factory, with the same arguments the
battery passes, and checks, against the checkpoint's ``model.safetensors``:

- every parameter is bfloat16 and no tensor is missing or unexpected;
- every loaded tensor equals the stored tensor rounded to bfloat16 (what the trainer's
  bf16 compute sees);
- the attention implementation of the top-level model, the decoder and the encoder.

With ``--reference <dir>`` it also counts how many stored tensors differ from another
checkpoint's (a warm start), so a checkpoint that is only a copy of its start shows up.

Needs a GPU for flash attention: run it through ``sbatch`` in the melt-eval venv with a
training checkout first on PYTHONPATH, like the evaluation itself:

    PYTHONPATH=<training checkout>:<melt-eval> python check_eval_load.py <checkpoint dir> \\
        --processor <dir with tokenizer and processor files> [--reference <dir>]

Exit status 1 when a check fails.
"""

import argparse
import sys
from collections import Counter

import torch
from safetensors import safe_open


def component(name: str) -> str:
    """``encoder``, ``adapter`` or ``decoder`` for a parameter name (``other`` for anything else)."""
    if name.startswith("audio_stack.encoder."):
        return "encoder"
    if name.startswith("audio_stack.adapter."):
        return "adapter"
    if name.startswith("text_decoder."):
        return "decoder"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--processor", required=True, help="directory with the tokenizer and processor files")
    p.add_argument("--reference", default=None, help="another checkpoint directory to compare the stored tensors with")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    from melteval.providers.router import create_model

    # The battery's -M arguments: processor and batch_size; dtype and attention are the provider's defaults.
    api = create_model(args.checkpoint, processor=args.processor, batch_size=args.batch_size)
    model = api.model
    failures: list[str] = []

    print(f"checkpoint: {args.checkpoint}")
    print(f"model.dtype: {model.dtype} | device: {next(model.parameters()).device}")
    for attr, label in (
        ("config", "top-level"),
        ("text_decoder.config", "decoder"),
        ("audio_stack.encoder.model.config", "encoder"),
    ):
        node = model
        for part in attr.split("."):
            node = getattr(node, part)
        impl = getattr(node, "_attn_implementation", None)
        print(f"attention implementation ({label}): {impl}")
        if label != "top-level" and impl != "flash_attention_2":
            failures.append(f"{label} attention is {impl}, not flash_attention_2")

    by_component: dict[str, Counter] = {}
    for name, param in model.named_parameters():
        by_component.setdefault(component(name), Counter())[str(param.dtype)] += param.numel()
    for comp, counts in sorted(by_component.items()):
        print(f"parameters, {comp}: " + ", ".join(f"{dt} {n / 1e6:.1f} M" for dt, n in counts.items()))
        if set(counts) != {"torch.bfloat16"}:
            failures.append(f"{comp} parameters are not all bfloat16: {dict(counts)}")

    state = model.state_dict()
    stored_dtypes: Counter = Counter()
    mismatched: list[str] = []
    seen: set[str] = set()
    with safe_open(f"{args.checkpoint}/model.safetensors", "pt") as f:
        for key in list(f.keys()):
            stored = f.get_tensor(key)
            stored_dtypes[str(stored.dtype)] += 1
            seen.add(key)
            if key not in state:
                failures.append(f"stored tensor {key} is not in the model")
                continue
            expected = stored.to(torch.bfloat16) if stored.is_floating_point() else stored
            if not torch.equal(state[key].cpu(), expected):
                mismatched.append(key)
    absent = sorted(set(state) - seen)
    print(f"stored tensors: {len(seen)} {dict(stored_dtypes)}")
    print(f"loaded tensors equal to the stored ones rounded to bfloat16: {len(seen) - len(mismatched)} of {len(seen)}")
    if mismatched:
        failures.append(f"{len(mismatched)} tensors differ from the file, e.g. {mismatched[:3]}")
    # Tied weights (lm_head sharing the embedding) are in the model but not in the file.
    print(f"model entries not in the file: {len(absent)} {absent[:4]}")

    if args.reference:
        same, differ = Counter(), Counter()
        with (
            safe_open(f"{args.checkpoint}/model.safetensors", "pt") as f,
            safe_open(f"{args.reference}/model.safetensors", "pt") as g,
        ):
            reference_keys = set(g.keys())
            for key in list(f.keys()):
                if key not in reference_keys:
                    differ[component(key)] += 1
                    continue
                a, b = f.get_tensor(key), g.get_tensor(key)
                a = a.to(torch.bfloat16) if a.is_floating_point() else a
                b = b.to(torch.bfloat16) if b.is_floating_point() else b
                (same if a.shape == b.shape and torch.equal(a, b) else differ)[component(key)] += 1
        for comp in sorted(set(same) | set(differ)):
            print(
                f"vs reference {args.reference.rstrip('/').rsplit('/', 1)[-1]}, {comp}: {same[comp]} tensors equal, {differ[comp]} differ"
            )

    print(
        "FAIL: " + "; ".join(failures)
        if failures
        else "OK: bfloat16 weights equal the stored fp32 tensors rounded to bfloat16"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
