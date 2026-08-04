"""Merge sharded FSDP weights into a single checkpoint.

Usage:
    python utils/merge_fsdp_weight.py \\
        --checkpoint_dir /path/to/fsdp/shards \\
        --output_path /path/to/output \\
        [--no_safe_serialization] \\
        [--remove_checkpoint_dir]
"""

import argparse

from accelerate.utils import merge_fsdp_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded FSDP model weights into a single combined checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the sharded FSDP checkpoints.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "Path where the merged checkpoint will be saved. "
            "The merged weights are written to {output_path}/model.safetensors "
            "(or pytorch_model.bin when --no_safe_serialization is set)."
        ),
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        default=False,
        help="Save as pytorch_model.bin instead of safetensors (not recommended).",
    )
    parser.add_argument(
        "--remove_checkpoint_dir",
        action="store_true",
        default=False,
        help="Delete the checkpoint directory after a successful merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    safe_serialization = not args.no_safe_serialization
    output_file = (
        f"{args.output_path}/model.safetensors"
        if safe_serialization
        else f"{args.output_path}/pytorch_model.bin"
    )

    print(f"Merging FSDP shards from : {args.checkpoint_dir}")
    print(f"Output path              : {args.output_path}")
    print(f"Safe serialization       : {safe_serialization}")
    print(f"Remove checkpoint dir    : {args.remove_checkpoint_dir}")

    merge_fsdp_weights(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        safe_serialization=safe_serialization,
        remove_checkpoint_dir=args.remove_checkpoint_dir,
    )

    print(f"Done. Merged weights saved to: {output_file}")


if __name__ == "__main__":
    main()
