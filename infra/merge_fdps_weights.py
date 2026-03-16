import argparse

import accelerate


def main(
    ckpt_dir, output_path, safe_serialization=True, remove_checkpoint_dir: bool = False
):
    accelerate.utils.merge_fsdp_weights(
        ckpt_dir,
        output_path,
        safe_serialization=safe_serialization,
        remove_checkpoint_dir=remove_checkpoint_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge FSDP weights into a single file."
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Path to the directory containing the FSDP checkpoint files.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the merged checkpoint file will be saved.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to use safe serialization (default: True).",
    )
    parser.add_argument(
        "--remove_checkpoint_dir",
        action="store_true",
        help="Whether to remove the original checkpoint directory after merging (default: False).",
    )
    args = parser.parse_args()
    main(
        args.ckpt_dir,
        args.output_path,
        safe_serialization=args.safe_serialization,
        remove_checkpoint_dir=args.remove_checkpoint_dir,
    )
