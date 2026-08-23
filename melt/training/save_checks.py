"""Post-save verification for training runs.

Under FSDP a run can finish successfully and still leave its output directory
without loadable weights.  ``Trainer.save_model`` writes weights only when the
accelerate plugin's ``state_dict_type`` is ``FULL_STATE_DICT``; with
``SHARDED_STATE_DICT`` (what ``config/accelerate/fsdp2.yaml`` ships) the FSDP
branch matches, the inner condition does not, and the call returns having
written nothing -- no exception, no warning, exit 0.  See issue #91.

The consequence is a directory that looks complete: ``config.json``,
``preprocessor_config.json``, ``chat_template.jinja`` and the tokenizer files
are all written by ``train.py`` itself, independently of the weights.  The
weights stay sharded in ``checkpoint-<step>/pytorch_model_fsdp_0/`` and must be
consolidated with ``utils/merge_fsdp_weight.py`` before the model can be loaded
with ``from_pretrained``.

This module does not change how saving works.  It checks the result and, when
weights are missing, says so loudly and prints the exact command to fix it.
"""

import re
from pathlib import Path

from ..logging_utils import get_logger


logger = get_logger(__name__)

# Globs, not exact names.  ``save_pretrained`` shards past ``max_shard_size``
# (5 GB by default), so a correctly saved multi-billion-parameter model is
# ``model-00001-of-0000N.safetensors`` plus an index -- never ``model.safetensors``.
# Matching the exact name would report a healthy run as broken.
WEIGHT_GLOBS = (
    "model*.safetensors",
    "model*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
)

# Directory accelerate writes sharded FSDP weights into, inside a checkpoint.
FSDP_SHARD_DIRNAME = "pytorch_model_fsdp_0"

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_saved_weights(output_dir: str | Path) -> list[Path]:
    """Return consolidated weight files directly inside ``output_dir``.

    Only the top level is searched: weights inside ``checkpoint-*`` are the
    thing we are checking *for the absence of*, not evidence of a good save.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    found: list[Path] = []
    for pattern in WEIGHT_GLOBS:
        found.extend(p for p in output_dir.glob(pattern) if p.is_file())
    return sorted(set(found))


def find_sharded_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-step ``checkpoint-*`` holding sharded FSDP weights.

    Ordering is by the integer step, not lexicographic: ``checkpoint-6250``
    must beat ``checkpoint-900``.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match is None:
            continue
        if (child / FSDP_SHARD_DIRNAME).is_dir():
            candidates.append((int(match.group(1)), child))

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def format_missing_weights_message(output_dir: str | Path) -> str:
    """Build the operator-facing report for a run that saved no weights."""
    output_dir = Path(output_dir)
    checkpoint = find_sharded_checkpoint(output_dir)

    lines = [
        "=" * 78,
        "NO MODEL WEIGHTS WERE SAVED -- this run's output directory is NOT loadable.",
        "=" * 78,
        f"  output_dir: {output_dir}",
        "",
        "  Under FSDP with fsdp_state_dict_type: SHARDED_STATE_DICT,",
        "  Trainer.save_model() writes no weights and reports no error, so the",
        "  config and tokenizer files above it land as usual and the directory",
        "  looks complete (issue #91).",
        "",
    ]

    if checkpoint is None:
        lines += [
            f"  No checkpoint-*/{FSDP_SHARD_DIRNAME} directory was found either, so",
            "  there is nothing to consolidate from.  Check that save_steps produced",
            "  at least one checkpoint, and that the run wrote to the directory above.",
        ]
    else:
        lines += [
            "  Training itself completed and the sharded weights are intact; only",
            "  the consolidated copy is missing.  Consolidate the last checkpoint",
            "  before loading this run with from_pretrained():",
            "",
            "    python utils/merge_fsdp_weight.py \\",
            f"        --checkpoint_dir {checkpoint / FSDP_SHARD_DIRNAME} \\",
            f"        --output_path {output_dir}",
            "",
            "  Run it as a BATCH job, not on a login node: the merge holds the whole",
            "  model in host RAM and a login node's per-user cgroup will SIGKILL it",
            "  (exit 137) with no useful message.  It needs no GPU -- on MN5,",
            "  `sbatch -p gp -q gp_ehpc` is enough.",
            "",
            "  Paths above are as the training process saw them.  Under a container",
            "  they are container paths; translate to the host bind target first.",
        ]

    lines.append("=" * 78)
    return "\n".join(lines)


def verify_saved_weights(output_dir: str | Path) -> bool:
    """Log whether ``output_dir`` ended up with loadable weights.

    Returns ``True`` when weights are present.  Never raises and never exits:
    the training run really did succeed, so failing the job here would make
    sacct lie about it.  The report is loud instead.
    """
    weights = find_saved_weights(output_dir)
    if weights:
        names = ", ".join(sorted(p.name for p in weights))
        logger.info(f"Verified model weights in {output_dir}: {names}")
        return True

    logger.error(format_missing_weights_message(output_dir))
    return False
