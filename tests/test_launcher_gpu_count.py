"""The launcher's GPU-count resolution.

These drive the real block out of `bash/run_train.sh` rather than a
reimplementation of it, so the test fails if the shipped logic changes.
"""

import re
import subprocess
from pathlib import Path

import pytest

RUN_TRAIN = Path(__file__).resolve().parents[1] / "bash" / "run_train.sh"

# The contiguous block that resolves GPUS_PER_NODE, lifted verbatim.
BLOCK_START = "_GPUS_PER_NODE_EXPLICIT="
BLOCK_END = "WORLD_SIZE="


def _gpu_block() -> str:
    text = RUN_TRAIN.read_text()
    start = text.index(BLOCK_START)
    end = text.index(BLOCK_END, start)
    return text[start:end]


def _resolve(env: dict[str, str], under_slurm: int) -> int:
    """Run the shipped block with a controlled environment; return GPUS_PER_NODE."""
    script = (
        f"RUNNING_UNDER_SLURM={under_slurm}\n"
        # Keep nvidia-smi out of the picture: these cases never reach it, and a
        # real GPU on the test host would otherwise leak into the result.
        "nvidia-smi() { return 1; }\n"
        "command() { return 1; }\n"
        f"{_gpu_block()}\n"
        'echo "$GPUS_PER_NODE"\n'
    )
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


def test_an_explicit_request_beats_the_nodes_total():
    """MN5 allocates `acc` nodes whole, so SLURM_GPUS_ON_NODE reports 4 even for
    a --gpus-per-node=2 job. The explicit request has to win: world_size is baked
    into a resumed lhotse sampler's partitioning, and a mismatch aborts the run."""
    assert _resolve({"GPUS_PER_NODE": "2", "SLURM_GPUS_ON_NODE": "4"}, under_slurm=1) == 2


def test_slurm_is_still_trusted_when_nothing_was_pinned():
    assert _resolve({"SLURM_GPUS_ON_NODE": "4"}, under_slurm=1) == 4


def test_it_falls_back_to_one_with_no_signal_at_all():
    assert _resolve({}, under_slurm=1) == 1


def test_an_explicit_request_survives_outside_slurm():
    assert _resolve({"GPUS_PER_NODE": "3"}, under_slurm=0) == 3


@pytest.mark.parametrize("requested", ["1", "2", "4", "8"])
def test_any_pinned_value_is_passed_through_untouched(requested):
    got = _resolve(
        {"GPUS_PER_NODE": requested, "SLURM_GPUS_ON_NODE": "4"}, under_slurm=1
    )
    assert got == int(requested)


def test_the_sbatch_wrapper_forwards_a_pinned_count():
    """run_train.sh only sees GPUS_PER_NODE if the sbatch wrapper forwards it
    through --cleanenv."""
    sbatch = (RUN_TRAIN.parent / "run_train_singularity.sbatch").read_text()
    assert re.search(
        r'MELT_GPUS_PER_NODE.*\n?.*container_env GPUS_PER_NODE "\$\{MELT_GPUS_PER_NODE\}"',
        sbatch,
    ), "sbatch must forward MELT_GPUS_PER_NODE into the container as GPUS_PER_NODE"
