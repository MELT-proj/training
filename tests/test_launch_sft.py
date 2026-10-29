"""projects/self-distill/sft/launch_sft.sh: the no-MA start and the stop-at-step option.

These run the shipped script with DRY_RUN=1, which prints the plan and the command and submits nothing, so
the tests fail if the script changes. Both options are off unless asked for, and a run that does not ask
for them has to get the command it always got.
"""

import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "projects" / "self-distill" / "sft" / "launch_sft.sh"


def _dry_run(**env: str) -> subprocess.CompletedProcess:
    """Run the launcher in DRY_RUN on a clean environment holding only *env* (plus the shell's own)."""
    clean = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL")}
    clean |= {"DRY_RUN": "1", "MELT_TIME": "01:00:00", "MIX": "eu24"} | env
    return subprocess.run(["bash", str(LAUNCHER)], env=clean, capture_output=True, text=True)


def _command(result: subprocess.CompletedProcess) -> list[str]:
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("[DRY_RUN] would run:"))
    return shlex.split(line.split("would run:", 1)[1])


def _without_flag(command: list[str], flag: str) -> list[str]:
    """*command* without *flag* and its value."""
    i = command.index(flag)
    return command[:i] + command[i + 2 :]


def test_init_none_builds_the_model_without_a_checkpoint_and_names_the_run_noma():
    result = _dry_run(INIT="none")
    command = _command(result)
    assert "--model.ckpt" not in command
    assert "--run.exp_name" in command
    assert command[command.index("--run.exp_name") + 1] == "SFT-EU24-from-noma-bd60-ga4-8g-s1337"
    assert "init:  none" in result.stdout


def test_init_none_differs_from_a_warm_start_only_in_the_checkpoint_and_the_name():
    """What makes the no-MA arm a comparison: the data, the schedule, the seeds and the batches are the same."""
    warm, none = _command(_dry_run(INIT="goldw")), _command(_dry_run(INIT="none"))
    warm = _without_flag(warm, "--model.ckpt")
    rename = lambda tokens: [t.replace("from-goldw", "from-X").replace("from-noma", "from-X") for t in tokens]  # noqa: E731
    assert rename(warm) == rename(none)


def test_the_seed_of_a_replicate_changes_the_name_and_the_three_seeds():
    command = _command(_dry_run(INIT="none", SEED="1338"))
    assert command[command.index("--run.exp_name") + 1] == "SFT-EU24-from-noma-bd60-ga4-8g-s1338"
    for flag in ("--trainer.seed", "--data.train_ds.seed", "--data.train_ds.shard_seed"):
        assert command[command.index(flag) + 1] == "1338"


def test_an_unknown_init_is_still_refused():
    result = _dry_run(INIT="bogus")
    assert result.returncode != 0
    assert "unknown INIT=bogus (goldw|sd|none)" in result.stderr


def test_stop_after_step_adds_one_flag_and_nothing_else():
    plain = _command(_dry_run(INIT="goldw"))
    stopped = _command(_dry_run(INIT="goldw", STOP_AFTER_STEP="2872"))
    assert stopped[-2:] == ["--run.stop_after_step", "2872"]
    assert stopped[:-2] == plain


def test_without_stop_after_step_nothing_about_it_is_printed_or_passed():
    """Run A's command, as its log recorded it, must come out unchanged."""
    result = _dry_run(INIT="goldw")
    assert "stop_after_step" not in result.stdout
    assert "stop:" not in result.stdout
    assert result.returncode == 0


def test_the_stop_flag_comes_before_extra_arguments_so_they_can_still_override_it():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--run.stop_after_step", "100"],
        env={**{k: v for k, v in os.environ.items() if k in ("PATH", "HOME")},
             "DRY_RUN": "1", "MELT_TIME": "01:00:00", "MIX": "eu24", "INIT": "goldw", "STOP_AFTER_STEP": "2872"},
        capture_output=True, text=True,
    )  # fmt: skip
    command = _command(result)
    last = len(command) - 1 - command[::-1].index("--run.stop_after_step")
    assert command[last + 1] == "100"


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "2.5", "11500"])
def test_a_bad_stop_after_step_is_refused_before_anything_is_queued(bad):
    result = _dry_run(INIT="none", STOP_AFTER_STEP=bad)
    assert result.returncode != 0
    assert "STOP_AFTER_STEP" in result.stderr
