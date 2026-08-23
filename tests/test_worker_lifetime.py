"""DataLoader workers must not outlive the process that forked them.

Workers are daemonic, which only covers a clean interpreter shutdown. When the
training process is SIGKILLed instead -- OOM killer, scheduler wall clock, a
supervisor escalating past SIGTERM -- nothing joins them and they are reparented
to init holding a full copy of the torch runtime each. See issue #97.

These tests drive the real `_die_with_parent` in a real process tree rather than
mocking prctl, because the whole point is what the *kernel* does when a parent
dies, and a mock cannot tell us that.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is a Linux facility",
)


# The child reports its own pid and what the kernel read back for PDEATHSIG, so
# one subprocess covers both "was the flag armed" and "did it actually fire".
_PARENT_SRC = """
import ctypes, os, sys, time
sys.path.insert(0, {repo!r})
from melt.training.data.audio.lhotse.dataloader import _die_with_parent

pid = os.fork()
if pid == 0:
    {arm}
    got = ctypes.c_int(-1)
    ctypes.CDLL(None).prctl(2, ctypes.byref(got), 0, 0, 0)  # PR_GET_PDEATHSIG
    with open({marker!r}, "w") as f:
        f.write("%d %d" % (os.getpid(), got.value))
        f.flush()
        os.fsync(f.fileno())
    time.sleep(120)
    os._exit(0)
time.sleep(120)
"""


def _is_alive(pid: int) -> bool:
    """True only for a process that still exists and is not a reaped-pending zombie.

    A zombie answers `kill(pid, 0)` just as a live process does, and the child
    here is reparented mid-test, so its reaping is up to whatever inherits it.
    Reading the state field out of /proc is the only way to tell the two apart.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    # ...) S ... -- the state letter follows the comm field's closing paren, and
    # comm itself may contain spaces, so split from the right of that paren.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _spawn_tree(tmp_path: Path, arm: str) -> tuple[subprocess.Popen, int, int]:
    """Start parent -> child, and wait for the child to announce itself."""
    marker = tmp_path / "child.txt"
    src = _PARENT_SRC.format(repo=str(REPO_ROOT), marker=str(marker), arm=arm)
    proc = subprocess.Popen([sys.executable, "-c", src])

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip():
            break
        if proc.poll() is not None:
            raise AssertionError(f"helper parent exited early with {proc.returncode}")
        time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("child never reported its pid")

    child_pid, pdeathsig = (int(x) for x in marker.read_text().split())
    return proc, child_pid, pdeathsig


def _wait_until_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_worker_is_killed_when_its_parent_is_sigkilled(tmp_path):
    proc, child_pid, pdeathsig = _spawn_tree(tmp_path, arm="_die_with_parent()")
    try:
        assert pdeathsig == int(signal.SIGKILL), (
            f"PDEATHSIG read back as {pdeathsig}, expected SIGKILL"
        )
        assert _is_alive(child_pid), "child died before the parent was touched"

        proc.kill()
        proc.wait(timeout=30)

        assert _wait_until_gone(child_pid), (
            f"child {child_pid} outlived the SIGKILLed parent -- this is the "
            "orphaned-worker leak from issue #97"
        )
    finally:
        # Belt and braces: never let this test be the thing that leaks a process.
        if _is_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()


def test_without_the_call_the_child_does_outlive_its_parent(tmp_path):
    """Negative control: without arming, the orphan really does survive.

    Without this the passing test above proves nothing -- a child that exits for
    some unrelated reason would satisfy it just as well.
    """
    proc, child_pid, pdeathsig = _spawn_tree(tmp_path, arm="pass")
    try:
        assert pdeathsig == 0, "expected no PDEATHSIG armed in the control"

        proc.kill()
        proc.wait(timeout=30)

        # Give it the same grace the positive case gets, then expect it *alive*.
        assert not _wait_until_gone(child_pid, timeout=3.0), (
            "control child died on its own, so the positive test above is vacuous"
        )
    finally:
        if _is_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)
            _wait_until_gone(child_pid)
        if proc.poll() is None:
            proc.kill()
