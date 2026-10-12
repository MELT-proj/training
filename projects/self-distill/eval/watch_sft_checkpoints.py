"""Watch the SFT runs for scheduled checkpoints and score each one on artemis as it lands.

Run on nyx, where the checkpoints are (or arrive from MN5, see ``pull_mn5_checkpoints.sh``):

    python3 watch_sft_checkpoints.py                      # the runs in scoring/runs.txt, until everything is scored
    python3 watch_sft_checkpoints.py --once --dry-run     # what would be submitted now, no ssh
    python3 watch_sft_checkpoints.py --status             # the state file as a table, sacct states included
    python3 watch_sft_checkpoints.py --runs RUN [RUN ...] # these runs instead of the file

The runs come from ``<outputs>/scoring/runs.txt`` (one run directory name per line, ``#`` starts a comment, a
trailing ``mn5`` marks a run whose checkpoints are pulled from MN5), read again at every poll, so a run added to
the file is picked up without a restart. A run name must contain ``-from-goldw-`` or ``-from-sd-azeros-``: that
decides which warm start supplies the tokenizer and processor.

Standard library only. Per run it looks for ``checkpoint-N`` at the scheduled steps (default 718, 1436,
2154, 2872, 4308, 5744, 8616: 1, 2, 3, 4, 6, 8 and 12 sixteenths of 11,499 steps) and for the final
model in the run root. A checkpoint counts as complete only when

- ``trainer_state.json`` parses and its ``global_step`` is N (the trainer writes it after the weights;
  a checkpoint pulled from MN5 carries ``.transfer_complete`` instead), and
- ``model.safetensors`` (or every shard of its index) is as long as its own header says, and
- ``config.json`` and ``generation_config.json`` are there,

and its newest file is at least ``--settle`` seconds old. The final model needs ``training_config.yaml``
(the last file of the final save), the tokenizer and processor files and valid weights.

A complete checkpoint gets one melt-eval job per set (``launch_eval_sft.sh``, copied to ``scoring/bin`` on the
shared scratch at startup and run on artemis over ssh, so no clone has to be synced):
fleurs24-st-xen-dev, fleurs24-st-enx-dev and fleurs24-asr-dev, plus sd-dev-per-source for the final model.
The launcher refuses to put more than ``--cap-st`` / ``--cap-asr`` of the user's jobs in gpu-short / gpu-medium
(running plus pending, whoever submitted them); the watcher then waits for the next poll.

State is kept in ``scoring/state.json``, one entry per (run, step, set) with the job id. A job is
**never submitted twice**: an entry is written the moment the launcher reports a job id, a restart
reads it back, and before every submission round the watcher also rebuilds missing entries from ``squeue``/``sacct``
(the launcher tags every job ``sft-score:<run>:<ckptN|final>:<set>`` in the SLURM comment, which squeue and
sacct keep), so a lost ssh connection or a lost state file cannot cause a second submission. A job that fails on
the cluster is reported, not resubmitted: delete its entry from the state file to retry. A submission that
errors before any job id exists is retried at most ``MAX_ATTEMPTS`` times. One watcher at a time (a lock
file); the prompt format file is pinned by its sha256 so every checkpoint is scored under the same prompts.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


OUTPUTS = Path("/mnt/scratch-nyx/giuseppe/melt/outputs-sft")
WARM_STARTS = {
    "goldw": "GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g",
    "sd-azeros": "S3-azeros_anchor-from-GOLDW-final-mix05-bd30-lr2e4-3000st-s43-2g",
}
DEFAULT_STEPS = (718, 1436, 2154, 2872, 4308, 5744, 8616)
SETS = ("fleurs24-st-xen-dev", "fleurs24-st-enx-dev", "fleurs24-asr-dev")
FINAL_ONLY_SETS = ("sd-dev-per-source",)
MAX_ATTEMPTS = 5
MAX_POLL_FAILURES = 12  # polls in a row that may fail before the watcher gives up (an hour at the default interval)
QOS_FULL = 3  # the launcher's exit status when a QOS has no free slot
TRANSIENT = 255  # ssh's own exit status for a connection that failed; also what a timeout is mapped to
TERMINAL_BAD = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")


@dataclass(frozen=True)
class Target:
    """One model to score: a scheduled checkpoint or the final model of a run."""

    run: str
    step: int | None  # None: the final model in the run root
    path: Path

    @property
    def label(self) -> str:
        return "final" if self.step is None else f"ckpt{self.step}"

    @property
    def sets(self) -> tuple[str, ...]:
        return SETS + (FINAL_ONLY_SETS if self.step is None else ())

    def key(self, set_name: str) -> str:
        return f"{self.run}|{self.label}|{set_name}"

    def comment(self, set_name: str) -> str:
        return f"sft-score:{self.run}:{self.label}:{set_name}"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str, path: Path | None = None) -> None:
    line = f"{utcnow()} {message}"
    print(line, flush=True)
    if path is not None:
        with open(path, "a") as f:
            f.write(line + "\n")


# --- completeness ----------------------------------------------------------------------------------------


def safetensors_complete(path: Path) -> bool:
    """True when the file is exactly as long as its own header says: 8 + header size + the last tensor's end."""
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len > 100_000_000:
                return False
            header = json.loads(f.read(header_len))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return path.stat().st_size == 8 + header_len + end
    except (OSError, ValueError, struct.error, KeyError):
        return False


def weights_complete(directory: Path) -> bool:
    """A single ``model.safetensors`` or every shard named by ``model.safetensors.index.json`` is whole."""
    single = directory / "model.safetensors"
    if single.is_file():
        return safetensors_complete(single)
    index = directory / "model.safetensors.index.json"
    try:
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    except (OSError, ValueError, KeyError):
        return False
    return bool(shards) and all(safetensors_complete(directory / shard) for shard in shards)


def checkpoint_complete(directory: Path, step: int, settle: float, now: float | None = None) -> tuple[bool, str]:
    """``(complete, why not)`` for ``checkpoint-<step>``."""
    if not directory.is_dir():
        return False, "absent"
    marker = directory / ".transfer_complete"
    state = directory / "trainer_state.json"
    if state.is_file():
        try:
            if json.loads(state.read_text()).get("global_step") != step:
                return False, "trainer_state.json names another step"
        except (OSError, ValueError):
            return False, "trainer_state.json unreadable (still being written?)"
        stamp = state
    elif marker.is_file():
        stamp = marker
    else:
        return False, "no trainer_state.json yet"
    for name in ("config.json", "generation_config.json"):
        if not (directory / name).is_file():
            return False, f"no {name}"
    if not weights_complete(directory):
        return False, "weights incomplete"
    age = (now if now is not None else time.time()) - stamp.stat().st_mtime
    if age < settle:
        return False, f"settling ({age:.0f}s < {settle:.0f}s)"
    return True, ""


def final_complete(run_dir: Path, settle: float, now: float | None = None) -> tuple[bool, str]:
    """``(complete, why not)`` for the final model in the run root."""
    for name in ("training_config.yaml", "config.json", "tokenizer.json", "processor_config.json"):
        if not (run_dir / name).is_file():
            return False, f"no {name}"
    if not weights_complete(run_dir):
        return False, "weights incomplete"
    age = (now if now is not None else time.time()) - (run_dir / "training_config.yaml").stat().st_mtime
    if age < settle:
        return False, f"settling ({age:.0f}s < {settle:.0f}s)"
    return True, ""


def is_complete(target: Target, settle: float, now: float | None = None) -> tuple[bool, str]:
    if target.step is None:
        return final_complete(target.path, settle, now)
    return checkpoint_complete(target.path, target.step, settle, now)


# --- runs and state ---------------------------------------------------------------------------------------


def targets_for(run: str, outputs: Path, steps: tuple[int, ...]) -> list[Target]:
    """The scheduled checkpoints of ``run`` in step order, then its final model."""
    run_dir = outputs / run
    return [Target(run, s, run_dir / f"checkpoint-{s}") for s in steps] + [Target(run, None, run_dir)]


def warm_start_of(run: str, outputs: Path) -> Path:
    """The directory holding the tokenizer and processor files the run started from."""
    match = re.search(r"-from-(goldw|sd-azeros)-", run)
    if not match:
        raise ValueError(f"cannot tell the warm start of {run!r}: expected '-from-goldw-' or '-from-sd-azeros-'")
    return outputs / WARM_STARTS[match.group(1)]


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "jobs": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def pending_sets(target: Target, state: dict) -> list[str]:
    """The sets of ``target`` with no state entry (a submitted job, or a submission given up on)."""
    return [s for s in target.sets if target.key(s) not in state["jobs"]]


def plan(
    targets: list[Target], state: dict, settle: float, now: float | None = None
) -> tuple[list[tuple[Target, list[str]]], dict[str, str]]:
    """Which complete targets still have sets to submit, in the order to submit them, and why the others wait."""
    todo, waiting = [], {}
    for target in targets:
        sets = pending_sets(target, state)
        if not sets:
            continue
        ok, why = is_complete(target, settle, now)
        if ok:
            todo.append((target, sets))
        else:
            waiting[f"{target.run}|{target.label}"] = why
    return todo, waiting


def read_runs_file(path: Path) -> list[tuple[str, str]]:
    """``(run directory name, site)`` for each line of the runs file; site is ``mn5`` or empty.

    A line is ``<run directory name> [mn5]``; blank lines and ``#`` comments are skipped.
    """
    runs = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        fields = raw.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) > 2 or (len(fields) == 2 and fields[1] != "mn5"):
            raise ValueError(f"{path}:{number}: expected '<run directory name> [mn5]', got {raw!r}")
        runs.append((fields[0], fields[1] if len(fields) == 2 else ""))
    return runs


def current_runs(args: argparse.Namespace) -> tuple[str, ...]:
    """The runs to watch: ``--runs`` if given, else the runs file (read afresh)."""
    if args.runs:
        return tuple(args.runs)
    if not args.runs_file.is_file():
        raise SystemExit(
            f"no runs to watch: pass --runs or create {args.runs_file} with one run directory name per line"
        )
    runs = tuple(name for name, _ in read_runs_file(args.runs_file))
    if not runs:
        raise SystemExit(f"{args.runs_file} lists no runs")
    return runs


# --- talking to artemis -----------------------------------------------------------------------------------


def ssh(host: str, remote: str, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run ``remote`` on ``host`` without ever prompting. A timeout comes back as ``TRANSIENT`` (ssh's own status
    for a connection that failed), never as an exception: a poll must survive a slow login node."""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, remote]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, TRANSIENT, "", f"ssh to {host} timed out after {timeout:.0f} s")


def parse_submitted(text: str) -> dict[str, int]:
    """``{comment: job id}`` from the launcher's ``SUBMITTED <comment> <job id>`` lines."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^SUBMITTED (\S+) (\d+)$", text, re.MULTILINE)}


def run_launcher(
    args: argparse.Namespace, target: Target, sets: list[str], processor: Path, dry_run: bool
) -> tuple[int, str]:
    """Run the launcher for ``sets`` of ``target``: on artemis over ssh, or here with ``--ssh-host ''``.

    A dry run executes the repository copy locally with DRY_RUN=1: it prints the sbatch commands, submits nothing.
    """
    argv = [str(target.path), str(processor), f"{target.run}:{target.label}", *sets]
    env = {"CAP_ST": str(args.cap_st), "CAP_ASR": str(args.cap_asr), "FORMAT_FILE": str(args.format_installed)}
    if args.time_limit:
        env["TIME_LIMIT"] = args.time_limit
    try:
        if dry_run:
            proc = subprocess.run(
                ["bash", args.launcher, *argv], env={**os.environ, **env, "DRY_RUN": "1"},
                capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
            )  # fmt: skip
        elif args.ssh_host:
            remote = (
                "env "
                + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
                + f" bash {shlex.quote(str(args.launcher_installed))} "
                + shlex.join(argv)
            )
            proc = ssh(args.ssh_host, remote)
        else:
            proc = subprocess.run(
                ["bash", str(args.launcher_installed), *argv], env={**os.environ, **env},
                capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL,
            )  # fmt: skip
    except subprocess.TimeoutExpired:
        return TRANSIENT, "the launcher timed out"
    return proc.returncode, proc.stdout + proc.stderr


def sacct_states(args: argparse.Namespace, job_ids: list[int]) -> dict[int, str]:
    """``{job id: state}`` from sacct on artemis; empty when it cannot be asked."""
    if not job_ids or not args.ssh_host:
        return {}
    proc = ssh(args.ssh_host, f"sacct -X -n -P -o JobID,State -j {','.join(str(j) for j in job_ids)}", timeout=120)
    if proc.returncode != 0:
        return {}
    states = {}
    for line in proc.stdout.splitlines():
        job, _, rest = line.partition("|")
        if job.isdigit() and rest.split():
            states[int(job)] = rest.split()[0]
    return states


def reconcile(args: argparse.Namespace, state: dict, runs: tuple[str, ...], logfile: Path | None) -> int | None:
    """Add to ``state`` the jobs the cluster knows under our comments; returns how many were missing, or None
    when the cluster could not be asked.

    The cluster is the source of truth for "was this submitted": it covers a state file that was lost and an ssh
    connection that dropped after sbatch had run but before its output came back. Slurm keeps a job's comment in
    two places: squeue shows it while the job is pending or running, sacct only once the job has ended (a
    running job has an empty Comment there), so both are read.
    """
    if not args.ssh_host:
        return 0
    remote = (
        f"squeue -h -u \"$(id -un)\" -o '%i|%k|%T'; sacct -X -n -P -S {shlex.quote(args.since)} -o JobID,Comment,State"
    )
    proc = ssh(args.ssh_host, remote, timeout=120)
    if proc.returncode != 0:
        log(f"reconcile: squeue/sacct failed ({proc.returncode}): {proc.stderr.strip()[:200]}", logfile)
        return None
    added = 0
    for line in proc.stdout.splitlines():
        job, _, rest = line.partition("|")
        comment, _, job_state = rest.partition("|")
        match = re.fullmatch(r"sft-score:([^:]+):(ckpt\d+|final):([^:]+)", comment)
        if not (job.isdigit() and match) or match.group(1) not in runs:
            continue
        key = "|".join(match.groups())
        if key not in state["jobs"]:
            state["jobs"][key] = {
                "job": int(job),
                "submitted_at": "?",
                "note": f"rebuilt from squeue/sacct ({job_state})",
            }
            added += 1
            log(f"reconcile: {key} is job {job} ({job_state}); not resubmitting", logfile)
    return added


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_files(args: argparse.Namespace, state: dict, logfile: Path | None) -> None:
    """Put the launcher and the prompt format where artemis jobs can read them (the shared scratch).

    The format file is pinned: once its sha256 is in the state file, a different file is refused, so the
    checkpoints are all scored under the same prompts.
    """
    scoring = args.format_installed.parent
    (scoring / "bin").mkdir(parents=True, exist_ok=True)
    (scoring / "slurm").mkdir(parents=True, exist_ok=True)
    source = Path(args.format_file)
    if not args.format_installed.exists():
        shutil.copyfile(source, args.format_installed)
    digest = sha256(args.format_installed)
    if sha256(source) != digest:
        raise SystemExit(
            f"{source} differs from the installed {args.format_installed}; the prompts must not change mid-experiment"
        )
    pinned = state.setdefault("format_sha256", digest)
    if pinned != digest:
        raise SystemExit(
            f"{args.format_installed} (sha256 {digest[:12]}) is not the file pinned in {args.state} ({pinned[:12]})"
        )
    if not args.launcher_installed.exists() or sha256(args.launcher_installed) != sha256(Path(args.launcher)):
        shutil.copyfile(args.launcher, args.launcher_installed)
        log(f"installed {args.launcher_installed} (sha256 {sha256(args.launcher_installed)[:12]})", logfile)


# --- the loop -------------------------------------------------------------------------------------------


def submit_pending(args: argparse.Namespace, todo: list[tuple[Target, list[str]]], state: dict, logfile: Path) -> bool:
    """Submit what the launcher accepts. Returns False once a QOS is full (the rest waits for the next poll)."""
    for target, sets in todo:
        processor = warm_start_of(target.run, args.outputs)
        code, out = run_launcher(args, target, sets, processor, args.dry_run)
        for line in out.strip().splitlines():
            log(f"  [{target.run} {target.label}] {line}", logfile)
        if args.dry_run:
            continue
        submitted = parse_submitted(out)
        for set_name in sets:
            job = submitted.get(target.comment(set_name))
            if job is not None:
                state["jobs"][target.key(set_name)] = {"job": job, "submitted_at": utcnow(), "model": str(target.path)}
        save_state(args.state, state)
        if code == QOS_FULL:
            log("a QOS is full; waiting for the next poll", logfile)
            return False
        missing = [s for s in sets if target.key(s) not in state["jobs"]]
        if code == TRANSIENT and missing:
            log(
                "could not reach artemis (or it timed out); retrying at the next poll, after checking sacct for jobs that did go through",
                logfile,
            )
            return False
        if code != 0 and missing:
            attempts = state.setdefault("attempts", {})
            for set_name in missing:
                key = target.key(set_name)
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] >= MAX_ATTEMPTS:
                    state["jobs"][key] = {
                        "job": None,
                        "submitted_at": utcnow(),
                        "note": f"gave up after {MAX_ATTEMPTS} failed submissions",
                    }
                    log(
                        f"GAVE UP on {key} after {MAX_ATTEMPTS} failed submissions; edit {args.state} to retry",
                        logfile,
                    )
            save_state(args.state, state)
            log(
                f"launcher exited {code} for {target.run} {target.label}; the missing sets are retried at the next poll",
                logfile,
            )
    return True


# What a job's stdout must not contain: transformers prints a LOAD REPORT when checkpoint keys do not match the
# model (missing ones are then randomly initialised and the run goes on), and a traceback means the run died.
STDOUT_FLAGS = ("LOAD REPORT", "Traceback (most recent call last)")


def scan_stdout(slurm_dir: Path, job: int) -> list[str]:
    """The ``STDOUT_FLAGS`` found in the stdout file(s) of ``job``."""
    found = []
    for path in sorted(slurm_dir.glob(f"*.{job}.out")):
        text = path.read_text(errors="replace")
        found += [f"{flag!r} in {path.name}" for flag in STDOUT_FLAGS if flag in text]
    return found


def refresh_job_states(args: argparse.Namespace, state: dict, logfile: Path | None) -> None:
    """Record each open job's sacct state in ``state``; report failures and suspicious stdout once."""
    open_jobs = {
        e["job"]: k
        for k, e in state["jobs"].items()
        if e.get("job") and e.get("state") not in ("COMPLETED", *TERMINAL_BAD)
    }
    for job, job_state in sacct_states(args, list(open_jobs)).items():
        key = open_jobs.get(job)
        if key is None:
            continue
        previous = state["jobs"][key].get("state")
        state["jobs"][key]["state"] = job_state
        if job_state == previous:
            continue
        if job_state.startswith(TERMINAL_BAD):
            log(
                f"JOB {job} ({key}) ended {job_state}; not resubmitting, delete its entry from {args.state} to retry",
                logfile,
            )
        elif job_state == "COMPLETED":
            log(f"job {job} ({key}) completed", logfile)
        if job_state == "COMPLETED" or job_state.startswith(TERMINAL_BAD):
            flags = scan_stdout(args.format_installed.parent / "slurm", job)
            if flags:
                state["jobs"][key]["stdout_flags"] = flags
                log(
                    f"WARNING job {job} ({key}): {'; '.join(flags)}. Read the log before trusting this score.", logfile
                )


def all_done(targets_by_run: dict[str, list[Target]], state: dict) -> bool:
    return all(
        state["jobs"].get(t.key(s), {}).get("state") == "COMPLETED"
        for targets in targets_by_run.values()
        for t in targets
        for s in t.sets
    )


def show_status(args: argparse.Namespace, steps: tuple[int, ...]) -> None:
    """Print what is on disk and what was submitted. Read-only: the state file is never written from here."""
    state = load_state(args.state)
    refresh_job_states(args, state, None)
    for run in current_runs(args):
        print(run)
        for target in targets_for(run, args.outputs, steps):
            ok, why = is_complete(target, args.settle)
            cells = []
            for s in target.sets:
                entry = state["jobs"].get(target.key(s))
                short = s.replace("fleurs24-", "").replace("-dev", "")
                cells.append(f"{short}: " + (f"{entry['job']} {entry.get('state', 'submitted')}" if entry else "-"))
            print(f"  {target.label:9s} {'on disk' if ok else why:34s} " + " | ".join(cells))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runs", nargs="+", default=None, help="run directory names under --outputs (default: --runs-file)"
    )
    p.add_argument("--runs-file", type=Path, default=None, help="default: <outputs>/scoring/runs.txt")
    p.add_argument("--steps", default=",".join(str(s) for s in DEFAULT_STEPS), help="checkpoint steps to score")
    p.add_argument("--outputs", type=Path, default=OUTPUTS)
    p.add_argument("--state", type=Path, default=None, help="default: <outputs>/scoring/state.json")
    p.add_argument("--log", type=Path, default=None, help="default: <outputs>/scoring/watcher.log")
    p.add_argument("--launcher", default=str(here / "launch_eval_sft.sh"), help="the launcher to install for artemis")
    p.add_argument("--format-file", default=str(here / "format-sft-random.yaml"), help="the prompt format to pin")
    p.add_argument("--ssh-host", default="artemis", help="where the launcher runs; '' runs it here")
    p.add_argument("--interval", type=float, default=300, help="seconds between polls")
    p.add_argument("--settle", type=float, default=120, help="seconds a checkpoint must be quiet before it is scored")
    p.add_argument(
        "--cap-st", type=int, default=3, help="most of the user's jobs in gpu-short (limit 4: one slot stays free)"
    )
    p.add_argument(
        "--cap-asr", type=int, default=2, help="most of the user's jobs in gpu-medium (limit 3: one slot stays free)"
    )
    p.add_argument("--time-limit", default="", help="per-job wall time passed to the launcher (its default: 02:00:00)")
    p.add_argument("--since", default="2026-10-12", help="reconcile with the sacct jobs submitted since this date")
    p.add_argument("--once", action="store_true", help="one poll, then exit")
    p.add_argument(
        "--dry-run", action="store_true", help="print the launcher's sbatch commands; no ssh, no state change"
    )
    p.add_argument("--status", action="store_true", help="print the runs' checkpoints and jobs, then exit")
    args = p.parse_args(argv)
    scoring = args.outputs / "scoring"
    args.runs_file = args.runs_file or scoring / "runs.txt"
    args.state = args.state or scoring / "state.json"
    args.log = args.log or scoring / "watcher.log"
    args.format_installed = scoring / "format-sft-random.yaml"
    args.launcher_installed = scoring / "bin" / "launch_eval_sft.sh"
    return args


def usable_runs(args: argparse.Namespace, logfile: Path | None, reported: set[str]) -> tuple[str, ...]:
    """The current runs whose warm start can be found; a run that cannot be mapped is reported once and skipped."""
    runs = []
    for run in current_runs(args):
        try:
            warm = warm_start_of(run, args.outputs)
            if not (warm / "tokenizer.json").is_file():
                raise ValueError(f"{warm} (the warm start of {run}) has no tokenizer.json")
        except ValueError as err:
            if run not in reported:
                reported.add(run)
                log(f"IGNORING {run}: {err}", logfile)
            continue
        runs.append(run)
    return tuple(runs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = tuple(int(s) for s in args.steps.split(","))
    for run in current_runs(args):  # a typo in a run name fails here, before anything is started
        try:
            warm = warm_start_of(run, args.outputs)
        except ValueError as err:
            raise SystemExit(str(err)) from err
        if not (warm / "tokenizer.json").is_file():
            raise SystemExit(f"{warm} (the warm start of {run}) has no tokenizer.json")
    if args.status:
        show_status(args, steps)
        return 0

    live = not args.dry_run
    if live:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        lock = open(args.state.with_name("watcher.lock"), "w")  # noqa: SIM115 - held for the life of the process
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another watcher holds {lock.name}; exiting", file=sys.stderr)
            return 2
        lock.write(f"{os.getpid()}\n")
        lock.flush()

    stop: list[int] = []
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop.append(1)) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return poll_loop(args, steps, stop, live)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


class Watcher:
    """What the polls share: the state file's contents, the run list and its targets, what was already reported."""

    def __init__(self, args: argparse.Namespace, steps: tuple[int, ...], live: bool):
        self.args, self.steps, self.live = args, steps, live
        self.logfile = args.log if live else None
        self.state = load_state(args.state)
        self.runs: tuple[str, ...] = ()
        self.targets_by_run: dict[str, list[Target]] = {}
        self.ignored: set[str] = set()
        self.last_waiting: str | None = None
        if live:
            install_files(args, self.state, args.log)
            save_state(args.state, self.state)
        elif not args.format_installed.exists():
            args.format_installed = Path(
                args.format_file
            )  # a dry run before the first start prints the repository copy

    def poll(self) -> bool:
        """One poll: refresh the run list, submit what is complete and not yet submitted, record the jobs' states.

        Returns True when every scheduled checkpoint and the final model of every run is scored.
        """
        args, state, logfile = self.args, self.state, self.logfile
        try:
            now_runs = usable_runs(args, logfile, self.ignored)
        except SystemExit as err:  # the runs file vanished or emptied while we were running: keep what we had
            now_runs = self.runs
            log(f"keeping the previous run list: {err}", logfile)
        if now_runs != self.runs:
            self.runs = now_runs
            self.targets_by_run = {run: targets_for(run, args.outputs, self.steps) for run in self.runs}
            steps = ",".join(map(str, self.steps))
            log(f"watching {', '.join(self.runs)} for steps {steps} and the final model; state {args.state}", logfile)
        all_targets = [t for targets in self.targets_by_run.values() for t in targets]
        todo, waiting = plan(all_targets, state, args.settle)
        if todo and self.live:
            added = reconcile(args, state, self.runs, args.log)
            if added is None:  # without the cluster's word on what is already submitted, submit nothing
                log("cannot ask the cluster what is already submitted; submitting nothing this round", args.log)
                todo = []
            elif added:
                save_state(args.state, state)
                todo, waiting = plan(all_targets, state, args.settle)
        # Lockstep: the same step of both runs before the next step of either.
        todo.sort(key=lambda item: (item[0].step is None, item[0].step or 0, item[0].run))
        if todo:
            log("to submit: " + ", ".join(f"{t.run} {t.label} ({len(s)} sets)" for t, s in todo), logfile)
            submit_pending(args, todo, state, logfile)
        else:
            summary = ", ".join(f"{k} ({v})" for k, v in waiting.items()) or "-"
            if summary != self.last_waiting or args.once or args.dry_run:  # a long wait is one line, not one per poll
                log("nothing to submit; waiting on: " + summary, logfile)
            self.last_waiting = summary
        if self.live:
            refresh_job_states(args, state, args.log)
            save_state(args.state, state)
            if self.runs and all_done(self.targets_by_run, state):
                log("every scheduled checkpoint and the final model of every run is scored; exiting", args.log)
                return True
        return False


def poll_loop(args: argparse.Namespace, steps: tuple[int, ...], stop: list[int], live: bool) -> int:
    """Poll until everything is scored (live), one pass (``--once``, ``--dry-run``) or a signal."""
    watcher = Watcher(args, steps, live)
    failures = 0
    while not stop:
        try:
            if watcher.poll():
                return 0
            failures = 0
        except Exception as err:  # noqa: BLE001 - a multi-day process must survive a glitch (a stale NFS handle, say)
            if args.once or args.dry_run:
                raise
            failures += 1
            log(f"ERROR in a poll ({failures} in a row): {type(err).__name__}: {err}", watcher.logfile)
            if failures >= MAX_POLL_FAILURES:
                log(f"giving up after {failures} failed polls in a row", watcher.logfile)
                return 1
        if args.once or args.dry_run:
            break
        deadline = time.monotonic() + args.interval
        while not stop and time.monotonic() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
