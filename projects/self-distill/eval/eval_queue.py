#!/usr/bin/env python3
"""Throttled submitter for melt-eval jobs on artemis A6000s, then a wait until they all finish.

usage: eval_queue.py JOBS.tsv --state STATE.jsonl [--poll 90] [--dry-run] [--wait-only]

JOBS.tsv, tab separated, '#' starts a comment, one job per line, in priority order:
    run <TAB> model_dir <TAB> label <TAB> frozen_set <TAB> qos1,qos2,... <TAB> time_limit

``run`` and ``label`` become the melt-eval metadata ``sft_run`` / ``sft_label`` through launch_eval_sft.sh
(tag ``run:label``). For each job, in order, the first QOS of its list that has a free slot (this user's
running + pending jobs below the QOS's job cap) takes it; a job whose QOSes are all full waits for the next
cycle, and later jobs may overtake it. Submissions go through the same launcher the dev evals used, so the
prompt format, batch size, dtype and attention implementation are the same; this script only chooses the QOS
and the time limit (clipped to the QOS's maximum wall time).

State: one JSON line per submitted job (run, label, set, job id, qos), so a restart resubmits nothing.
When everything is submitted it waits until every job has left the queue and prints a table of final states.
It never cancels or resubmits: a failed job is reported and left to the operator.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SSH = "artemis"
LAUNCHER = "/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring/bin/launch_eval_sft.sh"
MELT_EVAL = "/mnt/scratch-artemis/giuseppe/melt-proj/melt-eval-pr28"
SACREBLEU = "/mnt/scratch-artemis/giuseppe/sacrebleu"
FORMAT_FILE = "/mnt/scratch-nyx/giuseppe/melt/outputs-sft/scoring/format-repeatfirst.yaml"
CAPS = {"gpu-short": 4, "gpu-medium": 3, "gpu-long": 2}
MAX_WALL_HOURS = {"gpu-short": 4}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def ssh(command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=25",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
    ]
    return subprocess.run(["ssh", *options, SSH, command], capture_output=True, text=True, timeout=timeout)


def in_flight() -> dict[str, int] | None:
    """This user's running + pending jobs per QOS, or None if artemis could not be asked."""
    try:
        out = ssh("squeue -h -u giuseppe -o '%q'")
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    counts = dict.fromkeys(CAPS, 0)
    for qos in out.stdout.split():
        if qos in counts:
            counts[qos] += 1
    return counts


def clip_time(time_limit: str, qos: str) -> str:
    hours, minutes, seconds = (int(x) for x in time_limit.split(":"))
    cap = MAX_WALL_HOURS.get(qos)
    if cap is not None and (hours, minutes, seconds) > (cap, 0, 0):
        return f"{cap:02d}:00:00"
    return time_limit


def submit(job: dict, qos: str, dry_run: bool) -> tuple[int | None, str]:
    env = {
        "SACREBLEU": SACREBLEU,
        "MELT_EVAL": MELT_EVAL,
        "FORMAT_FILE": FORMAT_FILE,
        "QOS_ST": qos,
        "QOS_ASR": qos,
        "CAP_ST": "99",
        "CAP_ASR": "99",
        "TIME_LIMIT": clip_time(job["time"], qos),
    }
    env_part = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    tag = f"{job['run']}:{job['label']}"
    command = f"env {env_part} bash {LAUNCHER} {shlex.quote(job['model'])} {shlex.quote(job['model'])} {shlex.quote(tag)} {job['set']}"
    if dry_run:
        print(f"[{now()}] DRY {qos} {job['time']} {tag} {job['set']}")
        return -1, "dry"
    try:
        out = ssh(command, timeout=300)
    except subprocess.TimeoutExpired:
        return (
            None,
            "ssh to artemis timed out; the job may or may not have been submitted (check squeue before a retry)",
        )
    text = out.stdout + out.stderr
    for line in text.splitlines():
        if line.startswith("SUBMITTED "):
            return int(line.rsplit(" ", 1)[1]), line
    return None, text.strip()[-400:]


def load_jobs(path: Path) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        run, model, label, frozen, qos, time_limit = line.split("\t")
        jobs.append(
            {"run": run, "model": model, "label": label, "set": frozen, "qos": qos.split(","), "time": time_limit}
        )
    return jobs


def key(job: dict) -> str:
    return f"{job['run']}|{job['label']}|{job['set']}"


def wait_for(job_ids: list[int], poll: int) -> None:
    ids = ",".join(str(j) for j in job_ids)
    while True:
        try:
            out = ssh(f"squeue -h -j {ids} -o '%i %T'")
        except subprocess.TimeoutExpired:
            time.sleep(poll)
            continue
        if out.returncode == 0 and not out.stdout.strip():
            break
        states = {}
        for line in out.stdout.splitlines():
            states[line.split()[1]] = states.get(line.split()[1], 0) + 1
        print(f"[{now()}] waiting: {states}", flush=True)
        time.sleep(poll * 4)
    final = ssh(f"sacct -n -P -X -j {ids} --format=JobID,State,Elapsed,Comment%80")
    print(f"[{now()}] all jobs have left the queue; final states (JobID|State|Elapsed|Comment):")
    print(final.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs")
    parser.add_argument("--state", required=True)
    parser.add_argument("--poll", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-only", action="store_true")
    args = parser.parse_args()

    jobs = load_jobs(Path(args.jobs))
    state_path = Path(args.state)
    state = [json.loads(line) for line in state_path.read_text().splitlines()] if state_path.exists() else []
    done = {f"{s['run']}|{s['label']}|{s['set']}": s for s in state}
    failures: dict[str, int] = {}
    print(f"[{now()}] {len(jobs)} jobs in the list, {len(done)} already submitted", flush=True)

    while not args.wait_only:
        todo = [j for j in jobs if key(j) not in done and failures.get(key(j), 0) < 3]
        if not todo:
            break
        counts = in_flight()
        if counts is None:
            print(f"[{now()}] could not read the queue, retrying", flush=True)
            time.sleep(args.poll)
            continue
        placed = 0
        for job in todo:
            qos = next((q for q in job["qos"] if counts[q] < CAPS[q]), None)
            if qos is None:
                continue
            job_id, message = submit(job, qos, args.dry_run)
            if job_id is None:
                # A timed-out submit may have gone through: never retry it automatically (a duplicate wastes GPU time).
                failures[key(job)] = (
                    3 if message.startswith("ssh to artemis timed out") else failures.get(key(job), 0) + 1
                )
                print(f"[{now()}] SUBMIT FAILED ({failures[key(job)]}/3) {key(job)}: {message}", flush=True)
                continue
            counts[qos] += 1
            placed += 1
            record = {
                "run": job["run"],
                "label": job["label"],
                "set": job["set"],
                "job": job_id,
                "qos": qos,
                "at": now(),
            }
            done[key(job)] = record
            if not args.dry_run:
                with state_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            print(
                f"[{now()}] SUBMITTED {job_id} {qos} {job['run'].split('-')[0]}.. {job['set']}  "
                f"(waiting to place: {len(todo) - placed}; in flight {counts})",
                flush=True,
            )
        if args.dry_run:
            break
        time.sleep(args.poll)

    skipped = [key(j) for j in jobs if key(j) not in done]
    if skipped:
        print(f"[{now()}] NOT SUBMITTED after 3 failures: {skipped}", flush=True)
    ids = [int(s["job"]) for s in done.values() if int(s["job"]) > 0]
    if ids and not args.dry_run:
        print(f"[{now()}] everything is submitted ({len(ids)} jobs); waiting for them to finish", flush=True)
        wait_for(ids, args.poll)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
