#!/usr/bin/env python3
"""Selection-metric jobs for every landed checkpoint of the MA-utility arms (Tier 1, ``plan/04-regime.md``).

A front end for ``eval_queue.py``: it reuses that module's artemis helpers, so the launcher, the prompt format
(``format-repeatfirst.yaml``), batch size, dtype, attention and QOS caps are those of the models scored before. What it adds
is what intermediate checkpoints need:

* a checkpoint is queued once its weights have landed on nyx scratch (``pull_mn5_checkpoints.sh`` writes
  ``.transfer_complete``) and the tokenizer / processor files sit next to them (a ``checkpoint-N`` has none: they are
  copied in from GOLDW-final, whose processor every arm shares);
* the list of targets is re-read against the disk at every cycle, so checkpoints that land later are picked up;
* the order is the information order: the step-2,872 reads of every arm first, then 718, 1,436, 2,154, then T0's later steps.

Per checkpoint: six melt-eval jobs (``selection-fleurs26-dev200`` and ``selection-id-{en,de,es,fr,it}``). Melt-eval
metadata: ``sft_run`` = the run, ``sft_label`` = ``eu24-repeatfirst-ckpt<N>``; ``tier1_selection.py`` reads them back.

    tier1_queue.py [--state STATE.jsonl] [--poll 90] [--dry-run] [--assume-ready]

Like ``eval_queue.py`` it never cancels or resubmits (the state file makes a restart submit nothing twice), and a file named
PAUSE next to the state file stops new submissions until it is removed. Run it where nyx scratch and the ``artemis`` ssh alias
are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import eval_queue as q
from tier1_tables import ARMS, Arm
from watch_sft_checkpoints import OUTPUTS, WARM_STARTS


LABEL = "eu24-repeatfirst"
READS = (718, 1436, 2154, 2872)  # the step-5 schedule's first four saves: the study's readouts
LATE = {"T0": (4308, 5744, 8616, 11499)}  # T0 runs to the end; the others are cut at 2,872
PRIORITY = ("A", "B", "T0", "T0'", "A'", "T1", "T2", "T3", "T4", "T5")
FIRST_READS = (2872, 718)  # the two panels of the figure
TOKENIZER_FILES = ("chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json")
SETS = [
    ("selection-fleurs26-dev200", "03:00:00"),
    *((f"selection-id-{lang}", "01:30:00") for lang in ("en", "de", "es", "fr", "it")),
]
QOS = ["gpu-short", "gpu-medium", "gpu-long"]
BY_KEY = {arm.key: arm for arm in ARMS}


def run_of(arm: Arm) -> str:
    return f"SFT-EU24-from-{arm.init}-bd60-ga4-8g-s{arm.seed}"


def plan() -> list[tuple[Arm, int]]:
    """Every (arm, step) to score, most informative first."""
    order = [(BY_KEY[key], step) for step in FIRST_READS for key in PRIORITY]
    order += [(BY_KEY[key], step) for step in READS if step not in FIRST_READS for key in PRIORITY]
    order += [(BY_KEY[key], step) for step in (11499, 8616, 5744, 4308) for key in LATE if step in LATE[key]]
    return order


def target_dir(arm: Arm, step: int, outputs: Path = OUTPUTS) -> Path:
    return outputs / run_of(arm) / f"checkpoint-{step}"


def label_of(step: int) -> str:
    return f"{LABEL}-ckpt{step}"


def landed(path: Path) -> bool:
    return (path / ".transfer_complete").is_file() and (path / "model.safetensors").is_file()


def assemble(path: Path, source: Path) -> None:
    """Copy the tokenizer / processor files next to the weights, from ``source`` (idempotent, never overwrites)."""
    for name in TOKENIZER_FILES:
        target = path / name
        if not target.exists():
            shutil.copyfile(source / name, target)


def ready(path: Path) -> bool:
    return landed(path) and all(
        (path / n).is_file() for n in ("config.json", "tokenizer.json", "processor_config.json")
    )


def jobs_of(arm: Arm, step: int, outputs: Path = OUTPUTS) -> list[dict]:
    model = str(target_dir(arm, step, outputs))
    return [
        {"run": run_of(arm), "model": model, "label": label_of(step), "set": s, "qos": QOS, "time": t} for s, t in SETS
    ]


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=str(OUTPUTS / "scoring" / "tier1" / "selection_state.jsonl"))
    parser.add_argument("--log", default=None)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--poll", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assume-ready", action="store_true", help="dry runs only: treat every target as landed")
    args = parser.parse_args(argv)
    if args.assume_ready and not args.dry_run:
        parser.error("--assume-ready is for --dry-run")

    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else state_path.with_name("tier1_queue.log")
    pause = state_path.with_name("PAUSE")
    processor_source = args.outputs / WARM_STARTS["goldw"]
    order = plan()

    def say(message: str) -> None:
        line = f"[{stamp()}] {message}"
        print(line, flush=True)
        if not args.dry_run:
            with log_path.open("a") as handle:
                handle.write(line + "\n")

    state = [json.loads(line) for line in state_path.read_text().splitlines()] if state_path.exists() else []
    done = {q.key(s): s for s in state}
    failures: dict[str, int] = {}
    say(f"{len(order)} targets, {len(order) * len(SETS)} jobs; {len(done)} already submitted (state {state_path})")

    waiting_note = None
    while True:
        todo: list[dict] = []
        not_landed: list[str] = []
        for arm, step in order:
            path = target_dir(arm, step, args.outputs)
            if not args.assume_ready:
                # a dry run writes nothing into the checkpoints, so for it a landed target counts as ready
                if landed(path) and not args.dry_run:
                    assemble(path, processor_source)
                if not (landed(path) if args.dry_run else ready(path)):
                    not_landed.append(f"{arm.key}@{step}")
                    continue
            todo += [
                j for j in jobs_of(arm, step, args.outputs) if q.key(j) not in done and failures.get(q.key(j), 0) < 3
            ]
        if not todo and not not_landed:
            break
        listed = ", ".join(not_landed[:6]) + ("..." if len(not_landed) > 6 else "")
        note = f"{len(todo)} jobs ready to place, {len(not_landed)} targets not landed yet ({listed})"
        if note != waiting_note:
            say(note)
            waiting_note = note
        if todo and pause.exists():
            say("PAUSE file present: not submitting this cycle")
        elif todo:
            counts = q.in_flight()
            if counts is None:
                say("could not read the queue, retrying")
            else:
                placed = 0
                for job in todo:
                    qos = next((c for c in job["qos"] if counts[c] < q.CAPS[c]), None)
                    if qos is None:
                        continue
                    job_id, message = q.submit(job, qos, args.dry_run)
                    if job_id is None:
                        failures[q.key(job)] = (
                            3 if message.startswith("ssh to artemis timed out") else failures.get(q.key(job), 0) + 1
                        )
                        say(f"SUBMIT FAILED ({failures[q.key(job)]}/3) {q.key(job)}: {message}")
                        continue
                    counts[qos] += 1
                    placed += 1
                    record = {
                        "run": job["run"], "label": job["label"], "set": job["set"], "job": job_id, "qos": qos,
                        "at": stamp(),
                    }  # fmt: skip
                    done[q.key(job)] = record
                    if not args.dry_run:
                        with state_path.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                    say(f"SUBMITTED {job_id} {qos} {job['label']} {job['set']}  (in flight {counts})")
                if placed:
                    waiting_note = None
        if args.dry_run:
            break
        time.sleep(args.poll)

    skipped = [k for k in (q.key(j) for a, s in order for j in jobs_of(a, s, args.outputs)) if k not in done]
    if skipped and not args.dry_run:
        say(f"NOT SUBMITTED after 3 failures: {skipped}")
    ids = [int(s["job"]) for s in done.values() if int(s["job"]) > 0]
    if ids and not args.dry_run:
        say(f"everything is submitted ({len(ids)} jobs); waiting for them to finish")
        q.wait_for(ids, args.poll)
    return 1 if skipped and not args.dry_run else 0  # a dry run places one cycle's worth: the rest is not a failure


if __name__ == "__main__":
    sys.exit(main())
