#!/usr/bin/env python3
"""Drive the ablation campaign from a declarative grid.

    campaign.py status [--site mn5]        what has run, is running, is missing
    campaign.py plan [ID ...]              show the exact command(s), submit nothing
    campaign.py run ID [--resume] [-- ...] submit one arm

Three artifacts, three jobs, deliberately not merged:

  campaign.yaml  INTENT  -- every arm we mean to run, as axis values.
  plan_arm.py    RENDER  -- axes -> one exact command (pure, no side effects).
  arms.tsv       LEDGER  -- what was actually submitted, append-only.

`status` is the join of intent against reality, which is the question a saved
command string can never answer: a command records what ran, but cannot tell
you an arm is *missing*, because a row that was never typed leaves no trace.

Reality is read from the cluster, not from the ledger, because a ledger only
knows what went through this script. Jobs submitted before the grid existed
(or by hand, for debugging) still show up correctly.

Runs with `python3` + PyYAML in any dev shell -- deliberately no melt import,
no torch, same constraint as plan_arm.py.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

import yaml

import plan_arm
from plan_arm import ArmAxes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GRID_PATH = os.path.join(HERE, "campaign.yaml")
LEDGER_PATH = os.path.join(HERE, "arms.tsv")

# Axis fields a grid row may set. Anything else in a row is a typo, and we say
# so rather than silently ignoring it -- a mis-spelled `decoder_freze` that
# quietly did nothing is exactly how an arm ends up not being the ablation it
# claims to be.
AXIS_FIELDS = {
    "adapter", "adapter_freeze", "encoder", "encoder_freeze",
    "decoder", "decoder_freeze", "decoder_lora",
    "encoder_lr", "decoder_lr", "adapter_lr",
}
POLICY_FIELDS = {
    "seed", "eval_rounds", "keep_checkpoints", "checkpoint_count",
    "exp_name", "init_from",
}
RUN_FIELDS = {"site", "accelerate", "nodes", "gpus_per_node", "qos", "time"}
KNOWN_FIELDS = AXIS_FIELDS | POLICY_FIELDS | RUN_FIELDS | {"id", "stage", "config"}


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_grid(path: str = GRID_PATH) -> tuple[dict, list[dict]]:
    if not os.path.isfile(path):
        die(f"campaign grid not found: {path}")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    defaults = doc.get("defaults") or {}
    arms = doc.get("arms") or []
    if not arms:
        die(f"{path} declares no arms")

    seen: set[str] = set()
    for row in arms:
        if "id" not in row:
            die(f"an arm in {path} has no `id`")
        if row["id"] in seen:
            die(f"duplicate arm id {row['id']!r} in {path}")
        seen.add(row["id"])
        for key in row:
            if key not in KNOWN_FIELDS:
                die(
                    f"arm {row['id']!r}: unknown field {key!r}. Known fields: "
                    + ", ".join(sorted(KNOWN_FIELDS))
                )
        for required in ("stage", "config"):
            if required not in row:
                die(f"arm {row['id']!r} has no `{required}`")
    return defaults, arms


def resolve(row: dict, defaults: dict, arms: list[dict]) -> tuple[ArmAxes, dict]:
    """Grid row + defaults -> (axes for plan_arm, run settings)."""
    merged = {**defaults, **row}
    nodes = int(merged.get("nodes", 2))
    gpus = int(merged.get("gpus_per_node", 4))

    config_path = merged["config"]
    if not os.path.isabs(config_path):
        config_path = os.path.join(HERE, config_path)

    # init_from names another ARM, not a path: resolved to that arm's run
    # directory in container space. This is what keeps stage 2 from hardcoding
    # a stage-1 output path in its base YAML, where it goes stale silently the
    # moment a second stage-1 arm exists.
    init_from = merged.get("init_from")
    init_path = None
    if init_from:
        parent = next((a for a in arms if a["id"] == init_from), None)
        if parent is None:
            die(
                f"arm {row['id']!r}: init_from={init_from!r} does not match any "
                "arm id in the grid"
            )
        parent_axes, _ = resolve(parent, defaults, arms)
        init_path = "/workspace/outputs/" + plan_arm.plan(parent_axes).exp_name

    axes = ArmAxes(
        config=config_path,
        stage=merged["stage"],
        world_size=nodes * gpus,
        seed=int(merged.get("seed", 42)),
        eval_rounds=int(merged.get("eval_rounds", ArmAxes.eval_rounds)),
        keep_checkpoints=int(merged.get("keep_checkpoints", ArmAxes.keep_checkpoints)),
        checkpoint_count=merged.get("checkpoint_count"),
        exp_name=merged.get("exp_name"),
        init_from=init_path,
        **{f: str(merged.get(f, "")) if merged.get(f) is not None else "" for f in AXIS_FIELDS},
    )
    run = {
        "site": merged.get("site", "mn5"),
        "accelerate": merged.get("accelerate", "config/accelerate/ddp.yaml"),
        "nodes": nodes,
        "gpus_per_node": gpus,
        "qos": merged.get("qos", "acc_ehpc"),
        "time": merged.get("time"),
    }
    return axes, run


def render(row: dict, defaults: dict, arms: list[dict], extra: list[str] | None = None):
    axes, run = resolve(row, defaults, arms)
    p = plan_arm.plan(axes)
    cmd = [
        "infra/runners/submit-container.sh", run["site"], run["accelerate"],
        "--config", os.path.relpath(axes.config, REPO_ROOT),
        "--run.exp_name", p.exp_name,
        "--trainer.output_dir", f"/workspace/outputs/{p.exp_name}",
        *p.overrides,
        "--trainer.num_train_epochs", "1",
        "--trainer.eval_steps", str(p.eval_steps),
        "--trainer.save_steps", str(p.save_steps),
        "--trainer.save_total_limit", str(p.save_total_limit),
        "--trainer.seed", str(axes.seed),
        *(extra or []),
    ]
    env = {
        "MELT_NODES": str(run["nodes"]),
        "MELT_GPUS_PER_NODE": str(run["gpus_per_node"]),
        "MELT_QOS": run["qos"],
        "MELT_SEED": str(axes.seed),
        "MELT_TIME": run["time"] or p.time_default,
    }
    return p, run, env, cmd


# ---------------------------------------------------------------- reality


def probe(site: str, exp_names: list[str]) -> dict:
    """One ssh round-trip for every arm's on-disk state plus the live queue.

    Returns {exp_name: {"exists", "last_ckpt", "final"}} and a set of exp_names
    that currently have a job in the queue. Read-only.
    """
    site_file = os.path.join(REPO_ROOT, "infra", "runners", "sites", f"{site}.sh")
    if not os.path.isfile(site_file):
        die(f"unknown site {site!r} (no {site_file})")
    # Source the site file rather than hardcoding its paths: OUTPUT_DIR's
    # default lives there and has moved before (gpfs_projects -> gpfs_scratch).
    out_dir = subprocess.run(
        ["bash", "-c", f'set -a; source {shlex.quote(site_file)} >/dev/null 2>&1; printf "%s" "$OUTPUT_DIR"'],
        capture_output=True, text=True,
    ).stdout.strip()
    if not out_dir:
        die(f"could not read OUTPUT_DIR from {site_file}")

    ssh_host = subprocess.run(
        ["bash", "-c", f'set -a; source {shlex.quote(site_file)} >/dev/null 2>&1; printf "%s" "$REMOTE_SSH"'],
        capture_output=True, text=True,
    ).stdout.strip() or site

    # For each name: exists | highest checkpoint-N | whether a consolidated
    # final model was written. train.py only calls save_model() after train()
    # returns, so a top-level model*.safetensors is the completion signal;
    # checkpoints without one mean "interrupted, resumable".
    script_lines = [f'O={shlex.quote(out_dir)}']
    for name in exp_names:
        q = shlex.quote(name)
        script_lines.append(
            f'if [ -d "$O"/{q} ]; then '
            f'C=$(ls -1d "$O"/{q}/checkpoint-* 2>/dev/null | sed "s|.*checkpoint-||" | sort -n | tail -1); '
            f'F=$(ls -1 "$O"/{q}/model*.safetensors 2>/dev/null | head -1); '
            f'echo "DIR|{name}|1|${{C:-0}}|$([ -n "$F" ] && echo 1 || echo 0)"; '
            f'else echo "DIR|{name}|0|0|0"; fi'
        )
    script_lines.append('squeue -u "$USER" -h -o "JOB|%i|%T|%j" 2>/dev/null || true')
    remote = "; ".join(script_lines)

    res = subprocess.run(["ssh", ssh_host, remote], capture_output=True, text=True)
    if res.returncode != 0:
        die(f"probe of {site} failed: {res.stderr.strip()[:400]}")

    disk: dict[str, dict] = {}
    jobs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        parts = line.strip().split("|")
        if parts[0] == "DIR" and len(parts) == 5:
            disk[parts[1]] = {
                "exists": parts[2] == "1",
                "last_ckpt": int(parts[3] or 0),
                "final": parts[4] == "1",
            }
        elif parts[0] == "JOB" and len(parts) == 4:
            jobs.append((parts[1], parts[2]))
    return {"disk": disk, "jobs": jobs, "out_dir": out_dir}


def ledger_rows() -> list[dict]:
    if not os.path.isfile(LEDGER_PATH):
        return []
    rows = []
    with open(LEDGER_PATH) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            vals = line.rstrip("\n").split("\t")
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals)))
    return rows


def classify(state: dict, running_ids: set[str], led: list[dict], exp_name: str) -> tuple[str, str]:
    submitted = [r for r in led if r.get("exp_name") == exp_name]
    live = [r for r in submitted if r.get("job_id") in running_ids]
    if live:
        return "RUNNING", f"job {live[-1]['job_id']}"
    if not state or not state["exists"]:
        return ("MISSING", "never submitted" if not submitted else
                f"submitted ({len(submitted)}x) but no output dir")
    if state["final"]:
        return "DONE", f"final weights, last ckpt {state['last_ckpt']}"
    if state["last_ckpt"]:
        return "INCOMPLETE", f"ckpt {state['last_ckpt']}, needs --resume"
    return "STARTED", "output dir only, no checkpoint yet"


# ---------------------------------------------------------------- commands


def cmd_status(args) -> None:
    defaults, arms = load_grid()
    plans = {}
    for row in arms:
        p, _, _, _ = render(row, defaults, arms)
        plans[row["id"]] = p
    state = probe(args.site, [p.exp_name for p in plans.values()])
    running_ids = {jid for jid, st in state["jobs"] if st in ("RUNNING", "PENDING", "CONFIGURING")}
    led = ledger_rows()

    print(f"site {args.site}   outputs {state['out_dir']}")
    print(f"{'arm':<24} {'status':<11} {'steps':>6}  detail")
    print("-" * 88)
    counts: dict[str, int] = {}
    for row in arms:
        p = plans[row["id"]]
        st, detail = classify(state["disk"].get(p.exp_name), running_ids, led, p.exp_name)
        counts[st] = counts.get(st, 0) + 1
        print(f"{row['id']:<24} {st:<11} {p.steps:>6}  {detail}")
    print("-" * 88)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not led:
        print("\nNOTE: arms.tsv is empty -- nothing has been submitted through "
              "campaign.py yet, so RUNNING cannot be attributed to an arm by job "
              "id. Status above is derived from the output dirs alone.")


def cmd_plan(args) -> None:
    defaults, arms = load_grid()
    wanted = args.ids or [a["id"] for a in arms]
    for row in arms:
        if row["id"] not in wanted:
            continue
        p, run, env, cmd = render(row, defaults, arms, args.extra)
        print(f"=== {row['id']} ===")
        print(f"  exp_name    {p.exp_name}")
        print(f"  steps       {p.steps}  (eval/save every {p.eval_steps}/{p.save_steps}, keep {p.save_total_limit})")
        print(f"  topology    {run['nodes']}x{run['gpus_per_node']} = {run['nodes']*run['gpus_per_node']} ranks, {env['MELT_TIME']}, {run['qos']}")
        print("  command     " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()))
        print("              " + " ".join(shlex.quote(c) for c in cmd))
        print()


def cmd_run(args) -> None:
    defaults, arms = load_grid()
    row = next((a for a in arms if a["id"] == args.id), None)
    if row is None:
        die(f"no arm {args.id!r} in the grid (ids: {', '.join(a['id'] for a in arms)})")

    extra = list(args.extra)
    if args.resume:
        # True, not a path: HF scans output_dir for the last checkpoint itself.
        # Never point it at a checkpoint-N/ subdirectory -- train.py calls
        # get_last_checkpoint() on whatever it is given.
        extra += ["--trainer.resume_from_checkpoint", "True"]

    p, run, env, cmd = render(row, defaults, arms, extra)
    printable = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + \
                " ".join(shlex.quote(c) for c in cmd)
    if args.dry_run:
        print(printable)
        return

    # submit-container.sh calls sbatch directly, so `run` only works from the
    # cluster's login node -- the same "run from the repo root ON MN5"
    # convention the launchers document. Without this guard the submitter
    # echoes its command, dies on a missing sbatch, and looks like it worked.
    if subprocess.run(["bash", "-c", "command -v sbatch"], capture_output=True).returncode != 0:
        die(
            "sbatch is not on PATH, so this is not a cluster login node. "
            f"`run` must be executed there:\n"
            f"    ssh {run['site']} 'cd training && "
            f"python3 projects/ablation-campaign/campaign.py run {args.id}"
            f"{' --resume' if args.resume else ''}'\n"
            "`status`, `plan` and `run --dry-run` work fine from anywhere."
        )

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, **env},
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    job_id = ""
    for line in proc.stdout.splitlines():
        if line.startswith("Submitted batch job "):
            job_id = line.split()[-1]
    if proc.returncode != 0:
        die(f"submission failed for {args.id}")

    import datetime
    new = not os.path.isfile(LEDGER_PATH)
    with open(LEDGER_PATH, "a") as fh:
        if new:
            fh.write("timestamp_utc\texp_name\tjob_id\tcommand\n")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fh.write(f"{ts}\t{p.exp_name}\t{job_id or 'UNKNOWN'}\t{printable}\n")
    print(f"Recorded {args.id} ({p.exp_name}) as job {job_id or 'UNKNOWN'} in arms.tsv")


def main() -> None:
    # Split passthrough args off BEFORE argparse sees them. argparse.REMAINDER
    # would swallow this command's own flags too: `run ID --resume --dry-run`
    # parsed --resume/--dry-run as train.py overrides, leaving --dry-run false
    # and actually invoking the submitter. Everything after a literal `--` is
    # passthrough, everything before it is ours -- no ambiguity either way.
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pass overrides straight through to train.py after a literal `--`:\n"
               "  campaign.py run IFT-700-llama1b-ins -- --trainer.max_steps 10",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="what has run, is running, is missing")
    s.add_argument("--site", default="mn5")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("plan", help="show the exact command(s), submit nothing")
    s.add_argument("ids", nargs="*")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("run", help="submit one arm")
    s.add_argument("id")
    s.add_argument("--resume", action="store_true",
                   help="continue in the same output_dir from its last checkpoint")
    s.add_argument("--dry-run", action="store_true", help="print the command, submit nothing")
    s.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    args.extra = extra
    args.func(args)


if __name__ == "__main__":
    main()
