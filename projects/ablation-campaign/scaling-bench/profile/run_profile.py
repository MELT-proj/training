#!/usr/bin/env python3
"""Submit one step-profile job of the Fondue recipe and record it in arms.tsv.

    run_profile.py render --nodes 2 --grad-accum 40
    run_profile.py check  --nodes 2 --grad-accum 40
    run_profile.py run    --nodes 2 --grad-accum 40 [--steps 22] [--trace-steps 8]
    run_profile.py levers --nodes 2

Run from the repo root on the cluster, from a clone of your own. Every job is
``acc_debug`` (2 h, one submission at a time) and writes only under
``BENCH-profile-*`` in the site's OUTPUT_DIR; weights are never saved
(``save_strategy: "no"``, as ``bench.py`` already does).

The recipe is ``bench.yaml``'s ``fondue``, rendered by ``bench.bench_overrides``
and then changed only by what is passed here, so a profile point differs from
the scaling sweep by ``grad_accum`` and the ``run.step_profile`` switches.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))

import bench  # noqa: E402

#: Host libibverbs, the mlx5 provider and libnl copied from /usr/lib64 (see the board entry).
IB_STAGE = "/gpfs/scratch/epor48/outputs/BENCH-profile-ibstage"


def exp_name(nodes: int, grad_accum: int, tag: str | None) -> str:
    name = f"BENCH-profile-n{nodes}-ga{grad_accum}"
    return f"{name}-{tag}" if tag else name


def render(args: argparse.Namespace) -> tuple[str, dict[str, str], list[str]]:
    cfg = bench.load()
    recipe = next(r for r in cfg["recipes"] if r["id"] == "fondue")
    defaults = cfg["defaults"]
    name = exp_name(args.nodes, args.grad_accum, args.tag)
    env = {
        "MELT_NODES": str(args.nodes),
        "MELT_GPUS_PER_NODE": str(defaults["gpus_per_node"]),
        "MELT_QOS": "acc_debug",
        "MELT_TIME": args.time,
        "MELT_SEED": str(defaults["seed"]),
    }
    overrides = bench.bench_overrides(recipe, args.steps)
    # Later overrides win.
    overrides += [
        f"--trainer.gradient_accumulation_steps={args.grad_accum}",
        "--run.step_profile.enabled=true",
        "--run.step_profile.skip_final_save=true",
        f"--run.step_profile.trace_steps=[{','.join(str(s) for s in args.trace_steps)}]",
        f"--run.step_profile.trace_micro=[{args.trace_micro}]",
        f"--run.step_profile.ranks=[{','.join(str(r) for r in args.ranks)}]",
        f"--run.step_profile.trace_dir={bench.CONTAINER_OUTPUTS}/{name}/profile",
        *args.extra,
    ]
    argv = [
        "infra/runners/submit-container.sh",
        defaults["site"],
        defaults["accelerate"],
        "--config",
        os.path.relpath(bench.CAMPAIGN_DIR / recipe["config"], bench.REPO_ROOT),
        "--run.exp_name",
        name,
        "--trainer.output_dir",
        f"{bench.CONTAINER_OUTPUTS}/{name}",
        *overrides,
        "--trainer.seed",
        str(defaults["seed"]),
    ]
    return name, env, argv


def add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--nodes", type=int, required=True)
    sub.add_argument("--grad-accum", type=int, default=40)
    sub.add_argument("--steps", type=int, default=22, help="optimizer steps in the job")
    sub.add_argument("--trace-steps", type=int, nargs="*", default=[8], help="global_step values to trace")
    sub.add_argument("--trace-micro", type=int, default=2, help="non-final micro-batch to trace")
    sub.add_argument("--ranks", type=int, nargs="*", default=[0])
    sub.add_argument("--time", default="01:20:00")
    sub.add_argument("--tag", default=None)
    sub.add_argument(
        "--extra", action="append", default=[], help="a further override, appended last: --extra=--a.b=c (repeatable)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("render", "check", "run"):
        add_common(sub.add_parser(command))
    lever = sub.add_parser("levers", help="submit the isolated lever benchmarks (no data, no training)")
    lever.add_argument("--nodes", type=int, default=2)
    lever.add_argument("--time", default="00:40:00")
    lever.add_argument("--ib", action="store_true", help="bind the host InfiniBand userspace into the container")
    args = parser.parse_args()

    if args.command == "levers":
        submit_levers(args)
        return

    name, env, argv = render(args)
    if args.command == "render":
        print(bench.describe(name, env, argv))
        return
    env["_SITE"] = bench.load()["defaults"]["site"]
    if args.command == "check":
        bench.submit(name, env, argv, test_only=True)
        return
    submit_and_record(name, env, argv)


def submit_and_record(
    name: str, env: dict[str, str], argv: list[str], extra_env: dict[str, str] | None = None
) -> None:
    run_env = {**os.environ, **{k: v for k, v in env.items() if not k.startswith("_")}, **(extra_env or {})}
    result = subprocess.run(
        [str(bench.REPO_ROOT / argv[0]), *argv[1:]],
        cwd=bench.REPO_ROOT,
        env=run_env,
        capture_output=True,
        text=True,
    )
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(stream.strip())
    if result.returncode != 0:
        raise SystemExit(f"submission failed (exit {result.returncode})")
    match = re.search(r"Submitted batch job (\d+)", result.stderr + result.stdout)
    if not match:
        raise SystemExit("could not parse a job id out of sbatch output")
    bench.record(name, match.group(1), argv)


def submit_levers(args: argparse.Namespace) -> None:
    """lever_bench.py runs through the training launcher as MELT_TRAIN_MODULE."""
    cfg = bench.load()
    defaults = cfg["defaults"]
    name = f"BENCH-profile-levers-n{args.nodes}"
    env = {
        "MELT_NODES": str(args.nodes),
        "MELT_GPUS_PER_NODE": str(defaults["gpus_per_node"]),
        "MELT_QOS": "acc_debug",
        "MELT_TIME": args.time,
        "MELT_SEED": str(defaults["seed"]),
        "MELT_TRAIN_MODULE": "lever_bench",
    }
    extra_env = {
        "SINGULARITYENV_PYTHONPATH": "/workspace/training/projects/ablation-campaign/scaling-bench/profile",
        # Which transport NCCL picks (NET/IB or NET/Socket) is printed at init.
        "SINGULARITYENV_NCCL_DEBUG": "INFO",
        "SINGULARITYENV_NCCL_DEBUG_SUBSYS": "INIT,NET",
    }
    if args.ib:
        name += "-ib"
        stage = f"{IB_STAGE}:/opt/hostib:ro"
        env["MELT_EXTRA_BINDS"] = f"{stage},/dev/infiniband,/etc/libibverbs.d"
        extra_env["SINGULARITYENV_LD_LIBRARY_PATH"] = "/opt/hostib"
    argv = [
        "infra/runners/submit-container.sh",
        defaults["site"],
        defaults["accelerate"],
        "--out",
        f"{bench.CONTAINER_OUTPUTS}/{name}",
    ]
    submit_and_record(name, env, argv, extra_env)


if __name__ == "__main__":
    main()
