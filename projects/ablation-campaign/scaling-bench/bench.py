#!/usr/bin/env python3
"""Render, check and submit the scaling-benchmark points.

    bench.py list
    bench.py render  ift-qwen-n8 [--repeat 2] [--all]
    bench.py check   ift-qwen-n8 [--repeat 2]
    bench.py run     ift-qwen-n8 [--repeat 2]
    bench.py status

Run from the repo root (``infra/runners/submit-container.sh`` is resolved
relative to it, as that script requires).

Why this is a separate driver and not a row in ``campaign.yaml``: the campaign
grid is a research artefact whose 65 arms each answer a modelling question, and
its ledger (``arms.tsv``) is read by the efficiency and cost-axis tooling.
Benchmark points are instrumentation runs with production eval, saving and
preallocation turned off; adding them to the grid would mean every later
analysis has to filter them out. They still append to ``arms.tsv``, because that
is the one sanctioned place for a job id.

The recipe is derived from ``ABL-IFT-700.yaml`` with an override list, not a
rendered copy of it: a copy would silently drift from the campaign config the
benchmark is supposed to describe.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_DIR = HERE.parent
ARMS_TSV = CAMPAIGN_DIR / "arms.tsv"

sys.path.insert(0, str(CAMPAIGN_DIR))

import plan_arm  # noqa: E402  (needs the path above)

# Inside the container, outputs are at /workspace/outputs (the site's host
# OUTPUT_DIR is only the bind source); see infra/runners/submit-container.sh.
CONTAINER_OUTPUTS = "/workspace/outputs"


def load() -> dict:
    with open(HERE / "bench.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def recipe_of(cfg: dict, point: dict) -> dict:
    for recipe in cfg["recipes"]:
        if recipe["id"] == point["recipe"]:
            return recipe
    raise SystemExit(f"point {point['id']!r} names unknown recipe {point['recipe']!r}")


def find_point(cfg: dict, point_id: str) -> dict:
    for point in cfg["points"]:
        if point["id"] == point_id:
            return point
    known = ", ".join(p["id"] for p in cfg["points"])
    raise SystemExit(f"unknown point {point_id!r}; known: {known}")


def site_output_dir(site: str) -> str:
    """The host OUTPUT_DIR the site file declares, without hardcoding gpfs paths.

    Read from the site file rather than restated here: this path has moved once
    already (gpfs_projects -> gpfs_scratch) and a stale copy would fetch from a
    directory that no longer exists.
    """
    script = 'source "$1" >/dev/null 2>&1; printf "%s" "${OUTPUT_DIR}"'
    result = subprocess.run(
        ["bash", "-c", script, "_", str(REPO_ROOT / "infra" / "runners" / "sites" / f"{site}.sh")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        raise SystemExit(f"could not read OUTPUT_DIR from the {site} site file: {result.stderr.strip()}")
    return path


def ledger_job_id(name: str) -> str | None:
    """The most recent job id recorded for *name* in arms.tsv."""
    if not ARMS_TSV.exists():
        return None
    found = None
    for line in ARMS_TSV.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == name:
            found = parts[2]
    return found


def fetch(cfg: dict, point: dict, repeat: int, tag: str | None = None) -> None:
    """Copy a run's measurement files off the cluster into raw/<exp_name>/."""
    name = exp_name(point, repeat, tag)
    site = cfg["defaults"]["site"]
    remote_root = site_output_dir(site)
    dest = HERE / "raw" / name
    dest.mkdir(parents=True, exist_ok=True)

    files = ["perf.jsonl", "resolved_config.json"]
    for filename in files:
        src = f"{site}:{remote_root}/{name}/{filename}"
        result = subprocess.run(["scp", "-q", src, str(dest / filename)], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  missing {filename}: {result.stderr.strip()}")

    job_id = ledger_job_id(name)
    if job_id:
        log_name = f"melt-train-container.{job_id}.out"
        src = f"{site}:~/training/logs/{log_name}"
        result = subprocess.run(["scp", "-q", src, str(dest / log_name)], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  missing log for job {job_id}: {result.stderr.strip()}")
    print(f"fetched into {os.path.relpath(dest, REPO_ROOT)}")


def exp_name(point: dict, repeat: int, tag: str | None = None) -> str:
    """Short and sortable; the repeat suffix keeps each submission's output_dir unique.

    A tag ("smoke") makes a short dry run a separate run rather than overwriting
    the point it is standing in for -- ``overwrite_output_dir`` is on, so sharing
    a name would delete the earlier one's perf.jsonl.
    """
    name = f"BENCH-{point['recipe']}-n{point['nodes']}"
    if tag:
        name += f"-{tag}"
    if repeat > 1:
        name += f"-r{repeat}"
    return name


def bench_overrides(recipe: dict, max_steps: int | None = None) -> list[str]:
    """Everything held fixed across points, plus the timing-hygiene switches.

    Order matters only in that later overrides win; none here collide.
    ``max_steps`` shortens a smoke run; the point's own value is the default.
    """
    profile = plan_arm.DECODER_PROFILES.get(recipe["decoder"], {})
    overrides = [
        f"--model.encoder.name={recipe['encoder']}",
        # "Everything unfrozen" is the requested recipe: a full fine-tune, not
        # the campaign's frozen-encoder IFT. One line each to revert.
        "--model.encoder.freeze=false",
        "--model.adapter.freeze=false",
        "--model.decoder.freeze=false",
        f"--model.decoder.name={recipe['decoder']}",
        f"--data.train_ds.batch_duration={recipe['batch_duration']}",
        f"--trainer.gradient_accumulation_steps={recipe['grad_accum_steps']}",
        (
            "--trainer.gradient_checkpointing=true"
            if recipe.get("gradient_checkpointing")
            else "--trainer.gradient_checkpointing=false"
        ),
        f"--trainer.max_steps={max_steps if max_steps is not None else recipe['max_steps']}",
        "--trainer.num_train_epochs=1",
    ]
    if profile:
        overrides += [
            f"--model.decoder.eos_token={profile['eos_token']}",
            f"--model.decoder.pad_token={profile['pad_token']}",
            f"--data.chat_template_config={profile['chat_template_config']}",
        ]
    # Timing hygiene. Everything below removes work that is not training.
    overrides += [
        "--run.perf_logging=true",
        "--run.memory_preallocation=false",
        "--trainer.logging_steps=1",
        "--trainer.eval_on_start=false",
        "--trainer.do_eval=false",
        # The inner quotes are load-bearing: a bare `no` reaches OmegaConf as
        # boolean False. Same trap, same fix, as --data... below.
        '--trainer.eval_strategy="no"',
        '--trainer.save_strategy="no"',
        # Skips building the eval dataset (~100k cuts) in MELTTrainer.__init__:
        # several minutes of startup on every point, and we never evaluate.
        "--data.validation_ds.input_cfg=null",
        "--trainer.overwrite_output_dir=true",
    ]
    return overrides


def render(
    cfg: dict,
    point: dict,
    repeat: int = 1,
    max_steps: int | None = None,
    tag: str | None = None,
) -> tuple[str, dict[str, str], list[str]]:
    recipe = recipe_of(cfg, point)
    defaults = cfg["defaults"]
    name = exp_name(point, repeat, tag)
    env = {
        "MELT_NODES": str(point["nodes"]),
        "MELT_GPUS_PER_NODE": str(defaults["gpus_per_node"]),
        "MELT_QOS": point["qos"],
        "MELT_TIME": point.get("time") or defaults["time"],
        "MELT_SEED": str(defaults["seed"]),
    }
    argv = [
        "infra/runners/submit-container.sh",
        defaults["site"],
        defaults["accelerate"],
        "--config",
        os.path.relpath(CAMPAIGN_DIR / recipe["config"], REPO_ROOT),
        "--run.exp_name",
        name,
        "--trainer.output_dir",
        f"{CONTAINER_OUTPUTS}/{name}",
        *bench_overrides(recipe, max_steps),
        "--trainer.seed",
        str(defaults["seed"]),
    ]
    return name, env, argv


def describe(name: str, env: dict[str, str], argv: list[str]) -> str:
    lines = [f"# {name}", "env " + " ".join(f"{k}={v}" for k, v in env.items())]
    lines.append(" ".join(shlex.quote(a) for a in argv))
    return "\n".join(lines)


def assert_one_recipe(cfg: dict) -> None:
    """Points may differ only in node count and scheduler settings.

    This is the mechanical half of "only --nodes changes": if a second point
    ever grows a different recipe, the curve it produces is not a scaling curve.
    """
    ids = {p["recipe"] for p in cfg["points"]}
    if len(ids) > 1:
        print(f"note: {len(ids)} recipes in play ({sorted(ids)}); checking within each", file=sys.stderr)

    # The two flag/value pairs allowed to vary between points.
    variable = {"--run.exp_name", "--trainer.output_dir"}

    def strip(argv: list[str]) -> str:
        kept: list[str] = []
        skip_next = False
        for arg in argv:
            if skip_next:
                skip_next = False
                continue
            if arg in variable:
                skip_next = True
                continue
            kept.append(arg)
        return " ".join(kept)

    by_recipe: dict[str, list[str]] = {}
    for point in cfg["points"]:
        _, _, argv = render(cfg, point, repeat=1)
        by_recipe.setdefault(point["recipe"], []).append(strip(argv))
    for recipe, renders in by_recipe.items():
        if len(set(renders)) != 1:
            raise SystemExit(f"recipe {recipe!r} does not render identically at every node count")
        print(f"OK  {recipe}: identical argv at every node count")


def submit(name: str, env: dict[str, str], argv: list[str], test_only: bool = False) -> None:
    """Run the site's own SBATCH_ARGS, optionally through --test-only.

    Sourcing the site file rather than re-deriving QOS/account/partition from
    bench.yaml is deliberate: it is the same file the real submission uses, so
    a check that passes here cannot disagree with the submit.
    """
    run_env = {**os.environ, **{k: v for k, v in env.items() if not k.startswith("_")}}
    if test_only:
        site_file = REPO_ROOT / "infra" / "runners" / "sites" / f"{cfg_site(env)}.sh"
        script = (
            "set -euo pipefail\n"
            'source "$1"\n'
            'shift\n'
            'sbatch --test-only "${SBATCH_ARGS[@]}" bash/run_train_singularity.sbatch "$@"\n'
        )
        # argv[:3] is [submit-container.sh, site, accelerate]; the sbatch script
        # receives everything after those, exactly as submit-container.sh does.
        cmd = ["bash", "-c", script, "_", str(site_file), *argv[3:]]
    else:
        cmd = [str(REPO_ROOT / argv[0]), *argv[1:]]

    result = subprocess.run(cmd, cwd=REPO_ROOT, env=run_env, capture_output=True, text=True)
    # sbatch reports its verdict -- "Submitted batch job N", and the
    # --test-only start estimate -- on stderr, so printing only stdout makes a
    # successful check look like a silent no-op.
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(stream.strip())
    if result.returncode != 0:
        raise SystemExit(f"submission failed (exit {result.returncode})")


def cfg_site(env: dict[str, str]) -> str:
    """The site name, recorded in the env dict by render()."""
    return env["_SITE"]


def record(name: str, job_id: str, argv: list[str]) -> None:
    """Append to the campaign ledger -- the only place a job id belongs."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    command = " ".join(shlex.quote(a) for a in argv)
    with open(ARMS_TSV, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}\t{name}\t{job_id}\t{command}\n")
    print(f"recorded {name} job {job_id} in {os.path.relpath(ARMS_TSV, REPO_ROOT)}")


def _add_shortening_flags(sub) -> None:
    """Flags that make a point shorter or separately named, for smoke runs."""
    sub.add_argument("--max-steps", type=int, default=None,
                     help="override the recipe's step count (smoke runs)")
    sub.add_argument("--tag", default=None,
                     help="suffix the exp name, so a smoke run does not overwrite the real point")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="print the points")

    p_render = sub.add_parser("render", help="print the exact command for a point")
    p_render.add_argument("point", nargs="?")
    p_render.add_argument("--repeat", type=int, default=1)
    _add_shortening_flags(p_render)
    p_render.add_argument("--all", action="store_true", help="check every point renders one recipe")

    p_check = sub.add_parser("check", help="sbatch --test-only (validates QOS, nodes, walltime)")
    p_check.add_argument("point")
    p_check.add_argument("--repeat", type=int, default=1)
    _add_shortening_flags(p_check)

    p_run = sub.add_parser("run", help="submit the point and record it in arms.tsv")
    p_run.add_argument("point")
    p_run.add_argument("--repeat", type=int, default=1)
    _add_shortening_flags(p_run)

    p_fetch = sub.add_parser("fetch", help="copy perf.jsonl / resolved_config.json / log into raw/")
    p_fetch.add_argument("point")
    p_fetch.add_argument("--repeat", type=int, default=1)
    _add_shortening_flags(p_fetch)

    sub.add_parser("status", help="queue state for BENCH- jobs")

    args = parser.parse_args()
    cfg = load()

    if args.command == "list":
        for point in cfg["points"]:
            print(f"{point['id']:<16} nodes={point['nodes']:<3} qos={point['qos']:<10} "
                  f"time={point.get('time') or cfg['defaults']['time']}")
        return

    if args.command == "render":
        if args.all or not args.point:
            assert_one_recipe(cfg)
            return
        name, env, argv = render(cfg, find_point(cfg, args.point), args.repeat, args.max_steps, args.tag)
        print(describe(name, env, argv))
        return

    if args.command == "check":
        point = find_point(cfg, args.point)
        name, env, argv = render(cfg, point, args.repeat, args.max_steps, args.tag)
        env["_SITE"] = cfg["defaults"]["site"]
        submit(name, env, argv, test_only=True)
        return

    if args.command == "run":
        point = find_point(cfg, args.point)
        name, env, argv = render(cfg, point, args.repeat, args.max_steps, args.tag)
        result = subprocess.run(
            [str(REPO_ROOT / argv[0]), *argv[1:]],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"submission failed (exit {result.returncode})")
        job_id = result.stdout.strip().rsplit(" ", 1)[-1]
        if not job_id.isdigit():
            raise SystemExit(f"could not parse a job id out of sbatch output: {result.stdout!r}")
        record(name, job_id, argv)
        return

    if args.command == "fetch":
        fetch(cfg, find_point(cfg, args.point), args.repeat, args.tag)
        return

    if args.command == "status":
        fmt = "%.3i %.9P %.16j %.8T %.10M %.6D %R"
        subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", fmt],
            check=False,
        )
        return


if __name__ == "__main__":
    main()
