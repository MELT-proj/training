#!/usr/bin/env python3
"""Post-hoc efficiency row for a finished arm (plan/03-audio-stack.md section 4).

One row per arm, tab-separated, joinable against arms.tsv on `exp_name`:

    encoder+adapter parameters (and active parameters for a MoE adapter)
    decoder positions per audio second (the frame rate after the adapter)
    training GPU-h per 1,000 audio hours, summed over every job of the arm
    eval throughput in utterances per second (from eval_throughput.py)

Two halves with different requirements, run separately or together:

  accounting  SLURM + the run's logs/trainer_state. Needs only python3 and
              access to the cluster (`--ssh mn5`, or run it there). No torch.
  static      Builds the audio stack on the meta device from the run's
              config.json (no weights, no GPU) and asks the model's own shape
              bookkeeping for the frame rate. Needs torch + transformers, so on
              nyx it runs in the container (infrastructure.md, "nyx").

Examples (from the repo root):

    # accounting only, over the ssh alias, for one arm; jobs found from the logs
    python3 projects/ablation-campaign/efficiency.py row EXP_NAME --ssh mn5 --no-static

    # everything, from the nyx host: accounting over ssh, static half in the container
    python3 projects/ablation-campaign/efficiency.py row --names-file names.txt --ssh mn5 \
        --image /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_tf5.sif

    # cost of a stack that has no run yet
    python3 projects/ablation-campaign/efficiency.py static --config-json config.json \
        --set adapter_config._type=conformer

GPU-h. Primary figure is ElapsedRaw x AllocTRES gres/gpu / 3600, which does not
depend on how SLURM counts CPUs. It is cross-checked against the documented
CPUTimeRAW / 3600 / 40 (infrastructure.md section 1); the two must agree to 1% or
the row is flagged. `AllocCPUS` on MN5 counts SMT threads (320 for two nodes), which
is where an earlier 2x error in a budget report came from.

Decoder positions per audio second is NOT affected by a fixed-window encoder's
padding: the attention mask keeps only real frames. Encoder compute per utterance
is inflated by the padding, and that lives in the GPU-h columns and in
`window_padding_ratio`, which this tool deliberately leaves blank (timeline week 3,
Track B, "Whisper window-padding ratio").
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS_TSV = HERE / "arms.tsv"

# CPUTimeRAW per GPU-second on MN5 `acc`: 320 SMT threads / 8 GPUs on two nodes.
MN5_CPU_PER_GPU = 40

SITES = {
    "mn5": {
        "outputs": "/gpfs/scratch/epor48/outputs",
        "logs": "~/training/logs",
    },
}

COLUMNS = [
    "exp_name", "stage", "encoder", "adapter", "decoder", "stack_factor", "world_size",
    "encoder_params_m", "adapter_params_m", "adapter_active_params_m",
    "stack_params_m", "stack_active_params_m",
    "positions_per_audio_s", "encoder_window_s", "window_padding_ratio",
    "n_jobs", "n_failed", "job_ids",
    "gpu_h_total", "gpu_h_startup", "gpu_h_redone", "gpu_h_clean",
    "train_audio_h", "gpu_h_per_1k_audio_h", "gpu_h_per_1k_audio_h_clean",
    "steps", "steps_redone", "loop_s_per_step", "encoder_attn", "decoder_attn",
    "eval_utt_per_s", "eval_conditions", "notes",
]


# --------------------------------------------------------------------------- SLURM


def parse_sacct(text: str) -> dict[str, dict]:
    """Parse `sacct -X -P -o JobID,ElapsedRaw,CPUTimeRAW,NNodes,State,AllocTRES`."""
    rows = list(csv.reader(text.strip().splitlines(), delimiter="|"))
    if not rows:
        return {}
    head, body = rows[0], rows[1:]
    jobs = {}
    for r in body:
        d = dict(zip(head, r))
        tres = dict(kv.split("=", 1) for kv in d.get("AllocTRES", "").split(",") if "=" in kv)
        jobs[d["JobID"]] = {
            "job_id": d["JobID"],
            "elapsed_s": int(d["ElapsedRaw"] or 0),
            "cputime_raw": int(d["CPUTimeRAW"] or 0),
            "nodes": int(d["NNodes"] or 0),
            "gpus": int(tres.get("gres/gpu", 0)),
            "state": d["State"].split()[0],
        }
    return jobs


def job_gpu_hours(job: dict, cpu_per_gpu: int = MN5_CPU_PER_GPU) -> tuple[float, float]:
    """(primary, cross-check) GPU-hours of one job.

    Primary counts GPUs directly; the cross-check applies the documented
    CPUTimeRAW / 3600 / 40. Both are 0 for a job that never started.
    """
    primary = job["elapsed_s"] * job["gpus"] / 3600
    check = job["cputime_raw"] / 3600 / cpu_per_gpu
    return primary, check


# --------------------------------------------------------------------------- logs

_BAR = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")
_RESUME = re.compile(r"Resuming training from checkpoint with epoch \d+ and global step (\d+)")


def parse_training_bar(text: str, expected_total: int | None = None) -> dict | None:
    """Progress of the training loop from tqdm output.

    A run's log holds several bars (training, each eval loop, dataset maps). The
    training bar's total is the run's `max_steps`, so pass it as `expected_total` when
    it is known: without it a job that died before step 1 can pick up a stray bar and
    be credited with steps it never took. Otherwise the most common total among long
    bars is used, since the training bar is redrawn thousands of times.
    """
    bars = [(int(s), int(t), int(h or 0) * 3600 + int(m) * 60 + int(x))
            for s, t, h, m, x in _BAR.findall(text)]
    totals = Counter(t for _, t, _ in bars if t > 100)
    if expected_total is not None:
        totals = Counter({expected_total: 1}) if expected_total in totals else Counter()
    if not totals:
        return None
    total = totals.most_common(1)[0][0]
    pts = [(s, e) for s, t, e in bars if t == total]
    first, last = min(pts), max(pts)
    return {"total": total, "first_step": first[0], "last_step": last[0],
            "loop_elapsed_s": last[1]}


def parse_resume_step(text: str) -> int | None:
    m = _RESUME.search(text)
    return int(m.group(1)) if m else None


def summarise_jobs(jobs: list[dict], logs: dict[str, dict]) -> dict:
    """Fold per-job SLURM records and log summaries into the arm's accounting.

    `logs[job_id]` is {first_step, last_step, loop_elapsed_s, resume_step} from
    the job's own log. Splits the arm's GPU-h into what a single uninterrupted
    pass would have cost (clean) and what interruptions added: pre-loop startup
    (dataloader build, model load) for every job, and steps a timeout threw away
    because they came after the last checkpoint.
    """
    jobs = sorted(jobs, key=lambda j: int(j["job_id"]))
    total = startup = redone = 0.0
    steps_redone = 0
    prev = None
    mismatch = []
    for j in jobs:
        primary, check = job_gpu_hours(j)
        if primary and abs(primary - check) > 0.01 * primary:
            mismatch.append(j["job_id"])
        total += primary
        lg = logs.get(j["job_id"])
        if not lg:
            continue  # a job that died before its first step: all of it is startup
        lg = logs[j["job_id"]] = {**lg, "first_step": max(lg["first_step"], lg["resume_step"] or 0)}
        startup += max(j["elapsed_s"] - lg["loop_elapsed_s"], 0) * j["gpus"] / 3600
        if prev is not None:
            lost = max(prev["lg"]["last_step"] - (lg["resume_step"] or 0), 0)
            span = max(prev["lg"]["last_step"] - prev["lg"]["first_step"], 1)
            rate = prev["lg"]["loop_elapsed_s"] / span
            steps_redone += lost
            redone += lost * rate * prev["job"]["gpus"] / 3600
        prev = {"job": j, "lg": lg}
    for j in jobs:
        if j["job_id"] not in logs:
            startup += j["elapsed_s"] * j["gpus"] / 3600
    last = logs.get(jobs[-1]["job_id"]) if jobs else None
    loop = None
    if last:
        loop = last["loop_elapsed_s"] / max(last["last_step"] - last["first_step"], 1)
    return {
        "n_jobs": len(jobs),
        "n_failed": sum(j["state"] in ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL") for j in jobs),
        "job_ids": [j["job_id"] for j in jobs],
        "gpu_h_total": total,
        "gpu_h_startup": startup,
        "gpu_h_redone": redone,
        "gpu_h_clean": total - startup - redone,
        "steps_redone": steps_redone,
        "loop_s_per_step": loop,
        "gpu_h_mismatch_jobs": mismatch,
    }


# --------------------------------------------------------------------------- cluster access

# Runs on the cluster (python3, stdlib only). Prints one JSON object: what the
# analysis needs from the run directory and the SLURM logs, nothing more.
_REMOTE = r"""
import glob, json, os, re, subprocess, sys
exp, outputs, logs, mode, jobs = sys.argv[1], sys.argv[2], os.path.expanduser(sys.argv[3]), sys.argv[4], sys.argv[5:]
d = os.path.join(outputs, exp)
out = {"exp_name": exp}
try:
    rc = json.load(open(os.path.join(d, "resolved_config.json")))
    out["resolved"] = rc
    out["config_json"] = open(os.path.join(d, "config.json")).read()
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    ts = json.load(open(os.path.join(cks[-1], "trainer_state.json")))
    hrs = [h["train_hours/total"] for h in ts["log_history"] if "train_hours/total" in h]
    out["state"] = {"global_step": ts["global_step"], "max_steps": ts["max_steps"],
                    "train_hours": hrs[-1] if hrs else None}
except Exception as e:
    out["error"] = repr(e)
if mode == "scan":
    r = subprocess.run(["grep", "-a", "-l", "--", "--run.exp_name " + exp + " "]
                       + glob.glob(os.path.join(logs, "melt-train-container.*.out")),
                       capture_output=True, text=True)
    jobs = list(set(jobs) | {p.rsplit(".", 2)[1] for p in r.stdout.split()})
out["jobs"] = sorted(jobs, key=int)
out["logs"] = {}
for j in out["jobs"]:
    p = os.path.join(logs, "melt-train-container.%s.out" % j)
    if not os.path.exists(p):
        continue
    bars = subprocess.run(["grep", "-a", "-o", "-E", r"[0-9]+/[0-9]+ \[[0-9:]+<", p],
                          capture_output=True, text=True).stdout
    misc = subprocess.run(["grep", "-a", "-E",
                           "Resuming training from checkpoint|attention implementation|world_size=", p],
                          capture_output=True, text=True).stdout
    out["logs"][j] = {"bars": bars, "misc": misc}
print(json.dumps(out))
"""


def run_cmd(argv_prefix: list[str], argv: list[str], stdin: str | None = None) -> str:
    r = subprocess.run(argv_prefix + argv, input=stdin, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({' '.join(argv_prefix + argv[:2])} ...):\n{r.stderr[-800:]}")
    return r.stdout


def fetch_arm(exp: str, jobs: list[str], scan: bool, site: str, ssh: str | None,
              outputs: str | None, logs: str | None) -> dict:
    s = SITES[site]
    prefix = ["ssh", "-o", "BatchMode=yes", ssh] if ssh else []
    argv = ["python3", "-", exp, outputs or s["outputs"], logs or s["logs"],
            "scan" if scan else "only", *jobs]
    out = run_cmd(prefix, argv, stdin=_REMOTE)
    return json.loads(out[out.index("{"):])  # login banners precede the JSON


def sacct(job_ids: list[str], ssh: str | None) -> dict[str, dict]:
    prefix = ["ssh", "-o", "BatchMode=yes", ssh] if ssh else []
    text = run_cmd(prefix, ["sacct", "-X", "-P", "-j", ",".join(job_ids), "-o",
                            "JobID,ElapsedRaw,CPUTimeRAW,NNodes,State,AllocTRES%100"])
    return parse_sacct(text[text.index("JobID"):])


def ledger_jobs(exp: str) -> list[str]:
    if not ARMS_TSV.exists():
        return []
    with open(ARMS_TSV) as f:
        return [r["job_id"] for r in csv.DictReader(f, delimiter="\t") if r["exp_name"] == exp]


# --------------------------------------------------------------------------- static half


def build_stack(config):
    """Audio stack (encoder + adapter) with no weights, on the meta device."""
    import torch

    from melt.modeling.modeling_melt import MELTAudioStack

    with torch.device("meta"):
        return MELTAudioStack(config, load_pretrained=False)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def active_adapter_params(adapter) -> int:
    """Parameters used per token: for a routed MoE, top-k of its experts.

    The router, the shared expert (if any) and the norm are always on. Anything
    that is not a MoE (no `experts` list) is dense: active == total.
    """
    inner = adapter.adapter
    total = count_params(inner)
    experts = getattr(inner, "experts", None)
    if experts is None:
        return total
    e, k = inner.num_experts, inner.num_experts_per_tok
    return total - count_params(experts) * (e - k) // e


def positions_per_audio_second(stack, seconds: float = 60.0) -> dict:
    """Decoder positions per second of real audio, from the model's own bookkeeping.

    Builds an input of `seconds` of real audio, padded up to a whole window when the
    encoder has a fixed one (Whisper), with a mask over the real frames only, and asks
    `MELTAudioStack._get_output_features_shape` how many positions the decoder gets.
    Counting the mask rather than the tensor is the point: the padding never reaches
    the decoder. `seconds` should divide evenly by the stack factor and the window so
    rounding does not show; 60 s does for every configured stack.
    """
    import torch

    spec = stack.encoder.spec
    real = round(seconds / spec.frame_seconds)
    window = spec.window_frames
    padded = math.ceil(real / window) * window if window else real
    mask = torch.zeros(1, padded, dtype=torch.long)
    mask[:, :real] = 1
    feats = torch.zeros(1, padded, 1)
    _, out_mask = stack._get_output_features_shape(feats, mask)
    valid = int(out_mask.sum())
    return {
        "positions_per_audio_s": valid / seconds,
        "encoder_window_s": spec.window_seconds(),
    }


def static_numbers(config) -> dict:
    stack = build_stack(config)
    enc, ada = count_params(stack.encoder), count_params(stack.adapter)
    ada_active = active_adapter_params(stack.adapter)
    out = {
        "encoder_params_m": enc / 1e6,
        "adapter_params_m": ada / 1e6,
        "adapter_active_params_m": ada_active / 1e6,
        "stack_params_m": (enc + ada) / 1e6,
        "stack_active_params_m": (enc + ada_active) / 1e6,
    }
    out.update(positions_per_audio_second(stack))
    return out


def static_in_container(config_json: str, image: str) -> dict:
    """Run the static half in a container, for a host that has no torch (nyx).

    The container has no ssh, so the accounting half cannot run inside it; this lets
    the host process, which has ssh, hand the fetched config.json in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "config.json").write_text(config_json)
        inner = ("source /workspace/venv/bin/activate; export PYTHONPATH=/workspace/training; "
                 "cd /workspace/training; python projects/ablation-campaign/efficiency.py "
                 "static --config-json /cfg/config.json")
        out = run_cmd(["singularity", "exec", "--bind", f"{HERE.parents[1]}:/workspace/training",
                       "--bind", f"{tmp}:/cfg", image], ["bash", "-c", inner])
    return json.loads(out[out.index("{"):])


def load_config(config_json: str, overrides: list[str] = (), encoder: str | None = None,
                max_audio_seq_len: int | None = None):
    """MELTConfig from a run's config.json, optionally edited to describe another stack."""
    from transformers import AutoConfig

    from melt.modeling.configuration_melt import MELTConfig

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "config.json").write_text(config_json)
        config = MELTConfig.from_pretrained(tmp)
    if encoder:
        enc = AutoConfig.from_pretrained(encoder)
        window = {"whisper": 3000}.get(enc.model_type)
        enc.max_audio_seq_len = max_audio_seq_len or window or getattr(
            config.audio_encoder_config, "max_audio_seq_len", 1500)
        config.audio_encoder, config.audio_encoder_config = encoder, enc
    for item in overrides:
        key, val = item.split("=", 1)
        section, attr = key.split(".", 1)
        setattr(getattr(config, section), attr, json.loads(val) if val[:1] in '0123456789[{"-tfn' else val)
    return config


# --------------------------------------------------------------------------- rows


def arm_row(exp: str, args) -> dict:
    row = dict.fromkeys(COLUMNS, "")
    row.update(exp_name=exp, stage=exp.split("-", 1)[0])
    notes = []

    # Explicit --jobs is taken as the whole truth. Otherwise the union of the ledger and a
    # scan of the SLURM logs for the exp_name, so a partial ledger cannot undercount.
    jobs = args.jobs.split(",") if args.jobs else ledger_jobs(exp)
    data = fetch_arm(exp, jobs, not args.jobs, args.site, args.ssh, args.outputs, args.logs)
    if "error" in data:
        notes.append("run dir unreadable: " + data["error"])
    job_ids = data["jobs"]
    if not job_ids:
        notes.append("no jobs found: pass --jobs")
    else:
        sj = sacct(job_ids, args.ssh)
        logs = {}
        for j, lg in data["logs"].items():
            bar = parse_training_bar(lg["bars"], (data.get("state") or {}).get("max_steps"))
            if bar:
                logs[j] = {**bar, "resume_step": parse_resume_step(lg["misc"])}
        acc = summarise_jobs([sj[j] for j in job_ids if j in sj], logs)
        row.update({k: v for k, v in acc.items() if k in COLUMNS and k != "job_ids"})
        row["job_ids"] = ",".join(acc["job_ids"])
        if acc["gpu_h_mismatch_jobs"]:
            notes.append("GPU-h cross-check disagrees >1% for " + ",".join(acc["gpu_h_mismatch_jobs"]))
        running = [j for j in sj.values() if j["state"] in ("RUNNING", "PENDING")]
        if running:
            notes.append("arm still has running/pending jobs: totals are partial")

    st = data.get("state")
    rc = data.get("resolved")
    if rc:
        t = rc["data"]["train_ds"]
        eff = t["batch_duration"] * rc["trainer"]["gradient_accumulation_steps"] * rc["world_size"]
        row.update(world_size=rc["world_size"], stack_factor=rc["model"]["adapter"].get("stack_factor", ""),
                   encoder=rc["model"]["encoder"]["name"], decoder=rc["model"]["decoder"]["name"],
                   adapter=rc["model"]["adapter"]["_type"],
                   encoder_attn=rc["model"]["encoder"].get("attn_implementation", ""),
                   decoder_attn=rc["model"]["decoder"].get("attn_implementation", ""))
        if st:
            row["steps"] = st["global_step"]
            if st["global_step"] != st["max_steps"]:
                notes.append("final checkpoint is not the last step: arm not finished")
            hours = st["train_hours"]
            if hours is None:
                hours = st["global_step"] * eff / 3600
                notes.append("train_hours missing: audio hours are nominal (steps x effective batch)")
            row["train_audio_h"] = hours
            if row["gpu_h_total"] != "":
                row["gpu_h_per_1k_audio_h"] = row["gpu_h_total"] / hours * 1000
                row["gpu_h_per_1k_audio_h_clean"] = row["gpu_h_clean"] / hours * 1000

    if not args.no_static and data.get("config_json"):
        try:
            if args.image:
                row.update(static_in_container(data["config_json"], args.image))
            else:
                row.update(static_numbers(load_config(data["config_json"])))
        except Exception as e:  # e.g. the Q-Former, which cannot be instantiated today
            notes.append(f"static half failed, columns left blank: {type(e).__name__}: {str(e)[:120]}")

    if args.throughput_dir:
        p = Path(args.throughput_dir) / f"{exp}.json"
        if p.exists():
            t = json.loads(p.read_text())
            row["eval_utt_per_s"] = t["utt_per_s"]
            row["eval_conditions"] = (f"bs{t['batch_size']} {t['decoder_attn']}/{t['encoder_attn']} "
                                      f"{t['gpu']} sorted={t['sorted_by_duration']}")
    row["notes"] = "; ".join(notes)
    return row


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def emit(rows: list[dict], fmt_name: str) -> None:
    if fmt_name == "json":
        json.dump(rows, sys.stdout, indent=1)
        print()
        return
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([fmt(r[c]) for c in COLUMNS])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("row", help="efficiency row(s) for finished arm(s)")
    r.add_argument("exp_names", nargs="*")
    r.add_argument("--names-file", help="one exp_name per line")
    r.add_argument("--jobs", help="comma-separated job ids, taken as complete; default is arms.tsv plus a scan of the SLURM logs")
    r.add_argument("--site", default="mn5", choices=sorted(SITES))
    r.add_argument("--ssh", help="ssh alias to reach the cluster; omit when running on it")
    r.add_argument("--outputs", help="override the site's outputs directory")
    r.add_argument("--logs", help="override the site's SLURM logs directory")
    r.add_argument("--no-static", action="store_true", help="skip the torch half")
    r.add_argument("--image", help="run the torch half in this .sif (nyx host has no torch); "
                                   "default: in-process, which needs torch")
    r.add_argument("--throughput-dir", help="directory of eval_throughput.py outputs, <exp_name>.json")
    r.add_argument("--format", choices=["tsv", "json"], default="tsv")

    s = sub.add_parser("static", help="parameters and positions/s of a stack from a config.json")
    s.add_argument("--config-json", required=True)
    s.add_argument("--encoder", help="swap in this HF encoder config (needs it in the local HF cache)")
    s.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="e.g. adapter_config._type=conformer, adapter_config.stack_factor=5")
    s.add_argument("--max-audio-seq-len", type=int)

    a = ap.parse_args()
    if a.cmd == "static":
        cfg = load_config(Path(a.config_json).read_text(), a.set, a.encoder, a.max_audio_seq_len)
        print(json.dumps(static_numbers(cfg), indent=1))
        return
    names = list(a.exp_names)
    if a.names_file:
        names += [ln.strip() for ln in Path(a.names_file).read_text().splitlines() if ln.strip()]
    if not names:
        ap.error("give at least one exp_name")
    emit([arm_row(n, a) for n in names], a.format)


if __name__ == "__main__":
    main()
