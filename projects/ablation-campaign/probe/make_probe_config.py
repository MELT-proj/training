#!/usr/bin/env python3
"""Training config for the language-transfer probe (plan/02a-backbone-language-transfer.md, Step B).

Takes one IFT backbone's ``training_config.yaml`` and replaces its ``train_ds`` mix by

* the probe: ``--hours-per-lang`` hours of ASR in each of pt, ro, nl, da, sv, drawn from the corpora that hold usable
  text for that language, in proportion to the hours each holds, and
* replay: ``--replay-ratio`` times as many hours drawn from the original IFT mix (five languages, ASR and ST), in
  the original groups' hour proportions, so the probe does not measure forgetting.

Everything else (model, optimiser, per-arm batch duration) is the checkpoint's own. Weights are cut probabilities
that enforce the hours (``cut_weight`` below), as in build_campaign_config.py. The run is a continuation: the
checkpoint is ``model.ckpt``; step count and effective batch are set so the run draws the probe's total audio.

    make_probe_config.py --template CKPT/training_config.yaml --ckpt CKPT --gpus 4 --out probe-qwen35_2bInsT.yaml
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_campaign_config as bcc  # noqa: E402


PROBE_LANGS = ["pt", "ro", "nl", "da", "sv"]
# lang -> [(shar path under the datasets root, text_field or None for the supervision)]
# Only sources whose text is usable today (preprocessing docs/pnc_coverage.md, 2026-10): VoxPopuli nl and ro are
# MIXED (some splits lack pnc_text) and are left out; CommonVoice keeps the text outside the supervision.
SOURCES = {
    "pt": [
        ("cv22_sidon/pt/train", "custom.metadata.sentence"),
        ("mls_sidon/portuguese/train", "custom.pnc_text"),
        ("yodas-granary/Portuguese/asr_only", None),
        ("yodas-granary/Portuguese/ast", None),
        ("fleurs/pt_br/train", "custom.pnc_text"),
    ],
    "ro": [
        ("cv22_sidon/ro/train", "custom.metadata.sentence"),
        ("yodas-granary/Romanian/asr_only", None),
        ("yodas-granary/Romanian/ast", None),
        ("fleurs/ro_ro/train", "custom.pnc_text"),
    ],
    "nl": [
        ("cv22_sidon/nl/train", "custom.metadata.sentence"),
        ("mls_sidon/dutch/train", "custom.pnc_text"),
        ("yodas-granary/Dutch/asr_only", None),
        ("yodas-granary/Dutch/ast", None),
        ("fleurs/nl_nl/train", "custom.pnc_text"),
    ],
    "da": [
        ("cv22_sidon/da/train", "custom.metadata.sentence"),
        ("yodas-granary/Danish/asr_only", None),
        ("yodas-granary/Danish/ast", None),
        ("fleurs/da_dk/train", "custom.pnc_text"),
    ],
    "sv": [
        ("cv22_sidon/sv-SE/train", "custom.metadata.sentence"),
        ("yodas-granary/Swedish/asr_only", None),
        ("yodas-granary/Swedish/ast", None),
        ("fleurs/sv_se/train", "custom.pnc_text"),
    ],
}
SHAR_IN_CONTAINER = "/workspace/shar"
EFFECTIVE_BATCH_SECONDS = 4800.0  # batch_duration x grad_accum x GPUs of the IFT runs (30x20x8 = 75x16x4)


def leaf_path(shar_path: str) -> str:
    return shar_path.removeprefix(SHAR_IN_CONTAINER + "/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, type=Path, help="the IFT checkpoint's training_config.yaml")
    ap.add_argument("--ckpt", required=True, help="model.ckpt for the probe run (a finished IFT run root)")
    ap.add_argument("--datasets-root", type=Path, default=Path("/mnt/scratch-nyx/giuseppe/melt/melt-data/shar"))
    ap.add_argument("--cache", type=Path, default=None, help="measured-hours cache (infra/.config_check_cache.json)")
    ap.add_argument("--hours-per-lang", type=float, default=29.0)
    ap.add_argument("--replay-ratio", type=float, default=1.0)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--output-dir", default="/workspace/outputs")
    ap.add_argument("--summary", type=Path, default=None, help="write the hours accounting here as JSON")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.template.read_text())
    train = cfg["data"]["train_ds"]
    old_groups = train["input_cfg"]

    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}
    probe_paths = [p for lang in PROBE_LANGS for p, _ in SOURCES[lang]]
    replay_paths = [leaf_path(leaf["shar_path"]) for g in old_groups for leaf in g["input_cfg"]]
    need = set(probe_paths) | set(replay_paths)
    hours, cuts = bcc.measure(sorted(need), args.datasets_root, cache, args.cache, None, args.jobs, need)
    for p in sorted(need):
        if not hours.get(p) or not cuts.get(p):
            sys.exit(f"no hours/cuts for {p}: unreadable or empty source")

    def cut_sec(path: str) -> float:
        return hours[path] * 3600.0 / cuts[path]

    # --- probe groups: natural mix inside a language, equal hours across languages ---
    new_groups: list[dict] = []
    summary: dict = {"probe": {}, "replay": {}}
    for lang in PROBE_LANGS:
        pool = sum(hours[p] for p, _ in SOURCES[lang])
        want = args.hours_per_lang
        if pool < 0.98 * want:
            sys.exit(f"{lang}: pool {pool:.1f} h < {want} h per language")
        want = min(want, pool)  # a pool within 2% of the budget (da: 28.99 h) is drawn once, whole
        leaves, ncuts = [], []
        for path, text_field in SOURCES[lang]:
            drawn_h = want * hours[path] / pool
            n = drawn_h * 3600.0 / cut_sec(path)
            tags = {"task": "asr", "lang": lang}
            if text_field:
                tags["text_field"] = text_field
            leaves.append(
                {"type": "lhotse_shar", "shar_path": f"{SHAR_IN_CONTAINER}/{path}", "weight": n, "tags": tags}
            )
            ncuts.append(n)
            summary["probe"].setdefault(lang, {})[path] = {
                "pool_h": round(hours[path], 2),
                "drawn_h": round(drawn_h, 2),
            }
        total = sum(ncuts)
        for leaf in leaves:
            leaf["weight"] = leaf["weight"] / total
        new_groups.append(
            {"type": "group", "weight": total, "tags": {"task": "asr", "lang": lang}, "input_cfg": leaves}
        )

    # --- replay groups: the original mix, scaled to replay_ratio x the probe's hours ---
    probe_total_h = args.hours_per_lang * len(PROBE_LANGS)
    replay_h = probe_total_h * args.replay_ratio
    mean = []
    for g in old_groups:
        sec = sum(leaf["weight"] * cut_sec(leaf_path(leaf["shar_path"])) for leaf in g["input_cfg"])
        mean.append(sec)  # leaves' weights sum to 1 (cut probabilities), so this is the group's mean cut length
    share = [g["weight"] * m for g, m in zip(old_groups, mean)]
    share = [s / sum(share) for s in share]
    for g, m, s in zip(old_groups, mean, share):
        ng = copy.deepcopy(g)
        ng["weight"] = replay_h * s * 3600.0 / m
        new_groups.append(ng)
        key = "/".join(str(v) for v in g["tags"].values())
        summary["replay"][key] = round(replay_h * s, 2)

    gsum = sum(g["weight"] for g in new_groups)
    for g in new_groups:
        g["weight"] = round(g["weight"] / gsum, 8)
        for leaf in g["input_cfg"]:
            leaf["weight"] = round(leaf["weight"], 8)

    total_h = probe_total_h + replay_h
    train["input_cfg"] = new_groups
    train["total_hours"] = round(total_h, 2)
    train["shuffle"] = True

    # --- run, schedule, evaluation ---
    bd = train["batch_duration"]
    gpus = args.gpus
    ga = EFFECTIVE_BATCH_SECONDS / (bd * gpus)
    if ga != int(ga):
        sys.exit(f"batch_duration {bd} x {gpus} GPUs does not divide {EFFECTIVE_BATCH_SECONDS:.0f} s")
    ga = int(ga)
    steps = round(total_h * 3600.0 / EFFECTIVE_BATCH_SECONDS)
    tr = cfg["trainer"]
    name = args.exp_name or f"PROBE-{Path(args.ckpt).name}"
    cfg["run"]["exp_name"] = name
    cfg["run"]["config"] = "projects/ablation-campaign/probe/make_probe_config.py"
    cfg["model"]["ckpt"] = args.ckpt
    tr["output_dir"] = f"{args.output_dir}/{name}"
    tr["gradient_accumulation_steps"] = ga
    tr["max_steps"] = steps
    tr["warmup_steps"] = max(1, round(0.05 * steps))
    # checkpoints at 10 h per language (the first point of the learning curve) and at the end
    first = round(10.0 / args.hours_per_lang * steps)
    tr["save_steps"] = first
    tr["save_total_limit"] = 4
    tr["eval_strategy"] = "steps"
    tr["eval_steps"] = 10**9  # offline scoring with melt-eval; an in-trainer pass would use the old validation mix
    tr["eval_on_start"] = False
    tr["do_eval"] = False
    tr["resume_from_checkpoint"] = None
    cfg["data"]["validation_ds"]["max_samples"] = 1

    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000))
    summary.update(total_hours=total_h, steps=steps, grad_accum=ga, gpus=gpus, first_save_step=first)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=1))
    print(f"{args.out}: {total_h:.0f} h, {steps} steps, ga {ga} x {gpus} GPUs, save every {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
