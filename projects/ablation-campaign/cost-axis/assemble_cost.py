"""Join the cost collector's raw TSV with the cells' LR / QK-norm (from the run names) and the selection-metric scores.

usage: assemble_cost.py <joblist.tsv> <cost_raw.tsv> <scores-file-with-ranking> > table.tsv

Job ids are used only to join the job list and the raw TSV; they are not written to the output.
"""

import csv
import re
import sys
from pathlib import Path


COLS = [
    "cell",
    "role",
    "state",
    "adapter_lr",
    "qk_norm",
    "nodes",
    "gpus",
    "steps",
    "wall_h",
    "s_per_step",
    "gpu_h",
    "audio_h",
    "gpu_h_per_1k_audio_h",
    "wall_s_per_audio_h",
    "trainable_params_M",
    "peak_gb",
    "racks",
    "id_cer",
    "score",
]
RANK_ROW = re.compile(r"ma2100-(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def lr_of(exp: str) -> str:
    """Adapter LR of a run, from the `lr2e3` tag of its name (as `2e-3`)."""
    m = re.search(r"-lr(\d)e(\d)-", exp)
    return f"{m.group(1)}e-{m.group(2)}" if m else "?"


def read_scores(path: str) -> tuple[dict[str, float], dict[str, float]]:
    """ID median CER and score per cell, from the ranking table at the end of a scores file."""
    id_of: dict[str, float] = {}
    score_of: dict[str, float] = {}
    in_rank = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# Ranking of the"):
            in_rank = True
        elif in_rank and (m := RANK_ROW.match(line)):
            id_of[m.group(1)] = float(m.group(2))
            score_of[m.group(1)] = float(m.group(7))
    return id_of, score_of


def main(joblist: str, raw: str, scores: str) -> None:
    """Write the cost table (tab separated) to stdout."""
    exp_of: dict[tuple[str, str], str] = {}
    for line in Path(joblist).read_text(encoding="utf-8").splitlines():
        cell, role, _job, exp = line.split("\t")
        exp_of[(cell, role)] = exp
    id_of, score_of = read_scores(scores)
    with open(raw, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    out.writerow(COLS)
    for r in rows:
        cell, role = r["cell"], r["role"]
        exp = exp_of.get((cell, role), "")
        adapter = next((a for a in ("conformer", "qformer", "mlp", "moe") if a in cell), "?")
        qk = "yes" if "-qkn-" in exp else ("no" if adapter in ("conformer", "qformer") else "-")
        steps = int(r["steps"]) if r["steps"].isdigit() else 0
        wall = float(r["elapsed_h"])
        gpu_h = float(r["gpu_h"])
        audio = float(r["train_hours"]) if r["train_hours"] not in ("", "NA") else 0.0
        sps = wall * 3600 / steps if steps else float("nan")
        per1k = gpu_h / audio * 1000 if audio else float("nan")
        wsah = wall * 3600 / audio if audio else float("nan")
        params = int(r["trainable_params"]) / 1e6 if r["trainable_params"].isdigit() else float("nan")
        scored = role == "final" and cell in score_of
        out.writerow(
            [
                cell,
                role,
                r["state"],
                lr_of(exp),
                qk,
                r["nodes"],
                r["gpus"],
                steps,
                f"{wall:.2f}",
                f"{sps:.3f}",
                f"{gpu_h:.1f}",
                f"{audio:.0f}",
                f"{per1k:.2f}",
                f"{wsah:.2f}",
                f"{params:.1f}",
                r["peak_gb"],
                r["racks"],
                f"{id_of[cell]:.4f}" if scored else "",
                f"{score_of[cell]:.4f}" if scored else "",
            ]
        )


if __name__ == "__main__":
    main(*sys.argv[1:4])
