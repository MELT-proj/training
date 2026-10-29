"""Join the campaign's crossing rows to the ledger: one line per submitted run, 'cell<TAB>role<TAB>jobid<TAB>exp_name'.

usage: crossing_job_list.py <worktree>   (a checkout of main: reads campaign.yaml, arms.tsv and the planner)

role = final      the run the cell's current row renders to (the one the selection metric scored)
       discarded  every other crossing submission of the ledger (diverged, cancelled or superseded runs)

The job ids come out on stdout only, for the MN5 collector (collect_cost.sh); keep that output out of the repo.
"""

import re
import subprocess
import sys
from pathlib import Path


ENC = {"w2vbF": "w2vb", "mms1bF": "mms1b", "mhubert147F": "mhubert", "whisperlargeF": "whisper"}
ADA = {"mlpT": "mlp", "moeT": "moe", "qformerT": "qformer", "conformerT": "conformer"}


def cell_of(exp: str) -> str:
    """Cell label of a run name that no current row renders to: encoder-adapter plus its variant tags."""
    toks = exp.split("-")
    enc = next((ENC[t] for t in toks if t in ENC), "?")
    ada = next((ADA[t] for t in toks if t in ADA), "?")
    tags = [t for t in toks if t == "qkn" or re.fullmatch(r"lr\d+e\d+", t) or re.fullmatch(r"s\d+", t)]
    return "-".join([enc, ada, *tags])


def final_runs(worktree: str) -> dict[str, str]:
    """Map the exp_name each `MA-2100-crossing-*` row renders to onto the row's cell label."""
    root = Path(worktree).resolve()
    camp = root / "projects/ablation-campaign"
    text = (camp / "campaign.yaml").read_text(encoding="utf-8")
    ids = re.findall(r"^\s*- id: (MA-2100-crossing-\S+)\s*$", text, re.MULTILINE)
    plan = subprocess.run(
        ["python3", str(camp / "campaign.py"), "plan", *ids], capture_output=True, text=True, check=True, cwd=root
    ).stdout
    final: dict[str, str] = {}
    cur = None
    for line in plan.splitlines():
        m = re.match(r"=== (MA-2100-crossing-\S+) ===", line)
        if m:
            cur = m.group(1).removeprefix("MA-2100-crossing-")
        m = re.match(r"\s+exp_name\s+(\S+)", line)
        if m and cur:
            final[m.group(1)] = cur
    return final


def main(worktree: str) -> None:
    """Print one tab-separated line per crossing submission in the ledger."""
    final = final_runs(worktree)
    ledger = Path(worktree) / "projects/ablation-campaign/arms.tsv"
    for line in ledger.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[1].startswith("MA-2100asr-"):
            continue
        exp, job = parts[1], parts[2]
        if exp in final:
            print(f"{final[exp]}\tfinal\t{job}\t{exp}")
        else:
            print(f"{cell_of(exp)}\tdiscarded\t{job}\t{exp}")


if __name__ == "__main__":
    main(sys.argv[1])
