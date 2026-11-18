"""Stream F driver: noisy copies of every Stream A/C/E overlay and of the native LR tail, one `noise_mix.main` call each.

Sources (all on nyx scratch):
  A     giga_en, cv22_en, vp_en, granary_en       30% of the cuts that have a translation, one copy
  C     c_cv22_*, c_vp_*, y2lr_*                  same
  E     tts_en                                    same
  tail  the leaves of plan.json (plan_noisy_tail.py): every cut that passes the anneal WER band, two copies

Each source writes `<out_root>/<name>_noisy/train` and `noise_stats.json`; a source whose stats file exists is skipped,
so a rerun resumes. `--dry-run` sizes the work from the manifests alone (no catalog read, nothing written). At the end
`leaves.json` lists every noisy corpus with what a training leaf needs (path, source, stream, cuts, hours, tags).
Run inside synth.sif.
"""

import json
import time
from pathlib import Path

import tyro

from noise_mix import list_shards
from noise_mix import main as noise_mix


SHAR_OUT = "/mnt/scratch-nyx/giuseppe/synth/shar-out"
OUT_ROOT = "/mnt/scratch-nyx/giuseppe/synth/shar-noisy"
CATALOG = "/mnt/scratch-nyx/giuseppe/synth/noise/catalog.json"
A_NAMES = ("giga_en", "cv22_en", "vp_en", "granary_en")


def st_sources(shar_out: str) -> list[tuple[str, str]]:
    found = []
    for p in sorted(Path(shar_out).iterdir()):
        if not (p / "train").is_dir():
            continue
        if p.name in A_NAMES:
            found.append((p.name, "A"))
        elif p.name.startswith(("c_", "y2lr_")):
            found.append((p.name, "C"))
        elif p.name == "tts_en":
            found.append((p.name, "E"))
    return found


def main(
    shar_out: str = SHAR_OUT,
    out_root: str = OUT_ROOT,
    catalog: str = CATALOG,
    plan: str = f"{OUT_ROOT}/plan/plan.json",
    streams: tuple[str, ...] = ("A", "C", "E", "tail"),
    fraction: float = 0.3,
    tail_copies: int = 2,
    workers: int = 40,
    only: str = "",
    limit: int = 0,
    dry_run: bool = False,
) -> None:
    jobs = []
    for name, stream in st_sources(shar_out):
        if stream in streams:
            jobs.append({"name": name, "stream": stream, "kind": "st", "src": f"{shar_out}/{name}/train",
                         "args": {"fraction": fraction, "require_field": "synth.ok", "copies": 1}})  # fmt: skip
    if "tail" in streams:
        for p in json.loads(Path(plan).read_text())["plan"]:
            jobs.append({"name": p["name"].removesuffix("_noisy"), "stream": "tail", "kind": "asr", "src": p["src_dir"],
                         "args": {"keep_mask": p["mask"], "copies": tail_copies, "stream": "asr_tail"},
                         "tags": p["tags"], "rel": p["rel"]})  # fmt: skip
    jobs = [j for j in jobs if only in j["name"]]
    rows, leaves = [], []
    for j in jobs:
        out = f"{out_root}/{j['name']}_noisy"
        stats_file = Path(out) / "noise_stats.json"
        n_workers = max(1, min(workers, len(list_shards(j["src"]))))
        if dry_run:
            rep = noise_mix(src_dir=j["src"], out=out, catalog=catalog, workers=n_workers, dry_run=True, **j["args"])
            rows.append((j["stream"], j["name"], rep["selected_cuts"], rep["noisy_hours"]))
            continue
        if stats_file.exists():
            rep = json.loads(stats_file.read_text())
            print(f"skip {j['name']}: done ({rep['noisy_cuts']} cuts)")
        else:
            t0 = time.time()
            rep = noise_mix(src_dir=j["src"], out=out, catalog=catalog, workers=n_workers, limit=limit, **j["args"])
            with open(Path(out_root) / "ledger.tsv", "a") as f:
                f.write(f"{time.strftime('%FT%T')}\t{j['stream']}\t{j['name']}\t{rep['source_cuts']}\t"
                        f"{rep['noisy_cuts']}\t{rep['hours']}\t{round(time.time() - t0)}\n")  # fmt: skip
        leaves.append({"name": f"{j['name']}_noisy", "stream": j["stream"], "kind": j["kind"], "source": j["name"],
                       "path": f"{j['name']}_noisy/train", "cuts": rep["noisy_cuts"], "hours": rep["hours"],
                       "copies": rep["copies"], "tags": j.get("tags"), "rel": j.get("rel")})  # fmt: skip
    if dry_run:
        print(f"{'stream':6} {'source':32} {'cuts':>9} {'noisy h':>9}")
        for s, n, c, h in rows:
            print(f"{s:6} {n:32} {c:9d} {h:9.1f}")
        for s in sorted({r[0] for r in rows}):
            print(
                f"{s:6} {'TOTAL':32} {sum(r[2] for r in rows if r[0] == s):9d} {sum(r[3] for r in rows if r[0] == s):9.1f}"
            )
        print(f"all    {'TOTAL':32} {sum(r[2] for r in rows):9d} {sum(r[3] for r in rows):9.1f}")
        return
    leaves_file = Path(out_root, "leaves.json")
    merged = {x["name"]: x for x in json.loads(leaves_file.read_text())} if leaves_file.exists() else {}
    merged.update({x["name"]: x for x in leaves})  # a rerun of one stream keeps the others' entries
    leaves_file.write_text(json.dumps(list(merged.values()), indent=1))
    print(f"{len(merged)} noisy corpora, {sum(x['hours'] for x in merged.values()):.1f} h -> {leaves_file}")


if __name__ == "__main__":
    tyro.cli(main)
