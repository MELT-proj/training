"""Tables and the MA-hours scatter of the MA-utility study, Tier 1 (``projects/ablation-campaign/plan/04-regime.md``).

    python tier1_tables.py summaries.json langshare.json [--selection selection_tier1.json] [--out-dir DIR]
    python tier1_tables.py --collect [--logs-dir DIR] [--refs-dir DIR] --out-dir DIR      # build the two input files first

Every arm is the step-5 EU24 fine-tune (all three components trainable, the same batches, the 11,499-step schedule) and
differs only in the start it fine-tunes from; ``arms`` below lists the starts. The question: does modality alignment (MA)
before a full fine-tune buy anything that survives the fine-tune, in which format, and for how many hours?

Per arm and scored step the metrics are those of ``sft_tables.py`` (same helpers, same FLEURS-24 dev logs, same lowercase ST
scoring), so the rows of A and B equal the step-5 tables':

- ASR: WER over the five high-resource languages (HR5: en, de, fr, es, it) and CER over the other 19;
- X->en and en->X chrF, on the four HR sources / targets (de, fr, es, it) and on the other 19;
- the share of en->X outputs that are in the requested language (``lang_check.py``);
- runaways: ASR hypotheses more than twice the length of the reference, as clips out of all ASR clips;
- the campaign's selection score, when ``--selection`` is given (``selection_tier1.py``).

The figure plots one metric per row and one scored step (718, 2,872 by default) per column: x = hours of MA before the fine-tune
(symmetric log axis, so 0 and the low-hour arms are both visible), one marker per MA format, the CE arms joined by a line, and a
band around the two seed replicates (T0 / T0' at 0 h, A / A' at 1,000 h): a gap smaller than the band is not a result.
Standard library for the tables; matplotlib only for the figure.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sft_tables import FINAL_STEP, SEEN, SETS, asr_values, fmt, group_mean, mean, per_language, share_values


DEFAULT_STEPS = (718, 1436, 2154, 2872)
FIGURE_STEPS = (718, 2872)
RUN_RE = re.compile(r"^SFT-EU24-from-(?P<init>[a-z0-9-]+?)-bd60-ga4-8g-s(?P<seed>\d+)$")


@dataclass(frozen=True)
class Arm:
    """One fine-tune of the study: which start it fine-tunes from and what MA that start carries."""

    key: str
    label: str
    init: str  # the word after "-from-" in the run name
    seed: int
    ma_hours: float  # hours of MA before the fine-tune (MA 1,000 h of CE, then the extra stage)
    fmt: str  # "none", "CE", "SD-mix" or "SIFT"


ARMS = (
    Arm("T0", "T0 no MA", "noma", 1337, 0, "none"),
    Arm("T0'", "T0' no MA, seed 1338", "noma", 1338, 0, "none"),
    Arm("T1", "T1 CE 120 h", "ce120h", 1337, 120, "CE"),
    Arm("T2", "T2 CE 360 h", "ce360h", 1337, 360, "CE"),
    Arm("A", "A CE 1,000 h", "goldw", 1337, 1000, "CE"),
    Arm("A'", "A' CE 1,000 h, seed 1338", "goldw", 1338, 1000, "CE"),
    Arm("B", "B CE 1,000 + SD-mix 50 h", "sd-azeros", 1337, 1050, "SD-mix"),
    Arm("T3", "T3 CE 1,000 + CE 2,700 h", "ce24", 1337, 3700, "CE"),
    Arm("T4", "T4 CE 1,000 + SD-mix 2,700 h", "sd24", 1337, 3700, "SD-mix"),
    Arm("T5", "T5 CE 1,000 + SIFT 2,700 h", "sift24", 1337, 3700, "SIFT"),
)
BY_RUN_KEY = {(arm.init, arm.seed): arm for arm in ARMS}
REPLICATES = {"T0": "T0'", "A": "A'"}  # the arm and its seed replicate

METRICS = {  # key -> (heading, higher is better, unit shown in the table)
    "selection": ("selection score", False, ""),
    "wer_hr5": ("WER HR5 %", False, "%"),
    "cer_other": ("CER other-19 %", False, "%"),
    "xen_hr": ("X->en chrF HR4", True, ""),
    "xen_other": ("X->en chrF other-19", True, ""),
    "enx_hr": ("en->X chrF HR4", True, ""),
    "enx_other": ("en->X chrF other-19", True, ""),
    "share_hr": ("en->X in language HR4 %", True, "%"),
    "share_other": ("en->X in language other-19 %", True, "%"),
}


def classify_arm(model: str) -> dict | None:
    """``{cond, init, seed, step, final}`` for a checkpoint or final model of one of the arms, else None.

    Same keys as ``sft_tables.classify_model`` (which ``collect_sft_scores.select_logs`` expects); step 11,499 is the final model.
    """
    path = Path(model.split("//", 1)[-1])
    is_checkpoint = path.name.startswith("checkpoint-")
    run = path.parent.name if is_checkpoint else path.name
    match = RUN_RE.match(run)
    if not match or (match["init"], int(match["seed"])) not in BY_RUN_KEY:
        return None
    step = int(path.name.split("-", 1)[1]) if is_checkpoint else FINAL_STEP
    return {
        "cond": "EU24",
        "init": match["init"],
        "seed": int(match["seed"]),
        "step": step,
        "final": not is_checkpoint,
    }


def load(summaries_path: str, langshare_path: str | None) -> dict[tuple[str, int], dict[str, dict]]:
    """``cells[(arm key, step)][family] = summary row`` (``asr``, ``xen``, ``enx``) plus ``share = {target: share}``."""
    cells: dict[tuple[str, int], dict] = {}
    for row in json.loads(Path(summaries_path).read_text()):
        info = classify_arm(row["model"])
        if info is None or row["status"] != "success":
            continue
        arm = BY_RUN_KEY[(info["init"], info["seed"])]
        for family, set_name in SETS.items():
            if row["set"] == set_name:
                cells.setdefault((arm.key, info["step"]), {})[family] = row
    if langshare_path:
        counts: dict[str, dict[str, dict]] = {}
        for row in json.loads(Path(langshare_path).read_text())["rows"]:
            if row["set"] == SETS["enx"]:
                counts.setdefault(row["model"], {})[row["key"]] = row
        for model, per_key in counts.items():
            info = classify_arm(model)
            if info is not None:
                arm = BY_RUN_KEY[(info["init"], info["seed"])]
                cells.setdefault((arm.key, info["step"]), {})["share"] = share_values(per_key)
    return cells


def runaways(row: dict) -> tuple[int, int]:
    """``(runaway clips, clips)`` of an ASR summary row: hypotheses more than twice their reference's length."""
    clips = sum(v["n"] for v in row["lang"].values())
    return round(sum(v["n"] * v.get("runaway", 0.0) for v in row["lang"].values())), clips


def metrics_of(cell: dict, selection: float | None = None) -> dict:
    """The numbers of one (arm, step): ``None`` for a family that has not been scored."""
    out: dict = dict.fromkeys(METRICS) | {"runaways": None, "clips": None}
    asr = cell.get("asr")
    if asr:
        values = asr_values(asr)
        out["wer_hr5"] = 100 * mean([v for lang, v in values.items() if lang in SEEN])
        out["cer_other"] = 100 * mean([v for lang, v in values.items() if lang not in SEEN])
        out["runaways"], out["clips"] = runaways(asr)
    for family, hr, other in (("xen", "xen_hr", "xen_other"), ("enx", "enx_hr", "enx_other")):
        if cell.get(family):
            values = per_language(cell[family], "chrf")
            out[hr], out[other] = group_mean(values, True), group_mean(values, False)
    if cell.get("share"):
        out["share_hr"], out["share_other"] = (
            100 * group_mean(cell["share"], True),
            100 * group_mean(cell["share"], False),
        )
    out["selection"] = selection
    return out


def load_selection(path: str | None) -> dict[tuple[str, int], float]:
    """``{(arm key, step): selection score}`` from ``selection_tier1.json`` rows ``{arm, step, result: {score} | None}``."""
    if not path:
        return {}
    rows = json.loads(Path(path).read_text())
    return {(r["arm"], r["step"]): r["result"]["score"] for r in rows if r.get("result")}


def build(cells: dict, selection: dict, steps: tuple[int, ...] = DEFAULT_STEPS) -> dict:
    """Everything the tables and the figure show: ``rows[step][arm key] = metrics`` and the seed-replicate gaps."""
    seen_steps = sorted({step for (_, step) in cells} | {step for (_, step) in selection})
    rows: dict[int, dict[str, dict]] = {}
    for step in seen_steps:
        rows[step] = {}
        for arm in ARMS:
            cell = cells.get((arm.key, step))
            if cell is not None or (arm.key, step) in selection:
                rows[step][arm.key] = metrics_of(cell or {}, selection.get((arm.key, step)))
    gaps: dict[int, dict[str, dict]] = {}
    for step, by_arm in rows.items():
        for first, second in REPLICATES.items():
            if first in by_arm and second in by_arm:
                gaps.setdefault(step, {})[f"{first} vs {second}"] = {
                    m: abs(by_arm[first][m] - by_arm[second][m])
                    for m in METRICS
                    if by_arm[first][m] is not None and by_arm[second][m] is not None
                }
    return {"steps": [s for s in seen_steps if s in steps or s > max(steps)], "rows": rows, "replicate_gaps": gaps}


# --- markdown -----------------------------------------------------------------------------------------------------


def cell_text(value: float | None, key: str) -> str:
    if value is None:
        return "-"
    return fmt(value, 4 if key == "selection" else 1)


def table_for(by_arm: dict[str, dict], keys: tuple[str, ...], with_runaways: bool = True) -> str:
    head = ["arm", "MA h", *(METRICS[k][0] for k in keys), *(["runaways"] if with_runaways else [])]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|"]
    for arm in ARMS:
        m = by_arm.get(arm.key)
        if m is None:
            continue
        runaway = "-" if m["runaways"] is None else f"{m['runaways']}/{m['clips']}"
        cells = [
            arm.label,
            f"{arm.ma_hours:,.0f}",
            *(cell_text(m[k], k) for k in keys),
            *([runaway] if with_runaways else []),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(result: dict) -> str:
    out = [
        "# MA utility, Tier 1: fine-tunes from different starts",
        "",
        "Every row is the step-5 EU24 fine-tune (SEED 1337 unless the label says 1338; same batches); only the start differs. "
        "FLEURS-24 dev, 100 clips per language or direction, ST scored case-insensitively (plain chrF, as in the step-5 tables). "
        "HR5 = en, de, fr, es, it; HR4 = de, fr, es, it. One step is 1,920 s of audio nominal, about 85% filled. "
        "MA h counts the hours of the MA stages before the fine-tune.",
        "",
    ]
    asr_st = ("wer_hr5", "cer_other", "xen_hr", "xen_other", "enx_hr", "enx_other")
    for step in result["steps"]:
        by_arm = result["rows"].get(step, {})
        if not by_arm:
            continue
        out += [f"## Step {step}" + (" (final model)" if step == FINAL_STEP else ""), ""]
        out += [table_for(by_arm, ("selection", *asr_st)), ""]
        out += [table_for(by_arm, ("share_hr", "share_other"), with_runaways=False), ""]
    gaps = result["replicate_gaps"]
    if gaps:
        out += [
            "## Seed replicates",
            "",
            "Absolute gap between an arm and its seed-1338 replicate; a difference between arms below this is noise.",
            "",
        ]
        keys = [k for k in METRICS if any(k in g for per in gaps.values() for g in per.values())]
        out += ["| step | pair | " + " | ".join(METRICS[k][0] for k in keys) + " |", "|---|---|" + "---:|" * len(keys)]
        for step in sorted(gaps):
            for pair, gap in gaps[step].items():
                out.append(f"| {step} | {pair} | " + " | ".join(cell_text(gap.get(k), k) for k in keys) + " |")
        out.append("")
    return "\n".join(out)


OBJECTIVE = {  # the "MA objective" column of plan/04-regime.md §7
    "T0": "none",
    "T0'": "none",
    "T1": "CE",
    "T2": "CE",
    "A": "CE",
    "A'": "CE",
    "B": "CE + SD-mix",
    "T3": "CE + CE24",
    "T4": "CE + SD-mix24",
    "T5": "CE + SIFT24",
}
SECTION7_HEAD = (
    "point",
    "MA objective",
    "MA h",
    "step",
    "WER HR5",
    "CER other-19",
    "X→en HR4 / other",
    "en→X HR4 / other",
    "selection",
    "GPU-h",
)


def bare(value: float | None, digits: int, scale: float = 1.0) -> str:
    """A fraction as §7 prints it, without the leading zero (``.092``); ``-`` when the number is not there yet."""
    if value is None:
        return "-"
    return f"{value / scale:.{digits}f}".removeprefix("0")


def render_section7(result: dict, steps: tuple[int, ...] = FIGURE_STEPS) -> str:
    """The results table of ``plan/04-regime.md`` §7: one row per arm, the scored steps side by side inside each cell.

    WER and CER are fractions (the tables above hold percent), chrF pairs read ``HR4 / other-19``, and a step that has not
    been scored yet shows ``-``. The GPU-h column is left empty: it is the cost accounting's, not the scorer's.
    """
    lines = ["| " + " | ".join(SECTION7_HEAD) + " |", "|" + "|".join("---" for _ in SECTION7_HEAD) + "|"]
    for arm in ARMS:
        ms = [result["rows"].get(step, {}).get(arm.key) for step in steps]
        if not any(ms):
            continue

        def joined(key: str, digits: int = 3, scale: float = 100.0, ms=ms) -> str:
            return " / ".join(bare(m[key], digits, scale) if m else "-" for m in ms)

        def pairs(first: str, second: str, ms=ms) -> str:
            return ", ".join(
                f"{cell_text(m[first], first)} / {cell_text(m[second], second)}" if m else "-" for m in ms
            )

        cells = [
            arm.key,
            OBJECTIVE[arm.key],
            f"{arm.ma_hours:,.0f}",
            " / ".join(f"{step:,}" for step in steps),
            joined("wer_hr5"),
            joined("cer_other"),
            pairs("xen_hr", "xen_other"),
            pairs("enx_hr", "enx_other"),
            joined("selection", 4, 1.0),
            "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- the figure ---------------------------------------------------------------------------------------------------

MARKERS = {"none": ("X", "#555555"), "CE": ("o", "#1f77b4"), "SD-mix": ("s", "#d62728"), "SIFT": ("^", "#2ca02c")}


def figure(
    result: dict, path: Path, metric_keys: tuple[str, ...], steps: tuple[int, ...] = FIGURE_STEPS
) -> list[Path]:
    """Write ``path`` (.png) and its .pdf twin: one row per metric, one column per step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(metric_keys), len(steps), figsize=(4.6 * len(steps), 2.7 * len(metric_keys)), squeeze=False
    )
    for col, step in enumerate(steps):
        by_arm = result["rows"].get(step, {})
        for row, key in enumerate(metric_keys):
            ax = axes[row][col]
            points = {
                a.key: (a, by_arm[a.key][key]) for a in ARMS if a.key in by_arm and by_arm[a.key][key] is not None
            }
            for fmt_name, (marker, color) in MARKERS.items():
                xs = [a.ma_hours for a, _ in points.values() if a.fmt == fmt_name]
                ys = [y for a, y in points.values() if a.fmt == fmt_name]
                if xs:
                    ax.scatter(
                        xs,
                        ys,
                        marker=marker,
                        color=color,
                        s=46,
                        label=fmt_name if (row, col) == (0, 0) else None,
                        zorder=3,
                    )
            ce = sorted(
                (a.ma_hours, y)
                for a, y in points.values()
                if a.fmt in ("CE", "none") and a.key not in REPLICATES.values()
            )
            if len(ce) > 1:
                ax.plot(*zip(*ce), color="#1f77b4", alpha=0.35, zorder=2)
            for arm, y in points.values():
                ax.annotate(
                    arm.key, (arm.ma_hours, y), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#444"
                )
            for first, second in REPLICATES.items():
                if first in points and second in points:
                    pair = [points[first][1], points[second][1]]
                    x = points[first][0].ma_hours
                    ax.vlines(x, min(pair), max(pair), color="#888888", alpha=0.35, linewidth=9, zorder=1)
            ax.set_xscale("symlog", linthresh=60)
            ax.set_xlim(-20, 6000)
            ax.set_xticks([0, 120, 360, 1000, 3700])
            ax.set_xticklabels(["0", "120", "360", "1k", "3.7k"])
            if col == 0:
                ax.set_ylabel(METRICS[key][0])
            if row == 0:
                ax.set_title(f"after {step:,} fine-tune steps")
            ax.grid(alpha=0.25)
            if row == len(metric_keys) - 1:
                ax.set_xlabel("MA hours before the fine-tune")
    from matplotlib.lines import Line2D

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#888888", alpha=0.35, linewidth=9))
    labels.append("seed spread (T0/T0', A/A')")
    fig.tight_layout(rect=(0, 0.035, 1, 1))  # room under the panels for the legend, which then never covers a point
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    written = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=160)
        written.append(target)
    plt.close(fig)
    return written


# --- collecting the logs (needs the melt-eval venv) -------------------------------------------------------------------


def collect(logs_dir: str, out_dir: Path, refs_dir: str, since: str) -> None:
    """Write ``summaries.json`` and ``langshare.json`` for the arms, as ``collect_sft_scores.py`` does for A and B."""
    import collect_sft_scores as c

    c.collect(logs_dir, out_dir, refs_dir, since, classify=classify_arm)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries", nargs="?")
    p.add_argument("langshare", nargs="?")
    p.add_argument("--selection", default=None, help="selection_tier1.json: rows {arm, step, result: {score}}")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--collect", action="store_true", help="build summaries.json / langshare.json from the logs first")
    p.add_argument("--logs-dir", default="/mnt/scratch-artemis/giuseppe/melt-data/eval-logs")
    p.add_argument("--refs-dir", default="/mnt/scratch-artemis/giuseppe/tmp/fleurs-refs24")
    p.add_argument("--since", default="2026-10-11")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args.logs_dir, args.out_dir, args.refs_dir, args.since)
        args.summaries, args.langshare = str(args.out_dir / "summaries.json"), str(args.out_dir / "langshare.json")
    if not args.summaries:
        p.error("give summaries.json (and langshare.json) or --collect")
    result = build(load(args.summaries, args.langshare), load_selection(args.selection))
    (args.out_dir / "tier1_tables.md").write_text(render(result))
    (args.out_dir / "tier1_tables.json").write_text(
        json.dumps({"arms": [a.__dict__ for a in ARMS], **result}, indent=1)
    )
    section7 = render_section7(result)
    (args.out_dir / "tier1_section7.md").write_text(section7 + "\n")
    print(render(result))
    print("## Results table (plan/04-regime.md §7 layout)\n")
    print(section7)
    if not args.no_figure:
        written = figure(
            result, args.out_dir / "tier1_scatter", ("selection", "wer_hr5", "cer_other", "xen_hr", "enx_hr")
        )
        print("figure:", ", ".join(map(str, written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
