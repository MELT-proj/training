#!/usr/bin/env python3
"""Figure: FLEURS word error rate of our checkpoints next to public audio LLMs, language by language.

    python fig_fleurs_asr_vs_baselines.py [--context paper|poster] [--ymax 50]
        [--langs en de ...] [--ours KEY ...] [--baselines KEY ...] [--include-start] [--labels|--no-labels]
        [--master results_master.csv] [--out-dir figures] [--formats pdf png]

Standalone: standard library, numpy and matplotlib only, and it reads nothing but ``results_master.csv``. Every
number, label and bar order comes from that file (rows with ``kind`` ours / baseline and ``status`` active; bar order
is the ``order`` column), so the figure is a pure function of the master and the command line. The output files carry
the master's SHA-256 so a figure can be traced to the exact table it was drawn from.

What is compared: the baselines' published WER (set ``fleurs-reported``) against our checkpoints scored by melt-eval
on FLEURS dev (set ``fleurs24-asr-dev``). They are not scored alike (our clips are a frozen 100-per-language subset
of the dev split, our normaliser is Whisper's multilingual ``basic``; the table the baselines come from states
neither its split nor its normaliser), so the figure says so in the caption this script prints, and a language
enters only if at least one baseline and at least one of our models has a number for it.

Bars above ``--ymax`` are cut at the top with a break mark and carry their value, so the languages every model
handles (5-25 % WER) stay readable next to the ones some baselines fail (Polish, up to 99 %).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_MASTER = HERE / "results_master.csv"

# Order of the languages in the figure: the baseline table's own (English first), then the rest alphabetically.
LANG_ORDER = ("en", "de", "fr", "it", "es", "pl", "uk")
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "it": "Italian", "es": "Spanish", "pl": "Polish",
    "uk": "Ukrainian", "nl": "Dutch", "pt": "Portuguese", "cs": "Czech", "sk": "Slovak", "sl": "Slovenian",
    "hu": "Hungarian", "fi": "Finnish", "sv": "Swedish", "da": "Danish", "ro": "Romanian", "bg": "Bulgarian",
    "el": "Greek", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian", "hr": "Croatian", "ga": "Irish",
    "mt": "Maltese", "ru": "Russian",
}  # fmt: skip

# Presentation. Colour = recipe (the same in every decoder family), white hatch = EuroLLM decoder, greys = baselines.
RECIPE_COLOUR = {
    "full-ft": "#0072B2",
    "adapter-ce": "#56B4E9",
    "adapter-sd": "#D55E00",
    "adapter-sd-sift": "#E69F00",
    "start": "#009E73",
}
FAMILY_HATCH = {"eurollm-1.7b": "////"}
BASELINE_COLOURS = ("#37474F", "#607D8B", "#90A4AE", "#B0BEC5", "#E3E9EC")
FALLBACK_COLOUR = "#CC79A7"

CONTEXTS = {
    "paper": {"width": 7.0, "height": 3.2, "font": 7.0, "tick": 6.5, "legend": 6.3, "value": 4.4, "lw": 0.5, "dpi": 300, "labels": True},
    "poster": {"width": 14.0, "height": 6.6, "font": 17, "tick": 15, "legend": 14, "value": 9, "lw": 1.2, "dpi": 200, "labels": False},
}  # fmt: skip


def load_master(path: Path) -> dict[str, dict[str, str]]:
    """The master as ``{model: {column: text}}``, refusing a file with conflict markers or ragged rows."""
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            sys.exit(f"{path}:{number}: git conflict marker; see PROTOCOL.md, section Recovery")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if not header or header[0] != "model":
        sys.exit(f"{path}: not a master table (the first column must be 'model')")
    rows: dict[str, dict[str, str]] = {}
    for number, row in enumerate(reader, 2):
        if len(row) != len(header):
            sys.exit(f"{path}:{number}: {len(row)} fields, header has {len(header)}; run master.py validate")
        rows[row[0]] = dict(zip(header, row, strict=True))
    return rows


def number(row: dict[str, str], prefix: str, lang: str) -> float | None:
    cell = row.get(f"{prefix}.{lang}", "")
    return float(cell) if cell else None


def pick(rows, kind, prefix, keys, include_start):
    """Keys of the active rows of *kind* that have at least one cell under *prefix*, in ``order``."""
    chosen = []
    for key, row in rows.items():
        if row["kind"] != kind or row["status"] != "active":
            continue
        if keys and key not in keys:
            continue
        if row["recipe"] == "start" and not include_start and not keys:
            continue
        if any(v for col, v in row.items() if col.startswith(prefix + ".") and v):
            chosen.append(key)
    chosen.sort(key=lambda k: (float(rows[k]["order"] or 1e9), k))
    if keys:
        absent = [k for k in keys if k not in chosen]
        if absent:
            sys.exit(f"no {kind} data under {prefix} for: {', '.join(absent)}")
    return chosen


def bar_style(row: dict[str, str], baseline_index: int | None) -> dict:
    if row["kind"] == "baseline":
        return {"color": BASELINE_COLOURS[baseline_index % len(BASELINE_COLOURS)], "hatch": None, "edge": "#263238"}
    colour = RECIPE_COLOUR.get(row["recipe"], FALLBACK_COLOUR)
    return {"color": colour, "hatch": FAMILY_HATCH.get(row["family"]), "edge": "white"}


def draw_bars(ax, xs, values, width, style, ymax, ctx, show_values):
    """One model's bars at *xs*; values above *ymax* are cut with a break mark and keep their number."""
    for x, v in zip(xs, values, strict=True):
        if v is None:
            continue
        cut = v > ymax
        top = ymax if cut else v
        ax.bar(
            x, top, width=width, color=style["color"], edgecolor=style["edge"], hatch=style["hatch"],
            linewidth=ctx["lw"], zorder=3,
        )  # fmt: skip
        if cut:
            for shift in (0.0, 0.035 * ymax):
                y = 0.90 * ymax + shift
                ax.plot([x - width / 2, x + width / 2], [y - 0.02 * ymax, y + 0.02 * ymax], color="white", lw=ctx["lw"] * 2.4, zorder=5, solid_capstyle="butt")  # fmt: skip
        if cut or show_values:
            ax.text(x, top + 0.012 * ymax, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", va="bottom", rotation=90, fontsize=ctx["value"], zorder=6, fontweight="bold" if cut else "normal")  # fmt: skip


def build_caption(args, ours_prefix, base_prefix, n_clips):
    clips = f", {n_clips} clips per language" if n_clips else ""
    return (
        "FLEURS word error rate (%, lower is better; bars above "
        f"{args.ymax:g} are cut and carry their value). Public models: values as published in Table 35 of the Apertus "
        "1.5 report, where the FLEURS split and the text normaliser are not stated. This work: melt-eval "
        f"`{ours_prefix.split('.')[0]}`{clips}, prompt format {ours_prefix.split('.')[1]}, Whisper multilingual "
        "(basic) normaliser, so the two groups are not scored alike and a gap of a few points is within the noise "
        "of a 100-clip subset."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--stem", default="fig_fleurs_asr_vs_baselines")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--context", choices=sorted(CONTEXTS), default="paper")
    parser.add_argument("--ours-set", default="fleurs24-asr-dev")
    parser.add_argument("--ours-fmt", default="repeatfirst")
    parser.add_argument("--baseline-set", default="fleurs-reported")
    parser.add_argument("--baseline-fmt", default="native")
    parser.add_argument("--langs", nargs="+", help="language codes to show (default: those both sides have)")
    parser.add_argument("--ours", nargs="+", help="our model keys (default: every active one except start models)")
    parser.add_argument("--baselines", nargs="+", help="baseline model keys (default: all with data)")
    parser.add_argument("--include-start", action="store_true", help="also show the start checkpoints (A, A')")
    parser.add_argument("--ymax", type=float, default=50.0, help="cut bars above this WER (percent)")
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="print every bar's value (default: paper yes, poster no)",
    )
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    ctx = CONTEXTS[args.context]
    show_values = ctx["labels"] if args.labels is None else args.labels

    rows = load_master(args.master)
    ours_prefix, base_prefix = f"{args.ours_set}.{args.ours_fmt}.wer", f"{args.baseline_set}.{args.baseline_fmt}.wer"
    baselines = pick(rows, "baseline", base_prefix, args.baselines, True)
    ours = pick(rows, "ours", ours_prefix, args.ours, args.include_start)
    if not baselines or not ours:
        sys.exit(
            f"nothing to plot: {len(baselines)} baselines under {base_prefix}, {len(ours)} of ours under {ours_prefix}"
        )

    def langs_of(keys, prefix):
        return {c.rsplit(".", 1)[1] for k in keys for c, v in rows[k].items() if c.startswith(prefix + ".") and v}

    # The baseline table fixes the subset of languages; one of ours must have a number for a language to be shown.
    wanted = list(args.langs or langs_of(baselines, base_prefix))
    order = {lang: i for i, lang in enumerate(LANG_ORDER)}
    wanted.sort(key=lambda lang: (order.get(lang, len(order)), lang))
    langs = wanted if args.langs else [lang for lang in wanted if lang in langs_of(ours, ours_prefix)]
    dropped = [lang for lang in wanted if lang not in langs]
    if not langs:
        sys.exit("no language has numbers on both sides")

    models = baselines + ours
    values = {
        k: [number(rows[k], base_prefix if k in baselines else ours_prefix, lang) for lang in langs] for k in models
    }  # fmt: skip

    plt.rcParams.update({
        "font.size": ctx["font"], "axes.labelsize": ctx["font"], "xtick.labelsize": ctx["font"],
        "ytick.labelsize": ctx["tick"], "axes.linewidth": ctx["lw"] * 1.2, "xtick.major.width": ctx["lw"],
        "ytick.major.width": ctx["lw"], "xtick.major.size": 0, "ytick.major.size": ctx["tick"] * 0.4,
        "hatch.linewidth": ctx["lw"], "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
    })  # fmt: skip
    styles = {}
    for index, key in enumerate(baselines):
        styles[key] = bar_style(rows[key], index)
    for key in ours:
        styles[key] = bar_style(rows[key], None)

    fig, ax = plt.subplots(figsize=(ctx["width"], ctx["height"]))
    group, gap = 0.86, 0.7  # group width in x units; the gap between the two kinds of model, in bar widths
    width = group / (len(models) + gap)
    centres = np.arange(len(langs))
    for index, key in enumerate(models):
        slot = index + (gap if key in ours else 0.0)
        xs = centres - group / 2 + width * (slot + 0.5)
        draw_bars(ax, xs, values[key], width * 0.94, styles[key], args.ymax, ctx, show_values)
    ax.set_xticks(centres, [LANG_NAMES.get(lang, lang) for lang in langs])
    ax.set_xlim(-0.55, len(langs) - 0.45)
    step = 10 if args.ymax <= 60 else 20
    ax.set_yticks(np.arange(0, args.ymax + 1e-9, step))
    ax.set_ylim(0, args.ymax * 1.14)
    ax.set_ylabel("WER (%)")
    ax.yaxis.grid(True, color="#d0d0d0", linewidth=ctx["lw"], zorder=0)
    ax.spines["bottom"].set_zorder(10)  # above the bars, whose white edges would otherwise cut the axis line
    if args.title:
        ax.set_title(args.title, fontsize=ctx["font"] + 1)

    def handles(keys):
        return [
            Patch(
                facecolor=styles[k]["color"],
                edgecolor=styles[k]["edge"],
                hatch=styles[k]["hatch"],
                linewidth=ctx["lw"],
                label=rows[k]["label"],
            )  # fmt: skip
            for k in keys
        ]

    first = ax.legend(handles=handles(baselines), title="Public models (published WER)", loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=ctx["legend"], title_fontsize=ctx["legend"], handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)  # fmt: skip
    first._legend_box.align = "left"
    ax.add_artist(first)
    second = ax.legend(handles=handles(ours), title="This work (FLEURS dev, melt-eval)", loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2 if len(ours) <= 6 else 3, frameon=False, fontsize=ctx["legend"], title_fontsize=ctx["legend"], handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)  # fmt: skip
    second._legend_box.align = "left"

    digest = hashlib.sha256(args.master.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in args.formats:
        path = args.out_dir / f"{args.stem}.{fmt}"
        title, source = (
            "FLEURS WER of our checkpoints and public audio LLMs",
            f"{args.master.name} sha256:{digest[:16]}",
        )
        # Each backend accepts its own metadata keys; none of them gets a date, so the same input gives the same bytes.
        meta = {
            "pdf": {"Title": title, "Subject": source, "CreationDate": None, "Creator": Path(__file__).name},
            "png": {"Title": title, "Subject": source, "Software": Path(__file__).name},
            "svg": {"Title": title, "Description": source, "Date": None, "Creator": Path(__file__).name},
        }[fmt]  # fmt: skip
        fig.savefig(path, dpi=ctx["dpi"], bbox_inches="tight", metadata=meta)
        written.append(path)

    # What was drawn, for the log and the caption.
    print(f"master {args.master.name} sha256:{digest[:16]}")
    print(f"{'model':<22}" + "".join(f"{lang:>8}" for lang in langs))
    for key in models:
        print(f"{rows[key]['label']:<22}" + "".join(f"{'-' if v is None else f'{v:.1f}':>8}" for v in values[key]))
    if dropped:
        print(f"left out, the baseline table has them but none of our models does: {', '.join(dropped)}")
    sets_file = args.master.with_name("results_sets.csv")
    n_clips = None
    if sets_file.exists():
        with sets_file.open(newline="", encoding="utf-8") as handle:
            for entry in csv.DictReader(handle):
                if (entry["set"], entry["fmt"]) == (args.ours_set, args.ours_fmt):
                    n_clips = entry["n_per_unit"] or None
    print("\nSuggested caption:\n" + build_caption(args, ours_prefix, base_prefix, n_clips))
    print("\nwrote " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
