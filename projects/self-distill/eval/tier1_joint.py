#!/usr/bin/env python3
"""Joint ASR + ST score of the EU24 fine-tunes of the MA-utility study, and the ASR-vs-ST scatter with its Pareto frontier.

    python tier1_joint.py SUMMARIES.json [--langshare L.json] [--selection selection_tier1.json] --out-dir DIR

This is the headline score of the Tier-1 readout: ``tier1_tables.py`` has the per-metric tables, and the campaign's selection
score stays in the tables of this module for continuity.

The campaign's selection score (``projects/ablation-campaign/selection-metric``) was built for MA-stage checkpoints, where five
languages are trained and everything else is out of domain. It is ASR only, it weighs en/de/es/fr/it and five related languages
and gives the other 14 EU languages weight 0, and its groups are medians of five. The arms of this study are fine-tuned on all 24
EU languages, ASR and speech translation, so this module scores them on everything they were trained for, from the step-5 battery
that ``tier1_tables.py --collect`` writes to ``summaries.json`` (FLEURS-24 dev, 100 clips per language or direction):

- ``asr24``: geometric mean over the 24 EU languages of min(CER, 1), in %. Lower is better. Every language counts, and a relative
  change counts the same in a 3% language and a 40% one, so the 12-hour languages do not swamp the rest (``asr24_arith`` and
  ``asr24_med`` are the arithmetic mean and the median).
- ``xen23``: mean chrF of X->en over the 23 non-English sources (plain chrF, case-insensitive, as in the step-5 tables).
- ``enx23``: mean chrF of en->X over the 23 targets.
- ``st46``: (xen23 + enx23) / 2: translating into and out of English weigh the same.
- ``joint``: (100 - asr24 + xen23 + enx23) / 3, in points, higher is better. Over steps 718 to 2,872 the arms gain about as many
  points on each of the three parts (1.7 / 1.9 / 1.5), so one point of each is about the same amount of fine-tuning progress.

Diagnostics next to them: the three resource tiers of the EU24 results tables (``eu24_report.TIERS``) and, for ST, the directions
the EU24 mix trains against the ones it never saw (``sft_tables.training_data_languages``).

The seed spread is the mean absolute gap between an arm and its seed replicate (T0/T0' and A/A' at the four steps, eight gaps).
Two seeds per arm make it a rough yardstick, not a confidence interval.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import tier1_tables as t1
from sft_tables import per_language, training_data_languages


# Hours of transcribed audio per language in the MELT pools (``eu24_report.HOURS``): the tier rule of the EU24 results tables.
HOURS = {
    "en": 144001.2, "es": 28200.0, "fr": 14370.7, "de": 11128.2, "it": 6734.1, "nl": 3028.3, "pt": 2100.9,
    "pl": 930.7, "hu": 273.8, "fi": 193.7, "cs": 160.3, "ro": 130.6, "sv": 110.3,
    "sk": 54.2, "bg": 35.2, "lv": 30.5, "da": 29.0, "lt": 28.4, "el": 27.0, "et": 24.3, "hr": 21.2, "sl": 15.3,
    "ga": 12.7, "mt": 12.3,
}  # fmt: skip
TIERS = {
    "high": [lang for lang, h in HOURS.items() if h >= 1000],
    "mid": [lang for lang, h in HOURS.items() if 100 <= h < 1000],
    "low": [lang for lang, h in HOURS.items() if h < 100],
}
EU24 = list(HOURS)
MA_LANGS = ("en", "de", "es", "fr", "it")  # the five languages of the HR5 MA stage (A, B, T1, T2)
TRAINED = training_data_languages("EU24") or {}  # languages with ASR / X->en / en->X data in the EU24 mix
CER_CLIP = 1.0
CER_FLOOR = 0.002  # keeps the log of a perfect language finite
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # ASR, X->en, en->X
STEPS = (718, 1436, 2154, 2872)
# the reads after step 2,872: T0 and the continued arms run on to the end, A was scored there
LATE = (4308, 5744, 8616, 11499)
LATE_ARMS = ("T0", "A", "T3", "T5")  # the arms read at the late steps
LATE_MA = ("A", "T3", "T5")  # the MA starts whose lead over T0 is read at the end of the schedule
DECAY_START = 0.8 * t1.FINAL_STEP  # the last fifth of the schedule is the WSD decay
PAIRS = (("T0", "A"), ("T0'", "A'"))  # (no MA, MA) fine-tunes on identical batches
ARM_BY_KEY = {arm.key: arm for arm in t1.ARMS}
VARIANTS = {  # name -> (weights asr / X->en / en->X, ASR key)
    "thirds, geometric-mean CER (default)": (WEIGHTS, "asr24"),
    "thirds, arithmetic-mean CER": (WEIGHTS, "asr24_arith"),
    "thirds, median CER": (WEIGHTS, "asr24_med"),
    "ASR-heavy (1/2, 1/4, 1/4)": ((0.5, 0.25, 0.25), "asr24"),
    "ST-heavy (1/5, 2/5, 2/5)": ((0.2, 0.4, 0.4), "asr24"),
}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def joint_of(agg: dict, weights: tuple[float, float, float] = WEIGHTS, asr_key: str = "asr24") -> float:
    """The joint score of an ``aggregates`` row under other weights or another CER aggregate."""
    w_asr, w_xen, w_enx = weights
    return w_asr * (100 - agg[asr_key]) + w_xen * agg["xen23"] + w_enx * agg["enx23"]


def aggregates(cell: dict) -> dict | None:
    """The numbers of one (arm, step) cell, or None when one of the three families has not been scored.

    The private keys ``_cer``, ``_xen`` and ``_enx`` keep the per-language values for the paired comparison.
    """
    if not all(cell.get(family) for family in ("asr", "xen", "enx")):
        return None
    cer = {lang: min(v, CER_CLIP) for lang, v in per_language(cell["asr"], "cer").items()}
    xen, enx = per_language(cell["xen"], "chrf"), per_language(cell["enx"], "chrf")
    if set(cer) != set(EU24) or len(xen) != 23 or len(enx) != 23:
        raise ValueError(f"incomplete language set: {len(cer)} ASR, {len(xen)} X->en, {len(enx)} en->X")
    out: dict = {}
    out["asr24"] = 100 * math.exp(mean(math.log(max(v, CER_FLOOR)) for v in cer.values()))
    out["asr24_arith"] = 100 * mean(cer.values())
    out["asr24_med"] = 100 * statistics.median(cer.values())
    out["cer_ma5"] = 100 * mean(cer[lang] for lang in MA_LANGS)
    for tier, langs in TIERS.items():
        out[f"asr_{tier}"] = 100 * mean(cer[lang] for lang in langs)
    out["xen23"], out["enx23"] = mean(xen.values()), mean(enx.values())
    out["st46"] = (out["xen23"] + out["enx23"]) / 2
    for family, values in (("xen", xen), ("enx", enx)):
        with_data = TRAINED.get(family, set())
        out[f"{family}_trained"] = mean(v for lang, v in values.items() if lang in with_data)
        zero = [v for lang, v in values.items() if lang not in with_data]
        out[f"{family}_zero"] = mean(zero) if zero else None
    out["joint"] = joint_of(out)
    out["_cer"], out["_xen"], out["_enx"] = cer, xen, enx
    return out


def build(summaries: str, langshare: str | None, selection: str | None = None) -> dict[int, dict[str, dict]]:
    """``rows[step][arm key] = aggregates``, with the old selection score as ``selection`` when its file is given."""
    cells = t1.load(summaries, langshare)
    old = t1.load_selection(selection)
    id_median = {  # median CER of the five MA languages on their held-out CV22 / MLS / VoxPopuli sets, as the old score has it
        (r["arm"], r["step"]): 100 * r["result"]["medians"]["ID"]
        for r in (json.loads(Path(selection).read_text()) if selection else [])
        if r.get("result")
    }
    rows: dict[int, dict[str, dict]] = {}
    for (arm_key, step), cell in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        agg = aggregates(cell)
        if agg is not None:
            agg["selection"] = old.get((arm_key, step))
            agg["id_ma5"] = id_median.get((arm_key, step))
            rows.setdefault(step, {})[arm_key] = agg
    return rows


# --- noise, Pareto, paired comparison ------------------------------------------------------------------------------------


def replicate_gaps(rows: dict, keys: tuple[str, ...], steps: tuple[int, ...] = STEPS) -> dict[str, list[float]]:
    """Absolute gaps between an arm and its seed replicate (T0/T0', A/A'): one per pair and step."""
    gaps: dict[str, list[float]] = {key: [] for key in keys}
    for step in steps:
        for first, second in t1.REPLICATES.items():
            a, b = rows.get(step, {}).get(first), rows.get(step, {}).get(second)
            if a and b:
                for key in keys:
                    if a.get(key) is not None and b.get(key) is not None:
                        gaps[key].append(abs(a[key] - b[key]))
    return gaps


def seed_spread(gaps: dict[str, list[float]]) -> dict[str, float]:
    return {key: statistics.fmean(values) for key, values in gaps.items() if values}


def pareto(points: dict[str, tuple[float, float]]) -> set[str]:
    """Arms no other arm beats on both axes, for x = error (lower is better) and y = quality (higher is better)."""
    return {
        k
        for k, (x, y) in points.items()
        if not any(xo <= x and yo >= y and (xo < x or yo > y) for ko, (xo, yo) in points.items() if ko != k)
    }


def clearly_dominated(points: dict[str, tuple[float, float]], band_x: float, band_y: float) -> set[str]:
    """Arms that another arm beats on both axes by more than the seed spread of each axis."""
    return {
        k
        for k, (x, y) in points.items()
        if any(x - xo > band_x and yo - y > band_y for ko, (xo, yo) in points.items() if ko != k)
    }


def sign_counts(a: dict, b: dict) -> dict:
    """Across the 24 ASR, 23 X->en and 23 en->X cells, how many does *a* win against *b* (lower CER, higher chrF)?"""
    wins = losses = 0
    for family, lower_is_better in (("_cer", True), ("_xen", False), ("_enx", False)):
        for lang, value in a[family].items():
            delta = b[family][lang] - value if lower_is_better else value - b[family][lang]
            wins += delta > 0
            losses += delta < 0
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0  # two-sided exact binomial
    return {"wins": wins, "losses": losses, "p": p}


def advantage(ma: dict, no_ma: dict, key: str) -> float:
    """MA minus no MA, signed so that positive means the MA start is better (the ASR keys are errors)."""
    delta = ma[key] - no_ma[key]
    return -delta if key.startswith(("asr", "cer")) else delta


# --- tables --------------------------------------------------------------------------------------------------------------


def num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main_tables(rows: dict, spread: dict[str, float]) -> str:
    out = [
        "Seed spread (mean absolute gap between an arm and its seed replicate, T0/T0' and A/A' at the four steps): "
        + ", ".join(f"{k} {spread[k]:.2f}" for k in ("asr24", "xen23", "enx23", "st46", "joint")),
        "",
    ]
    for step in STEPS:
        by_arm = rows.get(step)
        if not by_arm:
            continue
        points = {k: (v["asr24"], v["st46"]) for k, v in by_arm.items()}
        frontier = pareto(points)
        dominated = clearly_dominated(points, spread["asr24"], spread["st46"])
        out += [
            f"### Step {step:,}",
            "",
            "| arm | MA h | ASR-24 CER % (geo) | X→en 23 | en→X 23 | ST-46 | joint | in-domain CER % (5 MA languages) | old selection | Pareto |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in t1.ARMS:
            v = by_arm.get(arm.key)
            if not v:
                continue
            flag = (
                "frontier"
                if arm.key in frontier
                else ("beaten beyond the spread" if arm.key in dominated else "within the spread")
            )
            out.append(
                f"| {arm.key} | {arm.ma_hours:,.0f} | {num(v['asr24'])} | {num(v['xen23'])} | {num(v['enx23'])} | "
                f"{num(v['st46'])} | {num(v['joint'])} | {num(v['id_ma5'])} | {num(v['selection'], 4)} | {flag} |"
            )
        out.append("")
    return "\n".join(out)


def comparison_table(rows: dict, spread: dict[str, float]) -> str:
    """MA against no MA on identical batches, per seed and pooled, in points (positive = the MA start is better)."""
    keys = ("asr24", "xen23", "enx23", "joint")
    out = [
        "MA advantage = A minus T0 (seed 1337) and A' minus T0' (seed 1338), same batches; positive = the MA start is better. "
        "ASR is the drop in geometric-mean CER, ST the rise in chrF, joint the rise in the joint score; the last column counts "
        "the 70 language cells (24 ASR + 23 + 23 ST) the MA start wins / loses, with an exact two-sided binomial p "
        "(the cells share one model, so p is optimistic).",
        "",
        "| step | pair | ASR-24 | X→en 23 | en→X 23 | joint | cells won / lost |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for step in STEPS:
        by_arm = rows.get(step, {})
        for no_ma, ma in PAIRS:
            if no_ma in by_arm and ma in by_arm:
                counts = sign_counts(by_arm[ma], by_arm[no_ma])
                out.append(
                    f"| {step:,} | {ma} − {no_ma} | "
                    + " | ".join(f"{advantage(by_arm[ma], by_arm[no_ma], k):+.2f}" for k in keys)
                    + f" | {counts['wins']} / {counts['losses']}, p={counts['p']:.2f} |"
                )
        if all(k in by_arm for k in ("T0", "T0'", "A", "A'")):
            pooled = [
                advantage(
                    {k: mean(by_arm[arm][k] for arm in ("A", "A'")) for k in keys},
                    {k: mean(by_arm[arm][k] for arm in ("T0", "T0'")) for k in keys},
                    key,
                )
                for key in keys
            ]
            out.append(
                f"| {step:,} | **pooled** (A, A') − (T0, T0') | "
                + " | ".join(f"**{v:+.2f}**" for v in pooled)
                + " | |"
            )
    out.append("| | seed spread | " + " | ".join(f"{spread[k]:.2f}" for k in keys) + " | |")
    return "\n".join(out)


def late_gaps(rows: dict) -> dict[str, dict[int, float]]:
    """D(s) = joint(X, s) - joint(T0, s) at the late steps where both were read, for each MA start X."""
    return {
        arm: {
            step: rows[step][arm]["joint"] - rows[step]["T0"]["joint"]
            for step in LATE
            if arm in rows.get(step, {}) and "T0" in rows.get(step, {})
        }
        for arm in LATE_MA
    }


def gate_verdict(gaps: dict[int, float], spread: float) -> str:
    """The reading of one MA start's lead over T0 at the end of the schedule, fixed before the late reads came in.

    *gone*: the lead at the last step and the mean of the stable-phase leads are both within the seed spread; *persists*: the lead at the
    last step is at least twice the spread and the stable-phase mean at least the spread; anything else is *between*. Before the three
    stable-phase reads and the last step are in, the verdict is *open*.
    """
    stable = [d for step, d in gaps.items() if step < DECAY_START]
    if len(stable) < 3 or t1.FINAL_STEP not in gaps:
        return "open"
    final, stable_mean = gaps[t1.FINAL_STEP], statistics.fmean(stable)
    if final <= spread and stable_mean <= spread:
        return "gone"
    if final >= 2 * spread and stable_mean >= spread:
        return "persists"
    return "between"


def recipe_verdict(rows: dict, spread: float) -> str:
    """T5 (SIFT) against T3 (CE24), the same 2,700 h of audio, at the last step."""
    last = rows.get(t1.FINAL_STEP, {})
    if "T5" not in last or "T3" not in last:
        return "open"
    gap = last["T5"]["joint"] - last["T3"]["joint"]
    if gap >= 2 * spread:
        return f"T5 - T3 = {gap:+.2f}: SIFT earns its extra MA"
    return f"T5 - T3 = {gap:+.2f}: " + ("CE24 does the same for less" if gap <= spread else "between")


def late_table(rows: dict, spread: dict[str, float]) -> str:
    """The reads after step 2,872 (T0, the continued arms and A), the lead of each MA start over T0, and the registered verdicts."""
    seen = [step for step in LATE if any(arm in rows.get(step, {}) for arm in LATE_ARMS)]
    if not seen:
        return "No read after step 2,872 yet."
    out = [
        "Steps 4,308 to 11,499 of the 11,499-step schedule (the WSD decay is its last fifth, from step 9,200). T0 and the continued arms (T3, T5) are read at "
        "these steps, A was scored there; D = joint minus T0's joint at the same step, positive = the arm is ahead.",
        "",
        "| step | arm | ASR-24 CER % (geo) | X→en 23 | en→X 23 | ST-46 | joint | D (joint − T0) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for step in seen:
        by_arm = rows[step]
        for key in LATE_ARMS:
            if key not in by_arm:
                continue
            v = by_arm[key]
            d = v["joint"] - by_arm["T0"]["joint"] if "T0" in by_arm and key != "T0" else None
            out.append(
                f"| {step:,} | {key} | {num(v['asr24'])} | {num(v['xen23'])} | {num(v['enx23'])} | {num(v['st46'])} | "
                f"{num(v['joint'])} | {'-' if d is None else f'{d:+.2f}'} |"
            )
    yardstick = spread["joint"]
    gaps = late_gaps(rows)
    out += [
        "",
        f"Reading fixed before the late reads came in: the yardstick is the seed spread of the joint score, {yardstick:.2f}. *gone*: D at "
        f"11,499 <= {yardstick:.2f} and the mean of the three stable-phase D <= {yardstick:.2f}; *persists*: D at 11,499 >= {2 * yardstick:.2f} "
        f"and the stable-phase mean >= {yardstick:.2f}; otherwise *between*; *open* until the 4,308, 5,744, 8,616 and 11,499 reads of the arm "
        "and of T0 are in.",
        "",
        "| MA start | " + " | ".join(f"D at {s:,}" for s in LATE) + " | stable-phase mean | verdict |",
        "|---|" + "---:|" * (len(LATE) + 1) + "---|",
    ]
    for arm in LATE_MA:
        stable = [d for step, d in gaps[arm].items() if step < DECAY_START]
        out.append(
            f"| {arm} | "
            + " | ".join("-" if s not in gaps[arm] else f"{gaps[arm][s]:+.2f}" for s in LATE)
            + f" | {'-' if not stable else f'{statistics.fmean(stable):+.2f}'} | **{gate_verdict(gaps[arm], yardstick)}** |"
        )
    out += ["", f"Recipe, SIFT against CE24 at 11,499: {recipe_verdict(rows, yardstick)}."]
    return "\n".join(out)


def variants_table(rows: dict) -> str:
    """The pooled MA advantage in the joint score under other weights and other CER aggregates."""
    out = [
        "Pooled MA advantage in the joint score, (A, A') minus (T0, T0'), under other definitions "
        "(positive = the MA start is better); the seed spread is that variant's own mean gap.",
        "",
        "| variant | " + " | ".join(f"{s:,}" for s in STEPS) + " | seed spread |",
        "|---|" + "---:|" * (len(STEPS) + 1),
    ]
    for name, (weights, asr_key) in VARIANTS.items():
        cells, replicate = [], []
        for step in STEPS:
            by_arm = rows.get(step, {})
            if not all(k in by_arm for k in ("T0", "T0'", "A", "A'")):
                cells.append("-")
                continue
            j = {k: joint_of(by_arm[k], weights, asr_key) for k in ("T0", "T0'", "A", "A'")}
            pooled = mean([j["A"], j["A'"]]) - mean([j["T0"], j["T0'"]])
            cells.append(f"{pooled:+.2f}")
            replicate += [abs(j["A"] - j["A'"]), abs(j["T0"] - j["T0'"])]
        out.append(f"| {name} | " + " | ".join(cells) + f" | {mean(replicate):.2f} |")
    return "\n".join(out)


def reliability_table(rows: dict) -> str:
    """Which aggregate separates arms from seed noise: progress between steps 718 and 2,872 against the replicate gap."""
    metrics = {  # key -> (label, lower is better, scale)
        "asr24": ("ASR-24 CER, geometric mean, %", True, 1.0),
        "asr24_arith": ("ASR-24 CER, arithmetic mean, %", True, 1.0),
        "asr24_med": ("ASR-24 CER, median, %", True, 1.0),
        "cer_ma5": ("CER of the five MA languages, %", True, 1.0),
        "asr_high": ("CER, high tier (7), %", True, 1.0),
        "asr_mid": ("CER, mid tier (6), %", True, 1.0),
        "asr_low": ("CER, low tier (11), %", True, 1.0),
        "xen23": ("X→en chrF, 23 sources", False, 1.0),
        "enx23": ("en→X chrF, 23 targets", False, 1.0),
        "st46": ("ST-46 chrF", False, 1.0),
        "joint": ("joint score", False, 1.0),
        "selection": ("old selection score (x100)", True, 0.01),
    }
    out = [
        "Progress = mean over the ten arms of the improvement from step 718 to 2,872; seed gap = mean absolute gap between an "
        "arm and its replicate (8 gaps); SNR = progress / gap. A higher SNR resolves a given difference with fewer replicates.",
        "",
        "| metric | progress | seed gap | SNR | sd of the ten arms at 2,872 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, (label, lower, scale) in metrics.items():
        progress, last = [], []
        for arm in t1.ARMS:
            a, b = rows.get(718, {}).get(arm.key, {}).get(key), rows.get(2872, {}).get(arm.key, {}).get(key)
            if a is not None and b is not None:
                progress.append((a - b) if lower else (b - a))
                last.append(b)
        gaps = replicate_gaps(rows, (key,))[key]
        if progress and gaps:
            p, g = statistics.fmean(progress), statistics.fmean(gaps)
            snr = f"{p / g:.1f}" if g else "-"  # identical replicates have no gap to divide by
            out.append(
                f"| {label} | {p / scale:.3g} | {g / scale:.3g} | {snr} | {statistics.stdev(last) / scale:.3g} |"
            )
    return "\n".join(out)


def tier_table(rows: dict, steps: tuple[int, ...] = (718, 2872)) -> str:
    head = [
        "arm",
        "CER high-7",
        "CER mid-6",
        "CER low-11",
        "X→en trained-21",
        "X→en zero-shot-2",
        "en→X trained-12",
        "en→X zero-shot-11",
    ]
    keys = ("asr_high", "asr_mid", "asr_low", "xen_trained", "xen_zero", "enx_trained", "enx_zero")
    out = []
    for step in steps:
        by_arm = rows.get(step)
        if not by_arm:
            continue
        out += [
            f"Step {step:,}: CER % (arithmetic mean) by resource tier (high >= 1,000 h of transcribed audio in the pools, "
            "mid 100-1,000 h, low < 100 h); chrF by direction, split into those the EU24 mix trains and those it never saw.",
            "",
            "| " + " | ".join(head) + " |",
            "|---|" + "---:|" * (len(head) - 1),
        ]
        for arm in t1.ARMS:
            if arm.key in by_arm:
                out.append(f"| {arm.key} | " + " | ".join(num(by_arm[arm.key][k]) for k in keys) + " |")
        out.append("")
    return "\n".join(out)


def render(rows: dict) -> str:
    spread = seed_spread(
        replicate_gaps(rows, ("asr24", "asr24_arith", "asr24_med", "xen23", "enx23", "st46", "joint"))
    )
    return "\n".join(
        [
            "# EU24 joint ASR + ST score, Tier 1",
            "",
            main_tables(rows, spread),
            "## MA against no MA, on identical batches",
            "",
            comparison_table(rows, spread),
            "",
            "## The end of the schedule",
            "",
            late_table(rows, spread),
            "",
            "## Under other definitions",
            "",
            variants_table(rows),
            "",
            "## Which aggregate resolves the arms best",
            "",
            reliability_table(rows),
            "",
            "## What the aggregates are made of",
            "",
            tier_table(rows),
        ]
    )


# --- figures -------------------------------------------------------------------------------------------------------------


def _save(fig, path: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    written = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=160)
        written.append(target)
    plt.close(fig)
    return written


def _seed_cross(ax, x: float, y: float, band: tuple[float, float]) -> None:
    """A cross whose arms are the mean seed gap on each axis, labelled above it."""
    ax.errorbar([x], [y], xerr=band[0] / 2, yerr=band[1] / 2, color="#444", capsize=3, lw=1.2, zorder=4)
    ax.annotate(
        "mean seed gap", (x, y), textcoords="offset points", xytext=(0, 9 + 0), fontsize=7, color="#444", ha="center"
    )


def _panel(
    ax, points: dict[str, tuple[float, float]], band: tuple[float, float], labels: bool = True, legend: bool = False
):
    """One scatter panel: error on x (axis inverted, better to the right), quality on y, frontier, seed pairs, seed cross."""
    frontier = pareto(points)
    beaten = clearly_dominated(points, *band)
    for fmt_name, (marker, color) in t1.MARKERS.items():
        chosen = [k for k in points if ARM_BY_KEY[k].fmt == fmt_name]
        for k in chosen:
            ax.scatter(*points[k], marker=marker, color=color, s=58, zorder=3, alpha=0.4 if k in beaten else 1.0)
        if legend and chosen:
            ax.scatter([], [], marker=marker, color=color, s=58, label=fmt_name)
    for first, second in t1.REPLICATES.items():
        if first in points and second in points:
            ax.plot(
                *zip(points[first], points[second]), color="#888", alpha=0.5, lw=5, zorder=1, solid_capstyle="round"
            )
    ordered = sorted(points.items(), key=lambda kv: kv[1][0])
    if labels:
        for i, (k, (x, y)) in enumerate(ordered):
            dy = (5, -11)[i % 2]
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, dy), fontsize=8, color="#222",
                        fontweight="bold" if k in frontier else "normal")  # fmt: skip
    front = sorted((points[k] for k in frontier), key=lambda p: p[0])
    if len(front) > 1:
        ax.step(
            [p[0] for p in front],
            [p[1] for p in front],
            where="post",
            color="black",
            lw=1.0,
            ls="--",
            alpha=0.7,
            zorder=2,
        )
    xs, ys = [p[0] for p in points.values()], [p[1] for p in points.values()]
    pad_x, pad_y = 0.08 * (max(xs) - min(xs)), 0.10 * (max(ys) - min(ys))
    ax.set_xlim(max(xs) + pad_x + band[0], min(xs) - pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y + band[1])
    cx, cy = max(xs) + pad_x * 0.4 + band[0] / 2, max(ys) + pad_y * 0.4 + band[1] / 2  # upper-left: usually empty
    _seed_cross(ax, cx, cy, band)
    ax.grid(alpha=0.25)


def pareto_figure(rows: dict, path: Path) -> list[Path]:
    """Four views: all steps on common axes, steps 718 and 2,872 zoomed, and 2,872 with the old selection score on x."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spread = seed_spread(replicate_gaps(rows, ("asr24", "st46", "selection")))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0))
    overview = axes[0][0]
    focus = ("T0", "T0'", "A", "A'")
    for key in focus:
        pts = [(rows[s][key]["asr24"], rows[s][key]["st46"]) for s in STEPS if key in rows.get(s, {})]
        marker, color = t1.MARKERS[ARM_BY_KEY[key].fmt]
        color = "black" if ARM_BY_KEY[key].fmt == "none" else color
        overview.plot(
            *zip(*pts),
            color=color,
            lw=1.4,
            ls=":" if key.endswith("'") else "-",
            marker=marker,
            ms=5,
            alpha=0.85,
            label=key,
        )
        for (x, y), step in zip(pts, STEPS):
            if not key.endswith("'"):
                overview.annotate(f"{step:,}", (x, y), textcoords="offset points", xytext=(5, -10 if key == "T0" else 5), fontsize=7,
                                  color=color)  # fmt: skip
    xs = [rows[s][k]["asr24"] for s in STEPS for k in focus]
    ys = [rows[s][k]["st46"] for s in STEPS for k in focus]
    overview.set_xlim(max(xs) + 0.5 + spread["asr24"], min(xs) - 0.2)
    overview.set_ylim(min(ys) - 0.2, max(ys) + 0.2 + spread["st46"])
    _seed_cross(overview, max(xs) + 0.25 + spread["asr24"] / 2, max(ys) + 0.1 + spread["st46"] / 2,
                (spread["asr24"], spread["st46"]))  # fmt: skip
    overview.legend(fontsize=8, loc="lower left", ncol=2)
    overview.set_title("no MA (T0, T0') against CE 1,000 h (A, A') over the steps", fontsize=10)
    overview.grid(alpha=0.25)
    for ax, step in ((axes[0][1], 718), (axes[1][0], 2872)):
        by_arm = rows[step]
        _panel(
            ax,
            {k: (v["asr24"], v["st46"]) for k, v in by_arm.items()},
            (spread["asr24"], spread["st46"]),
            legend=step == 718,
        )
        ax.set_title(f"after {step:,} fine-tune steps", fontsize=10)
    by_arm = rows[2872]
    if all(v.get("selection") is not None for v in by_arm.values()):
        _panel(
            axes[1][1],
            {k: (100 * v["selection"], v["st46"]) for k, v in by_arm.items()},
            (spread["selection"], spread["st46"]),
        )
        axes[1][1].set_title("after 2,872 steps, old selection score on x", fontsize=10)
        axes[1][1].set_xlabel("old selection score x100 (better →)")
    else:
        axes[1][1].axis("off")
    for ax in (overview, axes[0][1], axes[1][0]):
        ax.set_xlabel("ASR: geometric-mean CER over 24 languages, % (better →)")
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("ST: mean chrF over 46 directions (better ↑)")
    handles, labels = axes[0][1].get_legend_handles_labels()
    handles += [Line2D([0], [0], color="#888", alpha=0.5, lw=5), Line2D([0], [0], color="black", ls="--", lw=1)]
    labels += ["seed pair (T0/T0', A/A')", "Pareto frontier (bold; faded = beaten beyond the spread)"]
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    return _save(fig, path)


def hours_figure(rows: dict, path: Path, steps: tuple[int, ...] = (718, 2872)) -> list[Path]:
    """Each part and the joint score against the MA hours, one column per step, markers as in ``tier1_tables``.

    The y axis of a panel is never narrower than five seed gaps, so a spread inside the noise looks like one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = (
        ("asr24", "ASR: geo-mean CER, 24 lang. %"),
        ("xen23", "X→en chrF, 23 sources"),
        ("enx23", "en→X chrF, 23 targets"),
        ("joint", "joint score (points)"),
    )
    spread = seed_spread(replicate_gaps(rows, tuple(k for k, _ in panels)))
    fig, axes = plt.subplots(len(panels), len(steps), figsize=(4.6 * len(steps), 2.7 * len(panels)), squeeze=False)
    for r, (key, label) in enumerate(panels):
        values = [v[key] for s in steps for v in rows.get(s, {}).values()]
        half = max((max(values) - min(values)) / 2 * 1.15, 2.5 * spread[key])
        for c, step in enumerate(steps):
            ax = axes[r][c]
            by_arm = rows.get(step, {})
            pts = {a.key: (a, by_arm[a.key][key]) for a in t1.ARMS if a.key in by_arm}
            for fmt_name, (marker, color) in t1.MARKERS.items():
                sel = [(a.ma_hours, y) for a, y in pts.values() if a.fmt == fmt_name]
                if sel:
                    ax.scatter(*zip(*sel), marker=marker, color=color, s=46, zorder=3)
            ce = sorted(
                (a.ma_hours, y)
                for a, y in pts.values()
                if a.fmt in ("CE", "none") and a.key not in t1.REPLICATES.values()
            )
            if len(ce) > 1:
                ax.plot(*zip(*ce), color="#1f77b4", alpha=0.35, zorder=2)
            for a, y in pts.values():
                ax.annotate(
                    a.key, (a.ma_hours, y), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#444"
                )
            for first, second in t1.REPLICATES.items():
                if first in pts and second in pts:
                    pair = [pts[first][1], pts[second][1]]
                    ax.vlines(
                        pts[first][0].ma_hours,
                        min(pair),
                        max(pair),
                        color="#888888",
                        alpha=0.35,
                        linewidth=9,
                        zorder=1,
                    )
            mid = (min(y for _, y in pts.values()) + max(y for _, y in pts.values())) / 2
            ax.set_ylim(mid - half, mid + half)
            ax.set_xscale("symlog", linthresh=60)
            ax.set_xlim(-20, 6000)
            ax.set_xticks([0, 120, 360, 1000, 3700])
            ax.set_xticklabels(["0", "120", "360", "1k", "3.7k"])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_title(f"after {step:,} fine-tune steps")
            if r == len(panels) - 1:
                ax.set_xlabel("MA hours before the fine-tune")
            ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], marker=m, color=col, ls="", ms=7, label=name) for name, (m, col) in t1.MARKERS.items()]
    handles.append(Line2D([0], [0], color="#888888", alpha=0.35, linewidth=9, label="seed pair (T0/T0', A/A')"))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    return _save(fig, path)


def trajectory_figure(rows: dict, path: Path) -> list[Path]:
    """The three parts and the joint score against fine-tune steps, every arm (A also at its later steps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("asr24", "ASR: geometric-mean CER, 24 languages %"),
        ("xen23", "X→en chrF, 23 sources"),
        ("enx23", "en→X chrF, 23 targets"),
        ("joint", "joint score"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (key, title) in zip(axes, panels):
        for arm in t1.ARMS:
            pts = sorted((s, by[arm.key][key]) for s, by in rows.items() if arm.key in by)
            if not pts:
                continue
            marker, color = t1.MARKERS[arm.fmt]
            ax.plot(*zip(*pts), marker=marker, ms=4, lw=2.4 if arm.key in ("T0", "A") else 1.1,
                    ls=":" if arm.key in ("T0'", "A'") else "-", color="black" if arm.fmt == "none" else color,
                    alpha=0.9 if arm.key in ("T0", "A") else 0.6, label=arm.key)  # fmt: skip
        ax.axvline(
            0.8 * t1.FINAL_STEP, color="#999", ls="--", lw=0.8
        )  # the decay of the 11,499-step schedule starts here
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fine-tune steps")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries")
    parser.add_argument("--langshare", default=None)
    parser.add_argument(
        "--selection", default=None, help="selection_tier1.json: adds the old score to the tables and the scatter"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build(args.summaries, args.langshare, args.selection)
    text = render(rows)
    (args.out_dir / "joint_report.md").write_text(text + "\n")
    slim = {
        str(s): {a: {k: v for k, v in agg.items() if not k.startswith("_")} for a, agg in by.items()}
        for s, by in rows.items()
    }
    (args.out_dir / "joint_tables.json").write_text(json.dumps(slim, indent=1))
    print(text)
    if not args.no_figure:
        written = pareto_figure(rows, args.out_dir / "joint_pareto")
        written += hours_figure(rows, args.out_dir / "joint_vs_hours")
        written += trajectory_figure(rows, args.out_dir / "joint_trajectories")
        print("figures:", ", ".join(map(str, written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
