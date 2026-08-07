#!/usr/bin/env python3
"""Compute per-source mux sampling weights with two-tier language/corpus balancing.

Implements the policy from Section 3.3.1 of "Canary-1B-v2 & Parakeet-TDT-0.6B-v3:
Efficient and High-Performance Models for Multilingual ASR and AST"
(arXiv:2509.14128), which itself adapts Babu et al. (2021) by inverting the order:
corpora are balanced *within* a language first, and only then are languages
balanced against each other, so that a small corpus is not swamped before
cross-language balancing runs.

    1. Corpus balancing within a language l, for each corpus c in l:

           w_c = (n(c) / N_l) ** alpha        N_l = sum of n(c) over c in l
           p_c = w_c / sum(w_c over c in l)

    2. Language balancing across the mix:

           w_l = (n(l) / N_total) ** beta     n(l) = N_l,  N_total = sum of n(l)
           p_l = w_l / sum(w_l over all l)

    3. Final sampling probability for corpus c in language l:

           p_cl = p_l * p_c

n(.) is measured in HOURS of audio, not utterance counts. alpha < 1 flattens the
corpus distribution inside a language; beta < 1 flattens the language
distribution. The paper uses alpha = beta = 0.5 for pre-training (and mentions
alpha = 0.2 for a fine-tuning stage).

Following the paper, every translation DIRECTION is its own "language" entry:
ASR German is `de`, while en->de and de->en are the separate entries `en-de` and
`de-en`. That falls out of the config tags: ASR sources carry `lang`, ST sources
carry `src_lang` / `tgt_lang`.

Locale-tagged codes are folded onto their language via LOCALE_ALIASES before the
entries are formed, so `de_de` counts as `de` and `sv-SE_en` as `sv-en`. Without
that, one language split across two tag spellings would collect two
language-level shares under beta and be upsampled for no reason. The original
code survives per-cut as `region_code`, or as `src_region_code` /
`tgt_region_code` on ST sources where either side may carry a locale.

The resulting p_cl values are written straight into each source's `weight:` key.
Lhotse's ``CutSet.mux`` normalises whatever it is given, so the absolute scale
does not matter, but emitting a proper distribution keeps the config readable.

Usage:
    # full pass over every manifest (slow, exact) — results are cached
    python infra/compute_mix_weights.py --config config/train/SFT-v1.2.7.yaml

    # fast estimate from a few shards per source
    python infra/compute_mix_weights.py --config config/train/SFT-v1.2.7.yaml \
        --sample-shards 3

    # write the weights back into the config
    python infra/compute_mix_weights.py --config config/train/SFT-v1.2.7.yaml \
        --emit-yaml weights.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Every locale-tagged code in the mix, mapped to the language it belongs to.
#
# This table is exhaustive by design rather than derived from a rule: stripping
# whatever follows the first "-" or "_" would also mangle codes where the suffix
# is not a region, and a silent mis-split here quietly changes the sampling
# distribution. Adding a source with a new locale code means adding a row.
#
# A locale that gets its own language entry receives its own language-level
# share under beta, so splitting one language across two tag spellings
# upweights it purely for being spelled two ways. Collapsing them here is what
# prevents that; the original code is preserved per-cut as `region_code` (and
# `src_region_code` / `tgt_region_code` for ST), so nothing is lost.
LOCALE_ALIASES = {
    # FLEURS locale tags
    "de_de": "de",
    "en_us": "en",
    "es_419": "es",
    "fr_fr": "fr",
    "it_it": "it",
    "pt_br": "pt",
    # CoVoST2 / Common Voice
    "sv-se": "sv",
    "zh-cn": "zh",
    "zh-hk": "zh",
    "zh-tw": "zh",
}


def canonical_lang(code: str) -> str:
    """The language a possibly locale-tagged code belongs to."""
    return LOCALE_ALIASES.get(str(code).lower(), str(code))

ENV_RE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _expand_env(text: str) -> str:
    """Resolve ${oc.env:VAR} / ${oc.env:VAR,default} without pulling in OmegaConf."""
    def sub(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        val = os.environ.get(var)
        if val is None:
            if default is None:
                raise KeyError(
                    f"Config references ${{oc.env:{var}}} but {var} is not set. "
                    "Export it (or pass --datasets-root) and retry."
                )
            val = default.strip()
        return val
    return ENV_RE.sub(sub, text)


def load_sources(config_path: Path, datasets_root: str | None) -> list[dict]:
    """Return [{path, raw_path, tags, task, lang_key}] per train_ds input_cfg entry."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")

    if datasets_root:
        os.environ["LOCAL_DATASETS_DIR"] = datasets_root
    root = os.environ.get("LOCAL_DATASETS_DIR", "")
    # Placeholders for vars this script does not care about, so expansion of the
    # rest of the file cannot fail on an unrelated key.
    for var in ("OUTPUT_DIR", "HF_HOME"):
        os.environ.setdefault(var, "/unused")

    raw = _expand_env(config_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(raw)

    entries = cfg["data"]["train_ds"]["input_cfg"]
    sources = []
    for entry in entries:
        tags = entry.get("tags", {}) or {}
        task = str(tags.get("task", "asr")).lower()
        if task == "st":
            src, tgt = tags.get("src_lang"), tags.get("tgt_lang")
            if not src or not tgt:
                raise ValueError(
                    f"ST source {entry.get('shar_path')} is missing src_lang/tgt_lang"
                )
            # Each direction is its own language entry, on canonical codes so
            # that e.g. sv-SE_en and sv_en are one entry rather than two.
            lang_key = f"{canonical_lang(src)}-{canonical_lang(tgt)}"
            locales = {"src_region_code": str(src), "tgt_region_code": str(tgt)}
        else:
            lang = tags.get("lang")
            if not lang:
                raise ValueError(
                    f"ASR source {entry.get('shar_path')} is missing a lang tag"
                )
            lang_key = canonical_lang(lang)
            locales = {"region_code": str(lang)}
        path = str(entry["shar_path"])
        # Emitted configs keep the placeholder so the same file works on any
        # cluster, and so Smurf can consume it with its own datasets root.
        raw_path = (path.replace(root, "${oc.env:LOCAL_DATASETS_DIR}", 1)
                    if root and path.startswith(root) else path)
        sources.append({
            "path": path,
            "raw_path": raw_path,
            # Emitted configs carry the canonical language plus the locale it
            # came from, so the region survives without splitting the mixture.
            "tags": {**dict(tags), **_canonical_tags(tags, task), **locales},
            "task": task,
            "lang_key": lang_key,
        })
    return sources


def _canonical_tags(tags: dict, task: str) -> dict:
    """The language tags rewritten onto their canonical codes."""
    if task == "st":
        return {
            "src_lang": canonical_lang(tags["src_lang"]),
            "tgt_lang": canonical_lang(tags["tgt_lang"]),
        }
    return {"lang": canonical_lang(tags["lang"])}


def report_locale_folding(sources: list[dict]) -> None:
    """Show which locale codes were folded onto which language."""
    folded: dict[str, set[str]] = defaultdict(set)
    for s in sources:
        for key in ("region_code", "src_region_code", "tgt_region_code"):
            code = s["tags"].get(key)
            if code and canonical_lang(code) != code:
                folded[canonical_lang(code)].add(code)
    if not folded:
        return
    print("\nLocale codes folded onto their language "
          "(the original is kept per-cut as *region_code):")
    for lang, codes in sorted(folded.items()):
        print(f"    {lang:<6} <- {', '.join(sorted(codes))}")


# ---------------------------------------------------------------------------
# Duration measurement
# ---------------------------------------------------------------------------

def shard_seconds(shard: str) -> float:
    """Sum top-level cut durations in one gzipped JSONL manifest.

    Only the top-level ``duration`` is counted. A regex over the raw text would
    be faster but would also pick up ``duration`` inside each supervision and
    silently double-count.
    """
    total = 0.0
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line)["duration"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return total


def write_cache(cache: dict, path: Path) -> None:
    """Write the hours cache atomically, so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(path)


def plan_source(path: str, sample: int | None) -> tuple[list[str], int]:
    """Return (shards to read, total shard count) for one source."""
    shards = sorted(Path(path).glob("cuts.*.jsonl.gz"))
    chosen = shards if not sample else shards[:sample]
    return [str(s) for s in chosen], len(shards)


def measure_shard(task: tuple[str, str]) -> tuple[str, float]:
    """Return (source_path, seconds) for one shard.

    The unit of work is a shard rather than a source because shard counts span
    four orders of magnitude (1 to >10k here). Mapping over sources would leave
    one worker reading the largest corpus alone while the rest idle.
    """
    source, shard = task
    return source, shard_seconds(shard)


# ---------------------------------------------------------------------------
# The weighting policy
# ---------------------------------------------------------------------------

def compute_weights(sources: list[dict], alpha: float, beta: float) -> list[dict]:
    """Attach p_c, p_l and p_cl to each source per Section 3.3.1."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in sources:
        by_lang[s["lang_key"]].append(s)

    lang_hours = {l: sum(s["hours"] for s in ss) for l, ss in by_lang.items()}
    total_hours = sum(lang_hours.values())
    if total_hours <= 0:
        raise ValueError("Total hours is zero — no manifests were read.")

    # Step 1: corpora within each language.
    for lang, members in by_lang.items():
        n_l = lang_hours[lang]
        if n_l <= 0:
            for s in members:
                s["p_c"] = 1.0 / len(members)
            continue
        w = [(s["hours"] / n_l) ** alpha if s["hours"] > 0 else 0.0 for s in members]
        tot = sum(w)
        for s, wi in zip(members, w):
            s["p_c"] = (wi / tot) if tot > 0 else 1.0 / len(members)

    # Step 2: languages against each other.
    w_l = {l: (h / total_hours) ** beta if h > 0 else 0.0
           for l, h in lang_hours.items()}
    tot_l = sum(w_l.values())
    p_l = {l: (w / tot_l) if tot_l > 0 else 0.0 for l, w in w_l.items()}

    # Step 3: the product.
    for s in sources:
        s["p_l"] = p_l[s["lang_key"]]
        s["p_cl"] = s["p_l"] * s["p_c"]

    return sources


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Tag keys that define a language entry, and are therefore constant across every
# corpus in a group. Anything else (text_field, the *region_code keys, ...) stays
# on the leaf, since a group can hold several locales of one language.
GROUP_TAG_KEYS = ("task", "lang", "src_lang", "tgt_lang")


def emit_nemo_group_yaml(sources: list[dict], out: Path,
                         alpha: float, beta: float) -> None:
    """Write the mixture as a nested NeMo ``type: group`` input_cfg.

    The two tiers of the policy map onto the two levels of the schema: the group
    weight is p_l and each child weight is p_c. NeMo muxes the groups against
    each other and the corpora within a group, so the effective sampling
    probability is the product p_l * p_c = p_cl — no need to pre-multiply.

    This is the same schema NeMo's own ``estimate_data_weights.py`` produces, so
    the emitted file is consumable by NeMo Speech directly as well as by MELT.
    """
    lines = header_comment(sources, alpha, beta) + build_group_block(sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def header_comment(sources: list[dict], alpha: float, beta: float) -> list[str]:
    n_langs = len({s["lang_key"] for s in sources})
    total_hours = sum(s["hours"] for s in sources)
    return [
        "# Generated by infra/compute_mix_weights.py — do not hand-edit.",
        "# Two-tier language/corpus balancing, Section 3.3.1 of arXiv:2509.14128,",
        f"# with alpha={alpha} (corpus within language) and beta={beta} (across languages).",
        "#",
        "# Group weight   = p_l, the language entry's share of the mix.",
        "# Child weight   = p_c, the corpus's share within its language entry.",
        "# The two levels are muxed separately, so the sampling probability of a",
        "# corpus is the product p_l * p_c. Do not pre-multiply them.",
        "#",
        "# Do NOT set `reweight_temperature` alongside this file (NeMo only).",
        "# alpha/beta are already baked into the weights below; NeMo would raise",
        "# them to a further power. Both levels already sum to 1, so the",
        "# normalisation itself is a no-op.",
        "#",
        f"# {len(sources)} corpora, {n_langs} language entries, {total_hours:,.1f} h total.",
    ]


def build_group_block(sources: list[dict], indent: int = 0) -> list[str]:
    """The nested ``input_cfg:`` block, as lines at the given indent."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in sources:
        by_lang[s["lang_key"]].append(s)

    def hours_of(lang: str) -> float:
        return sum(s["hours"] for s in by_lang[lang])

    pad = " " * indent
    lines = [f"{pad}input_cfg:"]
    for lang in sorted(by_lang, key=lambda l: -hours_of(l)):
        members = sorted(by_lang[lang], key=lambda s: -s["p_c"])
        lines.append(f"{pad}  # {lang}: {hours_of(lang):,.1f} h "
                     f"across {len(members)} corpora")
        lines.append(f"{pad}  - type: group")
        lines.append(f"{pad}    weight: {members[0]['p_l']:.8f}")
        # Every member of a group shares these by construction, so reading them
        # off the first member is safe.
        group_tags = {k: v for k, v in members[0]["tags"].items()
                      if k in GROUP_TAG_KEYS}
        if group_tags:
            lines.append(f"{pad}    tags:")
            for k, v in group_tags.items():
                lines.append(f"{pad}      {k}: {v}")
        lines.append(f"{pad}    input_cfg:")
        for s in members:
            lines.append(f"{pad}      - type: lhotse_shar")
            lines.append(f"{pad}        shar_path: {s['raw_path']}")
            lines.append(f"{pad}        weight: {s['p_c']:.8f}")
            if s["tags"]:
                lines.append(f"{pad}        tags:")
                for k, v in s["tags"].items():
                    lines.append(f"{pad}          {k}: {v}")
    return lines


def emit_training_config(sources: list[dict], template: Path, out: Path,
                         alpha: float, beta: float,
                         exp_name: str | None = None) -> None:
    """Copy a training config, replacing train_ds.input_cfg with the weighted mix.

    The rest of the file is passed through byte for byte. A YAML round-trip
    would drop every comment and reorder the keys, and these configs are
    hand-maintained, so the block is spliced textually instead.
    """
    text = template.read_text(encoding="utf-8")
    lines = text.splitlines()

    start, end, indent = _find_block(lines, "train_ds", "input_cfg")
    block = ([f"{' ' * indent}{c}" for c in header_comment(sources, alpha, beta)]
             + build_group_block(sources, indent=indent))
    lines[start:end] = block

    # The mixture's measured hours are better than whatever the template said,
    # and train_ds.total_hours drives the steps-per-epoch estimate.
    total_hours = sum(s["hours"] for s in sources)
    _replace_scalar(lines, start + len(block), "total_hours", f"{total_hours:.1f}")

    if exp_name:
        _replace_scalar(lines, 0, "exp_name", exp_name)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_block(lines: list[str], section: str, key: str) -> tuple[int, int, int]:
    """Locate ``key:`` under ``section:``; return (start, end, indent of key)."""
    sec = next((i for i, l in enumerate(lines)
                if re.match(rf"^\s*{section}:\s*$", l)), None)
    if sec is None:
        raise ValueError(f"Could not find a '{section}:' section in the template")
    sec_indent = len(lines[sec]) - len(lines[sec].lstrip())

    for i in range(sec + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= sec_indent:
            break  # left the section without finding the key
        if re.match(rf"^\s*{key}:\s*$", line):
            # The block runs until the next non-blank line indented no further.
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if not nxt.strip():
                    continue
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    return i, j, indent
            return i, len(lines), indent
    raise ValueError(f"Could not find '{key}:' under '{section}:' in the template")


def _replace_scalar(lines: list[str], start: int, key: str, value: str) -> None:
    """Rewrite the first ``key: ...`` at or after start, within the same block."""
    for i in range(start, len(lines)):
        m = re.match(rf"^(\s*){key}:\s*\S.*$", lines[i])
        if m:
            lines[i] = f"{m.group(1)}{key}: {value}"
            return


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--datasets-root", default=os.environ.get("LOCAL_DATASETS_DIR"),
                   help="Value for LOCAL_DATASETS_DIR (default: from the environment).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Corpus-level subsampling factor (paper: 0.5).")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Language-level upsampling factor (paper: 0.5).")
    p.add_argument("--sample-shards", type=int, default=None,
                   help="Read only the first N shards per source and extrapolate.")
    p.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 4)))
    p.add_argument("--cache", type=Path, default=Path("infra/.mix_weights_hours.json"),
                   help="Where measured per-source hours are cached.")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--emit-yaml", type=Path, default=None,
                   help="Write a flat shar_path -> p_cl mapping to this file.")
    p.add_argument("--emit-nemo", type=Path, default=None,
                   help="Write a nested `type: group` input_cfg (p_l on the "
                        "group, p_c on each child). Consumable by NeMo Speech.")
    p.add_argument("--emit-config", type=Path, default=None,
                   help="Write a full training config: a copy of --config with "
                        "train_ds.input_cfg replaced by the weighted mixture.")
    p.add_argument("--exp-name", default=None,
                   help="Set run.exp_name in the emitted config.")
    args = p.parse_args()

    sources = load_sources(args.config, args.datasets_root)
    print(f"Parsed {len(sources)} sources from {args.config}")

    cache: dict[str, dict] = {}
    if args.cache.exists() and not args.no_cache:
        cache = json.loads(args.cache.read_text())

    todo = [s["path"] for s in sources
            if s["path"] not in cache
            or cache[s["path"]].get("sample") != args.sample_shards]
    if todo:
        tasks: list[tuple[str, str]] = []
        n_shards: dict[str, int] = {}
        n_read: dict[str, int] = {}
        for src in todo:
            chosen, total = plan_source(src, args.sample_shards)
            n_shards[src], n_read[src] = total, len(chosen)
            tasks.extend((src, sh) for sh in chosen)

        print(f"Measuring {len(todo)} sources / {len(tasks)} shards with "
              f"{args.jobs} workers "
              f"({'all shards' if not args.sample_shards else f'{args.sample_shards} shards each'})…")
        # A full pass reads tens of thousands of manifests over a network mount
        # and takes hours, so each source is cached the moment it completes.
        # Re-running then resumes instead of starting over.
        remaining = dict(n_read)
        for src in todo:
            if n_read[src] == 0:
                cache[src] = {"hours": 0.0, "shards": n_shards[src],
                              "sample": args.sample_shards}

        seconds: dict[str, float] = defaultdict(float)
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            # pool.map preserves input order, and tasks are grouped by source,
            # so a source is finished as soon as its last shard comes back.
            for src, secs in pool.map(measure_shard, tasks, chunksize=4):
                seconds[src] += secs
                remaining[src] -= 1
                done += 1
                if remaining[src]:
                    continue
                total = seconds[src]
                if args.sample_shards and n_read[src] < n_shards[src]:
                    # Shards within a source are near-uniform: scale by the ratio.
                    total *= n_shards[src] / n_read[src]
                cache[src] = {"hours": total / 3600.0, "shards": n_shards[src],
                              "sample": args.sample_shards}
                print(f"  [{done}/{len(tasks)} shards] {total / 3600.0:10.1f} h "
                      f"{n_shards[src]:6d} shards  {src}", flush=True)
                if not args.no_cache:
                    write_cache(cache, args.cache)
        if not args.no_cache:
            write_cache(cache, args.cache)
            print(f"Cached hours -> {args.cache}")
    else:
        print("All sources found in cache.")

    for s in sources:
        s["hours"] = cache[s["path"]]["hours"]
        s["shards"] = cache[s["path"]]["shards"]

    missing = [s["path"] for s in sources if s["shards"] == 0]
    if missing:
        print(f"\nWARNING: {len(missing)} sources had no cuts.*.jsonl.gz and count as 0 h:")
        for m in missing[:10]:
            print(f"    {m}")

    report_locale_folding(sources)
    compute_weights(sources, args.alpha, args.beta)

    total_hours = sum(s["hours"] for s in sources)
    n_langs = len({s["lang_key"] for s in sources})
    print(f"\nalpha={args.alpha}  beta={args.beta}  "
          f"{n_langs} language entries  {total_hours:,.1f} h total\n")

    print(f"{'p_cl':>9} {'nat.share':>10} {'boost':>7} {'hours':>11} "
          f"{'lang':<10} {'source'}")
    for s in sorted(sources, key=lambda x: -x["p_cl"]):
        natural = s["hours"] / total_hours if total_hours else 0.0
        boost = (s["p_cl"] / natural) if natural > 0 else float("inf")
        print(f"{s['p_cl']:>9.6f} {natural:>10.6f} {boost:>7.2f}x "
              f"{s['hours']:>11.1f} {s['lang_key']:<10} {s['path']}")

    if args.emit_yaml:
        lines = ["# Generated by infra/compute_mix_weights.py — do not hand-edit.",
                 f"# Section 3.3.1 of arXiv:2509.14128, alpha={args.alpha}, beta={args.beta}.",
                 "# Maps shar_path -> weight (p_cl). Paste each value into the",
                 "# matching input_cfg entry as `weight:`.",
                 "weights:"]
        for s in sorted(sources, key=lambda x: x["path"]):
            lines.append(f"  {s['path']}: {s['p_cl']:.8f}")
        args.emit_yaml.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.emit_yaml}")

    if args.emit_nemo:
        emit_nemo_group_yaml(sources, args.emit_nemo, args.alpha, args.beta)
        print(f"Wrote {args.emit_nemo} (nested group input_cfg)")

    if args.emit_config:
        emit_training_config(sources, args.config, args.emit_config,
                             args.alpha, args.beta, args.exp_name)
        print(f"Wrote {args.emit_config} (full training config)")


if __name__ == "__main__":
    main()
