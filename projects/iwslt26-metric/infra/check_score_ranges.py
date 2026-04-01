"""Check score ranges across Lhotse Shar datasets.

Reads a list of dataset entries from a YAML config (same input_cfg format as
config.yaml data sections), loads each CutSet, and reports the score range
found in the data. Warns when the normalize_factor looks inconsistent with
the observed range.

Usage:
    python projects/iwslt26-metric/infra/check_score_ranges.py --config projects/iwslt26-metric/config.yaml
    python projects/iwslt26-metric/infra/check_score_ranges.py --config my_check.yaml

The config YAML must have at least one of the following top-level keys with
an ``input_cfg`` list:
    data.train_ds.input_cfg
    data.validation_ds.input_cfg

Or you can pass a minimal YAML with just an ``input_cfg`` list at the top level:

    input_cfg:
      - type: lhotse_shar
        shar_path: /path/to/shar
        tags:
          target_field: custom.score
          normalize_factor: 100.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from lhotse import CutSet
from omegaconf import OmegaConf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested_value(obj, path: str):
    """Resolve a dot-notation path on a Lhotse Cut (or any nested dict/obj)."""
    parts = path.split(".")
    cur = obj
    for part in parts:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


@dataclass
class DatasetStats:
    shar_path: str
    target_field: str
    config_normalize_factor: float
    count: int = 0
    score_min: float = float("inf")
    score_max: float = float("-inf")
    errors: list[str] = field(default_factory=list)

    @property
    def range_ok(self) -> bool:
        """True when min/max raw scores lie in [0, 1] (already normalised)."""
        return self.score_min >= 0.0 and self.score_max <= 1.0

    @property
    def expected_normalize_factor(self) -> float | None:
        """Guess expected normalise factor from the raw range."""
        if self.score_max <= 1.0:
            return 1.0
        if self.score_max <= 100.0:
            return 100.0
        return None

    @property
    def factor_mismatch(self) -> bool:
        expected = self.expected_normalize_factor
        if expected is None:
            return False  # can't tell
        return abs(self.config_normalize_factor - expected) > 1e-6


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _collect_input_cfgs(cfg) -> list[dict]:
    """Extract all input_cfg entries from the loaded OmegaConf config."""
    entries: list[dict] = []

    def _harvest(node):
        if node is None:
            return
        cfg_dict = OmegaConf.to_container(node, resolve=True) if not isinstance(node, (dict, list)) else node
        if isinstance(cfg_dict, list):
            entries.extend(cfg_dict)
        elif isinstance(cfg_dict, dict) and "input_cfg" in cfg_dict:
            entries.extend(cfg_dict["input_cfg"])

    raw = OmegaConf.to_container(cfg, resolve=True)

    if isinstance(raw, list):
        # top-level list
        entries.extend(raw)
    elif isinstance(raw, dict):
        if "input_cfg" in raw:
            # minimal YAML with just input_cfg at top
            entries.extend(raw["input_cfg"])
        else:
            # Full training config — harvest train_ds and validation_ds
            for ds_key in ("train_ds", "validation_ds"):
                ds = raw.get("data", {}).get(ds_key, {})
                if isinstance(ds, dict) and "input_cfg" in ds:
                    entries.extend(ds["input_cfg"])

    return entries


def analyse_dataset(entry: dict, sample_limit: int | None) -> DatasetStats:
    shar_path = entry.get("shar_path", "<unknown>")
    tags = entry.get("tags", {}) or {}
    target_field = tags.get("target_field", "custom.score")
    normalize_factor = float(tags.get("normalize_factor", 1.0))

    stats = DatasetStats(
        shar_path=shar_path,
        target_field=target_field,
        config_normalize_factor=normalize_factor,
    )

    path = Path(shar_path)
    if not path.exists():
        stats.errors.append(f"Path does not exist: {shar_path}")
        return stats

    try:
        cuts = CutSet.from_shar(in_dir=shar_path)
    except Exception as exc:
        stats.errors.append(f"Failed to load CutSet: {exc}")
        return stats

    for i, cut in enumerate(cuts):
        if sample_limit is not None and i >= sample_limit:
            break
        raw = _get_nested_value(cut, target_field)
        if raw is None:
            stats.errors.append(f"Cut '{cut.id}': field '{target_field}' not found (first occurrence)")
            if len(stats.errors) >= 3:
                break
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            stats.errors.append(f"Cut '{cut.id}': cannot convert {raw!r} to float")
            continue

        stats.count += 1
        if score < stats.score_min:
            stats.score_min = score
        if score > stats.score_max:
            stats.score_max = score

    return stats


def print_report(all_stats: list[DatasetStats]) -> int:
    """Print a summary table. Returns number of warnings."""
    RESET  = "\033[0m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    GREEN  = "\033[32m"
    BOLD   = "\033[1m"

    warnings = 0

    print()
    print(f"{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  Score-range audit ({len(all_stats)} dataset(s)){RESET}")
    print(f"{BOLD}{'='*80}{RESET}")

    for stats in all_stats:
        short_path = stats.shar_path
        print()
        print(f"  {BOLD}{short_path}{RESET}")
        print(f"    target_field       : {stats.target_field}")
        print(f"    config norm factor : {stats.config_normalize_factor}")

        if stats.errors:
            for err in stats.errors:
                print(f"    {RED}ERROR: {err}{RESET}")
            warnings += 1
            continue

        if stats.count == 0:
            print(f"    {YELLOW}WARNING: no cuts found{RESET}")
            warnings += 1
            continue

        range_str = f"[{stats.score_min:.4f}, {stats.score_max:.4f}]"
        normalized_min = stats.score_min / stats.config_normalize_factor
        normalized_max = stats.score_max / stats.config_normalize_factor
        normalized_str = f"[{normalized_min:.4f}, {normalized_max:.4f}]"

        print(f"    cuts inspected     : {stats.count}")
        print(f"    raw score range    : {range_str}")
        print(f"    after normalising  : {normalized_str}")

        if stats.factor_mismatch:
            expected = stats.expected_normalize_factor
            print(
                f"    {RED}WARNING: normalize_factor={stats.config_normalize_factor} looks wrong "
                f"for range {range_str}  (expected ~{expected}){RESET}"
            )
            warnings += 1
        elif not stats.range_ok:
            print(
                f"    {YELLOW}WARNING: normalised range {normalized_str} is outside [0, 1] — "
                f"check normalize_factor{RESET}"
            )
            warnings += 1
        else:
            print(f"    {GREEN}OK: normalised range is within [0, 1]{RESET}")

    print()
    print(f"{BOLD}{'='*80}{RESET}")
    if warnings:
        print(f"  {RED}{BOLD}{warnings} warning(s) found — review datasets above.{RESET}")
    else:
        print(f"  {GREEN}{BOLD}All datasets look consistent.{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")
    print()

    return warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to YAML config (training config or minimal input_cfg YAML)")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Inspect only the first N cuts per dataset (default: all)",
    )
    parser.add_argument(
        "--ds",
        choices=["train", "val", "both"],
        default="both",
        help="Which dataset split to check when reading a full training config (default: both)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # If a full training config, optionally restrict to one split
    raw = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        filtered: dict = {}
        if args.ds in ("train", "both") and "train_ds" in data:
            filtered["train_ds"] = data["train_ds"]
        if args.ds in ("val", "both") and "validation_ds" in data:
            filtered["validation_ds"] = data["validation_ds"]
        cfg = OmegaConf.create({"data": filtered})

    entries = _collect_input_cfgs(cfg)
    if not entries:
        print("No input_cfg entries found in the provided config.", file=sys.stderr)
        sys.exit(1)

    # Filter to lhotse_shar entries only
    shar_entries = [e for e in entries if isinstance(e, dict) and e.get("type") == "lhotse_shar"]
    if not shar_entries:
        print("No 'lhotse_shar' entries found — nothing to check.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shar_entries)} lhotse_shar dataset(s). Inspecting scores...")

    all_stats = [
        analyse_dataset(entry, args.sample)
        for entry in tqdm(shar_entries, desc="Scanning datasets", unit="ds")
    ]

    warnings = print_report(all_stats)
    sys.exit(1 if warnings else 0)


if __name__ == "__main__":
    main()
