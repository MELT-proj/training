"""Lightweight import sanity checks.

Usage:
  python infra/check_imports.py

This script is intentionally dependency-light: it only imports the local code
and reports failures with a clear message.
"""

from __future__ import annotations

import importlib
import sys


MODULES = [
    "src",
    "src.ddp",
    "src.logging_utils",
    "src.modeling",
    "src.modeling.configuration_melt",
    "src.modeling.modeling_melt",
    "src.modeling.processing_melt",
    "src.training",
    "src.training.config",
    "src.training.trainer",
    "src.training.train",
    "src.evaluation",
    "src.evaluation.evaluate",
]


def main() -> None:
    failures: list[tuple[str, str]] = []

    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"OK: import {name}")
        except Exception as e:
            failures.append((name, repr(e)))
            print(f"FAIL: import {name} -> {e!r}")

    if failures:
        print("\nImport failures:")
        for name, err in failures:
            print(f"- {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    # Ensure repo root is on sys.path when invoked as `python infra/check_imports.py`.
    # (Python inserts the script directory; we want the repo root.)
    sys.path.insert(0, ".")
    main()
