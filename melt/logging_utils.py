"""Central logging utilities.

Why this module exists:
- In distributed training (Accelerate/DDP), every rank tends to emit logs.
- This project wants *only the global master process* (rank 0) to produce logs.

Note: we intentionally avoid naming this file `logging.py` because training is
launched as `python src/train.py`, which puts `src/` on `sys.path` and would
cause a local `src/logging.py` to shadow Python's standard-library `logging`.
"""

from __future__ import annotations

import logging
import os


def _is_global_master() -> bool:
    try:
        from . import ddp

        return (not ddp.is_distributed()) or ddp.is_global_master()
    except Exception:
        # Fallback for early startup/import edge-cases.
        # In distributed runs, non-zero RANK should stay silent.
        return int(os.environ.get("RANK", "0")) == 0


def configure_logging(
    *,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> None:
    """Configure Python + Transformers logging.

    For non-master ranks, this function silences Python logging completely.

    Safe to call multiple times.
    """

    is_master = _is_global_master()

    # `force=True` ensures we reset handlers even if other modules called
    # logging.basicConfig() earlier.
    logging.basicConfig(
        level=level if is_master else logging.ERROR,
        format=log_format,
        force=True,
    )

    if is_master:
        logging.disable(logging.NOTSET)
    else:
        # Silence all logging from non-master processes.
        logging.disable(logging.CRITICAL)

    # Keep Transformers' verbosity consistent with our policy.
    try:
        from transformers.utils import logging as hf_logging

        if is_master:
            hf_logging.set_verbosity_info()
        else:
            hf_logging.set_verbosity_error()
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Log emission is controlled centrally by `configure_logging()`.
    """

    return logging.getLogger(name)


# Print model layers for inspection/debugging and write to file
def _print_model_layers(m, out_dir: str | None = None):
    """Log leaf-level model modules (layers) by name and type and optionally write to file."""
    logger.info("Listing model layers (leaf modules):")
    lines = []
    for name, module in m.named_modules():
        if not name:
            continue
        # Consider leaf modules only (no child modules)
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            line = f"{name}: {module.__class__.__name__} | params={params:,}"
            logger.info("  %s", line)
            lines.append(line)
    if out_dir is not None:
        try:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file = out_path / "model_layers.txt"
            file.write_text("\n".join(lines) + "\n")
            logger.info("Wrote model layers list to %s", str(file))
        except Exception as e:
            logger.warning("Could not write model layers to %s: %s", out_dir, e)
