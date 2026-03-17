import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_utils import configure_logging, get_logger


def _emit_message(name: str, message: str) -> str:
    logger = get_logger(name)
    logger.handlers = []
    logger.propagate = False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(message)
    handler.flush()
    logger.removeHandler(handler)
    return stream.getvalue()


def test_configure_logging_silences_non_global_master(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    configure_logging(level=logging.INFO)

    assert logging.root.manager.disable == logging.CRITICAL
    assert _emit_message("tests.logging.non_master", "hidden") == ""


def test_configure_logging_enables_global_master(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    configure_logging(level=logging.INFO)

    assert logging.root.manager.disable == logging.NOTSET
    assert "visible" in _emit_message("tests.logging.master", "visible")
