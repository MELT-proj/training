#!/usr/bin/env python3
"""The master results table: one CSV, one row per model, one column per measured cell.

Every number a chart or a table quotes lives in ``results_master.csv`` next to this file, and this module is the
only sanctioned way to change it (``PROTOCOL.md`` says why and how). It is a library (the ``ingest_*.py`` scripts
use it) and a command line:

    python master.py where                        # the resolved paths and the lock state
    python master.py validate                     # the checks every writer and reader relies on
    python master.py show [-m MODEL ...] [-c GLOB ...] [--markdown]
    python master.py columns [GLOB ...] [--sets]  # columns and how many models fill each; the set registry
    python master.py upsert [--cells cells.csv] [--meta meta.csv] --who WHO --note TEXT
                            [--overwrite --reason TEXT] [--dry-run]
    python master.py add-set --set S --fmt F --task T --split SPLIT --n-per-unit N --metrics "m1 m2"
                             --description TEXT --who WHO
    python master.py export-long [--out FILE]     # tidy form (one row per measurement) for pandas / R
    python master.py install-git-driver           # once per clone: cell-level merge of the master (see below)

Layout. Rows are models (``model`` is a stable key). The first columns are metadata (``META_COLUMNS``); every
other column is one measurement named ``<set>.<fmt>.<metric>.<unit>``:

* ``set``    the frozen evaluation set (``fleurs24-asr-dev``) or, for published numbers, the source's set;
* ``fmt``    the prompt format the model was run under (``repeatfirst``, ``sft-random``; ``native`` for published);
* ``metric`` one of ``METRICS`` (``wer``, ``chrfpp-lc``, ...). A name is a promise about the scorer;
* ``unit``   a language code, a language group, or ``all``.

A blank cell is "not measured". Every ``set.fmt`` prefix must be registered in ``results_sets.csv``.

Writes are serialised by an exclusive lock file, applied as read-modify-write inside that lock, replace the file
atomically, never overwrite a different value unless asked to (``--overwrite`` plus a reason), and leave one line
per model in ``results_changelog.csv``. Readers need no lock: the file is replaced, never edited in place.

Sessions on different branches or worktrees each hold their own copy and share it through git. A line-based merge of
a wide CSV is useless here: two branches that each add a column change the header and every row, so git reports the
whole file as one conflict. The repository therefore maps the master to a merge driver (``merge-driver``, installed
per clone by ``install-git-driver``) that merges cell by cell: edits to different cells combine, only two different
values for one cell are a conflict.

Standard library only (Python >= 3.10), so it runs in any venv and in the training container.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENV_MASTER = "MELT_RESULTS_MASTER"
MASTER_NAME = "results_master.csv"
SETS_NAME = "results_sets.csv"
CHANGELOG_NAME = "results_changelog.csv"

META_COLUMNS = ("model", "label", "kind", "family", "recipe", "run", "status", "order", "source", "notes")
NEW_ROW_REQUIRES = ("label", "kind", "source")
KINDS = ("ours", "baseline", "reference")
STATUSES = ("active", "deprecated")
SETS_COLUMNS = ("set", "fmt", "task", "split", "n_per_unit", "metrics", "description")
CHANGELOG_COLUMNS = (
    "ts_utc", "who", "host", "op", "model", "added", "changed", "unchanged", "prefixes", "note", "reason", "changes",
)  # fmt: skip

#: metric name -> (scale, definition). A name is a promise about the scorer: never reuse one for a different scorer,
#: add a new name here (in a reviewed change) instead. Order is the column order inside a ``set.fmt``.
METRICS: dict[str, tuple[str, str]] = {
    "wer": (
        "percent",
        "Word error rate, corpus level per language. melt-eval normaliser `basic` (Whisper BasicTextNormalizer, the "
        "multilingual one: lowercase, bracketed text removed, symbols and punctuation to spaces) on reference and "
        "completion; the completion is scored as it stands; not clipped. Published numbers use their source's own "
        "normaliser.",
    ),
    "cer": ("percent", "Character error rate, same conventions as `wer`; raw, not clipped at 100."),
    "chrfpp": ("0-100", "chrF++ (sacrebleu CHRF, character 6-grams, word order 2), cased, corpus level per direction."),
    "chrfpp-lc": (
        "0-100",
        "chrF++ on lowercased reference and hypothesis: the primary ST score, because standard-training models write "
        "translations without capitals.",
    ),
    "spbleu": ("0-100", "spBLEU: sacrebleu BLEU with tokenize=flores200, cased."),
    "spbleu-lc": ("0-100", "spBLEU on lowercased reference and hypothesis."),
    "bleu": ("0-100", "sacrebleu BLEU, 13a tokenizer, cased (legacy comparability)."),
    "bleu-lc": ("0-100", "BLEU (13a) on lowercased reference and hypothesis."),
    "chrf": ("0-100", "chrF (character 6-grams, word order 0), cased."),
    "chrf-lc": ("0-100", "chrF on lowercased reference and hypothesis."),
    "nocaps": ("percent", "Share of translations written without a single capital letter."),
    "runaway": (
        "percent",
        "Share of samples whose per-sample character errors exceed the reference length (a decoder that loops).",
    ),
    "score": (
        "fraction",
        "Selection metric of the campaign: (1.0 med ID + 1.5 med OOD-train + 0.6 med OOD-related) / 3.1 over "
        "per-language CER clipped at 1; lower is better. See selection-metric/README.md.",
    ),
    "median": ("fraction", "Median of the clipped per-language CER of one selection-metric group."),
}  # fmt: skip
METRIC_ORDER = {name: i for i, name in enumerate(METRICS)}

#: Canonical unit order inside a metric: aggregates first, then the 24 EU languages by descending hours of training
#: audio (the order of the EU24 tables), then the two non-EU languages the selection metric adds.
UNIT_ORDER = (
    "all", "id", "ood-train", "ood-related", "ood-latin", "ood-script",
    "en", "es", "fr", "de", "it", "nl", "pt", "pl", "hu", "fi", "cs", "ro", "sv",
    "sk", "bg", "lv", "da", "lt", "el", "et", "hr", "sl", "ga", "mt", "ru", "uk",
)  # fmt: skip
UNIT_RANK = {unit: i for i, unit in enumerate(UNIT_ORDER)}

_TOKEN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MasterError(Exception):
    """The master, the registry or a write request is unusable; nothing was written."""


class SchemaError(MasterError):
    """A column name or a metadata field breaks the schema."""


class LockTimeout(MasterError):
    """Another writer holds the lock. It is never broken automatically."""


class OverwriteRefused(MasterError):
    """A write would change cells that already hold a different value."""


# --- paths ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """The files of one master; the registry, the changelog and the lock sit next to it."""

    master: Path

    @property
    def sets(self) -> Path:
        return self.master.with_name(SETS_NAME)

    @property
    def changelog(self) -> Path:
        return self.master.with_name(CHANGELOG_NAME)

    @property
    def lock(self) -> Path:
        return self.master.with_name(f".{self.master.name}.lock")


def resolve_paths(master: str | os.PathLike | None = None) -> Paths:
    """``master`` if given, else ``$MELT_RESULTS_MASTER``, else the file next to this module."""
    chosen = master or os.environ.get(ENV_MASTER) or (HERE / MASTER_NAME)
    return Paths(Path(chosen).expanduser().resolve())


# --- column grammar ------------------------------------------------------------------------------------------------


def parse_column(name: str) -> tuple[str, str, str, str]:
    """Split ``set.fmt.metric.unit`` into its four parts.

    Raises:
        SchemaError: If the name does not have exactly four lowercase hyphenated tokens.
    """
    parts = name.split(".")
    if len(parts) != 4 or not all(_TOKEN.match(part) for part in parts):
        raise SchemaError(
            f"column {name!r} is not <set>.<fmt>.<metric>.<unit> (four tokens of a-z, 0-9 and single hyphens)"
        )
    return parts[0], parts[1], parts[2], parts[3]


def column_sort_key(name: str) -> tuple:
    """Canonical column order: set, fmt, metric (``METRICS`` order), unit (``UNIT_ORDER``, then alphabetical)."""
    s, f, m, u = parse_column(name)
    return (s, f, METRIC_ORDER.get(m, len(METRIC_ORDER)), m, UNIT_RANK.get(u, len(UNIT_RANK)), u)


def check_column(column: str, sets: Mapping[tuple[str, str], Mapping[str, str]]) -> None:
    """The column's grammar, its metric and its ``set.fmt`` registration.

    Raises:
        SchemaError: On the first problem, with the fix in the message.
    """
    s, f, m, _ = parse_column(column)
    if m not in METRICS:
        raise SchemaError(
            f"{column}: unknown metric {m!r} (known: {', '.join(METRICS)}); a new metric is a reviewed change to "
            "METRICS in master.py, because its name promises a scorer"
        )
    registered = sets.get((s, f))
    if registered is None:
        raise SchemaError(f"{column}: {s}.{f} is not in {SETS_NAME}; register it first (master.py add-set)")
    if m not in registered["metrics"].split():
        raise SchemaError(f"{column}: metric {m!r} is not listed for {s}.{f} in {SETS_NAME} ({registered['metrics']})")


def fmt_value(value: float | int | str, column: str = "") -> str:
    """A finite number as the text stored in the CSV: 4 decimals (6 for fractions), trailing zeros dropped."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MasterError(f"{column or 'value'}: {value!r} is not a number") from exc
    if not math.isfinite(number):
        raise MasterError(f"{column or 'value'}: {value!r} is not finite; leave the cell blank instead")
    digits = 4
    if column:
        scale = METRICS.get(parse_column(column)[2], ("", ""))[0]
        digits = 6 if scale == "fraction" else 4
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


# --- reading -------------------------------------------------------------------------------------------------------


@dataclass
class Table:
    """The master in memory: ``header`` lists every column (metadata first), ``rows`` maps model -> column -> text."""

    header: list[str]
    rows: dict[str, dict[str, str]]

    @property
    def data_columns(self) -> list[str]:
        return self.header[len(META_COLUMNS) :]


def read_table(path: Path, *, missing_ok: bool = False) -> Table:
    """Read the master, refusing a file that is ragged, duplicated or carries git conflict markers.

    Raises:
        MasterError: If the file does not exist (unless *missing_ok*) or is structurally broken.
    """
    if not path.exists():
        if missing_ok:
            return Table(list(META_COLUMNS), {})
        raise MasterError(f"{path} does not exist")
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            raise MasterError(f"{path}:{number}: git conflict marker; PROTOCOL.md, section Recovery, says what to do")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None:
        raise MasterError(f"{path} is empty")
    if tuple(header[: len(META_COLUMNS)]) != META_COLUMNS:
        raise MasterError(f"{path}: the first columns must be {', '.join(META_COLUMNS)}")
    duplicates = [name for name, n in Counter(header).items() if n > 1]
    if duplicates:
        raise MasterError(f"{path}: duplicated columns: {', '.join(duplicates[:5])}")
    rows: dict[str, dict[str, str]] = {}
    for number, row in enumerate(reader, 2):
        if not row:
            continue
        if len(row) != len(header):
            raise MasterError(
                f"{path}:{number}: {len(row)} fields but the header has {len(header)} (a ragged row, usually a bad "
                "merge or a hand edit)"
            )
        key = row[0]
        if key in rows:
            raise MasterError(f"{path}:{number}: model {key!r} appears twice")
        rows[key] = dict(zip(header, row, strict=True))
    return Table(header, rows)


def read_sets(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """The registry of evaluation sets, keyed by ``(set, fmt)``."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SETS_COLUMNS:
            raise MasterError(f"{path}: the header must be {','.join(SETS_COLUMNS)}")
        sets: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            key = (row["set"], row["fmt"])
            if key in sets:
                raise MasterError(f"{path}: {key[0]}.{key[1]} is registered twice")
            sets[key] = row
    return sets


def validate_table(table: Table, sets: Mapping[tuple[str, str], Mapping[str, str]]) -> list[str]:
    """Every problem found in *table*, as readable lines (empty when it is sound)."""
    problems: list[str] = []
    for column in table.data_columns:
        try:
            check_column(column, sets)
        except SchemaError as exc:
            problems.append(str(exc))
    for model, row in table.rows.items():
        if not _MODEL.match(model):
            problems.append(f"model key {model!r}: use letters, digits, '.', '_' and '-' only")
        if not row["label"]:
            problems.append(f"{model}: empty label")
        if row["kind"] not in KINDS:
            problems.append(f"{model}: kind {row['kind']!r} is not one of {', '.join(KINDS)}")
        if row["status"] not in STATUSES:
            problems.append(f"{model}: status {row['status']!r} is not one of {', '.join(STATUSES)}")
        if row["order"]:
            try:
                float(row["order"])
            except ValueError:
                problems.append(f"{model}: order {row['order']!r} is not a number")
        for column in table.data_columns:
            cell = row[column]
            if not cell:
                continue
            try:
                if not math.isfinite(float(cell)):
                    raise ValueError
            except ValueError:
                problems.append(f"{model}: {column} = {cell!r} is not a finite number")
    return problems


# --- writing -------------------------------------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def locked(paths: Paths, who: str, timeout: float = 30.0) -> Iterator[None]:
    """Hold the writer lock (an exclusive-create file, safe on NFS and GPFS) for the ``with`` body.

    A lock whose holder died stays where it is: breaking it is the operator's call, and the error says who held it.

    Raises:
        LockTimeout: If the lock is still held after *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                try:
                    holder = paths.lock.read_text(encoding="utf-8").strip() or "(empty)"
                except OSError:
                    holder = "(unreadable)"
                raise LockTimeout(
                    f"{paths.lock} is held by {holder}. If that writer is gone (check the pid on that host), "
                    "remove the file yourself and say so in the changelog note of your next write."
                ) from None
            time.sleep(0.1)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "who": who, "since": utc_now()})
            )
        yield
    finally:
        paths.lock.unlink(missing_ok=True)


def atomic_write(path: Path, text: str) -> None:
    """Write *text* to a temporary file beside *path*, then rename it over *path*."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def render_table(table: Table) -> str:
    """The canonical text of *table*: metadata columns first, data columns in ``column_sort_key`` order, rows by model."""
    header = [*META_COLUMNS, *sorted(table.data_columns, key=column_sort_key)]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for model in sorted(table.rows, key=str.casefold):
        row = table.rows[model]
        writer.writerow([row.get(column, "") for column in header])
    return out.getvalue()


def _append_changelog(paths: Paths, lines: Sequence[Mapping[str, str | int]]) -> None:
    exists = paths.changelog.exists() and paths.changelog.stat().st_size > 0
    with paths.changelog.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, CHANGELOG_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(lines)


@dataclass
class UpsertReport:
    """What an upsert did (or, under ``dry_run``, would do)."""

    added: int = 0
    changed: int = 0
    unchanged: int = 0
    new_models: list[str] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)
    changes: list[tuple[str, str, str, str]] = field(default_factory=list)  # model, column, old, new
    wrote: bool = False

    def __str__(self) -> str:
        return (
            f"{self.added} cells added, {self.changed} changed, {self.unchanged} already identical; "
            f"{len(self.new_models)} new models, {len(self.new_columns)} new columns; "
            + ("written" if self.wrote else "nothing written")
        )


def upsert(
    cells: Iterable[tuple[str, str, float | int | str]],
    *,
    who: str,
    note: str,
    meta: Mapping[str, Mapping[str, str]] | None = None,
    overwrite: bool = False,
    reason: str = "",
    master: str | os.PathLike | None = None,
    lock_timeout: float = 30.0,
    dry_run: bool = False,
) -> UpsertReport:
    """Add cells (and metadata) to the master as one atomic, logged operation.

    Args:
        cells: ``(model, column, value)`` triples. A blank value is not a cell: leave it out.
        who: Who is writing (a session or person, as the changelog should show it); required.
        note: Where the numbers come from: the run, the eval logs or the report file, the table and page.
        meta: ``{model: {meta column: text}}``. A new model needs ``NEW_ROW_REQUIRES``; for an existing model blank
            metadata is filled in and a different value counts as a change.
        overwrite: Allow cells and metadata that hold a different value to be replaced; needs *reason*.
        reason: Why the old values were wrong, for the changelog.
        master: Path of the master (default ``resolve_paths``).
        lock_timeout: Seconds to wait for the lock.
        dry_run: Run every check and report, write nothing.

    Raises:
        OverwriteRefused: If cells would change and *overwrite* is false (nothing is written).
        SchemaError, MasterError, LockTimeout: On an invalid request, an invalid master, or a held lock.
    """
    if not who.strip():
        raise MasterError("who is required: the changelog must say who wrote")
    if not note.strip():
        raise MasterError("note is required: say where the numbers come from")
    if overwrite and not reason.strip():
        raise MasterError("overwrite needs a reason (why the old values are wrong)")
    paths = resolve_paths(master)
    wanted: dict[tuple[str, str], str] = {}
    for model, column, value in cells:
        text = fmt_value(value, column)
        if wanted.setdefault((model, column), text) != text:
            raise MasterError(
                f"{model} {column}: the input holds two different values ({wanted[model, column]}, {text})"
            )
    meta = {model: {k: str(v) for k, v in fields.items()} for model, fields in (meta or {}).items()}
    for model, fields in meta.items():
        unknown = set(fields) - set(META_COLUMNS[1:])
        if unknown:
            raise SchemaError(f"{model}: unknown metadata column(s) {', '.join(sorted(unknown))}")

    with locked(paths, who, lock_timeout):
        sets = read_sets(paths.sets)
        table = read_table(paths.master, missing_ok=True)
        broken = validate_table(table, sets)
        if broken:
            raise MasterError(f"{paths.master} is invalid; fix it before writing:\n  " + "\n  ".join(broken[:10]))
        for column in {column for _, column in wanted}:
            check_column(column, sets)

        report = UpsertReport()
        per_model: dict[str, Counter] = {}
        touched: dict[str, set[str]] = {}
        diffs: dict[str, list[list[str]]] = {}
        for model in sorted({m for m, _ in wanted} | set(meta), key=str.casefold):
            if not _MODEL.match(model):
                raise SchemaError(f"model key {model!r}: use letters, digits, '.', '_' and '-' only")
            row = table.rows.get(model)
            if row is None:
                missing = [f for f in NEW_ROW_REQUIRES if not meta.get(model, {}).get(f)]
                if missing:
                    raise MasterError(f"{model} is a new model: metadata {', '.join(missing)} is required")
                row = table.rows[model] = dict.fromkeys(table.header, "")
                row["model"], row["status"] = model, "active"
                report.new_models.append(model)
            per_model[model], touched[model], diffs[model] = Counter(), set(), []
            for name, text in meta.get(model, {}).items():
                old = row.get(name, "")
                if old == text:
                    continue
                touched[model].add("meta")
                if old:
                    report.changes.append((model, name, old, text))
                    diffs[model].append([name, old, text])
                    per_model[model]["changed"] += 1
                    report.changed += 1
                else:
                    per_model[model]["added"] += 1
                    report.added += 1
                row[name] = text
        for (model, column), text in sorted(
            wanted.items(), key=lambda kv: (kv[0][0].casefold(), column_sort_key(kv[0][1]))
        ):
            if column not in table.header:
                table.header.append(column)
                report.new_columns.append(column)
                for other in table.rows.values():
                    other.setdefault(column, "")
            row = table.rows[model]
            old = row[column]
            if old == text:
                per_model[model]["unchanged"] += 1
                report.unchanged += 1
                continue
            if old:
                report.changes.append((model, column, old, text))
                diffs[model].append([column, old, text])
                per_model[model]["changed"] += 1
                report.changed += 1
            else:
                per_model[model]["added"] += 1
                report.added += 1
            touched[model].add(".".join(parse_column(column)[:3]))
            row[column] = text

        if report.changes and not overwrite:
            shown = "\n  ".join(f"{m} {c}: {o} -> {n}" for m, c, o, n in report.changes[:10])
            raise OverwriteRefused(
                f"{len(report.changes)} existing value(s) would change; nothing was written. First ones:\n  {shown}\n"
                "If the new numbers are right and the old ones wrong, rerun with --overwrite --reason '<why>'."
            )
        problems = validate_table(table, sets)
        if problems:
            raise MasterError("the result would be invalid; nothing was written:\n  " + "\n  ".join(problems[:10]))
        text = render_table(table)
        current = paths.master.read_text(encoding="utf-8") if paths.master.exists() else None
        if dry_run or text == current:
            return report
        atomic_write(paths.master, text)
        report.wrote = True
        stamp, host = utc_now(), socket.gethostname()
        _append_changelog(
            paths,
            [
                {
                    "ts_utc": stamp, "who": who, "host": host, "op": "upsert", "model": model,
                    "added": counts["added"], "changed": counts["changed"], "unchanged": counts["unchanged"],
                    "prefixes": " ".join(sorted(touched[model])), "note": note, "reason": reason,
                    "changes": json.dumps(diffs[model], ensure_ascii=False) if diffs[model] else "",
                }
                for model, counts in per_model.items()
                if counts["added"] or counts["changed"]
            ],
        )  # fmt: skip
    return report


def add_set(
    entry: Mapping[str, str], *, who: str, master: str | os.PathLike | None = None, lock_timeout: float = 30.0
) -> bool:
    """Register an evaluation set (``results_sets.csv``) under the writer lock; ``True`` if the file changed.

    Raises:
        MasterError: If ``(set, fmt)`` is already registered with a different entry, or a metric is unknown.
    """
    if not who.strip():
        raise MasterError("who is required: the changelog must say who wrote")
    missing = [column for column in SETS_COLUMNS if column not in entry]
    if missing:
        raise MasterError(f"the set entry lacks {', '.join(missing)}")
    for part in (entry["set"], entry["fmt"]):
        if not _TOKEN.match(part):
            raise SchemaError(f"{part!r}: a set or fmt is lowercase letters, digits and single hyphens")
    unknown = [m for m in entry["metrics"].split() if m not in METRICS]
    if unknown:
        raise SchemaError(f"unknown metric(s) {', '.join(unknown)}; known: {', '.join(METRICS)}")
    paths = resolve_paths(master)
    with locked(paths, who, lock_timeout):
        sets = read_sets(paths.sets)
        key = (entry["set"], entry["fmt"])
        new = {column: str(entry[column]) for column in SETS_COLUMNS}
        if key in sets:
            if dict(sets[key]) == new:
                return False
            raise MasterError(
                f"{key[0]}.{key[1]} is already registered differently; edit {SETS_NAME} in a reviewed change"
            )
        sets[key] = new
        out = io.StringIO()
        writer = csv.DictWriter(out, SETS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sets[k] for k in sorted(sets))
        atomic_write(paths.sets, out.getvalue())
        _append_changelog(
            paths,
            [
                {
                    "ts_utc": utc_now(), "who": who, "host": socket.gethostname(), "op": "add-set", "model": "",
                    "added": 0, "changed": 0, "unchanged": 0, "prefixes": f"{key[0]}.{key[1]}",
                    "note": new["description"], "reason": "", "changes": "",
                }
            ],
        )  # fmt: skip
    return True


# --- merging two versions of the master -------------------------------------------------------------------------------


def merge3(base: Table, ours: Table, theirs: Table) -> tuple[Table, list[str]]:
    """Cell-level three-way merge: the merged table and the conflicts (cells both sides changed differently).

    A cell changed on one side only takes that side's value; the same change on both sides is one change. On a
    conflict the merged table keeps *ours*. Rows are never deleted, so a row missing on a side that had it in
    *base* is reported too, and the row is kept.
    """
    header = [
        *META_COLUMNS,
        *sorted({*base.data_columns, *ours.data_columns, *theirs.data_columns}, key=column_sort_key),
    ]
    conflicts: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for model in sorted({*base.rows, *ours.rows, *theirs.rows}, key=str.casefold):
        b, o, t = base.rows.get(model), ours.rows.get(model), theirs.rows.get(model)
        if b is not None and (o is None or t is None):
            conflicts.append(
                f"{model}: row deleted on one side (rows are never deleted; retire it with status=deprecated)"
            )
            kept = o if o is not None else t
            rows[model] = {column: kept.get(column, "") for column in header}
            continue
        merged = {}
        for column in header:
            bv, ov, tv = ((r or {}).get(column, "") for r in (b, o, t))
            if ov == tv or tv == bv:
                merged[column] = ov
            elif ov == bv:
                merged[column] = tv
            else:
                conflicts.append(
                    f"{model} {column}: base {bv or '(blank)'}, ours {ov or '(blank)'}, theirs {tv or '(blank)'}"
                )
                merged[column] = ov
        rows[model] = merged
    return Table(header, rows), conflicts


def _read_version(path: Path) -> Table:
    """One side of a merge; git hands over an empty file when a side has no common ancestor."""
    return Table(list(META_COLUMNS), {}) if path.stat().st_size == 0 else read_table(path)


def merge_driver(base: Path, ours: Path, theirs: Path) -> int:
    """Git merge driver (``%O %A %B``): write the merge into *ours* and return 0, or return 1 on a cell conflict."""
    merged, conflicts = merge3(_read_version(base), _read_version(ours), _read_version(theirs))
    if conflicts:
        print(
            f"results_master.csv: {len(conflicts)} conflict(s), the file is left as it is on our side:",
            file=sys.stderr,
        )
        for line in conflicts[:20]:
            print(f"  {line}", file=sys.stderr)
        print(
            "Decide which value is right and write it with master.py upsert --overwrite --reason ...", file=sys.stderr
        )
        return 1
    atomic_write(ours, render_table(merged))
    return 0


DRIVER_NAME = "melt-master"


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)  # noqa: S603, S607


def driver_warning(paths: Paths) -> str | None:
    """A warning when this clone does not merge the master cell by cell, else ``None`` (also outside a git repo)."""
    try:
        inside = _git("rev-parse", "--is-inside-work-tree", cwd=paths.master.parent)
    except OSError:
        return None
    if inside.returncode != 0:
        return None
    configured = _git("config", "--get", f"merge.{DRIVER_NAME}.driver", cwd=paths.master.parent)
    if configured.returncode == 0 and configured.stdout.strip():
        return None
    return (
        "WARNING: this clone has no cell-level merge driver for the master, so a git merge of two writers conflicts on "
        "the whole file. Run once: python master.py install-git-driver"
    )


def install_git_driver(paths: Paths) -> str:
    """Register the merge driver in this clone's git config (shared by all its worktrees); returns the command set."""
    script = Path(__file__).resolve()
    top = _git("rev-parse", "--show-toplevel", cwd=paths.master.parent)
    if top.returncode != 0:
        raise MasterError(f"{paths.master.parent} is not inside a git work tree")
    root = Path(top.stdout.strip())
    try:
        command = f"python3 {script.relative_to(root)} merge-driver %O %A %B"
    except ValueError:
        command = f"python3 {script} merge-driver %O %A %B"
    for key, value in ((f"merge.{DRIVER_NAME}.name", "cell-level merge of results_master.csv"),
                       (f"merge.{DRIVER_NAME}.driver", command)):  # fmt: skip
        result = _git("config", key, value, cwd=root)
        if result.returncode != 0:
            raise MasterError(f"git config {key} failed: {result.stderr.strip()}")
    return command


# --- reading helpers for other scripts -------------------------------------------------------------------------------


def load(master: str | os.PathLike | None = None) -> Table:
    """The validated master, for scripts that only read it (no lock needed).

    Raises:
        MasterError: If the file is missing or structurally broken. Schema problems are reported by ``validate``.
    """
    return read_table(resolve_paths(master).master)


def iter_long(table: Table) -> Iterator[dict[str, str]]:
    """One dict per filled cell: the metadata of its model plus set, fmt, metric, unit and value."""
    for row in table.rows.values():
        base = {name: row[name] for name in META_COLUMNS}
        for column in sorted(table.data_columns, key=column_sort_key):
            if row[column]:
                s, f, m, u = parse_column(column)
                yield {**base, "set": s, "fmt": f, "metric": m, "unit": u, "value": row[column]}


# --- command line ------------------------------------------------------------------------------------------------------


def _print_grid(rows: Sequence[Sequence[str]], markdown: bool) -> None:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for number, row in enumerate(rows):
        cells = [str(cell).ljust(width) for cell, width in zip(row, widths, strict=True)]
        print(("| " + " | ".join(cells) + " |") if markdown else "  ".join(cells).rstrip())
        if markdown and number == 0:
            print("|" + "|".join("-" * (width + 2) for width in widths) + "|")


def _cmd_where(paths: Paths, _: argparse.Namespace) -> int:
    if warning := driver_warning(paths):
        print(warning, file=sys.stderr)
    print(f"master     {paths.master}  ({'exists' if paths.master.exists() else 'MISSING'})")
    print(f"sets       {paths.sets}")
    print(f"changelog  {paths.changelog}")
    print(f"lock       {paths.lock}  ({'HELD: ' + paths.lock.read_text().strip() if paths.lock.exists() else 'free'})")
    return 0


def _cmd_validate(paths: Paths, _: argparse.Namespace) -> int:
    if warning := driver_warning(paths):
        print(warning, file=sys.stderr)
    table = read_table(paths.master)
    problems = validate_table(table, read_sets(paths.sets))
    for line in problems:
        print(f"PROBLEM  {line}")
    print(
        f"{paths.master.name}: {len(table.rows)} models, {len(table.data_columns)} data columns, {len(problems)} problems"
    )
    return 1 if problems else 0


def _cmd_show(paths: Paths, args: argparse.Namespace) -> int:
    table = read_table(paths.master)
    models = [m for m in table.rows if not args.model or m in args.model]
    if args.model and (absent := set(args.model) - set(table.rows)):
        raise MasterError(f"no such model: {', '.join(sorted(absent))}")
    if args.column:
        columns = [c for c in table.data_columns if any(fnmatch.fnmatchcase(c, g) for g in args.column)]
        if not columns:
            raise MasterError(f"no column matches {' '.join(args.column)}")
        grid = [["model", *columns]] + [[m, *(table.rows[m][c] or "-" for c in columns)] for m in models]
    else:
        names = ("label", "kind", "family", "recipe", "status", "order")
        grid = [["model", *names, "cells"]] + [
            [
                m,
                *(table.rows[m][n] or "-" for n in names),
                str(sum(bool(table.rows[m][c]) for c in table.data_columns)),
            ]
            for m in models
        ]
    _print_grid(grid, args.markdown)
    return 0


def _cmd_columns(paths: Paths, args: argparse.Namespace) -> int:
    if args.sets:
        sets = read_sets(paths.sets)
        grid = [list(SETS_COLUMNS[:6])] + [[v[c] for c in SETS_COLUMNS[:6]] for _, v in sorted(sets.items())]
        _print_grid(grid, False)
        return 0
    table = read_table(paths.master)
    patterns = args.glob or ["*"]
    for column in table.data_columns:
        if any(fnmatch.fnmatchcase(column, g) for g in patterns):
            filled = sum(bool(row[column]) for row in table.rows.values())
            print(f"{filled:>3}/{len(table.rows)}  {column}")
    return 0


def _read_cells(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) < {"model", "column", "value"}:
            raise MasterError(f"{path}: the header must contain model,column,value")
        return [(row["model"], row["column"], row["value"]) for row in reader if row["value"].strip() != ""]


def _read_meta(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "model" not in (reader.fieldnames or ()):
            raise MasterError(f"{path}: the header must contain model")
        return {row["model"]: {k: v for k, v in row.items() if k != "model" and v not in (None, "")} for row in reader}


def _cmd_upsert(paths: Paths, args: argparse.Namespace) -> int:
    if warning := driver_warning(paths):
        print(warning, file=sys.stderr)
    if not (args.cells or args.meta):
        raise MasterError("give --cells and/or --meta: there is nothing to write")
    report = upsert(
        _read_cells(Path(args.cells)) if args.cells else [],
        who=args.who,
        note=args.note,
        meta=_read_meta(Path(args.meta)) if args.meta else None,
        overwrite=args.overwrite,
        reason=args.reason or "",
        master=paths.master,
        lock_timeout=args.lock_timeout,
        dry_run=args.dry_run,
    )
    print(("DRY RUN: " if args.dry_run else "") + str(report))
    for model, column, old, new in report.changes:
        print(f"  changed {model} {column}: {old} -> {new}")
    return 0


def _cmd_add_set(paths: Paths, args: argparse.Namespace) -> int:
    entry = {
        "set": args.set, "fmt": args.fmt, "task": args.task, "split": args.split,
        "n_per_unit": args.n_per_unit, "metrics": args.metrics, "description": args.description,
    }  # fmt: skip
    print(
        "registered"
        if add_set(entry, who=args.who, master=paths.master, lock_timeout=args.lock_timeout)
        else "already registered"
    )
    return 0


def _cmd_merge_driver(_: Paths, args: argparse.Namespace) -> int:
    return merge_driver(Path(args.base), Path(args.ours), Path(args.theirs))


def _cmd_install_git_driver(paths: Paths, _: argparse.Namespace) -> int:
    print(f"installed: merge.{DRIVER_NAME}.driver = {install_git_driver(paths)}")
    return 0


def _cmd_export_long(paths: Paths, args: argparse.Namespace) -> int:
    rows = list(iter_long(read_table(paths.master)))
    names = [*META_COLUMNS, "set", "fmt", "metric", "unit", "value"]
    out = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout  # noqa: SIM115
    try:
        writer = csv.DictWriter(out, names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--master", help=f"path of the master (default: ${ENV_MASTER}, else the file next to this script)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("where", help="print the resolved paths and the lock state").set_defaults(run=_cmd_where)
    sub.add_parser("validate", help="structure, schema and values; exit 1 on any problem").set_defaults(
        run=_cmd_validate
    )
    show = sub.add_parser("show", help="print models (and optionally columns)")
    show.add_argument("-m", "--model", nargs="+")
    show.add_argument(
        "-c", "--column", nargs="+", help="glob(s) over column names, e.g. 'fleurs24-asr-dev.repeatfirst.wer.*'"
    )
    show.add_argument("--markdown", action="store_true")
    show.set_defaults(run=_cmd_show)
    columns = sub.add_parser("columns", help="list columns with how many models fill each, or the set registry")
    columns.add_argument("glob", nargs="*")
    columns.add_argument("--sets", action="store_true", help="print the registry of evaluation sets instead")
    columns.set_defaults(run=_cmd_columns)
    write = argparse.ArgumentParser(add_help=False)
    write.add_argument("--who", required=True, help="who is writing, e.g. 'claude:self-distill' or a person's name")
    write.add_argument("--lock-timeout", type=float, default=30.0)
    up = sub.add_parser("upsert", parents=[write], help="add cells (CSV model,column,value) and/or metadata (CSV)")
    up.add_argument("--cells", help="CSV with the columns model,column,value; a blank value is skipped")
    up.add_argument("--meta", help="CSV with a model column plus metadata columns (required for a new model)")
    up.add_argument("--note", required=True, help="where the numbers come from")
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--reason")
    up.add_argument("--dry-run", action="store_true")
    up.set_defaults(run=_cmd_upsert)
    add = sub.add_parser("add-set", parents=[write], help="register an evaluation set in results_sets.csv")
    for name in ("set", "fmt", "task", "split", "n-per-unit", "metrics", "description"):
        add.add_argument(f"--{name}", required=True)
    add.set_defaults(run=_cmd_add_set)
    driver = sub.add_parser("merge-driver", help="git merge driver: %%O %%A %%B (called by git, not by hand)")
    driver.add_argument("base")
    driver.add_argument("ours")
    driver.add_argument("theirs")
    driver.set_defaults(run=_cmd_merge_driver)
    sub.add_parser("install-git-driver", help="register the merge driver in this clone's git config").set_defaults(
        run=_cmd_install_git_driver
    )
    export = sub.add_parser("export-long", help="one row per filled cell")
    export.add_argument("--out")
    export.set_defaults(run=_cmd_export_long)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(resolve_paths(args.master), args)
    except MasterError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
