"""Tests for master.py: the schema, the write protocol (lock, atomic replace, no silent overwrite) and the readers.

python -m pytest projects/ablation-campaign/charts/test_master.py -q
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import master
import pytest


SETS = (
    "set,fmt,task,split,n_per_unit,metrics,description\n"
    'fleurs24-asr-dev,repeatfirst,asr,dev,100,wer cer,"test set, with a comma"\n'
    "fleurs24-st-xen-dev,repeatfirst,st,dev,100,chrfpp chrfpp-lc nocaps,xen\n"
    "selection-metric,repeatfirst,aggregate,dev,,score median,aggregates\n"
)
WER_EN = "fleurs24-asr-dev.repeatfirst.wer.en"
WER_DE = "fleurs24-asr-dev.repeatfirst.wer.de"
CER_EN = "fleurs24-asr-dev.repeatfirst.cer.en"
META = {"label": "Model A", "kind": "ours", "source": "unit test"}


@pytest.fixture
def paths(tmp_path: Path) -> master.Paths:
    (tmp_path / master.SETS_NAME).write_text(SETS, encoding="utf-8")
    return master.resolve_paths(tmp_path / master.MASTER_NAME)


def write(paths: master.Paths, cells, meta=None, **kwargs) -> master.UpsertReport:
    return master.upsert(cells, who="tester", note="unit test", master=paths.master, meta=meta, **kwargs)


def read_rows(paths: master.Paths) -> list[dict[str, str]]:
    with paths.master.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_parse_column_accepts_four_tokens_and_rejects_the_rest():
    assert master.parse_column(WER_EN) == ("fleurs24-asr-dev", "repeatfirst", "wer", "en")
    for bad in ("wer.en", "a.b.c", "a.b.c.d.e", "Fleurs.repeatfirst.wer.en", "a.b.c.d_e", "a..c.d", "a.b-.c.d"):
        with pytest.raises(master.SchemaError):
            master.parse_column(bad)


def test_column_order_is_set_fmt_metric_then_unit_rank():
    names = [
        "fleurs24-asr-dev.repeatfirst.cer.de",
        "fleurs24-asr-dev.repeatfirst.wer.de",
        "fleurs24-asr-dev.repeatfirst.wer.en",
        "fleurs24-asr-dev.repeatfirst.wer.all",
        "fleurs24-asr-dev.repeatfirst.wer.xx",
        "fleurs24-asr-dev.repeatfirst.wer.mt",
    ]
    assert sorted(names, key=master.column_sort_key) == [
        "fleurs24-asr-dev.repeatfirst.wer.all",
        "fleurs24-asr-dev.repeatfirst.wer.en",
        "fleurs24-asr-dev.repeatfirst.wer.de",
        "fleurs24-asr-dev.repeatfirst.wer.mt",
        "fleurs24-asr-dev.repeatfirst.wer.xx",
        "fleurs24-asr-dev.repeatfirst.cer.de",
    ]


def test_fmt_value_rounds_trims_and_refuses_non_numbers():
    assert master.fmt_value(7.45974) == "7.4597"
    assert master.fmt_value("15.2") == "15.2"
    assert master.fmt_value(100) == "100"
    assert master.fmt_value(-0.00001) == "0"
    assert master.fmt_value(0.0434319386, "selection-metric.repeatfirst.score.all") == "0.043432"
    assert master.fmt_value(0.0434319386, WER_EN) == "0.0434"
    for bad in (float("nan"), float("inf"), "n/a", None):
        with pytest.raises(master.MasterError):
            master.fmt_value(bad)


def test_upsert_creates_the_master_with_rows_columns_and_a_changelog_line(paths):
    report = write(paths, [("A", WER_EN, 7.5), ("A", WER_DE, 6.8)], {"A": META})
    assert report.wrote and report.added == 2 + 3 and report.new_models == ["A"] and len(report.new_columns) == 2
    rows = read_rows(paths)
    assert [r["model"] for r in rows] == ["A"]
    assert rows[0]["label"] == "Model A" and rows[0]["status"] == "active" and rows[0][WER_EN] == "7.5"
    assert master.validate_table(master.load(paths.master), master.read_sets(paths.sets)) == []
    log = list(csv.DictReader(paths.changelog.open(newline="", encoding="utf-8")))
    assert len(log) == 1 and log[0]["who"] == "tester" and log[0]["model"] == "A" and log[0]["added"] == "5"
    assert log[0]["prefixes"] == "fleurs24-asr-dev.repeatfirst.wer meta"
    assert not paths.lock.exists()


def test_the_same_write_twice_changes_nothing_and_logs_nothing(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    before = (paths.master.read_text(), paths.changelog.read_text())
    report = write(paths, [("A", WER_EN, 7.5)], {"A": META})
    assert not report.wrote and report.unchanged == 1 and report.added == 0
    assert (paths.master.read_text(), paths.changelog.read_text()) == before


def test_a_different_value_is_refused_until_overwrite_comes_with_a_reason(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    before = paths.master.read_text()
    with pytest.raises(master.OverwriteRefused, match="A fleurs24-asr-dev.repeatfirst.wer.en: 7.5 -> 8"):
        write(paths, [("A", WER_EN, 8.0), ("A", WER_DE, 6.8)])
    assert paths.master.read_text() == before  # all or nothing: the new de cell was not written either
    with pytest.raises(master.MasterError, match="reason"):
        write(paths, [("A", WER_EN, 8.0)], overwrite=True)
    report = write(paths, [("A", WER_EN, 8.0)], overwrite=True, reason="rescored with the fixed normaliser")
    assert report.changed == 1 and read_rows(paths)[0][WER_EN] == "8"
    last = list(csv.DictReader(paths.changelog.open(newline="", encoding="utf-8")))[-1]
    assert last["reason"] == "rescored with the fixed normaliser"
    assert json.loads(last["changes"]) == [[WER_EN, "7.5", "8"]]


def test_a_new_model_needs_its_metadata_and_a_valid_key(paths):
    with pytest.raises(master.MasterError, match="new model: metadata label, kind, source"):
        write(paths, [("A", WER_EN, 7.5)])
    with pytest.raises(master.MasterError, match="source"):
        write(paths, [("A", WER_EN, 7.5)], {"A": {"label": "x", "kind": "ours"}})
    with pytest.raises(master.SchemaError, match="model key"):
        write(paths, [("bad key", WER_EN, 7.5)], {"bad key": META})
    with pytest.raises(master.SchemaError, match="unknown metadata"):
        write(paths, [], {"A": {**META, "colour": "red"}})
    assert not paths.master.exists()


def test_unregistered_sets_and_unknown_metrics_are_refused(paths):
    with pytest.raises(master.SchemaError, match="not in results_sets.csv"):
        write(paths, [("A", "fleurs24-asr-test.repeatfirst.wer.en", 7.5)], {"A": META})
    with pytest.raises(master.SchemaError, match="unknown metric"):
        write(paths, [("A", "fleurs24-asr-dev.repeatfirst.cider.en", 7.5)], {"A": META})
    with pytest.raises(master.SchemaError, match="not listed for"):
        write(paths, [("A", "fleurs24-asr-dev.repeatfirst.chrfpp.en", 7.5)], {"A": META})
    assert not paths.master.exists()


def test_the_file_is_canonical_whatever_the_order_of_the_writes(paths, tmp_path):
    other = master.resolve_paths(tmp_path / "other" / master.MASTER_NAME)
    other.master.parent.mkdir()
    other.sets.write_text(SETS, encoding="utf-8")
    cells_a, cells_b = [("A", WER_DE, 6.8), ("A", CER_EN, 3.0)], [("B", WER_EN, 5.0)]
    write(paths, cells_a, {"A": META})
    write(paths, cells_b, {"B": {**META, "label": "Model B"}})
    write(other, cells_b, {"B": {**META, "label": "Model B"}})
    write(other, cells_a, {"A": META})
    assert paths.master.read_text() == other.master.read_text()
    assert paths.master.read_text().splitlines()[0].split(",")[len(master.META_COLUMNS) :] == [WER_EN, WER_DE, CER_EN]


def test_readers_refuse_ragged_rows_duplicates_and_conflict_markers(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    good = paths.master.read_text()
    lines = good.splitlines()
    for broken, message in (
        (good + "B,Model B,ours\n", "fields but the header has"),
        (good + lines[1] + "\n", "appears twice"),
        ("<<<<<<< HEAD\n" + good, "conflict marker"),
        (good.replace("model,label", "label,model", 1), "first columns must be"),
        (lines[0] + "," + WER_EN + "\n", "duplicated columns"),
    ):
        paths.master.write_text(broken, encoding="utf-8")
        with pytest.raises(master.MasterError, match=message):
            master.load(paths.master)
    paths.master.write_text(good, encoding="utf-8")
    assert master.load(paths.master).rows["A"][WER_EN] == "7.5"


def test_validate_reports_bad_cells_kinds_and_unregistered_columns(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    text = paths.master.read_text()
    text = text.replace(",ours,", ",weird,", 1).replace("7.5", "lots")
    text = text.replace(WER_EN, "fleurs24-asr-test.repeatfirst.wer.en")
    paths.master.write_text(text, encoding="utf-8")
    problems = master.validate_table(master.load(paths.master), master.read_sets(paths.sets))
    assert any("not in results_sets.csv" in p for p in problems)
    assert any("kind 'weird'" in p for p in problems)
    assert any("'lots' is not a finite number" in p for p in problems)
    assert master.main(["--master", str(paths.master), "validate"]) == 1


def test_a_broken_master_blocks_further_writes(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    paths.master.write_text(paths.master.read_text().replace("7.5", "lots"), encoding="utf-8")
    with pytest.raises(master.MasterError, match="is invalid; fix it before writing"):
        write(paths, [("A", WER_DE, 6.8)])


def test_a_held_lock_times_out_names_its_holder_and_is_never_broken(paths):
    paths.lock.write_text(json.dumps({"pid": 1, "host": "elsewhere", "who": "someone"}))
    with pytest.raises(master.LockTimeout, match="elsewhere"):
        write(paths, [("A", WER_EN, 7.5)], {"A": META}, lock_timeout=0.3)
    assert paths.lock.exists() and not paths.master.exists()


def test_the_lock_is_released_after_a_failed_write(paths):
    with pytest.raises(master.SchemaError, match="not in results_sets.csv"):  # raised while the lock is held
        write(paths, [("A", "fleurs24-asr-test.repeatfirst.wer.en", 7.5)], {"A": META})
    assert not paths.lock.exists()
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    assert not paths.lock.exists()
    assert not [p for p in paths.master.parent.iterdir() if ".tmp-" in p.name]


def test_dry_run_checks_everything_and_writes_nothing(paths):
    report = write(paths, [("A", WER_EN, 7.5)], {"A": META}, dry_run=True)
    assert not report.wrote and report.added == 4 and not paths.master.exists() and not paths.changelog.exists()
    with pytest.raises(master.SchemaError):
        write(paths, [("A", "nonsense", 7.5)], {"A": META}, dry_run=True)


def test_metadata_is_filled_when_blank_and_changed_only_with_overwrite(paths):
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    write(paths, [], {"A": {"family": "qwen3.5-2b"}})
    assert read_rows(paths)[0]["family"] == "qwen3.5-2b"
    with pytest.raises(master.OverwriteRefused, match="status"):
        write(paths, [], {"A": {"status": "deprecated"}})
    write(paths, [], {"A": {"status": "deprecated"}}, overwrite=True, reason="run discarded")
    assert read_rows(paths)[0]["status"] == "deprecated"


def test_add_set_registers_once_and_refuses_a_conflicting_redefinition(paths):
    entry = {
        "set": "fleurs24-asr-test", "fmt": "repeatfirst", "task": "asr", "split": "test", "n_per_unit": "",
        "metrics": "wer cer", "description": "the full test sets",
    }  # fmt: skip
    assert master.add_set(entry, who="tester", master=paths.master) is True
    assert master.add_set(entry, who="tester", master=paths.master) is False
    with pytest.raises(master.MasterError, match="already registered differently"):
        master.add_set({**entry, "n_per_unit": "100"}, who="tester", master=paths.master)
    with pytest.raises(master.SchemaError, match="unknown metric"):
        master.add_set({**entry, "set": "x", "metrics": "cider"}, who="tester", master=paths.master)
    write(paths, [("A", "fleurs24-asr-test.repeatfirst.wer.en", 7.5)], {"A": META})
    sets = master.read_sets(paths.sets)
    assert list(sets) == sorted(sets)


def test_export_long_gives_one_row_per_filled_cell(paths, capsys):
    write(
        paths, [("A", WER_EN, 7.5), ("A", WER_DE, 6.8), ("B", WER_EN, 5.0)], {"A": META, "B": {**META, "label": "B"}}
    )
    assert master.main(["--master", str(paths.master), "export-long"]) == 0
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 3
    assert {(r["model"], r["unit"], r["value"]) for r in rows} == {
        ("A", "en", "7.5"),
        ("A", "de", "6.8"),
        ("B", "en", "5"),
    }
    assert all(r["set"] == "fleurs24-asr-dev" and r["fmt"] == "repeatfirst" and r["metric"] == "wer" for r in rows)


def test_cli_upsert_reads_cells_and_meta_files(paths, tmp_path, capsys):
    cells, meta = tmp_path / "cells.csv", tmp_path / "meta.csv"
    cells.write_text(f"model,column,value\nA,{WER_EN},7.5\nA,{WER_DE},\n", encoding="utf-8")
    meta.write_text("model,label,kind,source\nA,Model A,ours,unit test\n", encoding="utf-8")
    argv = [
        "--master",
        str(paths.master),
        "upsert",
        "--cells",
        str(cells),
        "--meta",
        str(meta),
        "--who",
        "t",
        "--note",
        "n",
    ]
    assert master.main(argv) == 0
    assert "4 cells added" in capsys.readouterr().out  # one data cell and three metadata fields
    assert read_rows(paths)[0][WER_EN] == "7.5" and WER_DE not in read_rows(paths)[0]  # a blank value is no cell
    cells.write_text(f"model,column,value\nA,{WER_EN},9\n", encoding="utf-8")
    assert master.main(argv) == 2  # refused: a different value, no --overwrite
    retire = tmp_path / "retire.csv"
    retire.write_text("model,status\nA,deprecated\n", encoding="utf-8")
    base = ["--master", str(paths.master), "upsert", "--meta", str(retire), "--who", "t", "--note", "n"]
    assert master.main(base) == 2  # metadata alone works too, and a changed status needs --overwrite
    assert master.main([*base, "--overwrite", "--reason", "run discarded"]) == 0
    assert read_rows(paths)[0]["status"] == "deprecated"
    assert master.main(["--master", str(paths.master), "upsert", "--who", "t", "--note", "n"]) == 2  # nothing to write


def test_many_processes_writing_at_once_lose_nothing(paths, tmp_path):
    script = Path(master.__file__)
    procs = []
    for index in range(8):
        cells = tmp_path / f"cells{index}.csv"
        cells.write_text(
            f"model,column,value\nM{index},{WER_EN},{index + 1}\nM{index},{CER_EN},{index}.5\n", encoding="utf-8"
        )
        meta = tmp_path / f"meta{index}.csv"
        meta.write_text(f"model,label,kind,source\nM{index},Model {index},ours,race test\n", encoding="utf-8")
        cmd = [sys.executable, str(script), "--master", str(paths.master), "upsert", "--cells", str(cells),
               "--meta", str(meta), "--who", f"w{index}", "--note", "race", "--lock-timeout", "60"]  # fmt: skip
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outputs = [(p.wait(), p.stdout.read()) for p in procs]
    assert all(code == 0 for code, _ in outputs), outputs
    table = master.load(paths.master)
    assert sorted(table.rows) == [f"M{i}" for i in range(8)]
    assert [table.rows[f"M{i}"][WER_EN] for i in range(8)] == [str(i + 1) for i in range(8)]
    assert master.validate_table(table, master.read_sets(paths.sets)) == []
    assert len(list(csv.DictReader(paths.changelog.open(newline="", encoding="utf-8")))) == 8


# --- merging ---------------------------------------------------------------------------------------------------------


def table_of(cells_by_model: dict[str, dict[str, str]], extra_columns: tuple[str, ...] = ()) -> master.Table:
    """An in-memory master: every model gets the metadata of META and the given cells."""
    columns = sorted(
        {c for cells in cells_by_model.values() for c in cells} | set(extra_columns), key=master.column_sort_key
    )
    header = [*master.META_COLUMNS, *columns]
    rows = {}
    for model, cells in cells_by_model.items():
        row = dict.fromkeys(header, "")
        row |= {"model": model, "label": model, "kind": "ours", "source": "t", "status": "active"} | cells
        rows[model] = row
    return master.Table(header, rows)


def test_merge3_combines_changes_to_different_cells_of_the_same_row():
    base = table_of({"A": {WER_EN: "7.5"}})
    ours = table_of({"A": {WER_EN: "7.5", WER_DE: "6.8"}})
    theirs = table_of({"A": {WER_EN: "7.5", CER_EN: "3"}, "B": {WER_EN: "5"}})
    merged, conflicts = master.merge3(base, ours, theirs)
    assert conflicts == []
    assert {m: {c: r[c] for c in (WER_EN, WER_DE, CER_EN)} for m, r in merged.rows.items()} == {
        "A": {WER_EN: "7.5", WER_DE: "6.8", CER_EN: "3"},
        "B": {WER_EN: "5", WER_DE: "", CER_EN: ""},
    }
    assert master.render_table(master.merge3(base, theirs, ours)[0]) == master.render_table(merged)


def test_merge3_takes_a_change_made_on_one_side_and_accepts_the_same_change_on_both():
    base = table_of({"A": {WER_EN: "7.5", WER_DE: "6.8"}})
    ours = table_of({"A": {WER_EN: "8", WER_DE: "6.8"}})
    theirs = table_of({"A": {WER_EN: "7.5", WER_DE: "6.9"}})
    merged, conflicts = master.merge3(base, ours, theirs)
    assert conflicts == [] and (merged.rows["A"][WER_EN], merged.rows["A"][WER_DE]) == ("8", "6.9")
    both = master.merge3(base, ours, table_of({"A": {WER_EN: "8", WER_DE: "6.8"}}))
    assert both[1] == [] and both[0].rows["A"][WER_EN] == "8"


def test_merge3_reports_a_cell_both_sides_changed_differently_and_keeps_ours():
    base = table_of({"A": {WER_EN: "7.5"}})
    merged, conflicts = master.merge3(base, table_of({"A": {WER_EN: "8"}}), table_of({"A": {WER_EN: "9"}}))
    assert merged.rows["A"][WER_EN] == "8"
    assert conflicts == [f"A {WER_EN}: base 7.5, ours 8, theirs 9"]


def test_merge3_refuses_a_deleted_row_and_keeps_it():
    base = table_of({"A": {WER_EN: "7.5"}, "B": {WER_EN: "5"}})
    merged, conflicts = master.merge3(base, table_of({"A": {WER_EN: "7.5"}}), base)
    assert any("B: row deleted" in c for c in conflicts)
    assert merged.rows["B"][WER_EN] == "5" and merged.rows["B"]["label"] == "B"  # kept whole, not blanked


def test_merge_driver_writes_the_merge_into_ours_and_returns_one_on_conflict(tmp_path):
    def save(name: str, table: master.Table) -> Path:
        path = tmp_path / name
        path.write_text(master.render_table(table), encoding="utf-8")
        return path

    base = save("base", table_of({"A": {WER_EN: "7.5"}}))
    ours = save("ours", table_of({"A": {WER_EN: "7.5", WER_DE: "6.8"}}))
    theirs = save("theirs", table_of({"A": {WER_EN: "7.5"}, "B": {WER_EN: "5"}}))
    assert master.merge_driver(base, ours, theirs) == 0
    merged = master.read_table(ours)
    assert sorted(merged.rows) == ["A", "B"] and merged.rows["A"][WER_DE] == "6.8"
    # now both sides changed the same cell to different values
    ours = save("ours2", table_of({"A": {WER_EN: "8"}}))
    clash = save("clash", table_of({"A": {WER_EN: "9"}}))
    before = ours.read_text()
    assert master.merge_driver(base, ours, clash) == 1
    assert ours.read_text() == before


def test_merge_driver_treats_an_empty_ancestor_as_an_empty_table(tmp_path):
    empty = tmp_path / "base"
    empty.write_text("")
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    ours.write_text(master.render_table(table_of({"A": {WER_EN: "7.5"}})), encoding="utf-8")
    theirs.write_text(master.render_table(table_of({"B": {WER_EN: "5"}})), encoding="utf-8")
    assert master.merge_driver(empty, ours, theirs) == 0
    assert sorted(master.read_table(ours).rows) == ["A", "B"]


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository holding the master, its registry and changelog, with the cell-level merge driver wired in."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "tester")
    git(root, "config", "user.email", "tester@example.org")
    (root / master.SETS_NAME).write_text(SETS, encoding="utf-8")
    (root / ".gitattributes").write_text(
        f"{master.MASTER_NAME} merge={master.DRIVER_NAME}\n{master.CHANGELOG_NAME} merge=union\n", encoding="utf-8"
    )
    paths = master.resolve_paths(root / master.MASTER_NAME)
    assert master.driver_warning(paths) is not None  # not installed yet
    master.install_git_driver(paths)
    assert master.driver_warning(paths) is None
    git(
        root,
        "config",
        f"merge.{master.DRIVER_NAME}.driver",
        f"{sys.executable} {Path(master.__file__)} merge-driver %O %A %B",
    )
    write(paths, [("A", WER_EN, 7.5)], {"A": META})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start")
    return root


def test_two_branches_adding_different_models_and_cells_merge_without_a_conflict(repo):
    paths = master.resolve_paths(repo / master.MASTER_NAME)
    git(repo, "checkout", "-q", "-b", "one")
    write(paths, [("B", WER_EN, 5.0), ("B", CER_EN, 2.5)], {"B": {**META, "label": "Model B"}})
    git(repo, "commit", "-qam", "one adds B")
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "two")
    write(paths, [("C", WER_DE, 6.8), ("A", WER_DE, 6.1)], {"C": {**META, "label": "Model C"}})
    git(repo, "commit", "-qam", "two adds C and a cell of A")
    done = git(repo, "merge", "-q", "one", "-m", "merge", check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    table = master.load(paths.master)
    assert sorted(table.rows) == ["A", "B", "C"]
    assert (table.rows["A"][WER_EN], table.rows["A"][WER_DE], table.rows["B"][CER_EN]) == ("7.5", "6.1", "2.5")
    assert master.validate_table(table, master.read_sets(paths.sets)) == []
    log = list(csv.DictReader(paths.changelog.open(newline="", encoding="utf-8")))
    assert sorted(line["model"] for line in log) == ["A", "A", "B", "C"]  # the union merge kept every writer's line


def test_two_branches_setting_one_cell_to_different_values_stop_the_merge_loudly(repo):
    paths = master.resolve_paths(repo / master.MASTER_NAME)
    git(repo, "checkout", "-q", "-b", "one")
    write(paths, [("A", WER_EN, 8.0)], overwrite=True, reason="one")
    git(repo, "commit", "-qam", "one")
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "two")
    write(paths, [("A", WER_EN, 9.0)], overwrite=True, reason="two")
    git(repo, "commit", "-qam", "two")
    done = git(repo, "merge", "-q", "one", "-m", "merge", check=False)
    assert done.returncode != 0 and "base 7.5, ours 9, theirs 8" in done.stderr + done.stdout
    assert master.load(paths.master).rows["A"][WER_EN] == "9"  # our side stays readable, no conflict markers
    git(repo, "merge", "--abort")
