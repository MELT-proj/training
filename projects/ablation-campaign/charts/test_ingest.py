"""Tests for the two ingest scripts: eu24_report.json -> master, and the hand-copied published table -> master.

python -m pytest projects/ablation-campaign/charts/test_ingest.py -q
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import ingest_eu24_report
import ingest_published
import master
import pytest


HERE = Path(__file__).resolve().parent


def report(label: str = "fmt-repeatfirst", wer_en: float = 0.0677) -> dict:
    """A minimal eu24_report.json: one model with ASR, X->en, en->X and the selection metric."""
    st = {
        "log": "st.eval",
        "no_capitals_share": 0.0035,
        "cased": {"de": {"chrfpp": 50.0, "spbleu": 30.0, "bleu": 28.0, "chrf": 52.0}},
        "lower": {"de": {"chrfpp": 58.6, "spbleu": 37.5, "bleu": 33.0, "chrf": 60.3}},
    }
    return {
        "meta": {
            "label": label,
            "label_map": {},
            "generated": "2026-10-16 00:00 UTC",
            "eval_code": "abc1234",
            "sets_note": "fleurs24-asr-dev, fleurs24-st-xen-dev, fleurs24-st-enx-dev",
        },
        "models": {
            "SD24": {
                "asr": {"wer": {"en": wer_en, "de": 0.0787}, "cer": {"en": 0.03, "de": 0.04}, "log": "asr.eval"},
                "xen": st,
                "enx": st,
                "selection": {
                    "score": 0.0434319,
                    "medians": {"ID": 0.0587, "OOD-train": 0.0309, "OOD-related": 0.0492, "OOD-latin": 0.0621, "OOD-script": 0.4984},
                    "languages": {"ID": {"en": 0.0463}, "OOD-train": {"en": 0.0348}, "OOD-script": {"uk": 0.9059}},
                    "runaway": {"ID": {"samples": 3000, "runaways": 8}, "FLEURS": {"samples": 5171, "runaways": 39}},
                },
            }
        },
    }  # fmt: skip


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    shutil.copy(HERE / master.SETS_NAME, tmp_path / master.SETS_NAME)
    return tmp_path


def ingest(workdir: Path, data: dict, *extra: str) -> int:
    path = workdir / "eu24_report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    fmt = data["meta"]["label"].removeprefix("fmt-")
    argv = [str(path), "--fmt", fmt, "--who", "test", "--master", str(workdir / master.MASTER_NAME), *extra]
    return ingest_eu24_report.main(argv)


def cell(workdir: Path, model: str, column: str) -> str:
    return master.load(workdir / master.MASTER_NAME).rows[model][column]


def test_every_number_of_a_report_lands_in_the_column_that_names_it(workdir):
    assert ingest(workdir, report()) == 0
    table = master.load(workdir / master.MASTER_NAME)
    row = table.rows["SD24"]
    assert row["label"] == "SD24 (Qwen)" and row["recipe"] == "adapter-sd" and row["family"] == "qwen3.5-2b"
    expected = {
        "fleurs24-asr-dev.repeatfirst.wer.en": "6.77",  # a fraction in the report, percent in the master
        "fleurs24-asr-dev.repeatfirst.cer.de": "4",
        "fleurs24-st-xen-dev.repeatfirst.chrfpp-lc.de": "58.6",  # lowercased scoring gets the -lc suffix
        "fleurs24-st-xen-dev.repeatfirst.chrfpp.de": "50",
        "fleurs24-st-enx-dev.repeatfirst.spbleu-lc.de": "37.5",
        "fleurs24-st-xen-dev.repeatfirst.nocaps.all": "0.35",
        "selection-id.repeatfirst.cer.en": "4.63",  # the ID group lives in its own set
        "selection-fleurs26-dev200.repeatfirst.cer.uk": "90.59",
        "selection-metric.repeatfirst.score.all": "0.043432",  # the selection metric stays a fraction
        "selection-metric.repeatfirst.median.ood-script": "0.4984",
        "selection-id.repeatfirst.runaway.all": "0.2667",
        "selection-fleurs26-dev200.repeatfirst.runaway.all": "0.7542",
    }
    assert {column: row[column] for column in expected} == expected
    assert master.validate_table(table, master.read_sets(workdir / master.SETS_NAME)) == []


def test_ingesting_the_same_report_again_changes_nothing_and_a_changed_number_is_refused(workdir):
    assert ingest(workdir, report()) == 0
    before = (workdir / master.MASTER_NAME).read_text()
    assert ingest(workdir, report()) == 0
    assert (workdir / master.MASTER_NAME).read_text() == before
    assert ingest(workdir, report(wer_en=0.0700)) == 2
    assert (workdir / master.MASTER_NAME).read_text() == before


def test_the_second_prompt_format_is_added_beside_the_first(workdir):
    other = report("fmt-sft-random")
    del other["models"]["SD24"]["selection"]  # the selection sets exist for the repeat-first prompt only
    assert ingest(workdir, report()) == 0
    assert ingest(workdir, other) == 0
    assert cell(workdir, "SD24", "fleurs24-asr-dev.repeatfirst.wer.en") == "6.77"
    assert cell(workdir, "SD24", "fleurs24-asr-dev.sft-random.wer.en") == "6.77"


def test_a_report_filed_under_the_wrong_prompt_format_or_with_an_unknown_model_is_refused(workdir, capsys):
    path = workdir / "eu24_report.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    master_arg = str(workdir / master.MASTER_NAME)
    assert ingest_eu24_report.main([str(path), "--fmt", "sft-random", "--who", "t", "--master", master_arg]) == 2
    assert "not 'fmt-sft-random'" in capsys.readouterr().err
    unknown = report()
    unknown["models"]["Mystery"] = unknown["models"]["SD24"]
    assert ingest(workdir, unknown) == 2
    assert "Mystery" in capsys.readouterr().err
    assert not (workdir / master.MASTER_NAME).exists()


def test_a_report_on_a_split_that_is_not_registered_cannot_be_ingested(workdir, capsys):
    test_split = report()
    test_split["meta"]["sets_note"] = "fleurs24-asr-heldout, fleurs24-st-xen-heldout, fleurs24-st-enx-heldout"
    assert ingest(workdir, test_split) == 2
    assert "is not in results_sets.csv" in capsys.readouterr().err


# The published table, transcribed a second time from the screenshot as one list per model (the file holds it as one
# row per language): a typo in either copy fails here. Languages: en de fr it es pl uk.
APERTUS_TABLE_35 = {
    "Apertus-1.5-8B": [15.2, 16.2, 22.2, 7.6, 8.7, 43.9, 34.9],
    "Apertus-1.5-70B": [14.1, 17.8, 21.0, 7.3, 7.9, 44.7, 31.4],
    "Qwen2.5-Omni-7B": [7.4, 8.3, 10.6, 6.8, 8.0, 87.1, 57.6],
    "Qwen2-Audio-7B": [8.4, 16.8, 13.8, 12.1, 11.5, 98.6, 99.3],
    "Kimi-Audio-7B": [4.4, 23.7, 17.7, 13.1, 14.9, 65.3, 108.2],
}


def test_the_published_table_in_the_repository_matches_the_second_transcription(workdir):
    cells, meta = ingest_published.build(HERE / "published" / "apertus15-table35-fleurs-wer.csv")
    got: dict[str, dict[str, float]] = {}
    for model, column, value in cells:
        got.setdefault(model, {})[column.rsplit(".", 1)[1]] = float(value)
    langs = ["en", "de", "fr", "it", "es", "pl", "uk"]
    assert {m: [got[m][lang] for lang in langs] for m in got} == APERTUS_TABLE_35
    assert all(row["kind"] == "baseline" and row["recipe"] == "public" for row in meta.values())
    assert {c.rsplit(".", 1)[0] for _, c, _ in cells} == {"fleurs-reported.native.wer"}


def test_published_numbers_enter_the_master_once_and_are_not_averaged_or_rounded(workdir):
    master_arg = str(workdir / master.MASTER_NAME)
    table = str(HERE / "published" / "apertus15-table35-fleurs-wer.csv")
    assert ingest_published.main([table, "--who", "t", "--master", master_arg]) == 0
    assert cell(workdir, "Kimi-Audio-7B", "fleurs-reported.native.wer.uk") == "108.2"
    assert cell(workdir, "Apertus-1.5-70B", "fleurs-reported.native.wer.fr") == "21"  # printed 21.0
    before = (workdir / master.MASTER_NAME).read_text()
    assert ingest_published.main([table, "--who", "t", "--master", master_arg]) == 0
    assert (workdir / master.MASTER_NAME).read_text() == before
