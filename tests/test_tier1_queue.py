"""Tests for projects/self-distill/eval/tier1_queue.py: which checkpoints get selection-metric jobs, in which order, and when."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "eval"))

import tier1_queue as tq  # noqa: E402
import tier1_tables as t1  # noqa: E402
import watch_sft_checkpoints as wsc  # noqa: E402


def test_every_arm_and_step_is_planned_once():
    plan = tq.plan()
    keys = [(arm.key, step) for arm, step in plan]
    assert len(keys) == len(set(keys))
    cut = {arm.key for arm in t1.ARMS} - {"T0"}
    assert {(arm, step) for arm in cut for step in tq.READS} <= set(
        keys
    )  # the four reads of every cut arm, A and B included
    assert {step for arm, step in keys if arm == "T0"} == {718, 1436, 2154, 2872, 4308, 5744, 8616, 11499}
    assert all(
        step in tq.READS for arm, step in keys if arm != "T0"
    )  # nothing is waited on past a cut run's last step


def test_the_two_figure_panels_come_first_then_the_other_reads_then_t0s_late_steps():
    plan = [(arm.key, step) for arm, step in tq.plan()]
    first = len(tq.PRIORITY)
    assert {step for _, step in plan[:first]} == {2872}  # the step-2,872 read of every arm
    assert {step for _, step in plan[first : 2 * first]} == {718}
    assert [step for _, step in plan[-4:]] == [11499, 8616, 5744, 4308]
    assert all(arm == "T0" for arm, _ in plan[-4:])


def test_the_run_names_are_the_launchers():
    names = {arm.key: tq.run_of(arm) for arm in t1.ARMS}
    assert names["T0"] == "SFT-EU24-from-noma-bd60-ga4-8g-s1337"
    assert names["A'"] == "SFT-EU24-from-goldw-bd60-ga4-8g-s1338"
    assert names["T3"] == "SFT-EU24-from-ce24-bd60-ga4-8g-s1337"
    for key, name in names.items():  # every arm's processor comes from a directory the watcher knows
        expected = "sd-azeros" if key == "B" else "goldw"
        assert wsc.warm_start_of(name, Path("/o")) == Path("/o") / wsc.WARM_STARTS[expected]


def test_a_checkpoint_has_six_selection_jobs_labelled_by_step(tmp_path):
    arm = next(a for a in t1.ARMS if a.key == "T4")
    jobs = tq.jobs_of(arm, 718, tmp_path)
    assert [j["set"] for j in jobs] == [
        "selection-fleurs26-dev200",
        "selection-id-en",
        "selection-id-de",
        "selection-id-es",
        "selection-id-fr",
        "selection-id-it",
    ]
    assert {j["label"] for j in jobs} == {"eu24-repeatfirst-ckpt718"}
    assert {j["run"] for j in jobs} == {"SFT-EU24-from-sd24-bd60-ga4-8g-s1337"}
    assert {j["model"] for j in jobs} == {str(tmp_path / "SFT-EU24-from-sd24-bd60-ga4-8g-s1337" / "checkpoint-718")}
    assert all(j["qos"] == ["gpu-short", "gpu-medium", "gpu-long"] for j in jobs)


def test_a_checkpoint_is_ready_only_when_pulled_whole_and_given_the_processor_files(tmp_path):
    source = tmp_path / "gold"
    source.mkdir()
    for name in tq.TOKENIZER_FILES:
        (source / name).write_text(f"{name} of gold")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_text("w")
    (ckpt / "config.json").write_text("{}")
    assert not tq.landed(ckpt)  # the weights are there but the transfer is not marked complete
    (ckpt / ".transfer_complete").write_text("")
    assert tq.landed(ckpt) and not tq.ready(ckpt)  # no tokenizer yet
    tq.assemble(ckpt, source)
    assert tq.ready(ckpt)
    (ckpt / "tokenizer.json").write_text("the checkpoint's own")  # never overwritten
    tq.assemble(ckpt, source)
    assert (ckpt / "tokenizer.json").read_text() == "the checkpoint's own"


def test_a_dry_run_with_everything_assumed_landed_lists_every_job_and_submits_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tq.q, "in_flight", lambda: {"gpu-short": 0, "gpu-medium": 0, "gpu-long": 0})
    state = tmp_path / "state.jsonl"
    assert tq.main(["--state", str(state), "--dry-run", "--assume-ready", "--outputs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"{len(tq.plan())} targets, {len(tq.plan()) * len(tq.SETS)} jobs" in out
    assert out.count("DRY ") == 4 + 3 + 2  # one cycle places what the QOS caps allow (4 short, 3 medium, 2 long)
    assert not state.exists()


def test_a_dry_run_on_the_real_disk_writes_nothing_into_the_checkpoints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tq.q, "in_flight", lambda: {"gpu-short": 0, "gpu-medium": 0, "gpu-long": 0})
    gold = tmp_path / wsc.WARM_STARTS["goldw"]
    gold.mkdir()
    for name in tq.TOKENIZER_FILES:
        (gold / name).write_text(name)
    ckpt = tmp_path / "SFT-EU24-from-sd24-bd60-ga4-8g-s1337" / "checkpoint-2872"
    ckpt.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", ".transfer_complete"):
        (ckpt / name).write_text("")
    state = tmp_path / "state.jsonl"
    assert tq.main(["--state", str(state), "--dry-run", "--outputs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "eu24-repeatfirst-ckpt2872 selection-fleurs26-dev200" in out  # placed: a landed checkpoint counts as ready
    assert not any((ckpt / name).exists() for name in tq.TOKENIZER_FILES)  # but the dry run did not copy the files in
    assert not state.exists()


@pytest.mark.parametrize("pause", [True, False])
def test_a_pause_file_stops_submissions(tmp_path, monkeypatch, capsys, pause):
    monkeypatch.setattr(tq.q, "in_flight", lambda: {"gpu-short": 0, "gpu-medium": 0, "gpu-long": 0})
    if pause:
        (tmp_path / "PAUSE").write_text("")
    tq.main(["--state", str(tmp_path / "state.jsonl"), "--dry-run", "--assume-ready", "--outputs", str(tmp_path)])
    out = capsys.readouterr().out
    assert ("PAUSE file present" in out) == pause
    assert ("DRY " in out) != pause
