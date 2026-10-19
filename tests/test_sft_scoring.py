"""Tests for the SFT scoring pipeline in projects/self-distill/eval/.

``watch_sft_checkpoints.py`` decides when a checkpoint is whole and which melt-eval jobs to submit; its
promises are that a half-written checkpoint is never scored, that a job is never submitted twice
(restart, a full QOS, a launcher that fails) and that nothing is written into a training output
directory. They are tested here with a fake launcher in place of ssh and sbatch. ``sft_tables.py`` is
tested on its arithmetic (the sign test) and on a small synthetic comparison.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "self-distill" / "eval"))

import sft_tables as tables  # noqa: E402
import watch_sft_checkpoints as wsc  # noqa: E402


RUN_A = "SFT-HR5-from-goldw-bd60-ga16-2g-s1337"
RUN_B = "SFT-HR5-from-sd-azeros-bd60-ga16-2g-s1337"


def write_safetensors(path: Path, truncate: int = 0) -> None:
    """A valid one-tensor safetensors file (8 float32 values), optionally with its tail cut off."""
    header = json.dumps({"w": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}}).encode()
    data = struct.pack("<Q", len(header)) + header + bytes(32)
    path.write_bytes(data[: len(data) - truncate])


def write_checkpoint(
    directory: Path, step: int, *, state: bool = True, truncate: int = 0, age: float = 1000.0
) -> Path:
    directory.mkdir(parents=True)
    write_safetensors(directory / "model.safetensors", truncate)
    (directory / "config.json").write_text("{}")
    (directory / "generation_config.json").write_text("{}")
    if state:
        (directory / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        old = os.stat(directory / "trainer_state.json").st_mtime - age
        os.utime(directory / "trainer_state.json", (old, old))
    return directory


# --- completeness ----------------------------------------------------------------------------------------------


def test_safetensors_complete_checks_the_length_against_the_header(tmp_path):
    whole, cut = tmp_path / "whole.safetensors", tmp_path / "cut.safetensors"
    write_safetensors(whole)
    write_safetensors(cut, truncate=4)
    (tmp_path / "junk.safetensors").write_bytes(b"not a safetensors file at all")
    assert wsc.safetensors_complete(whole)
    assert not wsc.safetensors_complete(cut)
    assert not wsc.safetensors_complete(tmp_path / "junk.safetensors")
    assert not wsc.safetensors_complete(tmp_path / "absent.safetensors")


def test_sharded_weights_need_every_shard(tmp_path):
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors")
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", truncate=1)
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert not wsc.weights_complete(tmp_path)
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors")
    assert wsc.weights_complete(tmp_path)


def test_a_checkpoint_is_complete_only_after_trainer_state_json(tmp_path):
    ckpt = write_checkpoint(tmp_path / "checkpoint-718", 718, state=False)
    assert wsc.checkpoint_complete(ckpt, 718, settle=60) == (False, "no trainer_state.json yet")
    (ckpt / "trainer_state.json").write_text('{"global_step": 71')  # being written
    assert wsc.checkpoint_complete(ckpt, 718, settle=0)[0] is False
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 1436}))
    assert "another step" in wsc.checkpoint_complete(ckpt, 718, settle=0)[1]
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 718}))
    assert wsc.checkpoint_complete(ckpt, 718, settle=0) == (True, "")


def test_a_checkpoint_waits_for_truncated_weights_missing_configs_and_the_settle_time(tmp_path):
    cut = write_checkpoint(tmp_path / "cut" / "checkpoint-718", 718, truncate=5)
    assert wsc.checkpoint_complete(cut, 718, settle=0) == (False, "weights incomplete")
    bare = write_checkpoint(tmp_path / "bare" / "checkpoint-718", 718)
    (bare / "generation_config.json").unlink()
    assert wsc.checkpoint_complete(bare, 718, settle=0) == (False, "no generation_config.json")
    fresh = write_checkpoint(tmp_path / "fresh" / "checkpoint-718", 718, age=10.0)
    ok, why = wsc.checkpoint_complete(fresh, 718, settle=120)
    assert not ok and why.startswith("settling")
    assert wsc.checkpoint_complete(tmp_path / "absent" / "checkpoint-718", 718, settle=0) == (False, "absent")


def test_a_pulled_checkpoint_is_complete_with_the_transfer_marker(tmp_path):
    ckpt = write_checkpoint(tmp_path / "checkpoint-718", 718, state=False)
    marker = ckpt / ".transfer_complete"
    marker.write_text("")
    old = marker.stat().st_mtime - 1000
    os.utime(marker, (old, old))
    assert wsc.checkpoint_complete(ckpt, 718, settle=60) == (True, "")


def test_the_final_model_needs_the_last_file_of_the_final_save(tmp_path):
    run = tmp_path / RUN_A
    run.mkdir()
    write_safetensors(run / "model.safetensors")
    for name in ("config.json", "tokenizer.json", "processor_config.json"):
        (run / name).write_text("{}")
    assert wsc.final_complete(run, settle=0) == (False, "no training_config.yaml")
    (run / "training_config.yaml").write_text("a: 1\n")
    assert wsc.final_complete(run, settle=0) == (True, "")


# --- names and the plan --------------------------------------------------------------------------------------


def test_the_warm_start_follows_the_run_name(tmp_path):
    assert wsc.warm_start_of(RUN_A, tmp_path) == tmp_path / wsc.WARM_STARTS["goldw"]
    assert (
        wsc.warm_start_of("SFT-EU24-from-sd-azeros-bd60-ga4-8g-s1337", tmp_path)
        == tmp_path / wsc.WARM_STARTS["sd-azeros"]
    )
    with pytest.raises(ValueError):
        wsc.warm_start_of("SFT-HR5-from-nowhere-s1", tmp_path)


def test_targets_are_the_scheduled_steps_then_the_final_model(tmp_path):
    targets = wsc.targets_for(RUN_A, tmp_path, wsc.DEFAULT_STEPS)
    assert [t.label for t in targets] == [
        "ckpt718",
        "ckpt1436",
        "ckpt2154",
        "ckpt2872",
        "ckpt4308",
        "ckpt5744",
        "ckpt8616",
        "final",
    ]
    assert targets[0].sets == wsc.SETS
    assert targets[-1].sets == wsc.SETS + ("sd-dev-per-source",)  # in-domain ASR only for the final model
    assert targets[0].comment("fleurs24-asr-dev") == f"sft-score:{RUN_A}:ckpt718:fleurs24-asr-dev"


def test_parse_submitted_reads_the_launchers_lines():
    out = "[launch] x\nSUBMITTED sft-score:r:ckpt718:fleurs24-asr-dev 336200\nQOS_FULL gpu-short 3/3\n"
    assert wsc.parse_submitted(out) == {"sft-score:r:ckpt718:fleurs24-asr-dev": 336200}


def test_a_load_report_or_traceback_in_a_jobs_stdout_is_flagged(tmp_path):
    (tmp_path / "run__ckpt718__fleurs24-asr-dev.100.out").write_text(
        "loading...\n[transformers] MELTForCausalLM LOAD REPORT from: x\n"
    )
    (tmp_path / "run__ckpt718__fleurs24-st-xen-dev.101.out").write_text("fine\n")
    (tmp_path / "run__ckpt718__fleurs24-st-enx-dev.102.out").write_text(
        "Traceback (most recent call last):\n  File x\n"
    )
    assert len(wsc.scan_stdout(tmp_path, 100)) == 1 and "LOAD REPORT" in wsc.scan_stdout(tmp_path, 100)[0]
    assert wsc.scan_stdout(tmp_path, 101) == []
    assert "Traceback" in wsc.scan_stdout(tmp_path, 102)[0]


# --- the watcher end to end, with a fake launcher -----------------------------------------------------------------

FAKE_LAUNCHER = """#!/bin/bash
# model processor tag sets...; FAKE_STATE_DIR holds a job counter and a call log
set -u
MODEL="$1"; PROCESSOR="$2"; TAG="$3"; shift 3
echo "$TAG $*" >> "$FAKE_STATE_DIR/calls"
[[ -n "${FAKE_FAIL:-}" ]] && { echo "launcher failed" >&2; exit "$FAKE_FAIL"; }
for set in "$@"; do
    n=$(cat "$FAKE_STATE_DIR/jobs" 2>/dev/null || echo 1000)
    if [[ -n "${FAKE_FULL_AFTER:-}" && $n -ge $((1000 + FAKE_FULL_AFTER)) && ! -f "$FAKE_STATE_DIR/lifted" ]]; then
        echo "QOS_FULL gpu-short 3/3"; exit 3
    fi
    echo $((n + 1)) > "$FAKE_STATE_DIR/jobs"
    echo "SUBMITTED sft-score:$TAG:$set $((n + 1))"
done
"""


@pytest.fixture
def sim(tmp_path, monkeypatch):
    """An outputs root with the two warm starts, two runs with checkpoint-718, and a fake launcher."""
    outputs = tmp_path / "outputs-sft"
    for name in wsc.WARM_STARTS.values():
        (outputs / name).mkdir(parents=True)
        (outputs / name / "tokenizer.json").write_text("{}")
    for run in (RUN_A, RUN_B):
        write_checkpoint(outputs / run / "checkpoint-718", 718)
    launcher = tmp_path / "fake_launcher.sh"
    launcher.write_text(FAKE_LAUNCHER)
    fmt = tmp_path / "format-sft-random.yaml"
    fmt.write_text("data: {}\n")
    state_dir = tmp_path / "fake"
    state_dir.mkdir()
    monkeypatch.setenv("FAKE_STATE_DIR", str(state_dir))
    for var in ("FAKE_FAIL", "FAKE_FULL_AFTER"):
        monkeypatch.delenv(var, raising=False)

    def run_once(*extra: str) -> int:
        args = [
            "--outputs",
            str(outputs),
            "--launcher",
            str(launcher),
            "--format-file",
            str(fmt),
            "--ssh-host",
            "",
            "--settle",
            "0",
            "--once",
        ]
        if "--runs" not in extra and "--runs-file" not in extra:
            args += ["--runs", RUN_A, RUN_B]
        return wsc.main([*args, *extra])

    return {
        "outputs": outputs,
        "state": outputs / "scoring" / "state.json",
        "calls": state_dir / "calls",
        "run_once": run_once,
        "dir": state_dir,
    }


def read_state(sim) -> dict:
    return json.loads(sim["state"].read_text())["jobs"]


def test_a_complete_checkpoint_is_submitted_once_per_set_and_never_again(sim):
    assert sim["run_once"]() == 0
    jobs = read_state(sim)
    assert len(jobs) == 6  # two runs x three sets; steps 1436 and later do not exist yet
    assert {k.split("|")[1] for k in jobs} == {"ckpt718"}
    assert all(entry["job"] for entry in jobs.values())
    calls_after_first = sim["calls"].read_text().splitlines()
    assert len(calls_after_first) == 2  # one launcher call per run

    assert sim["run_once"]() == 0  # a restart: the state file is read back
    assert sim["calls"].read_text().splitlines() == calls_after_first
    assert len(read_state(sim)) == 6


def test_a_new_checkpoint_is_picked_up_without_touching_the_old_ones(sim):
    sim["run_once"]()
    first = dict(read_state(sim))
    write_checkpoint(sim["outputs"] / RUN_A / "checkpoint-1436", 1436)
    sim["run_once"]()
    jobs = read_state(sim)
    assert len(jobs) == 9
    assert all(jobs[k] == first[k] for k in first)  # job ids of the first six unchanged
    assert len(sim["calls"].read_text().splitlines()) == 3


def test_a_half_written_checkpoint_is_not_scored(sim):
    write_checkpoint(sim["outputs"] / RUN_A / "checkpoint-1436", 1436, state=False)  # no trainer_state.json yet
    write_checkpoint(sim["outputs"] / RUN_B / "checkpoint-1436", 1436, truncate=100)
    sim["run_once"]()
    assert {k.split("|")[1] for k in read_state(sim)} == {"ckpt718"}


def test_a_full_qos_defers_the_rest_and_nothing_is_submitted_twice(sim, monkeypatch):
    monkeypatch.setenv("FAKE_FULL_AFTER", "4")  # the launcher takes four jobs, then reports QOS_FULL
    sim["run_once"]()
    assert len(read_state(sim)) == 4
    (sim["dir"] / "lifted").write_text("")  # slots free up
    sim["run_once"]()
    jobs = read_state(sim)
    assert len(jobs) == 6
    ids = [e["job"] for e in jobs.values()]
    assert len(set(ids)) == 6  # six distinct jobs: the first four were not submitted again


def test_a_failing_launcher_is_retried_a_few_times_and_then_given_up_on(sim, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    for _ in range(wsc.MAX_ATTEMPTS):
        sim["run_once"]()
    jobs = read_state(sim)
    assert len(jobs) == 6 and all(e["job"] is None and "gave up" in e["note"] for e in jobs.values())
    n_calls = len(sim["calls"].read_text().splitlines())
    sim["run_once"]()
    assert len(sim["calls"].read_text().splitlines()) == n_calls  # no more attempts


def test_an_unreachable_artemis_is_never_counted_against_a_checkpoint(sim, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", str(wsc.TRANSIENT))  # what ssh exits with when the connection fails
    for _ in range(wsc.MAX_ATTEMPTS + 3):
        sim["run_once"]()
    assert not sim["state"].exists() or read_state(sim) == {}
    monkeypatch.delenv("FAKE_FAIL")
    sim["run_once"]()  # artemis is back: everything goes through
    assert len(read_state(sim)) == 6


def test_a_second_watcher_cannot_run_alongside(sim):
    import fcntl

    sim["state"].parent.mkdir(parents=True, exist_ok=True)
    with open(sim["state"].with_name("watcher.lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sim["run_once"]() == 2
    assert not sim["state"].exists()


def test_the_prompt_format_is_pinned(sim, tmp_path):
    sim["run_once"]()
    (sim["outputs"] / "scoring" / "format-sft-random.yaml").write_text("data: {changed: true}\n")
    with pytest.raises(SystemExit):
        sim["run_once"]()


def test_a_dry_run_changes_nothing_on_disk(sim):
    before = sorted(p for p in sim["outputs"].rglob("*"))
    assert wsc.main([
        "--outputs", str(sim["outputs"]), "--launcher", str(sim["dir"].parent / "fake_launcher.sh"), "--ssh-host", "",
        "--settle", "0", "--dry-run", "--runs", RUN_A, RUN_B,
    ]) == 0  # fmt: skip
    assert sorted(p for p in sim["outputs"].rglob("*")) == before


def test_the_runs_file_lists_runs_and_their_site(tmp_path):
    runs = tmp_path / "runs.txt"
    runs.write_text(f"# comment\n{RUN_A}\n\n{RUN_B}   # trailing comment\nSFT-EU24-from-goldw-bd60-ga4-8g-s1337 mn5\n")
    assert wsc.read_runs_file(runs) == [(RUN_A, ""), (RUN_B, ""), ("SFT-EU24-from-goldw-bd60-ga4-8g-s1337", "mn5")]
    runs.write_text(f"{RUN_A} slurm\n")
    with pytest.raises(ValueError, match="expected"):
        wsc.read_runs_file(runs)


def test_runs_come_from_the_runs_file_and_a_run_added_to_it_is_picked_up(sim):
    runs_file = sim["outputs"] / "scoring" / "runs.txt"
    runs_file.parent.mkdir(parents=True)
    runs_file.write_text(f"{RUN_A}\n")
    sim["run_once"]("--runs-file", str(runs_file))
    assert {k.split("|")[0] for k in read_state(sim)} == {RUN_A}
    runs_file.write_text(f"{RUN_A}\n{RUN_B}\n")  # the watcher re-reads the file at every poll
    sim["run_once"]("--runs-file", str(runs_file))
    assert {k.split("|")[0] for k in read_state(sim)} == {RUN_A, RUN_B}
    assert len(read_state(sim)) == 6


def test_a_bad_run_name_stops_a_start_but_is_only_reported_and_skipped_by_a_running_watcher(sim, capsys):
    runs_file = sim["outputs"] / "scoring" / "runs.txt"
    runs_file.parent.mkdir(parents=True)
    runs_file.write_text(f"{RUN_A}\nSFT-HR5-from-somewhere-else-s1\n")
    with pytest.raises(SystemExit, match="cannot tell the warm start"):
        sim["run_once"]("--runs-file", str(runs_file))  # a start fails loudly, before anything is submitted
    assert not sim["state"].exists()

    # a running watcher re-reads the file at every poll; a typo added meanwhile must not stop it
    args = wsc.parse_args(["--outputs", str(sim["outputs"]), "--runs-file", str(runs_file)])
    reported: set[str] = set()
    assert wsc.usable_runs(args, None, reported) == (RUN_A,)
    assert wsc.usable_runs(args, None, reported) == (RUN_A,)
    assert capsys.readouterr().out.count("IGNORING SFT-HR5-from-somewhere-else-s1") == 1  # reported once


def test_without_runs_or_a_runs_file_the_watcher_refuses_to_start(sim):
    with pytest.raises(SystemExit, match="no runs to watch"):
        sim["run_once"]("--runs-file", str(sim["outputs"] / "scoring" / "absent.txt"))


def test_nothing_is_written_into_a_run_directory(sim):
    before = {p: p.stat().st_mtime for p in (sim["outputs"] / RUN_A).rglob("*")}
    sim["run_once"]()
    assert {p: p.stat().st_mtime for p in (sim["outputs"] / RUN_A).rglob("*")} == before
    assert not (sim["outputs"] / RUN_A / "scoring").exists()


# --- the tables -------------------------------------------------------------------------------------------------


def test_the_sign_test_is_an_exact_two_sided_binomial():
    d = {f"l{i}": 1.0 for i in range(9)} | {"x": -1.0, "tie": 0.0}
    test = tables.sign_test(d, higher_is_better=True)
    assert (test["b_better"], test["a_better"], test["ties"], test["n"]) == (9, 1, 1, 10)
    assert test["p"] == pytest.approx(2 * (1 + 10) / 1024)
    lower = tables.sign_test(d, higher_is_better=False)  # WER: a positive delta means B is worse
    assert (lower["b_better"], lower["a_better"]) == (1, 9)
    assert tables.sign_test({}, True)["p"] == 1.0
    assert tables.sign_test({"a": 1.0, "b": -1.0}, True)["p"] == 1.0


def test_models_are_classified_by_run_and_step():
    root = "melt//mnt/scratch-nyx/giuseppe/melt/outputs-sft/"
    assert tables.classify_model(root + RUN_A + "/checkpoint-718") == {
        "cond": "HR5",
        "init": "goldw",
        "seed": 1337,
        "step": 718,
        "final": False,
    }
    assert tables.classify_model(root + "SFT-EU24-from-sd-azeros-bd60-ga4-8g-s1337") == {
        "cond": "EU24", "init": "sd-azeros", "seed": 1337, "step": tables.FINAL_STEP, "final": True,
    }  # fmt: skip
    assert tables.classify_model(root + tables.STEP0_MODELS["goldw"])["step"] == 0
    assert (
        tables.classify_model(root + tables.STEP0_MODELS["sd-azeros"] + "/checkpoint-5") is None
    )  # not a step-0 model
    assert tables.classify_model(root + "scoring/test/fake-run/checkpoint-100") is None


def fake_row(model: str, set_name: str, task: str, values: dict[str, float]) -> dict:
    if task == "asr":
        lang = {k: {"wer": v, "cer": v / 2} for k, v in values.items()}
    else:
        lang = {k: {"chrf": v, "bleu": v / 2, "chrf_vs_source": 0.0} for k, v in values.items()}
    return {"model": model, "set": set_name, "task": task, "status": "success", "lang": lang}


def test_the_comparison_tables_put_step_zero_first_and_mark_large_deltas(tmp_path):
    root = "melt//x/"
    langs = [*tables.SEEN, "nl", "pl", "pt"]
    rows = []
    for init, base in (("goldw", 0.0), ("sd-azeros", 10.0)):
        step0 = root + tables.STEP0_MODELS[init]
        run = root + f"SFT-HR5-from-{init}-bd60-ga16-2g-s1337/checkpoint-718"
        for model, bump in ((step0, 0.0), (run, 20.0)):
            rows.append(
                fake_row(
                    model,
                    "fleurs24-asr-dev",
                    "asr",
                    dict.fromkeys(langs, 0.10 + (0.001 if init == "sd-azeros" else 0.0)),
                )
            )
            rows.append(
                fake_row(
                    model, "fleurs24-st-xen-dev", "st", {lang: 20 + base + bump for lang in langs if lang != "en"}
                )
            )
            rows.append(
                fake_row(
                    model, "fleurs24-st-enx-dev", "st", {lang: 20 + base / 5 + bump for lang in langs if lang != "en"}
                )
            )
    summaries = tmp_path / "summaries.json"
    summaries.write_text(json.dumps(rows))
    data = tables.load(str(summaries), None)
    block = tables.build_block(data, "HR5", 1337)
    assert [e["step"] for e in block["steps"]] == [0, 718]
    step0, step718 = block["steps"]
    assert step0["summary"]["A"]["xen_hr"] == pytest.approx(20.0)
    assert step0["summary"]["B"]["xen_hr"] == pytest.approx(30.0)
    assert step718["summary"]["A"]["xen_hr"] == pytest.approx(40.0)  # A at 718 came from its own run, not from step 0
    assert step0["sign"]["xen"]["all"]["b_better"] == 7 and step0["sign"]["xen"]["all"]["a_better"] == 0
    assert step0["sign"]["asr"]["all"]["a_better"] == 8  # B's WER is 0.001 higher in every language
    text = tables.render_block(block)
    assert text.index("| 0 |") < text.index("| 718 |")
    assert "+10.0*" in text  # a 10 chrF delta is above twice the noise floor
    assert "+0.001 " in text  # ... and a 0.001 WER delta is not


def test_the_languages_with_training_data_are_read_from_the_rendered_mixes():
    hr5 = tables.training_data_languages("HR5")
    assert hr5 == {"asr": set(tables.SEEN), "xen": {"de", "fr", "es", "it"}, "enx": {"de", "fr", "es", "it"}}
    eu24 = tables.training_data_languages("EU24")
    assert len(eu24["asr"]) == 24
    assert eu24["xen"] == set(tables.training_data_languages("EU24")["asr"]) - {
        "en",
        "ga",
        "mt",
    }  # no X->en data for ga and mt
    assert eu24["enx"] == {"de", "et", "lv", "sl", "sv", "es", "fr", "pt", "nl", "it", "pl", "ro"}
    assert tables.training_data_languages("EU24", mix_dir=Path("/nonexistent")) is None


def test_eu24_gets_tables_by_training_data_and_hr5_does_not(tmp_path):
    root = "melt//x/"
    langs = list(tables.training_data_languages("EU24")["asr"])
    rows = []
    for cond in ("HR5", "EU24"):
        for init, bump in (("goldw", 0.0), ("sd-azeros", 5.0)):
            model = root + f"SFT-{cond}-from-{init}-bd60-ga4-8g-s1337/checkpoint-718"
            rows.append(
                fake_row(model, "fleurs24-st-xen-dev", "st", {lang: 30 + bump for lang in langs if lang != "en"})
            )
            rows.append(
                fake_row(
                    model,
                    "fleurs24-st-enx-dev",
                    "st",
                    {lang: 30 + bump + (10 if lang in ("nl", "pt") else 0) for lang in langs if lang != "en"},
                )
            )
    summaries = tmp_path / "summaries.json"
    summaries.write_text(json.dumps(rows))
    data = tables.load(str(summaries), None)
    hr5 = tables.render_block(tables.build_block(data, "HR5", 1337))
    eu24_block = tables.build_block(data, "EU24", 1337)
    eu24 = tables.render_block(eu24_block)
    assert "by training data" not in hr5
    assert "en->X chrF by training data in EU24: targets with data (12) and without (11)" in eu24
    assert "X->en chrF by training data in EU24: sources with data (21) and without (2)" in eu24
    step = eu24_block["steps"][0]
    assert step["summary"]["B"]["enx_with_data"] == pytest.approx(
        30 + 5 + 10 * 2 / 12
    )  # nl and pt are among the 12 with data
    assert step["summary"]["B"]["enx_without_data"] == pytest.approx(35.0)
    assert step["sign"]["enx"]["with_data"]["n"] == 12 and step["sign"]["enx"]["without_data"]["n"] == 11


# --- the launcher's sbatch commands ---------------------------------------------------------------------------------


@pytest.fixture
def launcher_tree(tmp_path):
    """The directories launch_eval_sft.sh checks for, with just enough in them for a dry run."""
    tree = {name: tmp_path / name for name in ("scoring", "training", "melt-eval", "eval-sets", "model", "processor")}
    (tree["training"] / "melt").mkdir(parents=True)
    tree["melt-eval"].mkdir()
    (tree["scoring"] / "format-sft-random.yaml").parent.mkdir()
    (tree["scoring"] / "format-sft-random.yaml").write_text("data: {}\n")
    for name in ("fleurs24-st-xen-dev", "fleurs24-st-enx-dev", "fleurs24-asr-dev", "sd-dev-per-source"):
        (tree["eval-sets"] / name).mkdir(parents=True)
        (tree["eval-sets"] / name / "frozen_set.json").write_text("{}")
    tree["model"].mkdir()
    (tree["model"] / "model.safetensors").write_bytes(b"")
    (tree["model"] / "config.json").write_text("{}")
    tree["processor"].mkdir()
    (tree["processor"] / "tokenizer.json").write_text("{}")
    (tree["processor"] / "processor_config.json").write_text("{}")
    return tree


def dry_run_launcher(tree, *sets: str, tag: str = f"{RUN_A}:ckpt718") -> list[str]:
    env = {
        **os.environ, "DRY_RUN": "1", "SCORING": str(tree["scoring"]), "TRAINING": str(tree["training"]),
        "MELT_EVAL": str(tree["melt-eval"]), "EVAL_SETS": str(tree["eval-sets"]),
    }  # fmt: skip
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "projects" / "self-distill" / "eval" / "launch_eval_sft.sh"), str(tree["model"]), str(tree["processor"]), tag, *sets],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return [line for line in proc.stdout.splitlines() if line.startswith("[DRY_RUN]")]


def test_the_launcher_loads_bf16_with_flash_attention_and_takes_the_processor_from_the_warm_start(launcher_tree):
    commands = dry_run_launcher(launcher_tree, "fleurs24-st-xen-dev", "fleurs24-asr-dev")
    assert len(commands) == 2
    for command in commands:
        assert " -M dtype=bfloat16 " in command and " -M attn_implementation=flash_attention_2" in command
        assert (
            f" -M processor={launcher_tree['processor']} " in command
            and f" -T tokenizer={launcher_tree['processor']} " in command
        )
        assert f" -T format_config={launcher_tree['scoring'] / 'format-sft-random.yaml'} " in command
        assert f"infra/runners/submit_eval.sh artemis {launcher_tree['model']} " in command
        assert f"--metadata sft_run={RUN_A} --metadata sft_label=ckpt718" in command
        assert f"MELT_JOB_TAG='sft-score:{RUN_A}:ckpt718:" in command
    assert "MELT_QOS=gpu-short" in commands[0] and "-T task_filter=st" in commands[0]
    assert "MELT_QOS=gpu-medium" in commands[1] and "-T task_filter=asr" in commands[1]


def test_only_the_en_to_x_set_skips_melt_evals_pooled_scoring(launcher_tree):
    xen, enx, asr = dry_run_launcher(launcher_tree, "fleurs24-st-xen-dev", "fleurs24-st-enx-dev", "fleurs24-asr-dev")
    assert "--no-score" not in xen and "--no-score" not in asr
    assert enx.endswith("--no-score")


def test_the_launcher_refuses_a_model_without_weights_and_a_processor_without_a_tokenizer(launcher_tree):
    (launcher_tree["model"] / "model.safetensors").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        dry_run_launcher(launcher_tree, "fleurs24-asr-dev")
    (launcher_tree["model"] / "model.safetensors").write_bytes(b"")
    (launcher_tree["processor"] / "tokenizer.json").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        dry_run_launcher(launcher_tree, "fleurs24-asr-dev")


def test_nothing_is_submitted_when_the_cluster_cannot_say_what_is_already_submitted(sim, monkeypatch):
    monkeypatch.setattr(wsc, "reconcile", lambda *args, **kwargs: None)  # squeue/sacct failed
    sim["run_once"]()
    assert not sim["calls"].exists()
    assert not sim["state"].exists() or read_state(sim) == {}


def loop_args(sim) -> wsc.argparse.Namespace:
    return wsc.parse_args([
        "--outputs", str(sim["outputs"]), "--launcher", str(sim["dir"].parent / "fake_launcher.sh"),
        "--format-file", str(sim["dir"].parent / "format-sft-random.yaml"), "--ssh-host", "", "--settle", "0",
        "--interval", "0.001", "--runs", RUN_A, RUN_B,
    ])  # fmt: skip


def test_a_poll_that_raises_is_logged_and_the_watcher_goes_on(sim, monkeypatch, capsys):
    calls = []
    stop: list[int] = []

    def flaky_poll(self):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("stale file handle")
        stop.append(1)  # the second poll works; a signal then ends the loop
        return False

    monkeypatch.setattr(wsc.Watcher, "poll", flaky_poll)
    assert wsc.poll_loop(loop_args(sim), wsc.DEFAULT_STEPS, stop, live=True) == 0
    assert len(calls) == 2
    assert "ERROR in a poll (1 in a row): OSError: stale file handle" in capsys.readouterr().out


def test_a_watcher_whose_polls_keep_failing_gives_up_with_status_1(sim, monkeypatch):
    def broken_poll(self):
        raise RuntimeError("broken")

    monkeypatch.setattr(wsc.Watcher, "poll", broken_poll)
    assert wsc.poll_loop(loop_args(sim), wsc.DEFAULT_STEPS, [], live=True) == 1


def test_an_ssh_that_times_out_is_a_transient_failure_not_an_exception(monkeypatch):
    def hang(command, **kwargs):
        raise wsc.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(wsc.subprocess, "run", hang)
    proc = wsc.ssh("artemis", "squeue", timeout=5)
    assert proc.returncode == wsc.TRANSIENT and "timed out" in proc.stderr
