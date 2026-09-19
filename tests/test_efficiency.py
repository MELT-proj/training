"""Tests for projects/ablation-campaign/efficiency.py: the accounting half.

The known answers are real MN5 jobs whose GPU-h is fixed by construction
(nodes x 4 GPUs x elapsed hours), which is how the 2x divisor error that once
produced a false budget report is kept from coming back. The static half needs
torch and the HF cache and is validated against the plan's parameter table by
running the tool, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign"))

import efficiency as eff  # noqa: E402

SACCT = """JobID|ElapsedRaw|CPUTimeRAW|NNodes|State|AllocTRES
45969297|8624|2759680|2|COMPLETED|billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2
46050285|6969|2230080|2|COMPLETED|billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2
45985946|59|18880|2|FAILED|billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2
"""


def test_gpu_hours_match_nodes_x_4_x_elapsed():
    jobs = eff.parse_sacct(SACCT)
    for job_id, nodes in (("45969297", 2), ("46050285", 2)):
        j = jobs[job_id]
        by_construction = nodes * 4 * j["elapsed_s"] / 3600
        primary, check = eff.job_gpu_hours(j)
        assert primary == pytest.approx(by_construction)
        assert check == pytest.approx(by_construction)  # divisor 40, not 20 or 80
    assert eff.job_gpu_hours(jobs["45969297"])[0] == pytest.approx(19.16, abs=0.01)
    assert eff.job_gpu_hours(jobs["46050285"])[0] == pytest.approx(15.49, abs=0.01)


def test_wrong_divisor_is_caught_by_the_cross_check():
    j = eff.parse_sacct(SACCT)["45969297"]
    primary, wrong = eff.job_gpu_hours(j, cpu_per_gpu=20)
    assert wrong == pytest.approx(2 * primary)


def _log(first, last, loop, resume=None):
    return {"first_step": first, "last_step": last, "loop_elapsed_s": loop, "resume_step": resume}


def test_resumed_arm_sums_every_job_and_splits_out_the_waste():
    # a timeout at step 7543 whose last checkpoint was 7336, then a resume
    text = SACCT.replace("46050285|6969|2230080", "46050286|1000|320000").replace(
        "45969297|8624|2759680", "46050285|500|160000"
    )
    jobs = eff.parse_sacct(text)
    a = eff.summarise_jobs(
        [jobs["46050285"], jobs["46050286"]],
        {"46050285": _log(0, 7543, 450), "46050286": _log(0, 8652, 900, resume=7336)},
    )
    assert a["n_jobs"] == 2
    assert a["gpu_h_total"] == pytest.approx((500 + 1000) * 8 / 3600)
    assert a["steps_redone"] == 7543 - 7336
    # loop rate of the resumed job counts from its resume step, not from 0
    assert a["loop_s_per_step"] == pytest.approx(900 / (8652 - 7336))
    assert a["gpu_h_clean"] == pytest.approx(a["gpu_h_total"] - a["gpu_h_startup"] - a["gpu_h_redone"])
    assert a["gpu_h_mismatch_jobs"] == []


def test_failed_job_counts_its_gpu_hours_but_no_steps():
    jobs = eff.parse_sacct(SACCT)
    a = eff.summarise_jobs(
        [jobs["45985946"], jobs["46050285"]],
        {"46050285": _log(0, 8652, 6876)},
    )
    assert a["n_failed"] == 1
    assert a["steps_redone"] == 0  # a job that died at startup is not a "previous attempt"
    assert a["gpu_h_total"] == pytest.approx((59 + 6969) * 8 / 3600)


def test_training_bar_ignores_eval_bars_and_pins_to_max_steps():
    text = "\n".join(
        ["1259/1259 [00:01<"] * 400
        + ["146/2000 [00:09<"]  # a stray bar from a job that never trained
        + ["262/8652 [04:10<", "8652/8652 [1:54:34<"]
    )
    assert eff.parse_training_bar(text, expected_total=8652)["last_step"] == 8652
    # the job that died before step 1 has only the stray bar: nothing to credit
    assert eff.parse_training_bar("146/2000 [00:09<", expected_total=8652) is None
    # without a pin the most common long bar wins, which is the eval bar here
    assert eff.parse_training_bar(text)["total"] == 1259


def test_resume_step_is_read_from_the_transformers_line():
    line = "[transformers]   Resuming training from checkpoint with epoch 2 and global step 7336"
    assert eff.parse_resume_step(line) == 7336
    assert eff.parse_resume_step("nothing here") is None
