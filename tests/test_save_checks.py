"""Tests for post-save weight verification (issue #91).

A run that finishes under FSDP with SHARDED_STATE_DICT writes config and
tokenizer files but no weights, and reports success.  These cover the check
that catches that, and -- just as important -- that it does not cry wolf on a
correctly saved run whose weights are sharded across several safetensors files.
"""

import logging

import pytest

from melt.training.save_checks import (
    FSDP_SHARD_DIRNAME,
    find_saved_weights,
    find_sharded_checkpoint,
    format_missing_weights_message,
    verify_saved_weights,
)


def _make_run_dir(tmp_path, weight_files=(), checkpoint_steps=()):
    """Build a run directory that always has the non-weight files a run writes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_config.yaml",
        "resolved_config.json",
    ):
        (run_dir / name).write_text("{}")
    for name in weight_files:
        (run_dir / name).write_bytes(b"\0")
    for step in checkpoint_steps:
        shard_dir = run_dir / f"checkpoint-{step}" / FSDP_SHARD_DIRNAME
        shard_dir.mkdir(parents=True)
        (shard_dir / "__0_0.distcp").write_bytes(b"\0")
    return run_dir


class TestFindSavedWeights:
    def test_the_reported_failure_finds_nothing(self, tmp_path):
        """The exact directory listing from issue #91: config yes, weights no."""
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(4000, 5000, 6000, 6250))
        assert find_saved_weights(run_dir) == []

    def test_single_file_save_is_found(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, weight_files=("model.safetensors",))
        assert [p.name for p in find_saved_weights(run_dir)] == ["model.safetensors"]

    def test_sharded_save_is_found(self, tmp_path):
        """A 9.8 GB model is split past max_shard_size; that is a GOOD save.

        Checking for the literal name "model.safetensors" would report this
        healthy run as broken, which is the failure mode that matters most --
        a false alarm here would train people to ignore the message.
        """
        run_dir = _make_run_dir(
            tmp_path,
            weight_files=(
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model.safetensors.index.json",
            ),
        )
        assert len(find_saved_weights(run_dir)) == 4

    def test_legacy_bin_save_is_found(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, weight_files=("pytorch_model.bin",))
        assert [p.name for p in find_saved_weights(run_dir)] == ["pytorch_model.bin"]

    def test_weights_inside_a_checkpoint_do_not_count(self, tmp_path):
        """Sharded checkpoint weights are what we are checking for the absence of."""
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(6250,))
        (run_dir / "checkpoint-6250" / "model.safetensors").write_bytes(b"\0")
        assert find_saved_weights(run_dir) == []

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert find_saved_weights(tmp_path / "nope") == []


class TestFindShardedCheckpoint:
    def test_picks_highest_step_numerically_not_lexicographically(self, tmp_path):
        """checkpoint-6250 must beat checkpoint-900; string sort gets this wrong."""
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(900, 6250))
        assert find_sharded_checkpoint(run_dir).name == "checkpoint-6250"

    def test_ignores_checkpoints_without_shards(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(1000,))
        (run_dir / "checkpoint-2000").mkdir()
        assert find_sharded_checkpoint(run_dir).name == "checkpoint-1000"

    def test_returns_none_when_there_is_nothing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        assert find_sharded_checkpoint(run_dir) is None


class TestMessage:
    def test_names_the_checkpoint_and_the_command(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(4000, 6250))
        message = format_missing_weights_message(run_dir)

        assert "utils/merge_fsdp_weight.py" in message
        assert str(run_dir / "checkpoint-6250" / FSDP_SHARD_DIRNAME) in message
        assert str(run_dir) in message
        # The login-node cgroup kill is silent and costs a debugging session.
        assert "BATCH job" in message
        assert "#91" in message

    def test_says_so_when_there_is_nothing_to_merge(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        message = format_missing_weights_message(run_dir)
        assert "nothing to consolidate" in message
        assert "utils/merge_fsdp_weight.py" not in message


class TestVerifySavedWeights:
    def test_missing_weights_log_an_error_and_return_false(self, tmp_path, caplog):
        run_dir = _make_run_dir(tmp_path, checkpoint_steps=(6250,))
        with caplog.at_level(logging.ERROR):
            assert verify_saved_weights(run_dir) is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "NOT loadable" in errors[0].getMessage()

    def test_present_weights_log_no_error(self, tmp_path, caplog):
        run_dir = _make_run_dir(tmp_path, weight_files=("model.safetensors",))
        with caplog.at_level(logging.INFO):
            assert verify_saved_weights(run_dir) is True
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    def test_never_raises_even_with_no_directory(self, tmp_path):
        """The run completed; the check must not be what kills the job."""
        assert verify_saved_weights(tmp_path / "does-not-exist") is False
