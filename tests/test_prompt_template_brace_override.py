"""An unquoted braced CLI override must fail with the cause, not the symptom.

`--data.prompt_template '{audio_token}'` becomes an OmegaConf dotlist entry, and
that parser reads a bare `{...}` as YAML flow-mapping syntax. The value reaching
the collator is then `{'audio_token': None}` -- a task->template dict with no
templates -- and the run used to die at the first batch with "Task 'asr' not
found", which names the symptom rather than the cause. See issue #94.
"""

import pytest
from omegaconf import OmegaConf

from melt.training.data.audio.lhotse.helpers import (
    _normalize_prompt_template,
    resolve_custom_template,
)


class TestTheReportedFailure:
    def test_omegaconf_really_does_parse_braces_as_a_mapping(self):
        """Pin the upstream behaviour the whole issue rests on.

        If a future OmegaConf stops doing this, the guard below becomes dead
        code and this test is what says so.
        """
        cfg = OmegaConf.from_dotlist(["data.prompt_template={audio_token}"])
        assert cfg.data.prompt_template == {"audio_token": None}

    def test_the_documented_command_now_fails_with_the_cause(self):
        cfg = OmegaConf.from_dotlist(["data.prompt_template={audio_token}"])
        with pytest.raises(ValueError) as excinfo:
            _normalize_prompt_template(cfg.data.prompt_template)

        message = str(excinfo.value)
        assert "unquoted" in message
        # The message must carry the fix, not just the diagnosis.
        assert "\"'{audio_token}'\"" in message

    def test_the_documented_workaround_still_produces_a_string(self):
        """The quoted form is what the launcher uses; it must keep working."""
        cfg = OmegaConf.from_dotlist(["data.prompt_template='{audio_token}'"])
        resolved = _normalize_prompt_template(cfg.data.prompt_template)
        assert resolved == "{audio_token}"
        assert isinstance(resolved, str)
        assert resolve_custom_template(resolved, "asr") == "{audio_token}"

    def test_a_multi_key_brace_expression_is_also_caught(self):
        cfg = OmegaConf.from_dotlist(["data.prompt_template={audio_token,lang}"])
        with pytest.raises(ValueError, match="unquoted"):
            _normalize_prompt_template(cfg.data.prompt_template)


class TestHealthyValuesAreUntouched:
    """The guard must not fire on anything legitimate."""

    def test_a_plain_string_is_returned_as_is(self):
        assert _normalize_prompt_template("Transcribe: {audio_token}") == (
            "Transcribe: {audio_token}"
        )

    def test_a_real_task_mapping_is_returned_as_is(self):
        value = {"asr": "Transcribe: {audio_token}", "st": "Translate: {audio_token}"}
        assert _normalize_prompt_template(value) == value

    def test_a_list_of_single_key_mappings_still_merges(self):
        value = [{"asr": "Transcribe:"}, {"st": "Translate:"}]
        assert _normalize_prompt_template(value) == {
            "asr": "Transcribe:",
            "st": "Translate:",
        }

    def test_none_is_still_none(self):
        assert _normalize_prompt_template(None) is None

    def test_an_empty_mapping_is_not_the_brace_artifact(self):
        """`{}` carries no key to have come from a brace expression."""
        assert _normalize_prompt_template({}) == {}

    def test_a_mapping_that_only_partly_lacks_values_is_left_alone(self):
        """Mixed None is a different mistake; blaming quoting would be a guess.

        It keeps falling through to resolve_custom_template's own error.
        """
        value = {"asr": "Transcribe: {audio_token}", "st": None}
        assert _normalize_prompt_template(value) == value


class TestYamlShapes:
    def test_a_task_key_with_the_value_left_off_is_caught(self):
        """`- asr:` in YAML produces the same valueless shape."""
        cfg = OmegaConf.create({"prompt_template": [{"asr": None}, {"st": None}]})
        with pytest.raises(ValueError, match="string value"):
            _normalize_prompt_template(cfg.prompt_template)

    def test_a_well_formed_yaml_mapping_survives_the_round_trip(self):
        cfg = OmegaConf.create(
            {"prompt_template": {"asr": "Transcribe: {audio_token}"}}
        )
        resolved = _normalize_prompt_template(cfg.prompt_template)
        assert resolve_custom_template(resolved, "asr") == "Transcribe: {audio_token}"
