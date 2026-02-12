"""Test save_pretrained / from_pretrained round-trip for MELTForCausalLM.

Creates a model, perturbs adapter weights, saves to disk, reloads and
verifies that every parameter matches exactly.
"""

import shutil
import tempfile

import pytest
import torch

from src.modeling import MELTConfig, MELTForCausalLM


# ---------------------------------------------------------------------------
# Small model identifiers for fast CI.
# ---------------------------------------------------------------------------
AUDIO_ENCODER = "facebook/wav2vec2-base"
TEXT_DECODER = "gpt2"
SAVE_DIR = None  # populated per-test via tmp_path


def _make_tiny_config():
    """Return a MELTConfig with trimmed layers for speed."""
    config = MELTConfig(
        audio_encoder=AUDIO_ENCODER,
        text_decoder=TEXT_DECODER,
        adapter_config={"_type": "mlp"},
    )
    # Trim layers so model construction is fast
    config.audio_encoder_config.num_hidden_layers = 1
    config.text_decoder_config.n_layer = 1
    return config


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """Verify that save_pretrained → from_pretrained preserves all weights."""

    @pytest.fixture
    def save_dir(self, tmp_path):
        """Provide a temporary directory for saving checkpoints."""
        d = tmp_path / "melt_checkpoint"
        d.mkdir()
        return str(d)

    def test_roundtrip_preserves_all_weights(self, save_dir):
        """Create → perturb adapter → save → load → compare all params."""
        config = _make_tiny_config()
        model = MELTForCausalLM(config)

        # -- Perturb adapter parameters so they differ from init defaults ----
        with torch.no_grad():
            for name, param in model.audio_stack.adapter.named_parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Snapshot full state dict before saving
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # -- Save & reload --------------------------------------------------
        model.save_pretrained(save_dir)
        loaded_model = MELTForCausalLM.from_pretrained(save_dir)

        loaded_state = loaded_model.state_dict()

        # -- Compare ---------------------------------------------------------
        assert set(original_state.keys()) == set(loaded_state.keys()), (
            "Key mismatch between original and loaded state dicts.\n"
            f"  Missing in loaded: {set(original_state) - set(loaded_state)}\n"
            f"  Extra in loaded:   {set(loaded_state) - set(original_state)}"
        )

        mismatches = []
        for key in original_state:
            if not torch.equal(original_state[key], loaded_state[key]):
                max_diff = (original_state[key].float() - loaded_state[key].float()).abs().max().item()
                mismatches.append(f"  {key}: max_diff={max_diff:.6e}")

        assert len(mismatches) == 0, "Weight mismatch after save/load round-trip:\n" + "\n".join(mismatches)

    def test_adapter_weights_are_not_default_after_roundtrip(self, save_dir):
        """Confirm that adapter weights survive the round-trip with the
        exact perturbation we applied (not re-initialised to defaults)."""
        config = _make_tiny_config()
        model = MELTForCausalLM(config)

        # Snapshot adapter weights *before* perturbation
        pre_perturb = {}
        for name, param in model.audio_stack.adapter.named_parameters():
            pre_perturb[name] = param.clone()

        # Apply a deterministic perturbation
        torch.manual_seed(42)
        with torch.no_grad():
            for name, param in model.audio_stack.adapter.named_parameters():
                param.add_(torch.randn_like(param) * 0.5)

        # Save & reload
        model.save_pretrained(save_dir)
        loaded_model = MELTForCausalLM.from_pretrained(save_dir)

        # The loaded adapter weights should differ from the pre-perturbation
        # snapshot, proving they are NOT re-initialised.
        for name, param in loaded_model.audio_stack.adapter.named_parameters():
            assert not torch.equal(param, pre_perturb[name]), (
                f"Adapter param '{name}' was re-initialised to default after round-trip "
                f"instead of retaining the perturbed values."
            )

    def test_text_decoder_weights_preserved(self, save_dir):
        """Ensure text decoder weights are identical after round-trip."""
        config = _make_tiny_config()
        model = MELTForCausalLM(config)

        original_td = {k: v.clone() for k, v in model.text_decoder.state_dict().items()}

        model.save_pretrained(save_dir)
        loaded_model = MELTForCausalLM.from_pretrained(save_dir)

        for key, orig_val in original_td.items():
            loaded_val = loaded_model.text_decoder.state_dict()[key]
            assert torch.equal(orig_val, loaded_val), f"Text decoder param '{key}' differs after round-trip"

    def test_audio_encoder_weights_preserved(self, save_dir):
        """Ensure audio encoder weights are identical after round-trip."""
        config = _make_tiny_config()
        model = MELTForCausalLM(config)

        original_enc = {k: v.clone() for k, v in model.audio_stack.encoder.state_dict().items()}

        model.save_pretrained(save_dir)
        loaded_model = MELTForCausalLM.from_pretrained(save_dir)

        for key, orig_val in original_enc.items():
            loaded_val = loaded_model.audio_stack.encoder.state_dict()[key]
            assert torch.equal(orig_val, loaded_val), f"Audio encoder param '{key}' differs after round-trip"
