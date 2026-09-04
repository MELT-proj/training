#!/usr/bin/env python3
"""Integration test: save_pretrained / from_pretrained round-trip for MELTForCausalLM.

Creates a model, perturbs adapter weights, saves to disk, reloads and
verifies that every parameter matches exactly.

Run from the training/ directory:

    python tests/integration/io/check_save_load_roundtrip.py
"""

import shutil
import sys
import tempfile

import torch

from melt.modeling import MELTConfig, MELTForCausalLM, MELTForSequenceClassification


# ---------------------------------------------------------------------------
# Small model identifiers for fast execution.
# ---------------------------------------------------------------------------
AUDIO_ENCODER = "facebook/wav2vec2-base"
TEXT_DECODER = "Qwen/Qwen3-0.6B"

_PASSED = 0
_FAILED = 0


def _report(name: str, ok: bool, detail: str = "") -> None:
    global _PASSED, _FAILED
    if ok:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        print(f"  FAIL  {name}")
        if detail:
            for line in detail.splitlines():
                print(f"        {line}")


def _make_tiny_config():
    """Return a MELTConfig with trimmed layers for speed."""
    config = MELTConfig(
        audio_encoder=AUDIO_ENCODER,
        text_decoder=TEXT_DECODER,
        adapter_config={"_type": "mlp"},
    )
    config.audio_encoder_config.num_hidden_layers = 1
    config.text_decoder_config.n_layer = 1
    # A sample count, not a frame count: wav2vec2-base is a raw-waveform encoder.
    config.audio_encoder_config.max_audio_seq_len = 16_000
    return config


# ---------------------------------------------------------------------------
# Test helpers — each returns True on pass.
# ---------------------------------------------------------------------------


def check_roundtrip_preserves_all_weights(save_dir: str) -> bool:
    """Create → perturb adapter → save → load → compare all params."""
    config = _make_tiny_config()
    model = MELTForCausalLM(config)

    with torch.no_grad():
        for param in model.audio_stack.adapter.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.save_pretrained(save_dir)
    loaded_model = MELTForCausalLM.from_pretrained(save_dir)
    loaded_state = loaded_model.state_dict()

    if set(original_state.keys()) != set(loaded_state.keys()):
        missing = set(original_state) - set(loaded_state)
        extra = set(loaded_state) - set(original_state)
        _report(
            "roundtrip_preserves_all_weights",
            False,
            f"Key mismatch.\nMissing in loaded: {missing}\nExtra in loaded: {extra}",
        )
        return False

    mismatches = []
    for key in original_state:
        if not torch.equal(original_state[key], loaded_state[key]):
            max_diff = (original_state[key].float() - loaded_state[key].float()).abs().max().item()
            mismatches.append(f"  {key}: max_diff={max_diff:.6e}")

    ok = len(mismatches) == 0
    _report("roundtrip_preserves_all_weights", ok, "\n".join(mismatches) if mismatches else "")
    return ok


def check_adapter_weights_are_not_default_after_roundtrip(save_dir: str) -> bool:
    """Confirm adapter weights survive the round-trip (not re-initialised)."""
    config = _make_tiny_config()
    model = MELTForCausalLM(config)

    pre_perturb = {}
    for name, param in model.audio_stack.adapter.named_parameters():
        pre_perturb[name] = param.clone()

    torch.manual_seed(42)
    with torch.no_grad():
        for param in model.audio_stack.adapter.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    model.save_pretrained(save_dir)
    loaded_model = MELTForCausalLM.from_pretrained(save_dir)

    for name, param in loaded_model.audio_stack.adapter.named_parameters():
        if torch.equal(param, pre_perturb[name]):
            _report(
                "adapter_weights_are_not_default_after_roundtrip",
                False,
                f"Adapter param '{name}' was re-initialised instead of retaining perturbed values.",
            )
            return False

    _report("adapter_weights_are_not_default_after_roundtrip", True)
    return True


def check_text_decoder_weights_preserved(save_dir: str) -> bool:
    """Ensure text decoder weights are identical after round-trip."""
    config = _make_tiny_config()
    model = MELTForCausalLM(config)

    original_td = {k: v.clone() for k, v in model.text_decoder.state_dict().items()}
    model.save_pretrained(save_dir)
    loaded_model = MELTForCausalLM.from_pretrained(save_dir)

    for key, orig_val in original_td.items():
        loaded_val = loaded_model.text_decoder.state_dict()[key]
        if not torch.equal(orig_val, loaded_val):
            _report(
                "text_decoder_weights_preserved",
                False,
                f"Text decoder param '{key}' differs after round-trip",
            )
            return False

    _report("text_decoder_weights_preserved", True)
    return True


def check_audio_encoder_weights_preserved(save_dir: str) -> bool:
    """Ensure audio encoder weights are identical after round-trip."""
    config = _make_tiny_config()
    model = MELTForCausalLM(config)

    original_enc = {k: v.clone() for k, v in model.audio_stack.encoder.state_dict().items()}
    model.save_pretrained(save_dir)
    loaded_model = MELTForCausalLM.from_pretrained(save_dir)

    for key, orig_val in original_enc.items():
        loaded_val = loaded_model.audio_stack.encoder.state_dict()[key]
        if not torch.equal(orig_val, loaded_val):
            _report(
                "audio_encoder_weights_preserved",
                False,
                f"Audio encoder param '{key}' differs after round-trip",
            )
            return False

    _report("audio_encoder_weights_preserved", True)
    return True


def check_load_sequence_classification_from_causal_checkpoint(save_dir: str) -> bool:
    """Load MELTForSequenceClassification from MELTForCausalLM checkpoint."""
    config = _make_tiny_config()
    causal_model = MELTForCausalLM(config)
    causal_model.save_pretrained(save_dir)

    seq_model, loading_info = MELTForSequenceClassification.from_pretrained(
        save_dir,
        output_loading_info=True,
    )

    causal_state = causal_model.state_dict()
    seq_state = seq_model.state_dict()

    shared_keys = set(causal_state.keys()).intersection(seq_state.keys())
    for key in shared_keys:
        if not torch.equal(causal_state[key], seq_state[key]):
            _report(
                "load_sequence_classification_from_causal_checkpoint",
                False,
                f"Shared param '{key}' was not restored from causal checkpoint",
            )
            return False

    seq_only_text_keys = {
        key for key in (set(seq_state.keys()) - set(causal_state.keys())) if key.startswith("text_decoder.")
    }
    if len(seq_only_text_keys) == 0:
        _report(
            "load_sequence_classification_from_causal_checkpoint",
            False,
            "Expected sequence-classification-specific text-decoder head params",
        )
        return False

    missing_keys = set(loading_info["missing_keys"])
    for key in missing_keys:
        if not (key in seq_only_text_keys or key.startswith("text_decoder.")):
            _report(
                "load_sequence_classification_from_causal_checkpoint",
                False,
                f"Unexpected missing key while loading from causal checkpoint: {key}",
            )
            return False

    seq_model_2 = MELTForSequenceClassification.from_pretrained(save_dir)
    seq_state_2 = seq_model_2.state_dict()
    if not any(not torch.equal(seq_state[key], seq_state_2[key]) for key in seq_only_text_keys):
        _report(
            "load_sequence_classification_from_causal_checkpoint",
            False,
            "Sequence-classification head appears loaded from checkpoint instead of random init",
        )
        return False

    _report("load_sequence_classification_from_causal_checkpoint", True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global _PASSED, _FAILED

    save_dir = tempfile.mkdtemp(prefix="melt_roundtrip_test_")
    print(f"Save dir: {save_dir}")
    print()

    try:
        check_roundtrip_preserves_all_weights(save_dir)
        check_adapter_weights_are_not_default_after_roundtrip(save_dir)
        check_text_decoder_weights_preserved(save_dir)
        check_audio_encoder_weights_preserved(save_dir)
        check_load_sequence_classification_from_causal_checkpoint(save_dir)
    finally:
        print(f"\nCleaning up {save_dir} ...")
        shutil.rmtree(save_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Results: {_PASSED} passed, {_FAILED} failed  ({_PASSED + _FAILED} total)")
    print(f"{'='*50}")

    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
