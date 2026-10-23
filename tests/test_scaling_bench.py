"""Tests for projects/ablation-campaign/scaling-bench/flops.py.

The arithmetic in that module is what the whole scaling report rests on, and
its outputs are quoted with three significant figures in a paper, so the
hand-computable cases are pinned exactly and the anchor point is pinned to the
value the campaign plan already committed to in prose.

No torch and no cluster access: this is pure arithmetic, which is the reason it
was factored out of the measurement scripts in the first place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "scaling-bench"))

import flops  # noqa: E402


def make_shape(**overrides) -> flops.ModelShape:
    """A shape with every component zeroed, so a test can switch on one at a time."""
    base = {
        "encoder_params": 0.0,
        "encoder_layers": 0,
        "encoder_dim": 0,
        "encoder_chunk_frames": 1500,
        "encoder_trainable": True,
        "encoder_conv_flops_per_frame": 0.0,
        "adapter_params": 0.0,
        "adapter_trainable": True,
        "decoder_body_params": 0.0,
        "decoder_head_params": 0.0,
        "decoder_full_attn_layers": 0,
        "decoder_dim": 0,
        "decoder_trainable": True,
        "gradient_checkpointing": False,
    }
    base.update(overrides)
    return flops.ModelShape(**base)


def make_window(**overrides) -> flops.Window:
    base = {
        "audio_s": 600.0,
        "enc_padded_frames": 600.0 * 50,
        "enc_valid_frames": 600.0 * 50,
        "enc_chunks": 20.0,
        "dec_positions": 30_000.0,
        "dec_sequences": 20.0,
        "sup_tokens": 500.0,
    }
    base.update(overrides)
    return flops.Window(**base)


# --------------------------------------------------------------------------
# Body multipliers
# --------------------------------------------------------------------------


def test_body_flops_are_6N_times_tokens_without_checkpointing():
    shape = make_shape(decoder_body_params=1e9, gradient_checkpointing=False)
    window = make_window(dec_positions=1000.0)
    assert flops.decoder_flops(shape, window) == pytest.approx(6 * 1e9 * 1000)


def test_body_flops_are_8N_under_activation_checkpointing():
    shape = make_shape(decoder_body_params=1e9, gradient_checkpointing=True)
    window = make_window(dec_positions=1000.0)
    assert flops.decoder_flops(shape, window) == pytest.approx(8 * 1e9 * 1000)


def test_frozen_component_costs_one_forward_pass():
    shape = make_shape(decoder_body_params=1e9, decoder_trainable=False)
    window = make_window(dec_positions=1000.0)
    assert flops.decoder_flops(shape, window) == pytest.approx(2 * 1e9 * 1000)


def test_checkpointing_does_not_inflate_a_frozen_component():
    """A frozen module has no backward, so there is no recompute to pay for."""
    frozen = make_shape(decoder_body_params=1e9, decoder_trainable=False,
                        gradient_checkpointing=True)
    window = make_window(dec_positions=1000.0)
    assert flops.decoder_flops(frozen, window) == pytest.approx(2 * 1e9 * 1000)


def test_head_is_counted_with_the_body():
    shape = make_shape(decoder_body_params=900.0, decoder_head_params=100.0)
    window = make_window(dec_positions=10.0)
    assert flops.decoder_flops(shape, window) == pytest.approx(6 * 1000 * 10)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def test_decoder_attention_is_12_T2_d_per_full_layer():
    shape = make_shape(decoder_full_attn_layers=2, decoder_dim=512)
    # One sequence of 100 tokens; mean length is exactly the position count.
    window = make_window(dec_positions=100.0, dec_sequences=1.0)
    expected = 12 * 100**2 * 512 * 2
    assert flops.decoder_flops(shape, window) == pytest.approx(expected)


def test_attention_uses_mean_sequence_length_not_total_positions():
    """Two sequences of 100 cost 2x one sequence of 100, not 1x one of 200."""
    shape = make_shape(decoder_full_attn_layers=1, decoder_dim=512)
    one = make_window(dec_positions=100.0, dec_sequences=1.0)
    two = make_window(dec_positions=200.0, dec_sequences=2.0)
    assert flops.decoder_flops(shape, two) == pytest.approx(2 * flops.decoder_flops(shape, one))


def test_linear_attention_layers_carry_no_quadratic_term():
    """18 of Qwen3.5-2B's 24 layers are Gated-DeltaNet; only full layers count."""
    with_full = make_shape(decoder_full_attn_layers=6, decoder_dim=2048,
                           decoder_body_params=1.2e9)
    without = make_shape(decoder_full_attn_layers=0, decoder_dim=2048,
                         decoder_body_params=1.2e9)
    window = make_window(dec_positions=979_200.0, dec_sequences=640.0)
    assert flops.decoder_flops(without, window) < flops.decoder_flops(with_full, window)


def test_encoder_attention_is_quadratic_within_chunks_only():
    """A 3000-frame cut is two 1500-frame chunks, not one 3000-frame window."""
    shape = make_shape(encoder_layers=1, encoder_dim=1024)
    one_chunk = make_window(enc_padded_frames=1500.0, enc_chunks=1.0)
    two_chunks = make_window(enc_padded_frames=3000.0, enc_chunks=2.0)
    assert flops.encoder_flops(shape, two_chunks) == pytest.approx(
        2 * flops.encoder_flops(shape, one_chunk)
    )


# --------------------------------------------------------------------------
# Totals and the trainable split
# --------------------------------------------------------------------------


def test_total_sums_every_component():
    shape = make_shape(
        encoder_params=10.0, encoder_trainable=True,
        adapter_params=20.0, adapter_trainable=True,
        decoder_body_params=30.0, decoder_trainable=True,
    )
    window = make_window(enc_padded_frames=100.0, enc_valid_frames=100.0,
                         dec_positions=100.0, enc_chunks=0.0, dec_sequences=0.0)
    out = flops.step_flops(shape, window)
    assert out["total"] == pytest.approx(out["encoder"] + out["adapter"] + out["decoder"])
    assert out["trainable"] == pytest.approx(out["total"])


def test_frozen_components_are_excluded_from_the_trainable_column():
    shape = make_shape(
        encoder_params=10.0, encoder_trainable=False,
        adapter_params=20.0, adapter_trainable=False,
        decoder_body_params=30.0, decoder_trainable=True,
    )
    window = make_window(enc_padded_frames=100.0, enc_valid_frames=100.0,
                         dec_positions=100.0, enc_chunks=0.0, dec_sequences=0.0)
    out = flops.step_flops(shape, window)
    assert out["trainable"] == pytest.approx(out["decoder"])
    assert out["trainable"] < out["total"]


# --------------------------------------------------------------------------
# Fill
# --------------------------------------------------------------------------


def test_encoder_fill_is_the_padded_encoder_frame_ratio():
    """Padding efficiency, not batch fill: collect.py computes batch fill."""
    window = make_window(enc_padded_frames=1000.0, enc_valid_frames=860.0)
    assert window.encoder_fill == pytest.approx(0.86)


def test_encoder_fill_of_an_empty_window_is_zero_not_a_division_error():
    assert make_window(audio_s=0.0, enc_padded_frames=0.0).encoder_fill == 0.0


# --------------------------------------------------------------------------
# Throughput and MFU
# --------------------------------------------------------------------------


def test_tflops_and_mfu_divide_by_world_size_and_peak():
    """Only the decoder is on, so the hand-computed 6N*T is the whole numerator."""
    shape = make_shape(decoder_body_params=1e9)
    window = make_window(dec_positions=1000.0, dec_sequences=0.0,
                         enc_padded_frames=0.0, enc_valid_frames=0.0,
                         enc_chunks=0.0)
    out = flops.point_metrics(shape, window, s_per_step=2.0, world_size=8,
                              peak_tflops=100.0)
    # 6e12 FLOPs in 2 s across 8 GPUs.
    assert out["tflops_per_gpu_total"] == pytest.approx(6e12 / 2.0 / 8 / 1e12)
    assert out["mfu_total"] == pytest.approx(6e12 / 2.0 / 8 / 100e12)


def test_mfu_rejects_a_non_positive_step_time():
    shape = make_shape()
    window = make_window()
    with pytest.raises(ValueError):
        flops.point_metrics(shape, window, s_per_step=0.0, world_size=8)


def test_audio_h_per_wall_h_equals_audio_s_per_wall_s():
    """Both ratios scale numerator and denominator by 3600."""
    shape = make_shape(decoder_body_params=1.0)
    out = flops.point_metrics(shape, make_window(audio_s=6000.0), s_per_step=10.0,
                              world_size=4)
    assert out["audio_h_per_wall_h"] == pytest.approx(600.0)


def test_rates_are_invariant_to_window_length():
    """A window's counters are totals while s_per_step is a mean.

    Dividing one by the other without accounting for the window length scales
    every rate by the number of steps -- a 100-step benchmark would report 100x
    the throughput, and nothing in the output would look wrong.
    """
    shape = make_shape(decoder_body_params=1e9)
    one_step = make_window(audio_s=600.0, dec_positions=1000.0,
                           enc_padded_frames=0.0, enc_valid_frames=0.0, enc_chunks=0.0,
                           steps=1.0)
    ten_steps = make_window(audio_s=6000.0, dec_positions=10_000.0,
                            enc_padded_frames=0.0, enc_valid_frames=0.0, enc_chunks=0.0,
                            steps=10.0)
    a = flops.point_metrics(shape, one_step, s_per_step=10.0, world_size=4)
    b = flops.point_metrics(shape, ten_steps, s_per_step=10.0, world_size=4)
    assert b["audio_s_per_wall_s"] == pytest.approx(a["audio_s_per_wall_s"])
    assert b["mfu_total"] == pytest.approx(a["mfu_total"])
    assert b["tflops_per_gpu_total"] == pytest.approx(a["tflops_per_gpu_total"])


# --------------------------------------------------------------------------
# Walltime
# --------------------------------------------------------------------------


def test_walltime_for_100k_hours_at_the_measured_8_node_rate():
    """The anchor from plan/06-fondue.md: 594.4 audio-s per wall-s at 8 nodes."""
    out = flops.walltime_for(audio_s_per_wall_s=594.4, gpus=32, hours=100_000)
    assert out["wall_h"] == pytest.approx(168.2, abs=0.1)
    assert out["wall_days"] == pytest.approx(7.01, abs=0.01)
    assert out["gpu_h"] == pytest.approx(5_383.0, rel=0.01)
    assert out["gpu_h_per_1k_audio_h"] == pytest.approx(53.8, abs=0.1)


def test_walltime_halves_per_node_doubling_when_the_rate_doubles():
    a = flops.walltime_for(594.4, 32, 100_000)
    b = flops.walltime_for(2 * 594.4, 64, 100_000)
    assert b["wall_h"] == pytest.approx(a["wall_h"] / 2)
    assert b["gpu_h"] == pytest.approx(a["gpu_h"])


def test_walltime_rejects_a_zero_rate():
    with pytest.raises(ValueError):
        flops.walltime_for(0.0, 32, 1000)


# --------------------------------------------------------------------------
# Nominal batch
# --------------------------------------------------------------------------


def test_nominal_audio_per_step_is_the_exact_product_when_quadratic_is_unset():
    """quadratic_duration: null makes lhotse charge a cut exactly its duration."""
    assert flops.nominal_audio_s_per_step(30, 20, 32) == pytest.approx(19_200)


# --------------------------------------------------------------------------
# The anchor the report quotes, end to end
# --------------------------------------------------------------------------


def test_anchor_point_lands_at_the_mfu_the_plan_predicts():
    """8 nodes, Qwen3.5-2B fully unfrozen, 32.3 s/step, measured 2026-09-16.

    Parameter counts are real, not assumed: counted on the meta device with
    transformers 5.16.1 -- w2v-BERT-2.0 is 580.5 M, Qwen3.5-2B is 1.882 B of
    which 0.509 B is the (tied) embedding/head matrix. The head is counted
    because it is used as a matmul on every position; the embedding *lookup*
    is not, which is why body + head is the whole parameter count.

    The plan predicts ~2% MFU here and treats it as the thing to explain, not a
    bug to hide. If this test moves, the report's headline number moved.
    """
    shape = make_shape(
        encoder_params=580.5e6, encoder_layers=24, encoder_dim=1024,
        encoder_trainable=True,
        adapter_params=6.3e6, adapter_trainable=True,
        decoder_body_params=1.373e9, decoder_head_params=0.509e9,
        decoder_full_attn_layers=6, decoder_dim=2048,
        decoder_trainable=True, gradient_checkpointing=True,
    )
    window = make_window(
        audio_s=19_200.0,
        enc_padded_frames=19_200.0 * 50,
        enc_valid_frames=19_200.0 * 50,
        enc_chunks=640.0,
        dec_positions=640.0 * 1530.0,
        dec_sequences=640.0,
        sup_tokens=640.0 * 30.0,
    )
    out = flops.point_metrics(shape, window, s_per_step=32.3, world_size=32)
    assert out["mfu_total"] == pytest.approx(0.0197, abs=0.0005)
    assert out["tflops_per_gpu_total"] == pytest.approx(19.5, abs=0.3)
    assert out["audio_s_per_wall_s"] == pytest.approx(594.4, abs=0.5)
    # A fully-unfrozen encoder is a quarter of the step's FLOPs: the column
    # that says how much of the compute buys no gradient when it is frozen.
    assert out["encoder_share"] == pytest.approx(0.25, abs=0.02)


def test_bench_overrides_pin_the_recipe_invariants():
    """A missing override here silently measures a different model.

    `model.ckpt` is the dangerous one. The base config is a stage-2 YAML whose
    checkpoint points at a stage-1 run directory; while it is set, MELTConfig --
    and so the decoder, the adapter and the vocabulary -- come from that
    checkpoint and `model.decoder.name` is ignored. A smoke run did exactly that
    before this test existed, quietly training Llama-3.2-1B instead of
    Qwen3.5-2B, with nothing in the logs looking wrong.
    """
    sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "scaling-bench"))
    import bench

    cfg = bench.load()
    overrides = bench.bench_overrides(cfg["recipes"][0])
    for required in (
        "--model.ckpt=null",
        "--model.encoder.freeze=false",
        "--model.adapter.freeze=false",
        "--model.decoder.freeze=false",
        "--run.perf_logging=true",
        "--run.memory_preallocation=false",
        "--trainer.eval_on_start=false",
        "--trainer.do_eval=false",
        '--trainer.eval_strategy="no"',
        '--trainer.save_strategy="no"',
        "--data.validation_ds.input_cfg=null",
        "--trainer.ddp_find_unused_parameters=true",
    ):
        assert required in overrides, f"bench lost the {required!r} override"


def test_bench_max_steps_override_replaces_the_recipe_value():
    sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "scaling-bench"))
    import bench

    cfg = bench.load()
    recipe = cfg["recipes"][0]
    default = [o for o in bench.bench_overrides(recipe) if o.startswith("--trainer.max_steps=")]
    smoke = [o for o in bench.bench_overrides(recipe, max_steps=40) if o.startswith("--trainer.max_steps=")]
    assert default == [f"--trainer.max_steps={recipe['max_steps']}"]
    assert smoke == ["--trainer.max_steps=40"]


# --------------------------------------------------------------------------
# collect.py: reading the node count back out of a run name
# --------------------------------------------------------------------------


def test_exp_name_parses_nodes_tag_and_repeat():
    """`-r2` and `-smoke` are both "dash then letters"; the repeat must win.

    A mis-parse here is silent -- a repeat would be read as a tag and collapse
    into the wrong point -- which is why the lookahead is pinned by a test.
    """
    sys.path.insert(0, str(REPO_ROOT / "projects" / "ablation-campaign" / "scaling-bench"))
    from collect import EXP_PATTERN

    cases = {
        "BENCH-ift-qwen-n1": ("1", None, None),
        "BENCH-ift-qwen-n8-r2": ("8", None, "2"),
        "BENCH-ift-qwen-n16-smoke": ("16", "smoke", None),
        "BENCH-ift-qwen-n32-smoke-r3": ("32", "smoke", "3"),
    }
    for name, (nodes, tag, repeat) in cases.items():
        match = EXP_PATTERN.search(name)
        assert match is not None, name
        assert match["nodes"] == nodes
        assert match["tag"] == tag
        assert match["repeat"] == repeat


# --------------------------------------------------------------------------
# The trainer side: MELTTrainer._perf_observe_batch / _perf_emit
#
# Imported lazily so this file still runs its arithmetic on a machine without
# torch -- that is the whole reason flops.py is torch-free.
# --------------------------------------------------------------------------


def _perf_stub(tmp_path, step=0):
    """A real MELTTrainer instance with only the attributes the perf path touches.

    ``__new__`` rather than ``__init__``: building a trainer means a model, a
    processor and a config, and none of them are read by the perf code. Using a
    real instance (not a stub object) keeps ``_perf_append_jsonl`` and friends
    bound, so the test exercises the code that ships.
    """
    from collections import deque
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from melt.training.trainer import MELTTrainer

    trainer = object.__new__(MELTTrainer)
    trainer._perf = True
    trainer._perf_t_last = None
    trainer._perf_step_last = step
    trainer._perf_log_s = 0.0
    trainer._perf_step_times = deque(maxlen=100)
    trainer._perf_window = {}
    trainer.state = SimpleNamespace(global_step=step)
    trainer.config = OmegaConf.create(
        {
            "model": {
                "encoder": {"max_audio_seq_len": 1500},
                "decoder": {"audio_token": "<|audio|>"},
                "adapter": {"stack_factor": 1},
            }
        }
    )
    trainer.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda _t: 99)
    )
    trainer.args = SimpleNamespace(device=torch.device("cpu"), output_dir=str(tmp_path))
    return trainer


def _batch(batch_size=2, frames=1500, valid=2900, seq_len=100, supers=120):
    import torch

    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[:, : supers // batch_size] = 7
    mask = torch.ones(batch_size, frames, dtype=torch.long)
    flat = mask.view(-1)
    flat[valid:] = 0
    return {
        "input_features": torch.zeros(batch_size, frames, 80),
        "features_attention_mask": mask,
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "labels": labels,
    }


def test_perf_window_counts_match_the_batch(tmp_path):
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    stub._perf_observe_batch(_batch(), [10.0, 20.0])

    w = stub._perf_window
    assert w["audio_s"] == pytest.approx(30.0)
    assert w["enc_padded_frames"] == pytest.approx(2 * 1500)
    assert w["enc_valid_frames"] == pytest.approx(2900.0)
    assert w["enc_chunks"] == pytest.approx(2.0)  # ceil(1500/1500) x 2 cuts
    # 200 input_ids tokens, no audio placeholders in this batch, plus the
    # 2,900 valid frames that get INSERTED into the decoder sequence.
    assert w["dec_positions"] == pytest.approx(200.0 + 2900.0)
    assert w["dec_sequences"] == pytest.approx(2.0)
    assert w["sup_tokens"] == pytest.approx(120.0)


def test_decoder_positions_include_the_inserted_audio_embeddings(tmp_path):
    """MELT *inserts* audio into the decoder sequence; input_ids alone undercounts.

    `_inject_tensor` replaces the `<|audio|>` placeholder with the adapter's
    embeddings by insertion, so the decoder runs on
    `input_ids - placeholders + audio_embeddings`. On a 50 Hz encoder that is
    about nine times the raw `input_ids.numel()`, and using the raw count
    deflates every FLOPs and MFU figure derived from it.
    """
    torch = pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    batch = _batch()
    batch["input_ids"][:, 0] = 99  # the audio placeholder the stub's tokenizer resolves to
    stub._perf_observe_batch(batch, [10.0, 20.0])

    # 2 x 100 input_ids, 2 placeholders removed, 2,900 valid audio frames inserted.
    assert stub._perf_window["dec_positions"] == pytest.approx(200.0 - 2.0 + 2900.0)


def test_decoder_positions_respect_the_adapter_stack_factor(tmp_path):
    """Stacking frames before the decoder divides the injected positions."""
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    stub.config.model.adapter.stack_factor = 8
    stub._perf_observe_batch(_batch(), [10.0, 20.0])
    assert stub._perf_window["dec_positions"] == pytest.approx(200.0 + 2900.0 / 8)


def test_perf_counters_accumulate_across_microbatches(tmp_path):
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    stub._perf_observe_batch(_batch(), [10.0, 20.0])
    stub._perf_observe_batch(_batch(), [10.0, 20.0])
    assert stub._perf_window["audio_s"] == pytest.approx(60.0)
    assert stub._perf_window["dec_sequences"] == pytest.approx(4.0)


def test_emit_divides_by_steps_and_subtracts_log_time(tmp_path, monkeypatch):
    """s_per_step is the window, not the step count times anything else."""
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    stub._perf_observe_batch(_batch(), [30.0, 30.0])

    from melt.training import trainer as trainer_module

    # A step began at t=100; log() has already cost 2 s somewhere in between.
    stub._perf_t_last = 100.0
    stub._perf_log_s = 2.0
    stub.state.global_step = 6  # 5 steps since _perf_step_last
    stub._perf_step_last = 1
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: 162.0)

    logs: dict[str, float] = {}
    stub._perf_emit(logs)

    # 162 - 100 = 62 s of wall for 5 steps; log time is *not* subtracted from
    # the interval itself but from the step-boundary detector, so emit reports
    # the raw window mean. assert the arithmetic the report quotes.
    assert logs["perf/s_per_step"] == pytest.approx(62.0 / 5)
    assert logs["perf/audio_s"] == pytest.approx(60.0)
    assert logs["perf/audio_s_per_wall_s"] == pytest.approx(60.0 / 62.0)


def test_emit_writes_one_jsonl_record_and_clears_the_window(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    stub._perf_observe_batch(_batch(), [30.0, 30.0])
    stub._perf_t_last = 0.0
    stub._perf_step_last = 0
    stub.state.global_step = 1

    import json

    from melt.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: 30.0)
    stub._perf_emit({})

    path = tmp_path / "perf.jsonl"
    assert path.exists()
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["step"] == 1
    assert record["s_per_step"] == pytest.approx(30.0)
    assert stub._perf_window == {}

    # A second emit appends rather than truncating: a job killed at the wall
    # still leaves every completed record on disk. The flow it mirrors is
    # HF's: observe_batch for step 1 (which moves the boundary to 1), then
    # optimizer.step() raising global_step to 2, then log().
    stub._perf_observe_batch(_batch(), [30.0, 30.0])
    assert stub._perf_step_last == 1
    stub.state.global_step = 2
    stub._perf_t_last = 30.0
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: 60.0)
    stub._perf_emit({})
    assert len(path.read_text().splitlines()) == 2


def test_emit_is_a_noop_before_any_step_has_boundary(tmp_path):
    """Called before _perf_observe_batch sets a start time, it must not divide."""
    pytest.importorskip("torch")
    stub = _perf_stub(tmp_path)
    logs: dict[str, float] = {}
    stub._perf_emit(logs)
    assert logs == {}


def test_whisper_counters_are_in_encoder_positions_not_mel_frames(tmp_path):
    """Whisper's input is 10 ms mel frames; its stride-2 conv halves them.

    Counting input frames would double the encoder's body FLOPs and the audio
    positions the decoder is credited with.
    """
    pytest.importorskip("torch")
    from transformers import WhisperFeatureExtractor

    stub = _perf_stub(tmp_path)
    stub.config.model.encoder.max_audio_seq_len = 3000
    stub.processor.feature_extractor = WhisperFeatureExtractor()
    stub._perf_observe_batch(_batch(frames=3000, valid=4000), [20.0, 20.0])

    w = stub._perf_window
    assert w["enc_padded_frames"] == pytest.approx(2 * 3000 / 2)
    assert w["enc_valid_frames"] == pytest.approx(4000 / 2)
    assert w["enc_chunks"] == pytest.approx(2.0)  # one 3000-frame window per cut
    assert w["dec_positions"] == pytest.approx(200.0 + 4000 / 2)


def test_moe_adapter_flops_use_the_active_experts():
    import calibrate_flops

    total, active = calibrate_flops.moe_adapter_params(1280, 2048, 1, num_experts=8, top_k=2, intermediate=1024)
    expert = 2 * 1280 * 1024 + 1024 * 2048
    fixed = 2 * 2048 + 1
    assert total == 1280 * 8 + 8 * expert + fixed
    assert active == 1280 * 8 + 2 * expert + fixed
    assert 3.9 < total / active < 4.1


def test_collect_picks_the_static_file_of_the_runs_recipe(tmp_path):
    import collect

    default = tmp_path / "flops-static.json"
    default.write_text("{}")
    (tmp_path / "flops-static-fondue.json").write_text("{}")
    assert collect.static_for("BENCH-fondue-n4", default).name == "flops-static-fondue.json"
    assert collect.static_for("BENCH-fondue-n4-r2", default).name == "flops-static-fondue.json"
    assert collect.static_for("BENCH-ift-qwen-n1", default) == default


def test_fondue_recipe_pins_the_requested_configuration():
    import bench

    cfg = bench.load_config() if hasattr(bench, "load_config") else None
    if cfg is None:
        import yaml

        cfg = yaml.safe_load((Path(bench.__file__).parent / "bench.yaml").read_text())
    recipe = next(r for r in cfg["recipes"] if r["id"] == "fondue")
    argv = " ".join(bench.bench_overrides(recipe))
    assert "--model.encoder.name=openai/whisper-large-v3" in argv
    assert "--model.encoder.attn_implementation=flash_attention_2" in argv
    assert "--model.encoder.max_audio_seq_len=3000" in argv
    assert "--model.adapter._type=moe" in argv
    assert "--model.decoder.attn_implementation=flash_attention_2" in argv
    assert "--trainer.gradient_checkpointing=false" in argv
    assert "--model.ckpt=null" in argv
    for part in ("encoder", "adapter", "decoder"):
        assert f"--model.{part}.freeze=false" in argv


def test_ledger_job_id_prefers_the_newest_row_across_both_copies(tmp_path, monkeypatch):
    import bench

    home = tmp_path / "arms.tsv"
    home.write_text("2026-10-23T22:00:00Z\tBENCH-x-n1-smoke\t100\tcmd\n")
    monkeypatch.setattr(bench, "ARMS_TSV", home)
    remote = "2026-10-23T23:00:00Z\tBENCH-x-n1-smoke\t200\tcmd\n"
    assert bench.ledger_job_id("BENCH-x-n1-smoke") == "100"
    assert bench.ledger_job_id("BENCH-x-n1-smoke", remote) == "200"
    assert bench.ledger_job_id("BENCH-unknown", remote) is None
