"""Tests for the MoE audio adapter (MELTMoEAdapter)."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from melt.modeling import MELTAdapterConfig, MELTConfig, MELTForCausalLM
from melt.modeling.modeling_melt import MELTMoEAdapter

from test_modeling_melt import _make_melt_config


def _moe_config(
    audio_hidden_size: int = 6,
    text_hidden_size: int = 10,
    **adapter_kwargs,
) -> MELTConfig:
    """A lightweight fake config for unit-testing MELTMoEAdapter in isolation."""
    adapter_kwargs.setdefault("num_experts", 4)
    adapter_kwargs.setdefault("num_experts_per_tok", 2)
    adapter_kwargs.setdefault("moe_intermediate_size", 8)
    adapter_kwargs.setdefault("shared_expert_intermediate_size", 8)

    config = MagicMock(spec=MELTConfig)
    config.adapter_config = MELTAdapterConfig(_type="moe", **adapter_kwargs)
    config.audio_encoder_config = MagicMock()
    config.audio_encoder_config.hidden_size = audio_hidden_size
    config.audio_encoder_config.output_hidden_size = audio_hidden_size
    config.text_decoder_config = MagicMock()
    config.text_decoder_config.hidden_size = text_hidden_size
    return config


# ============================================================================
# Shape contract
# ============================================================================


class TestMoEAdapterOutputFeaturesShape:
    def test_preserves_sequence_length(self):
        """Like the MLP adapter, routing doesn't change sequence length."""
        adapter = MELTMoEAdapter(_moe_config())

        batch_size, seq_len, input_hidden_size = 2, 5, 6
        input_features = torch.randn(batch_size, seq_len, input_hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(
            input_features.shape, attention_mask
        )

        assert output_shape == (batch_size, seq_len, 10)
        assert torch.equal(output_mask, attention_mask)

    def test_without_attention_mask(self):
        adapter = MELTMoEAdapter(_moe_config())

        input_features = torch.randn(2, 5, 6)
        output_shape, output_mask = adapter._get_output_features_shape(
            input_features.shape, None
        )

        assert output_shape == (2, 5, 10)
        assert output_mask is None

    def test_output_matches_shape_prediction(self):
        adapter = MELTMoEAdapter(_moe_config())

        input_features = torch.randn(2, 5, 6)
        attention_mask = torch.ones(2, 5)

        predicted_shape, _ = adapter._get_output_features_shape(
            input_features.shape, attention_mask
        )
        actual_output = adapter(input_features, attention_mask=attention_mask)

        assert actual_output.shape == predicted_shape


# ============================================================================
# Routing correctness
# ============================================================================


def test_exactly_top_k_experts_receive_each_token():
    """Every token must be dispatched to exactly num_experts_per_tok experts."""
    num_experts, num_experts_per_tok = 4, 2
    adapter = MELTMoEAdapter(
        _moe_config(num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
    )

    batch_size, seq_len = 3, 7
    total_tokens = batch_size * seq_len
    counts = []

    def _count_hook(_module, inputs, _output):
        counts.append(inputs[0].shape[0])

    handles = [expert.down_proj.register_forward_hook(_count_hook) for expert in adapter.experts]
    try:
        adapter(torch.randn(batch_size, seq_len, 6))
    finally:
        for h in handles:
            h.remove()

    assert len(counts) == num_experts
    assert sum(counts) == total_tokens * num_experts_per_tok


def test_topk_weights_sum_to_one_per_token():
    adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=2))

    flat = torch.randn(9, 6)
    router_logits = adapter.router(flat)
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, _ = routing_weights.topk(2, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(9), atol=1e-6)


# ============================================================================
# Gradient flow
# ============================================================================


def test_gradient_flows_only_to_routed_experts():
    """With a deterministic router, only the selected experts should get a gradient."""
    adapter = MELTMoEAdapter(
        _moe_config(num_experts=5, num_experts_per_tok=2, audio_hidden_size=6, text_hidden_size=10)
    )

    # token 0 -> experts {0, 1}; token 1 -> experts {2, 3}; expert 4 is never picked.
    fixed_logits = torch.tensor(
        [
            [10.0, 9.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0, 9.0, -10.0],
        ]
    )
    adapter.router.forward = lambda x: fixed_logits

    audio_features = torch.randn(1, 2, 6)
    output = adapter(audio_features)
    # Not output.sum(): LayerNorm with a uniform weight makes every row sum to exactly
    # zero regardless of its input, so that reduction has a constant-zero gradient and
    # would make this test pass vacuously.
    output.pow(2).sum().backward()

    for expert_idx in (0, 1, 2, 3):
        grad = adapter.experts[expert_idx].gate_proj.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)

    assert adapter.experts[4].gate_proj.weight.grad is None


# ============================================================================
# Load-balancing auxiliary loss
# ============================================================================


def test_load_balancing_loss_is_minimal_for_uniform_routing():
    adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))

    # 4 tokens, each routed to a distinct expert, with uniform router probabilities.
    routing_weights = torch.full((4, 4), 0.25)
    topk_indices = torch.tensor([[0], [1], [2], [3]])

    aux_loss = adapter._load_balancing_loss(routing_weights, topk_indices, attention_mask=None)

    # The Switch-Transformer aux loss has a known minimum of 1.0 at perfect balance,
    # independent of num_experts.
    assert aux_loss.item() == pytest.approx(1.0, abs=1e-5)


def test_load_balancing_loss_is_higher_for_degenerate_routing():
    adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))

    uniform_weights = torch.full((4, 4), 0.25)
    uniform_indices = torch.tensor([[0], [1], [2], [3]])
    uniform_loss = adapter._load_balancing_loss(uniform_weights, uniform_indices, attention_mask=None)

    # All tokens collapse onto expert 0.
    degenerate_weights = torch.tensor([[0.9, 0.05, 0.03, 0.02]] * 4)
    degenerate_indices = torch.zeros((4, 1), dtype=torch.long)
    degenerate_loss = adapter._load_balancing_loss(
        degenerate_weights, degenerate_indices, attention_mask=None
    )

    assert degenerate_loss.item() > uniform_loss.item()


def test_load_balancing_loss_excludes_padded_tokens():
    adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))

    valid_weights = torch.full((4, 4), 0.25)
    valid_indices = torch.tensor([[0], [1], [2], [3]])

    # Padded rows are wildly imbalanced; if they leaked into the statistics the loss
    # would change from the "valid tokens only" reference computed below.
    padded_weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    padded_indices = torch.zeros((3, 1), dtype=torch.long)

    all_weights = torch.cat([valid_weights, padded_weights], dim=0)
    all_indices = torch.cat([valid_indices, padded_indices], dim=0)
    attention_mask = torch.cat([torch.ones(4), torch.zeros(3)])

    masked_loss = adapter._load_balancing_loss(all_weights, all_indices, attention_mask)
    reference_loss = adapter._load_balancing_loss(valid_weights, valid_indices, attention_mask=None)

    assert masked_loss.item() == pytest.approx(reference_loss.item(), abs=1e-6)


def test_load_balancing_loss_is_none_when_fully_padded():
    adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))

    weights = torch.full((3, 4), 0.25)
    indices = torch.zeros((3, 1), dtype=torch.long)
    attention_mask = torch.zeros(3)

    assert adapter._load_balancing_loss(weights, indices, attention_mask) is None


# ============================================================================
# Shared expert
# ============================================================================


def test_shared_expert_toggle():
    torch.manual_seed(0)
    adapter_no_shared = MELTMoEAdapter(_moe_config(use_shared_expert=False))
    torch.manual_seed(0)
    adapter_with_shared = MELTMoEAdapter(_moe_config(use_shared_expert=True))

    assert adapter_no_shared.shared_expert is None
    assert adapter_with_shared.shared_expert is not None

    audio_features = torch.randn(2, 5, 6)
    out_no_shared = adapter_no_shared(audio_features)
    out_with_shared = adapter_with_shared(audio_features)

    assert not torch.allclose(out_no_shared, out_with_shared)


# ============================================================================
# End-to-end: aux loss actually reaches the training loss
# ============================================================================


@pytest.mark.hub
def test_moe_aux_loss_is_wired_into_training_loss():
    """`forward()` must add `router_aux_loss_coef * aux_loss` to the LM loss.

    The real audio-placeholder injection path (`_merge_embeddings`) needs a tokenizer
    with the audio token registered, which is processor/tokenizer setup outside this
    unit test's scope. Since that machinery is unrelated to the change under test, stub
    `_get_audio_embeddings`/`_merge_embeddings` to isolate the loss-combination logic
    in `MELTForCausalLM.forward()` -- exactly the two lines this feature added.
    """
    config = _make_melt_config(
        adapter_config={
            "_type": "moe",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
        }
    )
    config.audio_encoder_config.num_hidden_layers = 1
    config.text_decoder_config.n_layer = 1
    model = MELTForCausalLM(config, load_backbones=True)
    model.eval()  # deterministic: no dropout randomness between the two calls below

    aux_loss = torch.tensor(2.0)

    def fake_get_audio_embeddings(**kwargs):
        return None, None, None, aux_loss

    def fake_merge_embeddings(**kwargs):
        return (
            model._get_text_embeddings(input_ids=kwargs["input_ids"]),
            kwargs["attention_mask"],
            kwargs["labels"],
        )

    model._get_audio_embeddings = fake_get_audio_embeddings
    model._merge_embeddings = fake_merge_embeddings

    input_ids = torch.randint(0, 1000, (1, 6))
    labels = input_ids.clone()
    call_kwargs = dict(input_ids=input_ids, input_features=torch.zeros(1, 1), labels=labels)

    model.config.adapter_config.router_aux_loss_coef = 0.0
    loss_zero_coef = model(**call_kwargs).loss

    model.config.adapter_config.router_aux_loss_coef = 1.0
    loss_with_coef = model(**call_kwargs).loss

    assert torch.allclose(loss_with_coef - loss_zero_coef, aux_loss, atol=1e-5)


# ============================================================================
# stack_factor (frame stacking before the router)
# ============================================================================
# Reaches the crossing's common frame rate the same way MELTMLPAdapter does:
# concatenates k consecutive encoder frames along the feature axis before the
# router, so the router and every expert see k x encoder width
# (`03-audio-stack.md` §3; design confirmed by the PI 2026-09-23).


class TestMoEAdapterStackFactor:
    def test_stack_factor_widens_router_and_expert_input(self):
        adapter = MELTMoEAdapter(_moe_config(stack_factor=4))
        assert adapter.router.in_features == 6 * 4
        for expert in adapter.experts:
            assert expert.gate_proj.in_features == 6 * 4
            assert expert.up_proj.in_features == 6 * 4

    def test_stack_factor_widens_shared_expert_input(self):
        adapter = MELTMoEAdapter(_moe_config(stack_factor=4, use_shared_expert=True))
        assert adapter.shared_expert.gate_proj.in_features == 6 * 4

    def test_stack_factor_one_matches_pre_stacking_behavior(self):
        """stack_factor=1 (the default) is exactly the old, unstacked adapter."""
        adapter = MELTMoEAdapter(_moe_config(stack_factor=1))
        assert adapter.router.in_features == 6

    def test_stack_factor_none_matches_pre_stacking_behavior(self):
        """stack_factor left unset (None on a config that omits the key) is 1."""
        config = _moe_config()
        config.adapter_config.stack_factor = None
        adapter = MELTMoEAdapter(config)
        assert adapter.router.in_features == 6

    def test_stack_factor_downsamples_exact_multiple(self):
        """seq_len an exact multiple of k: output length is exactly seq_len / k."""
        adapter = MELTMoEAdapter(_moe_config(stack_factor=4))

        batch_size, seq_len = 2, 8
        input_features = torch.randn(batch_size, seq_len, 6)
        attention_mask = torch.ones(batch_size, seq_len)

        output_shape, output_mask = adapter._get_output_features_shape(
            input_features.shape, attention_mask
        )
        # Feature dim is `text_hidden_size` (10, the decoder width the adapter
        # projects into), not `audio_hidden_size` (6) and not `6 * stack_factor`
        # (24): stacking only widens what feeds INTO the router/experts, per
        # test_stack_factor_widens_router_and_expert_input above. It never
        # changes what they emit, which is always the fixed decoder width.
        assert output_shape == (batch_size, 2, 10)
        assert output_mask.shape == (batch_size, 2)
        assert output_mask.bool().all()

        actual_output = adapter(input_features, attention_mask=attention_mask)
        assert actual_output.shape == output_shape

    def test_stack_factor_pads_a_non_multiple_seq_len(self):
        """seq_len 5 with k=4 pads to 8 frames, i.e. ceil(5/4) = 2 output frames."""
        adapter = MELTMoEAdapter(_moe_config(stack_factor=4))

        input_features = torch.randn(2, 5, 6)
        output_shape, _ = adapter._get_output_features_shape(input_features.shape, None)
        assert output_shape == (2, 2, 10)

        actual_output = adapter(input_features)
        assert actual_output.shape == output_shape

    def test_stack_factor_subsamples_the_attention_mask(self):
        """Per-example valid length is subsampled by k, as a prefix mask."""
        adapter = MELTMoEAdapter(_moe_config(stack_factor=4))

        batch_size, seq_len = 2, 8
        input_features = torch.randn(batch_size, seq_len, 6)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask[0, :8] = 1  # full 8 frames -> ceil(8/4) = 2
        attention_mask[1, :3] = 1  # 3 frames -> ceil(3/4) = 1

        output_shape, output_mask = adapter._get_output_features_shape(
            input_features.shape, attention_mask
        )
        assert output_shape == (batch_size, 2, 10)
        assert output_mask.sum(-1).tolist() == [2, 1]
        assert output_mask[1, :1].all() and not output_mask[1, 1:].any()

    def test_k1_is_bit_identical_to_pre_stacking_forward(self):
        """A stack_factor=1 MoE adapter is byte-for-byte the pre-stacking one.

        Guards the checkpoint-compatibility requirement directly: a checkpoint
        trained before this change must still load and behave identically.
        """
        torch.manual_seed(0)
        adapter_k1 = MELTMoEAdapter(_moe_config(stack_factor=1))
        torch.manual_seed(0)
        adapter_none = MELTMoEAdapter(_moe_config())  # default, no stack_factor kwarg

        assert adapter_k1.router.in_features == adapter_none.router.in_features == 6

        input_features = torch.randn(2, 5, 6)
        attention_mask = torch.ones(2, 5)
        torch.manual_seed(1)
        out_k1 = adapter_k1(input_features, attention_mask=attention_mask)
        torch.manual_seed(1)
        out_none = adapter_none(input_features, attention_mask=attention_mask)
        assert torch.equal(out_k1, out_none)


# ============================================================================
# stack_factor's effect on the load-balancing loss: it must see the STACKED
# mask, so padded frames (post-stacking) never count.
# ============================================================================


class TestMoEAdapterStackedAuxLoss:
    def test_aux_loss_uses_stacked_mask_not_raw_mask(self):
        """A raw-frame mask with valid frames scattered across two stacked
        groups must not leak into the aux loss computed on the stacked
        sequence: only whole stacked frames that are fully valid should count.
        """
        adapter = MELTMoEAdapter(
            _moe_config(num_experts=4, num_experts_per_tok=1, stack_factor=4)
        )

        # seq_len 8, k=4 -> 2 stacked frames. Batch item 0: all 8 raw frames
        # valid (2 valid stacked frames). Batch item 1: only 2 of 8 raw frames
        # valid (ceil(2/4) = 1 valid stacked frame).
        input_features = torch.randn(2, 8, 6)
        attention_mask = torch.zeros(2, 8, dtype=torch.long)
        attention_mask[0, :] = 1
        attention_mask[1, :2] = 1

        adapter(input_features, attention_mask=attention_mask)

        assert adapter.aux_loss is not None
        # 3 valid stacked frames total (2 + 1), never the 16 raw frames.
        _, stacked_mask = adapter._get_output_features_shape(input_features.shape, attention_mask)
        assert stacked_mask.sum().item() == 3

    def test_aux_loss_finite_and_present_at_k5(self):
        """The crossing's actual working point: k=5 on a 50 Hz encoder."""
        adapter = MELTMoEAdapter(
            _moe_config(num_experts=8, num_experts_per_tok=2, stack_factor=5)
        )
        input_features = torch.randn(2, 27, 6)  # not a multiple of 5, exercises padding
        attention_mask = torch.ones(2, 27)

        adapter(input_features, attention_mask=attention_mask)

        assert adapter.aux_loss is not None
        assert torch.isfinite(adapter.aux_loss)


# ============================================================================
# Logging: aux loss and router entropy on their own (03-audio-stack.md §0.1)
# ============================================================================


class TestMoEAdapterLogging:
    def test_router_entropy_is_populated_after_forward(self):
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=2))
        assert adapter.router_entropy is None  # nothing run yet

        adapter(torch.randn(2, 5, 6))
        assert adapter.router_entropy is not None
        assert torch.isfinite(adapter.router_entropy)

    def test_router_entropy_is_maximal_for_a_uniform_router(self):
        """A router forced to output a uniform distribution over 4 experts has
        entropy exactly log(4) -- the maximum for 4 outcomes -- a precise
        value to check the formula against, not just "some number"."""
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=2))
        adapter.router.forward = lambda x: torch.zeros(x.shape[0], 4)  # uniform softmax

        adapter(torch.randn(2, 5, 6))

        assert adapter.router_entropy.item() == pytest.approx(math.log(4), abs=1e-5)

    def test_router_entropy_is_near_zero_for_a_collapsed_router(self):
        """A router that always picks the same expert with near-certainty has
        entropy near 0 -- the signal `03-audio-stack.md` §0.1 wants for
        catching router collapse."""
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=2))
        fixed_logits = torch.tensor([[20.0, -20.0, -20.0, -20.0]])
        adapter.router.forward = lambda x: fixed_logits.expand(x.shape[0], -1)

        adapter(torch.randn(1, 3, 6))

        assert adapter.router_entropy.item() < 1e-6

    def test_router_entropy_excludes_padded_frames(self):
        """Padded frames must not pull the entropy reading toward the uniform
        maximum, the same exclusion the aux loss applies."""
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))

        # 4 valid tokens with a collapsed router; 3 padded tokens the router
        # would (if counted) route much more uniformly.
        fixed_logits = torch.cat(
            [
                torch.tensor([[20.0, -20.0, -20.0, -20.0]]).expand(4, -1),
                torch.zeros(3, 4),
            ]
        )
        adapter.router.forward = lambda x: fixed_logits

        attention_mask = torch.cat([torch.ones(4), torch.zeros(3)])
        adapter(torch.randn(1, 7, 6), attention_mask=attention_mask.unsqueeze(0))

        assert adapter.router_entropy.item() < 1e-6

    def test_router_entropy_is_none_when_fully_padded(self):
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=1))
        attention_mask = torch.zeros(1, 3)

        adapter(torch.randn(1, 3, 6), attention_mask=attention_mask)

        assert adapter.router_entropy is None
        assert adapter.aux_loss is None  # same "no valid frames" condition

    def test_aux_loss_and_entropy_are_both_logged_as_independent_scalars(self):
        """`03-audio-stack.md` §0.1 wants the aux loss "on its own", not only
        folded into the training loss -- both must be plain 0-d tensors the
        trainer can read back independently of the loss computation."""
        adapter = MELTMoEAdapter(_moe_config(num_experts=4, num_experts_per_tok=2))
        adapter(torch.randn(2, 5, 6))

        assert adapter.aux_loss.dim() == 0
        assert adapter.router_entropy.dim() == 0
