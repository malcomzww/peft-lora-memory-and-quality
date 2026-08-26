"""Memory arithmetic, checked against closed forms rather than against itself.

An arithmetic repo is only worth anything if the arithmetic is right, and a
plausible-looking wrong number is the specific hazard. So these tests derive
expected values independently where they can, rather than asserting that the
code agrees with the code.
"""

from __future__ import annotations

import pytest

from peft_lora_memory_and_quality import (
    SHAPES,
    ModelShape,
    full_finetune_budget,
    lora_budget,
    lora_params,
    qlora_budget,
)

ATTN = ("q_proj", "v_proj")
ALL_LINEAR = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def test_parameter_count_is_close_to_published_size():
    """A sanity floor: these shapes should land near their advertised sizes.

    Not exact -- the published figure for a model usually excludes or ties the
    embeddings, and the shapes here keep them untied. But an order-of-magnitude
    error in the counting would show up immediately.
    """
    assert 100e6 < SHAPES["smol-135m"].total_params() < 250e6
    assert 400e6 < SHAPES["qwen-0.5b"].total_params() < 800e6
    assert 1.0e9 < SHAPES["llama-1b"].total_params() < 2.0e9
    assert 7.0e9 < SHAPES["llama-8b"].total_params() < 9.0e9


def test_gqa_reduces_kv_projection_size():
    """Grouped-query attention exists to shrink K and V.

    Counting all four projections at hidden x hidden -- the easy mistake --
    would make this test fail, which is why it is here.
    """
    base = SHAPES["llama-8b"]
    mha = ModelShape(
        "mha", base.layers, base.hidden, base.heads, base.heads,
        base.vocab, base.intermediate,
    )
    assert base.attention_params() < mha.attention_params()


def test_adapter_size_is_linear_in_rank():
    shape = SHAPES["llama-8b"]
    r8 = lora_params(shape, 8, ATTN)
    r16 = lora_params(shape, 16, ATTN)
    r32 = lora_params(shape, 32, ATTN)
    assert r16 == 2 * r8
    assert r32 == 4 * r8


def test_adapter_size_matches_the_closed_form():
    """rank * (d_in + d_out) per adapted matrix, times layers.

    Derived here by hand rather than by calling the same helper, so the test
    would catch a change to the formula.
    """
    shape = SHAPES["llama-1b"]
    rank = 16
    hd = shape.kv_heads * shape.head_dim

    expected_per_layer = (
        rank * (shape.hidden + shape.hidden)  # q_proj
        + rank * (shape.hidden + hd)  # v_proj
    )
    assert lora_params(shape, rank, ATTN) == shape.layers * expected_per_layer


def test_adapting_more_modules_costs_more():
    shape = SHAPES["llama-8b"]
    assert lora_params(shape, 16, ALL_LINEAR) > lora_params(shape, 16, ATTN)


def test_mlp_dominates_the_adapter_budget():
    """Where the parameters actually are.

    The finding this supports: switching from attention-only to all-linear is a
    bigger change than a 4x rank increase.
    """
    shape = SHAPES["llama-8b"]
    attn_r64 = lora_params(shape, 64, ATTN)
    all_r16 = lora_params(shape, 16, ALL_LINEAR)
    assert all_r16 > attn_r64


def test_lora_does_not_reduce_weight_memory():
    """The term everyone forgets, and the reason the saving plateaus."""
    shape = SHAPES["llama-8b"]
    assert lora_budget(shape, 16, ATTN).weights == full_finetune_budget(shape).weights


def test_optimizer_state_is_two_moments_in_fp32():
    """Adam keeps exp_avg and exp_avg_sq.

    Counting them in bf16 understates a full fine-tune's requirement by about a
    third, which is enough to turn a job that does not fit into one that
    appears to.
    """
    shape = SHAPES["qwen-0.5b"]
    budget = full_finetune_budget(shape)
    assert budget.optimizer == shape.total_params() * 4 * 2


def test_full_finetune_is_dominated_by_optimizer_state():
    """In bf16 weights with fp32 moments, Adam is the largest single term."""
    budget = full_finetune_budget(SHAPES["llama-8b"])
    assert budget.optimizer > budget.weights
    assert budget.optimizer > budget.gradients


def test_lora_budget_is_dominated_by_frozen_weights():
    budget = lora_budget(SHAPES["llama-8b"], 16, ATTN)
    assert budget.weights / budget.total > 0.9


def test_qlora_attacks_the_dominant_term():
    shape = SHAPES["llama-8b"]
    lora = lora_budget(shape, 16, ATTN)
    qlora = qlora_budget(shape, 16, ATTN)
    assert qlora.weights < lora.weights
    assert qlora.total < lora.total / 3


def test_the_trainable_ratio_is_not_the_memory_ratio():
    """The headline finding, asserted rather than only written down."""
    shape = SHAPES["llama-8b"]
    full = full_finetune_budget(shape)
    lora = lora_budget(shape, 16, ATTN)

    trainable_ratio = full.trainable / lora.trainable
    memory_ratio = full.total / lora.total

    assert trainable_ratio > 1000
    assert memory_ratio < 10
    assert trainable_ratio > 100 * memory_ratio


def test_saving_ratio_is_stable_across_model_sizes():
    """It is a ratio of dtype widths, not of parameter counts."""
    ratios = []
    for shape in SHAPES.values():
        full = full_finetune_budget(shape)
        lora = lora_budget(shape, 16, ATTN)
        ratios.append(full.total / lora.total)
    assert max(ratios) - min(ratios) < 0.5


@pytest.mark.parametrize("rank", [4, 8, 16, 32, 64, 128])
def test_rank_barely_moves_the_total(rank):
    """A 32x rank sweep must not change the total materially.

    This is the evidence for "rank is not the memory knob".
    """
    shape = SHAPES["llama-8b"]
    baseline = lora_budget(shape, 4, ATTN)
    budget = lora_budget(shape, rank, ATTN)
    assert budget.total / baseline.total < 1.15


def test_totals_are_positive_and_ordered():
    shape = SHAPES["llama-8b"]
    full = full_finetune_budget(shape)
    lora = lora_budget(shape, 16, ATTN)
    qlora = qlora_budget(shape, 16, ATTN)
    assert 0 < qlora.total < lora.total < full.total
