"""The term the weights-and-states budget excludes, and what it does to the headline.

`memory.py` counts weights, gradients and optimizer state, which is the budget
people quote and the one that produces "LoRA saves 5.97x". The README lists
activations as out of scope. These check what including them changes -- and the
answer is that the headline is a short-sequence claim rather than a property of
the method.
"""

from __future__ import annotations

import pytest

from peft_lora_memory_and_quality.activations import (
    activation_bytes,
    attention_dominance_sequence,
    checkpointing_saving,
    crossover_sequence,
    full_budget,
    max_sequence_within,
    saving_ratio,
    tokens_at_capacity,
)
from peft_lora_memory_and_quality.memory import SHAPES

SHAPE = SHAPES["llama-8b"]


def test_activations_scale_linearly_with_batch():
    """Batch multiplies every term, including the quadratic one."""
    one = activation_bytes(SHAPE, 1, 2048).total
    four = activation_bytes(SHAPE, 4, 2048).total
    assert four == pytest.approx(4 * one)


def test_the_attention_term_is_quadratic_in_sequence():
    """The reason activations behave unlike anything memory.py counts.

    Doubling sequence length doubles the per-layer term and quadruples the
    attention term, so the mix shifts as context grows.
    """
    short = activation_bytes(SHAPE, 1, 2048)
    long = activation_bytes(SHAPE, 1, 4096)
    assert long.per_layer == pytest.approx(2 * short.per_layer)
    assert long.attention == pytest.approx(4 * short.attention)


def test_attention_dominance_is_where_the_two_terms_cross():
    """Closed form rather than scanned, so it can be checked directly."""
    crossover = attention_dominance_sequence(SHAPE)
    below = activation_bytes(SHAPE, 1, crossover // 2)
    above = activation_bytes(SHAPE, 1, crossover * 2)
    assert below.attention < below.per_layer
    assert above.attention > above.per_layer


def test_activations_are_negligible_at_short_sequences():
    """Which is why excluding them was defensible for the original tables."""
    budget = full_budget("lora", SHAPE, 1, 512)
    assert budget.activation_share < 0.15


def test_activations_exceed_the_counted_budget_at_long_context():
    """The finding.

    At 8k tokens the excluded term is larger than everything memory.md counts,
    so the published budget is describing the smaller half of the problem.
    """
    crossover = crossover_sequence(SHAPE)
    assert crossover is not None
    assert crossover <= 8192
    budget = full_budget("lora", SHAPE, 1, crossover)
    assert budget.activations.total > budget.states.total


def test_loras_advantage_decays_with_sequence_length():
    """What this does to the repo's headline.

    LoRA removes gradient and optimizer state, which do not depend on sequence
    length. It does not reduce activations at all -- the forward pass through
    the frozen base is unchanged and backprop still traverses it. So the ratio
    falls monotonically: the fixed saving is being divided by a growing total.
    """
    ratios = [saving_ratio(SHAPE, 1, s) for s in (512, 2048, 8192, 32768)]
    for earlier, later in zip(ratios, ratios[1:], strict=False):
        assert later < earlier


def test_the_headline_saving_is_a_short_sequence_number():
    """5.97x at negligible context, 1.41x at 32k. Both are true; one is quoted."""
    assert saving_ratio(SHAPE, 1, 512) > 5.0
    assert saving_ratio(SHAPE, 1, 32768) < 1.6


def test_lora_never_loses_to_a_full_finetune():
    """The advantage decays toward 1.0 but does not invert.

    Activations are identical between the two, so the ratio is bounded below by
    one. Worth pinning, because "the advantage shrinks" is easy to overstate
    into "LoRA stops being worth it", and that is not what this shows.
    """
    for sequence in (512, 8192, 65536, 262144):
        assert saving_ratio(SHAPE, 1, sequence) > 1.0


def test_checkpointing_removes_the_quadratic_term():
    """It is the one lever that acts on activations rather than on states."""
    plain = activation_bytes(SHAPE, 1, 16384, checkpointing=False)
    checked = activation_bytes(SHAPE, 1, 16384, checkpointing=True)
    assert plain.attention > 0
    assert checked.attention == 0.0
    assert checked.total < plain.total


def test_checkpointing_helps_more_at_longer_sequences():
    """Because the term it removes is the one that grows fastest."""
    fractions = [checkpointing_saving(SHAPE, 1, s) for s in (2048, 8192, 32768)]
    for earlier, later in zip(fractions, fractions[1:], strict=False):
        assert later < earlier
    assert fractions[-1] < 0.10


def test_checkpointing_moves_the_crossover_far_out():
    """From 8k to 64k on this model -- an eightfold change in usable context."""
    plain = crossover_sequence(SHAPE)
    checked = crossover_sequence(SHAPE, checkpointing=True)
    assert plain is not None and checked is not None
    assert checked >= 8 * plain


def test_a_full_finetune_does_not_fit_on_a_single_80gib_card():
    """Its states alone are 89.7 GiB, so no sequence length helps.

    Zero here means "does not fit at any length", which is a different answer
    from "short sequences only" and the reason the function returns zero rather
    than the shortest length.
    """
    assert max_sequence_within("full", SHAPE, 80.0) == 0


def test_quantising_the_base_buys_context_not_just_headroom():
    """QLoRA's saving converts directly into sequence length.

    The 11.2 GiB it removes from the weights term is 11.2 GiB available for
    activations, which is the practical reason to reach for it at long context.
    """
    lora = max_sequence_within("lora", SHAPE, 80.0)
    qlora = max_sequence_within("qlora", SHAPE, 80.0)
    assert qlora > lora


def test_checkpointing_buys_more_context_than_quantising_does():
    """Which lever to pull first, at long context.

    Checkpointing takes LoRA from 8k to 128k on an 80 GiB card -- a 16x gain.
    Quantising the base takes it from 8k to 16k. They compose, but the ordering
    is worth knowing: the quadratic term is the expensive one.
    """
    lora_ck = max_sequence_within("lora", SHAPE, 80.0, checkpointing=True)
    qlora_plain = max_sequence_within("qlora", SHAPE, 80.0)
    assert lora_ck > qlora_plain


def test_with_checkpointing_only_the_token_count_matters():
    """A property I did not expect to be exact.

    Removing the quadratic term leaves activations linear in batch * sequence,
    so every allocation with the same token count costs the same memory. Batch
    4 at 32k and batch 1 at 128k are the same number of bytes -- which means the
    batch/sequence split is a throughput and convergence decision, not a memory
    one, once checkpointing is on.
    """
    a = full_budget("lora", SHAPE, 4, 32_768, checkpointing=True).total
    b = full_budget("lora", SHAPE, 1, 131_072, checkpointing=True).total
    assert a == pytest.approx(b)


def test_without_checkpointing_the_split_does_matter():
    """The contrast that makes the previous test meaningful.

    The quadratic term punishes long sequences specifically, so at equal token
    counts a long-sequence allocation costs strictly more.
    """
    a = full_budget("lora", SHAPE, 4, 8192).total
    b = full_budget("lora", SHAPE, 1, 32_768).total
    assert b > a


def test_tokens_at_capacity_fits_within_the_card():
    batch, sequence, tokens = tokens_at_capacity(
        "lora", SHAPE, 80.0, checkpointing=True
    )
    assert tokens == batch * sequence
    assert full_budget(
        "lora", SHAPE, batch, sequence, checkpointing=True
    ).gib <= 80.0


def test_smaller_models_reach_the_crossover_sooner():
    """The opposite of what I assumed, and the mechanism is worth stating.

    I expected small models to hit the crossover later -- fewer layers means
    fewer activations. They hit it sooner: 1k tokens on smol-135m against 8k on
    llama-8b, monotonically across all four shapes.

    The reason is the embedding table. It is 2 * vocab * hidden parameters that
    cost weights, gradients and optimizer state while producing no per-layer
    activations at all, and on a small model it is most of the parameter count.
    So a small model carries a large states budget relative to its activation
    footprint -- exactly backwards from the intuition that small models are
    activation-light.

    Practically: the excluded term matters most on the models people prototype
    with, which is where a memory plan is least likely to be checked.
    """
    lengths = [
        crossover_sequence(SHAPES[name])
        for name in ("smol-135m", "qwen-0.5b", "llama-1b", "llama-8b")
    ]
    assert all(v is not None for v in lengths)
    for earlier, later in zip(lengths, lengths[1:], strict=False):
        assert earlier < later


@pytest.mark.parametrize("bad", [(0, 2048), (1, 0), (-1, 2048)])
def test_invalid_shapes_are_rejected(bad):
    batch, sequence = bad
    with pytest.raises(ValueError):
        activation_bytes(SHAPE, batch, sequence)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        full_budget("adapter-soup", SHAPE, 1, 2048)


def test_nonpositive_capacity_is_rejected():
    with pytest.raises(ValueError):
        max_sequence_within("lora", SHAPE, 0.0)
