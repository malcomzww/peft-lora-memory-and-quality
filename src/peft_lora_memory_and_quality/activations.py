"""The term the weights-and-states budget leaves out, and when it dominates.

`memory.py` counts weights, gradients and optimizer state. That is the budget
people quote, and it is the one that produces "LoRA saves 5.97x". It excludes
activations, which the README says plainly -- and which turns out to be the
difference between a plan that fits and a job that OOMs at step one.

Activations behave nothing like the counted terms. Weights are fixed by the
architecture; activations scale with batch size and sequence length, and the
attention term scales with the *square* of sequence length. So the excluded term
has a growth rate the included ones do not, and past some context length it is
the whole problem.

That has a specific consequence for the repo's headline. LoRA removes gradient
and optimizer state, which are activation-independent. It does **not** reduce
activations -- the forward pass through the frozen base is unchanged, and
backprop still has to traverse it to reach the adapters. So LoRA's advantage
shrinks as sequence length grows, and the 5.97x figure is a short-sequence
number rather than a property of the method.

Gradient checkpointing is the lever that acts on this term, trading recomputation
for memory, and it is the one decision here that is not independent of the
others. This module quantifies all of it.

Everything remains closed-form arithmetic over stated shapes. No run is
performed, so the same caveat applies: it is a prediction, not a measurement,
and it is a floor rather than a target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .memory import (
    BYTES_PER_DTYPE,
    SHAPES,
    MemoryBudget,
    ModelShape,
    full_finetune_budget,
    lora_budget,
    qlora_budget,
)

GIB = 1024**3


@dataclass(frozen=True)
class ActivationBudget:
    """Activation memory for one training step, in bytes.

    Split into the two terms because they scale differently and the split is the
    whole reason this module exists: `per_layer` is linear in sequence length,
    `attention` is quadratic.
    """

    shape: str
    batch: int
    sequence: int
    per_layer: float
    attention: float
    checkpointed: bool

    @property
    def total(self) -> float:
        return self.per_layer + self.attention

    @property
    def gib(self) -> float:
        return self.total / GIB

    @property
    def attention_share(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.attention / self.total


def activation_bytes(
    shape: ModelShape,
    batch: int,
    sequence: int,
    dtype: str = "bf16",
    checkpointing: bool = False,
) -> ActivationBudget:
    """Activation memory held for the backward pass.

    Two contributions:

    `per_layer` -- the intermediate tensors every layer stores so backprop can
    use them. Counted as a multiple of `batch * sequence * hidden` plus the MLP
    intermediate, which is the dominant width in a SwiGLU block. The multiple is
    an approximation of how many tensors a real implementation keeps live; it is
    a stated parameter like everything else here.

    `attention` -- the score matrix, `batch * heads * sequence^2`. This is the
    quadratic term. FlashAttention-style kernels avoid materialising it, which is
    why `checkpointing=True` removes it here rather than merely reducing it.

    With checkpointing, only layer boundaries are retained and the rest is
    recomputed, so the per-layer term collapses to a single layer's worth. The
    cost is roughly a third more compute, which this module does not model
    because it is not a memory quantity.
    """
    if batch < 1:
        raise ValueError("batch must be at least 1")
    if sequence < 1:
        raise ValueError("sequence must be at least 1")

    width = BYTES_PER_DTYPE[dtype]
    tokens = batch * sequence

    # Tensors kept live per layer, as a multiple of the two characteristic
    # widths. Approximate, stated, and the same for every variant so that
    # comparisons between them are unaffected by the exact multiple.
    per_layer_one = width * tokens * (6 * shape.hidden + 2 * shape.intermediate)

    if checkpointing:
        retained = per_layer_one + width * tokens * shape.hidden * shape.layers
        attention = 0.0
    else:
        retained = per_layer_one * shape.layers
        attention = width * batch * shape.heads * sequence * sequence

    return ActivationBudget(
        shape=shape.name,
        batch=batch,
        sequence=sequence,
        per_layer=retained,
        attention=attention,
        checkpointed=checkpointing,
    )


@dataclass(frozen=True)
class FullBudget:
    """Weights, states and activations together -- the number that has to fit."""

    variant: str
    states: MemoryBudget
    activations: ActivationBudget

    @property
    def total(self) -> float:
        return self.states.total + self.activations.total

    @property
    def gib(self) -> float:
        return self.total / GIB

    @property
    def activation_share(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.activations.total / self.total


def full_budget(
    variant: str,
    shape: ModelShape,
    batch: int,
    sequence: int,
    rank: int = 16,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    checkpointing: bool = False,
) -> FullBudget:
    """One variant's complete footprint at a given batch and sequence length."""
    if variant == "full":
        states = full_finetune_budget(shape)
    elif variant == "lora":
        states = lora_budget(shape, rank, targets)
    elif variant == "qlora":
        states = qlora_budget(shape, rank, targets)
    else:
        raise ValueError(f"unknown variant: {variant}")

    return FullBudget(
        variant=variant,
        states=states,
        activations=activation_bytes(
            shape, batch, sequence, checkpointing=checkpointing
        ),
    )


def saving_ratio(
    shape: ModelShape,
    batch: int,
    sequence: int,
    rank: int = 16,
    checkpointing: bool = False,
) -> float:
    """LoRA's memory advantage over a full fine-tune, activations included.

    The headline 5.97x is this function at a sequence length short enough for
    activations to round away. It falls monotonically as sequence grows, because
    the term LoRA removes is fixed while the term it does not touch is growing.
    """
    full = full_budget("full", shape, batch, sequence, rank, checkpointing=checkpointing)
    lora = full_budget("lora", shape, batch, sequence, rank, checkpointing=checkpointing)
    if lora.total <= 0:
        return 0.0
    return full.total / lora.total


def crossover_sequence(
    shape: ModelShape,
    batch: int = 1,
    rank: int = 16,
    checkpointing: bool = False,
    limit: int = 262_144,
) -> int | None:
    """Sequence length at which activations exceed LoRA's weights-and-states.

    The planning number: past this point the budget in `memory.md` is describing
    the smaller half of the problem. Scanned over powers of two because that is
    the resolution the decision is made at.
    """
    length = 128
    while length <= limit:
        budget = full_budget(
            "lora", shape, batch, length, rank, checkpointing=checkpointing
        )
        if budget.activations.total > budget.states.total:
            return length
        length *= 2
    return None


def max_sequence_within(
    variant: str,
    shape: ModelShape,
    capacity_gib: float,
    batch: int = 1,
    rank: int = 16,
    checkpointing: bool = False,
) -> int:
    """Longest power-of-two sequence fitting in `capacity_gib`.

    Zero means the variant does not fit at any length -- its weights and states
    already exceed the card, which is a different answer from "short sequences
    only".
    """
    if capacity_gib <= 0:
        raise ValueError("capacity_gib must be positive")

    best = 0
    length = 128
    while length <= 262_144:
        budget = full_budget(
            variant, shape, batch, length, rank, checkpointing=checkpointing
        )
        if budget.gib > capacity_gib:
            break
        best = length
        length *= 2
    return best


def checkpointing_saving(
    shape: ModelShape, batch: int, sequence: int
) -> float:
    """Activation memory with checkpointing as a fraction of without.

    Small at short sequences and dramatic at long ones, because it removes the
    quadratic term entirely rather than scaling it.
    """
    without = activation_bytes(shape, batch, sequence, checkpointing=False).total
    with_ck = activation_bytes(shape, batch, sequence, checkpointing=True).total
    if without <= 0:
        return 1.0
    return with_ck / without


def tokens_at_capacity(
    variant: str,
    shape: ModelShape,
    capacity_gib: float,
    rank: int = 16,
    checkpointing: bool = False,
) -> tuple[int, int, int]:
    """Best (batch, sequence, tokens) fitting in `capacity_gib`.

    Included because batch and sequence trade against each other in the linear
    term but not the quadratic one, so the token-maximising point is not always
    the longest sequence -- and a training run cares about tokens per step.
    """
    best = (0, 0, 0)
    for batch in (1, 2, 4, 8, 16, 32):
        length = 128
        while length <= 262_144:
            budget = full_budget(
                variant, shape, batch, length, rank, checkpointing=checkpointing
            )
            if budget.gib > capacity_gib:
                break
            tokens = batch * length
            if tokens > best[2]:
                best = (batch, length, tokens)
            length *= 2
    return best


def attention_dominance_sequence(shape: ModelShape, batch: int = 1) -> int:
    """Sequence length where the quadratic term overtakes the linear one.

    Solvable in closed form -- the two terms are equal when

        heads * s = layers * (6*hidden + 2*intermediate)

    -- so this is exact rather than scanned, and it is where "activations scale
    with sequence length" stops being a linear intuition.
    """
    linear_coefficient = shape.layers * (6 * shape.hidden + 2 * shape.intermediate)
    return math.ceil(linear_coefficient / shape.heads)


__all__ = [
    "SHAPES",
    "ActivationBudget",
    "FullBudget",
    "activation_bytes",
    "attention_dominance_sequence",
    "checkpointing_saving",
    "crossover_sequence",
    "full_budget",
    "max_sequence_within",
    "saving_ratio",
    "tokens_at_capacity",
]
