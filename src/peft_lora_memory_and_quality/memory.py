"""Closed-form memory accounting for full fine-tuning versus LoRA.

Every number here is arithmetic, not measurement. That is deliberate and it is
the repo's main claim to being useful: the memory a training run needs is
*predictable* before the run starts, and an engineer who can predict it does not
discover at hour three that the job will not fit.

The four consumers of memory during training:

    weights          the model itself, in its storage dtype
    gradients        one value per *trainable* parameter
    optimizer state  Adam keeps two moments per trainable parameter
    activations      depends on batch, sequence length and depth

LoRA changes exactly one of those directly -- the trainable count -- and that
cascades into gradients and optimizer state. It does **not** reduce the weight
memory, and it barely touches activations. That asymmetry is where most of the
confusion about LoRA's savings comes from, and it is what the tables here make
explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_DTYPE = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "int8": 1,
    "nf4": 0.5,
}

# Adam and AdamW both keep exp_avg and exp_avg_sq.
ADAM_STATES = 2


@dataclass(frozen=True)
class ModelShape:
    """Enough of a transformer's shape to count its parameters.

    Deliberately not a config loader: the point is that these numbers follow
    from four integers, so they can be worked out on a whiteboard before any
    code is written.
    """

    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    vocab: int
    intermediate: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def attention_params(self) -> int:
        """Q, K, V, O projections, accounting for grouped-query attention.

        K and V are narrower than Q under GQA -- that is the whole point of it --
        so counting all four at hidden x hidden overstates the model.
        """
        q = self.hidden * self.hidden
        kv = 2 * self.hidden * (self.kv_heads * self.head_dim)
        o = self.hidden * self.hidden
        return q + kv + o

    def mlp_params(self) -> int:
        """Gate, up and down projections for a SwiGLU MLP."""
        return 3 * self.hidden * self.intermediate

    def embedding_params(self) -> int:
        """Input embedding plus an untied output head."""
        return 2 * self.vocab * self.hidden

    def total_params(self) -> int:
        per_layer = self.attention_params() + self.mlp_params()
        return self.layers * per_layer + self.embedding_params()


# Shapes taken from published architecture descriptions. No weights are
# downloaded; only the integers matter here.
SHAPES = {
    "smol-135m": ModelShape("smol-135m", 30, 576, 9, 3, 49152, 1536),
    "qwen-0.5b": ModelShape("qwen-0.5b", 24, 896, 14, 2, 151936, 4864),
    "llama-1b": ModelShape("llama-1b", 16, 2048, 32, 8, 128256, 8192),
    "llama-8b": ModelShape("llama-8b", 32, 4096, 32, 8, 128256, 14336),
}


def lora_params(shape: ModelShape, rank: int, targets: tuple[str, ...]) -> int:
    """Trainable parameters introduced by LoRA adapters.

    Each adapted matrix of shape (d_in, d_out) gains two factors: A is
    (d_in, rank) and B is (rank, d_out). So the cost is rank * (d_in + d_out)
    per matrix -- linear in rank, which is why doubling rank doubles the adapter
    and why rank is the only knob that matters for adapter size.
    """
    per_layer = 0
    hd = shape.kv_heads * shape.head_dim

    if "q_proj" in targets:
        per_layer += rank * (shape.hidden + shape.hidden)
    if "k_proj" in targets:
        per_layer += rank * (shape.hidden + hd)
    if "v_proj" in targets:
        per_layer += rank * (shape.hidden + hd)
    if "o_proj" in targets:
        per_layer += rank * (shape.hidden + shape.hidden)
    for name in ("gate_proj", "up_proj"):
        if name in targets:
            per_layer += rank * (shape.hidden + shape.intermediate)
    if "down_proj" in targets:
        per_layer += rank * (shape.intermediate + shape.hidden)

    return shape.layers * per_layer


@dataclass(frozen=True)
class MemoryBudget:
    """Bytes required, broken down by consumer."""

    weights: float
    gradients: float
    optimizer: float
    trainable: int
    total_params: int

    @property
    def total(self) -> float:
        return self.weights + self.gradients + self.optimizer

    @property
    def gib(self) -> float:
        return self.total / (1024**3)


def full_finetune_budget(
    shape: ModelShape, weight_dtype: str = "bf16", optimizer_dtype: str = "fp32"
) -> MemoryBudget:
    """Every parameter is trainable.

    Optimizer state is counted in fp32 even when the weights are bf16, because
    that is what a mixed-precision setup actually does -- keeping Adam moments
    in half precision is a known source of silent training failure. Counting
    them at 2 bytes is the most common way this arithmetic is got wrong, and it
    understates the requirement by a third.
    """
    n = shape.total_params()
    w = BYTES_PER_DTYPE[weight_dtype]
    o = BYTES_PER_DTYPE[optimizer_dtype]

    return MemoryBudget(
        weights=n * w,
        gradients=n * w,
        optimizer=n * o * ADAM_STATES,
        trainable=n,
        total_params=n,
    )


def lora_budget(
    shape: ModelShape,
    rank: int,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    weight_dtype: str = "bf16",
    optimizer_dtype: str = "fp32",
) -> MemoryBudget:
    """Base weights frozen; only adapters carry gradients and optimizer state.

    The base weights are still resident. LoRA does not shrink them, and any
    account of its savings that implies otherwise is wrong.
    """
    n = shape.total_params()
    trainable = lora_params(shape, rank, targets)
    w = BYTES_PER_DTYPE[weight_dtype]
    o = BYTES_PER_DTYPE[optimizer_dtype]

    return MemoryBudget(
        weights=n * w,
        gradients=trainable * w,
        optimizer=trainable * o * ADAM_STATES,
        trainable=trainable,
        total_params=n,
    )


def qlora_budget(
    shape: ModelShape,
    rank: int,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    base_dtype: str = "nf4",
    adapter_dtype: str = "bf16",
    optimizer_dtype: str = "fp32",
) -> MemoryBudget:
    """Quantised base weights, adapters in half precision.

    This is the variant that reduces the term LoRA alone cannot: the frozen
    weights themselves.
    """
    n = shape.total_params()
    trainable = lora_params(shape, rank, targets)

    return MemoryBudget(
        weights=n * BYTES_PER_DTYPE[base_dtype],
        gradients=trainable * BYTES_PER_DTYPE[adapter_dtype],
        optimizer=trainable * BYTES_PER_DTYPE[optimizer_dtype] * ADAM_STATES,
        trainable=trainable,
        total_params=n,
    )
