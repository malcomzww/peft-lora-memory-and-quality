"""LoRA memory arithmetic: what the adapters actually save, and what they do not.

The question: does the memory saving match the closed-form prediction, and
which term does LoRA fail to reduce?
"""

__version__ = "0.1.0"

from .activations import (
    ActivationBudget,
    FullBudget,
    activation_bytes,
    attention_dominance_sequence,
    checkpointing_saving,
    crossover_sequence,
    full_budget,
    max_sequence_within,
    saving_ratio,
    tokens_at_capacity,
)
from .memory import (
    BYTES_PER_DTYPE,
    SHAPES,
    MemoryBudget,
    ModelShape,
    full_finetune_budget,
    lora_budget,
    lora_params,
    qlora_budget,
)

__all__ = [
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
    "BYTES_PER_DTYPE",
    "MemoryBudget",
    "ModelShape",
    "SHAPES",
    "full_finetune_budget",
    "lora_budget",
    "lora_params",
    "qlora_budget",
]
