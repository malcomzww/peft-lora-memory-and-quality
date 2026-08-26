"""LoRA memory arithmetic: what the adapters actually save, and what they do not.

The question: does the memory saving match the closed-form prediction, and
which term does LoRA fail to reduce?
"""

__version__ = "0.1.0"

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
    "BYTES_PER_DTYPE",
    "MemoryBudget",
    "ModelShape",
    "SHAPES",
    "full_finetune_budget",
    "lora_budget",
    "lora_params",
    "qlora_budget",
]
