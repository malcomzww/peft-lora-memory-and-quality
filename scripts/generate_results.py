"""Generate results/memory.md.

Every number in the README comes from here. The script asserts each claim
before writing, so a change that breaks a stated finding aborts the run.

    uv run python scripts/generate_results.py

All values are closed-form byte counts and ratios. Nothing here is timed or
profiled, so nothing here is a property of the machine that ran it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_lora_memory_and_quality import (  # noqa: E402
    SHAPES,
    full_finetune_budget,
    lora_budget,
    lora_params,
    qlora_budget,
)

RANKS = (4, 8, 16, 32, 64, 128)
ATTENTION_ONLY = ("q_proj", "v_proj")
ALL_LINEAR = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
OUT = Path(__file__).resolve().parents[1] / "results" / "memory.md"
GIB = 1024**3


def main() -> int:
    ref = SHAPES["llama-8b"]

    full = full_finetune_budget(ref)
    lora = lora_budget(ref, 16, ATTENTION_ONLY)
    qlora = qlora_budget(ref, 16, ATTENTION_ONLY)

    trainable_ratio = full.trainable / lora.trainable
    memory_ratio = full.total / lora.total

    # --- assertions on the claims the README makes -----------------------
    # The headline: the trainable-parameter reduction is orders of magnitude
    # larger than the memory reduction. If these ever converge, the accounting
    # has stopped including the frozen base weights.
    assert trainable_ratio > 100 * memory_ratio, (
        "expected the trainable ratio to dwarf the memory ratio"
    )

    # LoRA cannot reduce weight memory. This is the term people forget.
    assert lora.weights == full.weights, "LoRA must not change weight memory"

    # Under LoRA the frozen weights must dominate the budget, which is why the
    # saving plateaus.
    weight_share = lora.weights / lora.total
    assert weight_share > 0.9, "expected frozen weights to dominate the LoRA budget"

    # QLoRA is the variant that attacks that term, so it must beat LoRA
    # substantially on total bytes.
    assert qlora.total < lora.total / 3, "expected QLoRA to cut the dominant term"

    # Adapter size is linear in rank: doubling rank must double the adapter.
    small = lora_params(ref, 8, ATTENTION_ONLY)
    large = lora_params(ref, 16, ATTENTION_ONLY)
    assert abs(large - 2 * small) < 1e-6, "adapter size must be linear in rank"

    lines: list[str] = []
    lines.append("# LoRA memory arithmetic\n")
    lines.append(
        "Closed-form byte counts, not measurements. Weights and gradients in "
        "bf16, Adam moments in fp32 -- which is what mixed precision actually "
        "does, and counting the moments at 2 bytes is the most common way this "
        "arithmetic is got wrong.\n"
    )

    lines.append("## The headline\n")
    lines.append(
        f"For {ref.name} at rank 16 on `q_proj`/`v_proj`:\n"
    )
    lines.append("| | full fine-tune | LoRA | QLoRA |")
    lines.append("|---|---|---|---|")
    for label, key in (
        ("weights", "weights"),
        ("gradients", "gradients"),
        ("optimizer", "optimizer"),
    ):
        lines.append(
            f"| {label} | {getattr(full, key) / GIB:.2f} GiB | "
            f"{getattr(lora, key) / GIB:.2f} GiB | "
            f"{getattr(qlora, key) / GIB:.2f} GiB |"
        )
    lines.append(
        f"| **total** | **{full.gib:.2f} GiB** | **{lora.gib:.2f} GiB** | "
        f"**{qlora.gib:.2f} GiB** |"
    )
    lines.append(
        f"| trainable params | {full.trainable / 1e6:,.1f}M | "
        f"{lora.trainable / 1e6:,.1f}M | {qlora.trainable / 1e6:,.1f}M |"
    )
    lines.append("")

    lines.append(
        f"**LoRA trains {trainable_ratio:,.0f}x fewer parameters and saves "
        f"{memory_ratio:.2f}x the memory.** Those are not the same number and "
        "they are not close. The gap is the frozen base weights, which LoRA "
        f"does not touch: they are {weight_share * 100:.1f}% of the LoRA budget "
        "and set a floor no rank can go below.\n"
    )

    lines.append("## Why the saving plateaus\n")
    lines.append("| rank | adapter params | LoRA total | vs full |")
    lines.append("|---|---|---|---|")
    for rank in RANKS:
        b = lora_budget(ref, rank, ATTENTION_ONLY)
        lines.append(
            f"| {rank} | {b.trainable / 1e6:.1f}M | {b.gib:.2f} GiB | "
            f"{full.total / b.total:.2f}x |"
        )
    lines.append("")
    lines.append(
        "Rank moves the total by a fraction of a gigabyte across a 32x sweep. "
        "**Choosing rank for memory reasons is choosing the wrong knob** -- "
        "pick it for quality and let the weight dtype carry the memory.\n"
    )

    lines.append("## Target modules matter more than rank\n")
    lines.append("| targets | rank | adapter params | LoRA total |")
    lines.append("|---|---|---|---|")
    for targets, label in ((ATTENTION_ONLY, "q,v"), (ALL_LINEAR, "all linear")):
        for rank in (16, 64):
            b = lora_budget(ref, rank, targets)
            lines.append(
                f"| {label} | {rank} | {b.trainable / 1e6:.1f}M | {b.gib:.2f} GiB |"
            )
    lines.append("")

    attn = lora_params(ref, 16, ATTENTION_ONLY)
    allin = lora_params(ref, 16, ALL_LINEAR)
    lines.append(
        f"Adapting every linear layer instead of just attention is a "
        f"{allin / attn:.1f}x larger adapter at the same rank -- a bigger "
        "change than a 4x rank increase. The MLP is where the parameters are.\n"
    )

    lines.append("## Across model sizes\n")
    lines.append(
        "| model | params | full | LoRA r=16 | QLoRA r=16 | LoRA saving |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, shape in SHAPES.items():
        f = full_finetune_budget(shape)
        lo = lora_budget(shape, 16, ATTENTION_ONLY)
        q = qlora_budget(shape, 16, ATTENTION_ONLY)
        lines.append(
            f"| `{name}` | {shape.total_params() / 1e6:,.0f}M | {f.gib:.2f} GiB | "
            f"{lo.gib:.2f} GiB | {q.gib:.2f} GiB | {f.total / lo.total:.2f}x |"
        )
    lines.append("")
    lines.append(
        "The saving ratio is nearly constant across three orders of magnitude "
        "of model size, because it is a ratio of dtype widths rather than of "
        "parameter counts. That is what makes it predictable on a whiteboard.\n"
    )

    lines.append("## What this does not cover\n")
    lines.append(
        "- **Activations.** They depend on batch size, sequence length and "
        "checkpointing policy, and at long sequence lengths they can exceed "
        "everything counted here. This is a weights-and-states budget, not a "
        "full training footprint.\n"
        "- **Fragmentation and allocator overhead.** A real run needs headroom "
        "above these numbers; they are a floor, not a target.\n"
        "- **Quality.** Nothing here says which rank trains a better model. "
        "The point is narrower: rank is not the memory knob.\n"
    )

    lines.append("## Reproduce\n")
    lines.append("```\nuv run python scripts/generate_results.py\n```\n")
    lines.append(
        "Pure arithmetic from four integers per architecture, so the output is "
        "identical on any machine.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    print(
        f"{trainable_ratio:,.0f}x fewer trainable params, "
        f"{memory_ratio:.2f}x less memory, weights are "
        f"{weight_share * 100:.1f}% of the LoRA budget"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
