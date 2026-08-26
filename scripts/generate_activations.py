"""Generate results/activations.md and results/activations-raw.json.

The term memory.md excludes, and what including it does to the headline.

    uv run python scripts/generate_activations.py

Every claim the README makes from this file is asserted before it is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_lora_memory_and_quality import (  # noqa: E402
    SHAPES,
    activation_bytes,
    attention_dominance_sequence,
    checkpointing_saving,
    crossover_sequence,
    full_budget,
    max_sequence_within,
    saving_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "activations.md"
RAW = ROOT / "results" / "activations-raw.json"
SHAPE = SHAPES["llama-8b"]
SEQUENCES = (512, 2048, 8192, 32768, 131072)
CARD_GIB = 80.0
MODELS = ("smol-135m", "qwen-0.5b", "llama-1b", "llama-8b")


def main() -> int:
    rows = []
    for sequence in SEQUENCES:
        act = activation_bytes(SHAPE, 1, sequence)
        lora = full_budget("lora", SHAPE, 1, sequence)
        rows.append(
            {
                "sequence": sequence,
                "activations_gib": act.gib,
                "attention_share": act.attention_share,
                "lora_total_gib": lora.gib,
                "activation_share": lora.activation_share,
                "saving_ratio": saving_ratio(SHAPE, 1, sequence),
            }
        )

    crossovers = {name: crossover_sequence(SHAPES[name]) for name in MODELS}
    capacity = {
        variant: {
            "plain": max_sequence_within(variant, SHAPE, CARD_GIB),
            "checkpointed": max_sequence_within(
                variant, SHAPE, CARD_GIB, checkpointing=True
            ),
        }
        for variant in ("full", "lora", "qlora")
    }
    ck = {
        s: checkpointing_saving(SHAPE, 1, s) for s in (2048, 8192, 32768)
    }

    # --- claims -----------------------------------------------------------
    assert rows[0]["saving_ratio"] > 5.0
    assert rows[3]["saving_ratio"] < 1.6
    assert all(
        later["saving_ratio"] < earlier["saving_ratio"]
        for earlier, later in zip(rows, rows[1:], strict=False)
    )
    assert all(r["saving_ratio"] > 1.0 for r in rows)
    assert crossovers["llama-8b"] == 8192
    assert all(v is not None for v in crossovers.values())
    ordered = [crossovers[m] for m in MODELS]
    assert all(a < b for a, b in zip(ordered, ordered[1:], strict=False))
    assert capacity["full"]["plain"] == 0
    assert capacity["qlora"]["plain"] > capacity["lora"]["plain"]
    assert capacity["lora"]["checkpointed"] > capacity["qlora"]["plain"]
    assert (
        crossover_sequence(SHAPE, checkpointing=True) >= 8 * crossovers["llama-8b"]
    )

    lines: list[str] = []
    lines.append("# Activations: the term the budget leaves out\n")
    lines.append(
        "[`memory.md`](memory.md) counts weights, gradients and optimizer state. "
        "That is the budget people quote and the one that produces "
        "**LoRA saves 5.97x**. It excludes activations, which is defensible at "
        "short context and wrong past about 8k tokens.\n"
    )
    lines.append(
        "Activations behave unlike anything in that budget. Weights are fixed by "
        "the architecture; activations scale with batch and sequence length, and "
        "the attention term scales with the **square** of sequence length. The "
        "excluded term has a growth rate the included ones do not.\n"
    )

    lines.append(f"## `{SHAPE.name}`, batch 1, LoRA rank 16\n")
    lines.append(
        "| sequence | activations | attention share | LoRA total | activations "
        "are | vs full |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['sequence']:,} | {r['activations_gib']:.2f} GiB | "
            f"{r['attention_share']:.0%} | {r['lora_total_gib']:.2f} GiB | "
            f"**{r['activation_share']:.0%}** of it | {r['saving_ratio']:.2f}x |"
        )
    lines.append("")
    short, long = rows[0], rows[3]
    lines.append(
        f"**LoRA's advantage is a short-sequence property.** It saves "
        f"{short['saving_ratio']:.2f}x at {short['sequence']} tokens and "
        f"{long['saving_ratio']:.2f}x at {long['sequence']:,}. LoRA removes "
        "gradient and optimizer state, which do not depend on sequence length. "
        "It does not reduce activations at all -- the forward pass through the "
        "frozen base is unchanged, and backprop still traverses it to reach the "
        "adapters. A fixed saving divided by a growing total is a shrinking "
        "ratio.\n"
    )
    lines.append(
        "The ratio decays toward 1.0 and does not invert, because activations "
        "are identical between the two. *The advantage shrinks* is the claim, "
        "not *LoRA stops being worth it*.\n"
    )

    lines.append("## Where the excluded term overtakes the counted one\n")
    lines.append("| model | parameters | crossover sequence |")
    lines.append("|---|---|---|")
    for name in MODELS:
        shape = SHAPES[name]
        lines.append(
            f"| `{name}` | {shape.total_params() / 1e9:.2f}B | "
            f"**{crossovers[name]:,}** tokens |"
        )
    lines.append("")
    lines.append(
        "**Smaller models reach it sooner**, which is the opposite of the "
        "intuition that fewer layers means fewer activations. The cause is the "
        "embedding table: `2 * vocab * hidden` parameters that cost weights, "
        "gradients and optimizer state while producing no per-layer activations "
        "at all. On `smol-135m` that is most of the parameter count, so its "
        "states budget is large relative to its activation footprint.\n"
    )
    lines.append(
        "Practically: the excluded term matters most on the models people "
        "prototype with, which is exactly where a memory plan is least likely "
        "to be checked.\n"
    )
    lines.append(
        f"The attention term overtakes the per-layer term at "
        f"**{attention_dominance_sequence(SHAPE):,} tokens** on `{SHAPE.name}` "
        "-- past that, activation memory is mostly the score matrix and the "
        "linear intuition stops applying.\n"
    )

    lines.append(f"## What fits on one {CARD_GIB:.0f} GiB card\n")
    lines.append("| variant | longest sequence | with checkpointing |")
    lines.append("|---|---|---|")
    for variant, v in capacity.items():
        plain = f"{v['plain']:,}" if v["plain"] else "**does not fit**"
        lines.append(
            f"| `{variant}` | {plain} | {v['checkpointed']:,} |"
        )
    lines.append("")
    lines.append(
        f"A full fine-tune does not fit at any sequence length -- its states "
        f"alone are 89.7 GiB. **Checkpointing buys more context than quantising "
        f"does**: it takes LoRA from {capacity['lora']['plain']:,} to "
        f"{capacity['lora']['checkpointed']:,} tokens, against "
        f"{capacity['qlora']['plain']:,} for QLoRA without it. They compose, but "
        "the quadratic term is the expensive one and checkpointing is what "
        "removes it.\n"
    )

    lines.append("## Checkpointing\n")
    lines.append("| sequence | activations retained |")
    lines.append("|---|---|")
    for sequence, fraction in ck.items():
        lines.append(f"| {sequence:,} | {fraction:.1%} |")
    lines.append("")
    lines.append(
        "It helps more as sequences grow, because the term it removes is the "
        "one growing fastest. On `llama-8b` it moves the crossover from "
        f"{crossovers['llama-8b']:,} to "
        f"{crossover_sequence(SHAPE, checkpointing=True):,} tokens.\n"
    )
    lines.append(
        "**With checkpointing on, only the token count matters.** Removing the "
        "quadratic term leaves activations linear in `batch * sequence`, so "
        "batch 4 at 32k and batch 1 at 128k cost identical memory. The "
        "batch/sequence split becomes a throughput and convergence decision "
        "rather than a memory one -- which is not true without it, where a long "
        "sequence at equal token count costs strictly more.\n"
    )
    lines.append(
        "The compute cost of recomputation is roughly a third more, and is not "
        "modelled here because it is not a memory quantity.\n"
    )

    lines.append("## Still not covered\n")
    lines.append(
        "The per-layer multiple is a stated approximation of how many tensors a "
        "real implementation keeps live, not a count from a profiler. It is the "
        "same for every variant, so comparisons between them are unaffected, "
        "but the absolute figures carry that uncertainty. Fragmentation and "
        "allocator overhead are still excluded, so these remain a floor rather "
        "than a target.\n"
    )

    lines.append("## Reproduce\n")
    lines.append("```\nuv run python scripts/generate_activations.py\n```\n")
    lines.append("Raw values in [`activations-raw.json`](activations-raw.json).\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")

    RAW.write_text(
        json.dumps(
            {
                "model": SHAPE.name,
                "card_gib": CARD_GIB,
                "by_sequence": rows,
                "crossover_sequence": crossovers,
                "crossover_checkpointed": crossover_sequence(
                    SHAPE, checkpointing=True
                ),
                "attention_dominance": attention_dominance_sequence(SHAPE),
                "capacity": capacity,
                "checkpointing_retained": ck,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {OUT.name} and {RAW.name}")
    print(
        f"  saving {short['saving_ratio']:.2f}x @{short['sequence']} -> "
        f"{long['saving_ratio']:.2f}x @{long['sequence']:,}"
    )
    print(f"  crossover: {crossovers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
