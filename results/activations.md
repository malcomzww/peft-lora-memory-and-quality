# Activations: the term the budget leaves out

[`memory.md`](memory.md) counts weights, gradients and optimizer state. That is the budget people quote and the one that produces **LoRA saves 5.97x**. It excludes activations, which is defensible at short context and wrong past about 8k tokens.

Activations behave unlike anything in that budget. Weights are fixed by the architecture; activations scale with batch and sequence length, and the attention term scales with the **square** of sequence length. The excluded term has a growth rate the included ones do not.

## `llama-8b`, batch 1, LoRA rank 16

| sequence | activations | attention share | LoRA total | activations are | vs full |
|---|---|---|---|---|---|
| 512 | 1.64 GiB | 1% | 16.66 GiB | **10%** of it | 5.48x |
| 2,048 | 6.75 GiB | 4% | 21.77 GiB | **31%** of it | 4.43x |
| 8,192 | 30.00 GiB | 13% | 45.02 GiB | **67%** of it | 2.66x |
| 32,768 | 168.00 GiB | 38% | 183.02 GiB | **92%** of it | 1.41x |
| 131,072 | 1440.00 GiB | 71% | 1455.02 GiB | **99%** of it | 1.05x |

**LoRA's advantage is a short-sequence property.** It saves 5.48x at 512 tokens and 1.41x at 32,768. LoRA removes gradient and optimizer state, which do not depend on sequence length. It does not reduce activations at all -- the forward pass through the frozen base is unchanged, and backprop still traverses it to reach the adapters. A fixed saving divided by a growing total is a shrinking ratio.

The ratio decays toward 1.0 and does not invert, because activations are identical between the two. *The advantage shrinks* is the claim, not *LoRA stops being worth it*.

## Where the excluded term overtakes the counted one

| model | parameters | crossover sequence |
|---|---|---|
| `smol-135m` | 0.16B | **1,024** tokens |
| `qwen-0.5b` | 0.63B | **2,048** tokens |
| `llama-1b` | 1.50B | **4,096** tokens |
| `llama-8b` | 8.03B | **8,192** tokens |

**Smaller models reach it sooner**, which is the opposite of the intuition that fewer layers means fewer activations. The cause is the embedding table: `2 * vocab * hidden` parameters that cost weights, gradients and optimizer state while producing no per-layer activations at all. On `smol-135m` that is most of the parameter count, so its states budget is large relative to its activation footprint.

Practically: the excluded term matters most on the models people prototype with, which is exactly where a memory plan is least likely to be checked.

The attention term overtakes the per-layer term at **53,248 tokens** on `llama-8b` -- past that, activation memory is mostly the score matrix and the linear intuition stops applying.

## What fits on one 80 GiB card

| variant | longest sequence | with checkpointing |
|---|---|---|
| `full` | **does not fit** | 0 |
| `lora` | 8,192 | 131,072 |
| `qlora` | 16,384 | 131,072 |

A full fine-tune does not fit at any sequence length -- its states alone are 89.7 GiB. **Checkpointing buys more context than quantising does**: it takes LoRA from 8,192 to 131,072 tokens, against 16,384 for QLoRA without it. They compose, but the quadratic term is the expensive one and checkpointing is what removes it.

## Checkpointing

| sequence | activations retained |
|---|---|
| 2,048 | 10.4% |
| 8,192 | 9.4% |
| 32,768 | 6.7% |

It helps more as sequences grow, because the term it removes is the one growing fastest. On `llama-8b` it moves the crossover from 8,192 to 65,536 tokens.

**With checkpointing on, only the token count matters.** Removing the quadratic term leaves activations linear in `batch * sequence`, so batch 4 at 32k and batch 1 at 128k cost identical memory. The batch/sequence split becomes a throughput and convergence decision rather than a memory one -- which is not true without it, where a long sequence at equal token count costs strictly more.

The compute cost of recomputation is roughly a third more, and is not modelled here because it is not a memory quantity.

## Still not covered

The per-layer multiple is a stated approximation of how many tensors a real implementation keeps live, not a count from a profiler. It is the same for every variant, so comparisons between them are unaffected, but the absolute figures carry that uncertainty. Fragmentation and allocator overhead are still excluded, so these remain a floor rather than a target.

## Reproduce

```
uv run python scripts/generate_activations.py
```

Raw values in [`activations-raw.json`](activations-raw.json).
