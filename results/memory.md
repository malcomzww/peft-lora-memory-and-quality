# LoRA memory arithmetic

Closed-form byte counts, not measurements. Weights and gradients in bf16, Adam moments in fp32 -- which is what mixed precision actually does, and counting the moments at 2 bytes is the most common way this arithmetic is got wrong.

## The headline

For llama-8b at rank 16 on `q_proj`/`v_proj`:

| | full fine-tune | LoRA | QLoRA |
|---|---|---|---|
| weights | 14.96 GiB | 14.96 GiB | 3.74 GiB |
| gradients | 14.96 GiB | 0.01 GiB | 0.01 GiB |
| optimizer | 59.83 GiB | 0.05 GiB | 0.05 GiB |
| **total** | **89.74 GiB** | **15.02 GiB** | **3.80 GiB** |
| trainable params | 8,030.0M | 6.8M | 6.8M |

**LoRA trains 1,178x fewer parameters and saves 5.97x the memory.** Those are not the same number and they are not close. The gap is the frozen base weights, which LoRA does not touch: they are 99.6% of the LoRA budget and set a floor no rank can go below.

## Why the saving plateaus

| rank | adapter params | LoRA total | vs full |
|---|---|---|---|
| 4 | 1.7M | 14.97 GiB | 5.99x |
| 8 | 3.4M | 14.99 GiB | 5.99x |
| 16 | 6.8M | 15.02 GiB | 5.97x |
| 32 | 13.6M | 15.08 GiB | 5.95x |
| 64 | 27.3M | 15.21 GiB | 5.90x |
| 128 | 54.5M | 15.46 GiB | 5.80x |

Rank moves the total by a fraction of a gigabyte across a 32x sweep. **Choosing rank for memory reasons is choosing the wrong knob** -- pick it for quality and let the weight dtype carry the memory.

## Target modules matter more than rank

| targets | rank | adapter params | LoRA total |
|---|---|---|---|
| q,v | 16 | 6.8M | 15.02 GiB |
| q,v | 64 | 27.3M | 15.21 GiB |
| all linear | 16 | 41.9M | 15.35 GiB |
| all linear | 64 | 167.8M | 16.52 GiB |

Adapting every linear layer instead of just attention is a 6.2x larger adapter at the same rank -- a bigger change than a 4x rank increase. The MLP is where the parameters are.

## Across model sizes

| model | params | full | LoRA r=16 | QLoRA r=16 | LoRA saving |
|---|---|---|---|---|---|
| `smol-135m` | 163M | 1.82 GiB | 0.31 GiB | 0.08 GiB | 5.83x |
| `qwen-0.5b` | 630M | 7.04 GiB | 1.18 GiB | 0.30 GiB | 5.95x |
| `llama-1b` | 1,498M | 16.75 GiB | 2.81 GiB | 0.71 GiB | 5.97x |
| `llama-8b` | 8,030M | 89.74 GiB | 15.02 GiB | 3.80 GiB | 5.97x |

The saving ratio is nearly constant across three orders of magnitude of model size, because it is a ratio of dtype widths rather than of parameter counts. That is what makes it predictable on a whiteboard.

## What this does not cover

- **Activations.** They depend on batch size, sequence length and checkpointing policy, and at long sequence lengths they can exceed everything counted here. This is a weights-and-states budget, not a full training footprint.
- **Fragmentation and allocator overhead.** A real run needs headroom above these numbers; they are a floor, not a target.
- **Quality.** Nothing here says which rank trains a better model. The point is narrower: rank is not the memory knob.

## Reproduce

```
uv run python scripts/generate_results.py
```

Pure arithmetic from four integers per architecture, so the output is identical on any machine.
