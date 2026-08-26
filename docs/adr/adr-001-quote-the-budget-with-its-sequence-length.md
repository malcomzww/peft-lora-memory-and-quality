# ADR 001: quote the memory budget with its sequence length

## Status

Accepted.

## Context

The repo's headline is that LoRA saves **5.97x** the memory of a full
fine-tune. It is arithmetically correct and it is a short-sequence number.

Adding activations to the budget:

| sequence | LoRA vs full | activations are |
|---|---|---|
| 512 | 5.48x | 10% of the total |
| 8,192 | 2.66x | 67% |
| 32,768 | **1.41x** | **92%** |

LoRA removes gradient and optimizer state, both independent of sequence length.
It does not reduce activations — the forward pass through the frozen base is
unchanged, and backprop traverses it to reach the adapters. So a fixed saving is
divided by a growing total.

Two further findings change what advice this repo can give:

- **Smaller models cross over sooner** — 1,024 tokens on `smol-135m` against
  8,192 on `llama-8b`. The embedding table costs states while producing no
  per-layer activations, and on a small model it is most of the parameters. The
  excluded term bites hardest on the models people prototype with.
- **Checkpointing buys more context than quantisation.** LoRA goes from 8k to
  128k tokens on an 80 GiB card with checkpointing, against 16k with a
  quantised base. Only checkpointing removes the quadratic term.

A memory figure quoted without its sequence length is not wrong so much as
unfalsifiable — the reader cannot tell which regime it describes.

## Decision

Every memory figure this repo publishes carries **the sequence length and batch
size it was computed at**, and the weights-and-states tables are labelled as a
floor rather than a budget.

- **The 5.97x claim is stated as "at negligible context"** wherever it appears,
  with the 32k figure alongside it. Quoting one without the other is the failure
  this ADR exists to prevent.
- **Long-context recommendations lead with checkpointing.** Quantising the base
  is the right lever for fitting weights; it is the wrong lever for fitting
  context, and the two questions get confused because both are answered in
  gigabytes.
- **Rank remains a quality decision, not a memory one.** That conclusion is
  unaffected — a 32x rank sweep moves the total by half a gigabyte, and
  activations do not depend on rank at all, so adding them strengthens rather
  than qualifies it.

## Consequences

- The headline number gets a qualifier, which makes it less quotable and more
  true.
- The per-layer activation multiple is a stated approximation of how many
  tensors an implementation keeps live, not a profiler count. It is identical
  across variants, so comparisons between them hold; absolute figures carry that
  uncertainty and the results file says so.
- Recomputation cost is not modelled. Checkpointing is presented as a memory
  lever with a compute price named but not quantified, because this repo counts
  bytes.

## What would change this

A profiled activation trace from a real run would replace the stated multiple
with a measurement and turn these figures from a prediction into a calibration.

Modelling sequence-parallel or ring attention would change the quadratic term's
scaling entirely, which is the main reason these numbers should not be treated
as an upper bound on what is achievable at long context — only on what a
single-device, single-shard setup can do.
