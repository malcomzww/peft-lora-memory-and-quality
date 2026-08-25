# peft-lora-memory-and-quality

> **Not started.** This repo is scaffolded — CI, test harness and the results
> pipeline are wired, but no code has been written and there are no
> measurements. The page below is the plan, not a report. It will be rewritten
> around the result when there is one.

LoRA/QLoRA rank sweep with the memory math verified against measured VRAM, and an honest answer to when full fine-tuning wins.

**The one question this repo will answer:**

> At what rank does quality plateau, and does the memory math match reality?

## Planned method

Decision guide: rank and target-module selection table keyed to VRAM budget, and where quality plateaus.

Constraints this repo inherits from the portfolio:

- **No GPU.** 24-core CPU, 32 GB RAM, `torch.cuda.is_available()` is False.
  Anything specced for an accelerator is re-scoped to a CPU-measurable
  question or shipped with the untested path explicitly labelled.
- **No live model calls in CI.** Recorded fixtures, so the suite is free and
  deterministic.
- **Every committed number is generated** by `scripts/generate_results.py`,
  carrying its date, hardware, model revision, seed, reproduce command and raw
  artifact path. Nothing is typed by hand.
- **Committed results must be machine-independent** — ratios, orderings and
  invariants. Absolute timings go to a gitignored raw file, because CI
  regenerates results and fails on any diff.

## Concepts covered

- 1B LoRA: rank, alpha, target modules, initialization
- 1B QLoRA: NF4, double quantization, paged optimizers
- 1B DoRA/rsLoRA/LoRA+/PiSSA (one sentence each)
- 1B adapter merging and multi-LoRA serving
- 1B full-vs-LoRA memory math (TRY-E)

## Status

| | |
|---|---|
| scaffold, CI, test harness | done |
| implementation | **not started** |
| measurements | **none** |

## License

MIT
