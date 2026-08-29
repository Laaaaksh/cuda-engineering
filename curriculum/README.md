# The curriculum

Nine stages, in order. Each one names what you'll be able to do at the end,
what to read or watch, roughly how long it takes, and what to build to prove
it stuck. Do them in order the first time through - later stages assume
earlier ones.

| Stage | You'll be able to... | Time | Build |
|---|---|---|---|
| [0 — Toolchain setup](00-toolchain-setup.md) | Compile and run CUDA, on borrowed or owned hardware | 30–90 min | `nvcc --version` runs |
| [1 — First kernel](01-first-kernel.md) | Launch and time a kernel; explain host/device | 2–3 hr | `code/01-hello-kernel`, `code/02-vector-add` |
| [2 — Thread hierarchy](02-thread-hierarchy-and-hardware.md) | Index 1D/2D problems correctly | 1–2 hr | 2D variant of `code/01-hello-kernel` |
| [3 — Memory & coalescing](03-memory-model-and-coalescing.md) | Explain and measure coalescing | 2–3 hr | `code/07-coalescing-demo` |
| [4 — Matmul: naive → tiled](04-matmul-naive-to-tiled.md) | Explain why naive matmul is memory-bound; fix it with shared memory | 3–5 hr | `code/03-matmul-naive`, `code/04-matmul-tiled` |
| [5 — Reductions](05-reductions.md) | Implement a parallel reduction; spot addressing-pattern bugs | 2–4 hr | `code/05-reduction-naive`, `code/06-reduction-optimized` |
| [6 — Profiling](06-profiling.md) | Read Nsight Systems/Compute output | 2–4 hr | Profile the matmul pair, name the metric that changed |
| [7 — Occupancy & divergence](07-occupancy-and-warp-divergence.md) | Explain warps, divergence, and why max occupancy isn't the goal | 2–3 hr | Explain the reduction pair's speed gap in terms of warps |
| [8 — Beyond raw CUDA](08-beyond-raw-cuda.md) | Name Triton/CUTLASS/torch.compile and when to reach for each | 2–3 hr | Triton's official vector-add tutorial |

**Total: roughly 20–30 hours** of focused work, spread over however long
that takes you. There's no clock running.

## Prerequisites

You should already be comfortable writing and debugging code in some
language - C, C++, Python, whatever. You do **not** need prior GPU,
parallel programming, or computer architecture experience. Comfort with
pointers and arrays in C or C++ will make Stage 1 faster, but isn't
required going in - Stage 1 covers what you need.

## How each stage is structured

- **What to read or watch** - a short, sequenced list, not an unordered
  pile. Full annotated details (what's covered well, how current it is, and
  what's stale but still worth it) live in
  [`resources/curated-resources.md`](../resources/curated-resources.md) -
  each stage links the specific entries relevant to it.
- **What to build** - a runnable sample in [`code/`](../code), with its own
  README explaining what to look for. Building it is not optional; reading
  about warp divergence and watching it show up in a timing number on your
  own run are different things.
- **Checkpoint questions** - answer these without looking anything up
  before moving on. They're not a quiz to pass, they're a way to notice
  what didn't actually land.

## If something stops you

[`resources/common-pitfalls.md`](../resources/common-pitfalls.md) collects
the specific things that stop most beginners - toolchain version mismatches,
what the first incomprehensible profiler screen means, and the
misconceptions almost everyone starts with (thread hierarchy vs. hardware,
warp divergence, why occupancy isn't itself the goal). Check there before
assuming you've found a new problem.
