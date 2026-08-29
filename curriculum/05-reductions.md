# Stage 5 — Reductions, done wrong then right

**You'll be able to:** implement a parallel sum reduction, and recognize
the addressing-pattern bug class that makes the naive version slow.

**Time:** 2–4 hours.

**Build:** [`code/05-reduction-naive`](../code/05-reduction-naive) then
[`code/06-reduction-optimized`](../code/06-reduction-optimized). Run both
at the same `n` and compare.

## Read

1. **[Mark Harris, "Optimizing Parallel Reduction in CUDA"](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)**
   (NVIDIA, PDF slide deck) - the canonical reference for this exact
   progression. It's old (pre-Fermi hardware in its benchmarks), and it goes
   several steps further than this repo's two samples - first-add-during-
   load, unrolling the last warp, warp-shuffle instructions. The divergence
   and addressing concepts it teaches are unchanged on current hardware;
   treat its specific timing numbers as historical, not something to
   reproduce.
2. **[Lei Mao, "CUDA Reduction"](https://leimao.github.io/blog/CUDA-Reduction/)**
   (2024-07) - a modern rewrite of the same progression using warp-shuffle
   instructions, benchmarked on current hardware. Read this if Harris's
   30-slide deck feels dated in its presentation even though the ideas
   hold up.

## What "done wrong" means here

`code/05-reduction-naive` uses **interleaved addressing**: at each step,
threads whose ID is a multiple of `2*s` add the element `s` away into
themselves (`if (tid % (2*s) == 0)`). It produces the correct sum. Its
problem is which threads that condition selects - a scattered subset within
each warp, not a clean split - which causes warp divergence on almost every
step. You've already met warp divergence in passing; this is where you feel
its cost directly, before Stage 7 explains the "why" in hardware terms.

`code/06-reduction-optimized` changes exactly one thing: **sequential
addressing** (`if (tid < s)`), which is true for a contiguous run of the
lowest thread IDs in the block - so within any warp, it's either
unanimously true or unanimously false. Same tree, same number of steps, no
more divergence.

## Do

Run both samples at the same `n` (the default, 2^24, is a reasonable
starting point) and compare kernel time. Then read
[Stage 7](07-occupancy-and-warp-divergence.md) - it uses this exact pair of
kernels to explain warps in hardware terms, so having run them first will
make that stage land better.

## For real code, don't hand-roll this

Both samples exist to teach the concept. Production code should use
`cub::DeviceReduce::Sum` (CUB, ships with the CUDA toolkit) or
`thrust::reduce` (Thrust, also bundled) - both are faster than a
hand-written reduction and handle edge cases these teaching samples don't
bother with.

## Checkpoint

- Both kernels use the exact same number of `__syncthreads()` calls and the
  exact same number of loop iterations. If the *amount* of work is
  identical, what specifically costs the naive version time?
- `code/05-reduction-naive` and `code/06-reduction-optimized` both use
  `atomicAdd` to combine each block's partial sum into a single output.
  What would go wrong if you instead had every block write to
  `output[blockIdx.x]` without atomics, and summed those on the CPU instead -
  and why might that actually be a reasonable design, not just a workaround?

Next: [Stage 6 — reading a profiler](06-profiling.md).
