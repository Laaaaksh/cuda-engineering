# 06 — Reduction, done right(er)

Same sum, same input size, same shared-memory tree as
[`05-reduction-naive`](../05-reduction-naive) - but sequential addressing
(`tid < s`) instead of interleaved (`tid % (2*s) == 0`), which keeps each
warp's threads either all active or all idle at every step.

Companion to [`curriculum/05-reductions.md`](../../curriculum/05-reductions.md)
and [`curriculum/07-occupancy-and-warp-divergence.md`](../../curriculum/07-occupancy-and-warp-divergence.md).

## Build and run

```bash
make
./reduction_optimized.out           # n = 2^24 by default, matches 05's default
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/06-reduction-optimized \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- Compare kernel time against `05-reduction-naive` at the same `n`. Expect
  this one to be faster on real hardware - divergence isn't free, and this
  removes it without changing the algorithm's shape or its result.
- This still isn't the fastest possible reduction. Real implementations
  (see `cub::DeviceReduce` in NVIDIA's CUB library) also: do the first
  add during the global-memory load (so half the threads aren't wasted
  loading a value they immediately discard), unroll the last warp (skip
  `__syncthreads()` once you're down to 32 threads - a single warp is
  already lockstep), and use warp shuffle instructions (`__shfl_down_sync`)
  to avoid shared memory entirely for the last few steps. This sample stops
  at "fix the divergence" on purpose, to isolate that one idea - see Mark
  Harris's classic writeup (linked from
  [`resources/curated-resources.md`](../../resources/curated-resources.md))
  for the full progression.
- For real reductions in production code, use `cub::DeviceReduce::Sum` or
  Thrust's `thrust::reduce` - both ship with the CUDA toolkit and are
  faster than anything reasonable to hand-write. This sample and 05 exist to
  teach the concept, not to replace them.
