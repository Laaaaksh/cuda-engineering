# 05 — Reduction, done wrong

Sum ~16M floats on the GPU using the reduction pattern most people write the
first time: a shared-memory tree per block, with interleaved addressing
(`tid % (2*s) == 0`). It's correct - the answer matches the CPU sum. It's
also the textbook case of warp divergence.
[`06-reduction-optimized`](../06-reduction-optimized) fixes it.

Companion to [`curriculum/05-reductions.md`](../../curriculum/05-reductions.md)
and [`curriculum/07-occupancy-and-warp-divergence.md`](../../curriculum/07-occupancy-and-warp-divergence.md).

## Build and run

```bash
make
./reduction_naive.out           # n = 2^24 by default
./reduction_naive.out 1000000
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/05-reduction-naive \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- Run this and `06-reduction-optimized` at the same `n` and compare kernel
  time. Same algorithm shape, same number of steps, same total work - the
  only difference is which threads within a warp are active at each step.
- `tid % (2*s) == 0` scatters the active threads across a warp instead of
  keeping them contiguous. A warp (32 threads) that has to run both the
  "true" and "false" side of an `if` serially - because different lanes
  disagree - is diverging. This loop causes that on nearly every iteration
  for most warps.
- Profile it with Nsight Compute (`ncu ./reduction_naive.out`, needs a real
  GPU) and look at the "Warp State Statistics" section - see
  [`curriculum/06-profiling.md`](../../curriculum/06-profiling.md) for how
  to read that output at all.
