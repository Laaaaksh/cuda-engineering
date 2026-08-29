# 04 — Tiled matmul (shared memory)

Same `C = A * B` as [`03-matmul-naive`](../03-matmul-naive), same output, same
FLOP count - but each block first cooperatively loads a `16x16` tile of `A`
and `B` into shared memory, and every thread reuses those tiles instead of
re-reading global memory for every multiply-add.

Companion to [`curriculum/04-matmul-naive-to-tiled.md`](../../curriculum/04-matmul-naive-to-tiled.md).

## Build and run

```bash
make
./matmul_tiled.out            # n = 512 by default, matches 03's default
./matmul_tiled.out 1024
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/04-matmul-tiled \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- Run this and `03-matmul-naive` at the same `n` and compare GFLOP/s. On
  real hardware, expect the tiled version to be several times faster - the
  exact multiple depends on the GPU and `n`, which is itself worth noticing:
  the naive kernel's cost is dominated by memory traffic, so the speedup
  scales with how much slower global memory is than shared memory on your
  specific GPU.
- The two `__syncthreads()` calls are not optional and not there "to be
  safe." Delete the first one and the kernel will sometimes compute wrong
  answers (a thread reads a shared-memory slot before another thread has
  written it) - and it may not fail every run, which is what makes missing
  barriers such a dangerous class of bug. Delete the second one and a fast
  thread can start overwriting the tile for the next iteration while a slow
  thread is still reading the current one.
- `TILE` is 16, chosen so `TILE*TILE = 256` threads per block - a common,
  reasonable block size. Try `TILE = 32` (1024 threads/block, the hardware
  max) and see whether it's actually faster on your GPU. It usually isn't
  the free win it looks like: bigger tiles use more shared memory and more
  registers per block, which can reduce how many blocks fit on an SM at
  once. This is a preview of the occupancy trade-off in
  [`curriculum/07-occupancy-and-warp-divergence.md`](../../curriculum/07-occupancy-and-warp-divergence.md).
- This is still far from what a production GEMM (cuBLAS, CUTLASS) achieves -
  those add register-level tiling, double buffering, and tensor-core
  instructions. This sample stops at "shared memory reuse" on purpose, to
  isolate that one idea. See
  [`curriculum/08-beyond-raw-cuda.md`](../../curriculum/08-beyond-raw-cuda.md)
  for where to go next.
