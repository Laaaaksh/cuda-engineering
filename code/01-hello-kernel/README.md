# 01 — Hello, kernel

Your first CUDA program. It launches a grid of threads and each one prints
its own `blockIdx`/`threadIdx`. There's no real computation - the point is
seeing the thread hierarchy turn into concrete numbers.

Companion to [`curriculum/01-first-kernel.md`](../../curriculum/01-first-kernel.md).

## Build and run

```bash
make
./hello_kernel.out
```

No GPU? Compile-check it with Docker (see the repo root README for why this
still catches real errors):

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/01-hello-kernel \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- The output lines are **not** in order. Run it a few times - the order
  shifts. Threads run concurrently; nothing here asked them to finish in any
  particular sequence, so nothing guarantees one.
- `global thread = blockIdx.x * blockDim.x + threadIdx.x` is the formula
  you'll use in almost every kernel you write from here on, to turn "which
  block, which thread within the block" into a single flat index.
- Change `numBlocks` and `threadsPerBlock` in `main()` and rebuild. Try
  `threadsPerBlock = 1025` - it should fail with an "invalid configuration"
  error. That's not a bug: hardware caps the threads per block (1024 on
  every architecture since Kepler), and `CUDA_CHECK_LAST()` is what surfaces
  that failure instead of silently doing nothing.
