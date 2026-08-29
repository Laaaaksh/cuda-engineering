# 02 — Vector add

`c[i] = a[i] + b[i]` over ~1M elements. This is the template every later
sample follows: allocate host memory, allocate device memory, copy input
host→device, launch, copy result device→host, verify, free everything.

Companion to [`curriculum/01-first-kernel.md`](../../curriculum/01-first-kernel.md).

## Build and run

```bash
make
./vector_add.out
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/02-vector-add \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- `if (i < n)` in the kernel: the launch config rounds the block count up,
  so the grid usually has more threads than there are elements. Delete that
  check and rebuild - on this problem size it will likely still "work" by
  luck (the extra threads write past the buffer into memory CUDA happened
  to allocate nearby), which is worse than crashing: it's how out-of-bounds
  writes hide until they corrupt something that matters. Run it under
  `compute-sanitizer ./vector_add.out` (ships with the CUDA toolkit) to see
  it flagged properly.
- The two `cudaMemcpy` calls (host→device, device→host) usually cost more
  time than the addition itself. That ratio - compute is cheap, moving data
  is not - is the single most important intuition for everything that
  follows in this curriculum.
- `cudaEvent` timing brackets only the kernel, not the copies. Add timing
  around the memcpys too and compare.
