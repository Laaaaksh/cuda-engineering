# 03 — Naive matmul

`C = A * B`, square N×N, one thread per output element. This is what you'd
write if you translated the textbook triple loop straight onto the GPU. It's
correct and it's slow - not because the GPU is slow, but because of how it
reads memory. [`04-matmul-tiled`](../04-matmul-tiled) fixes exactly this.

Companion to [`curriculum/04-matmul-naive-to-tiled.md`](../../curriculum/04-matmul-naive-to-tiled.md).

## Build and run

```bash
make
./matmul_naive.out           # n = 512 by default
./matmul_naive.out 1024      # or pass n explicitly
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/03-matmul-naive \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- Note the GFLOP/s the program prints. Run [`04-matmul-tiled`](../04-matmul-tiled)
  at the same `n` on the same GPU and compare. The gap is the whole lesson -
  identical math, identical FLOP count, radically different memory traffic.
- Every thread in a warp computing one row reads the *same* row of `A`
  independently, one `float` at a time, straight from global memory. That
  redundancy - not the arithmetic - is the bottleneck. `nvprof`/Nsight
  Compute would show this kernel is memory-bound, not compute-bound, despite
  doing real floating-point work.
- Try n=2048 or n=4096. The naive kernel's relative slowness gets worse as n
  grows, because the redundant global reads scale with it.
