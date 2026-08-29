# 07 — Memory coalescing demo

Two kernels, same number of threads, same number of loads/stores/multiplies.
`copyCoalesced` has thread `i` touch element `i`. `copyStrided` has thread
`i` touch element `i * stride`. Nothing else differs. The timing gap between
them *is* the cost of memory coalescing - not a claim about it, a number.

Companion to [`curriculum/03-memory-model-and-coalescing.md`](../../curriculum/03-memory-model-and-coalescing.md).

## Build and run

```bash
make
./coalescing_demo.out                 # 2^20 threads, stride 32
./coalescing_demo.out 1048576 128     # custom thread count and stride
```

No GPU? Compile-check with Docker:

```bash
docker run --rm -v "$PWD/../..:/work" -w /work/code/07-coalescing-demo \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

## What to look for

- The printed "Slowdown" line: on real hardware, expect it to be
  substantially greater than 1x, and to grow as you increase `stride`
  (try 2, 4, 8, 16, 32, 64). At small strides the hardware can still merge
  some threads' addresses into the same transaction; past a point every
  thread in a warp lands in a different segment and the transaction count
  per warp maxes out.
- This is deliberately the simplest possible demonstration: a copy, not a
  useful computation. Real coalescing problems usually show up disguised -
  e.g. reading a matrix by column instead of by row, or an array-of-structs
  layout where each thread needs one field that's interleaved with fields it
  doesn't need. `03-matmul-naive`'s slowness is partly a coalescing story
  too, tangled up with the redundant-reads story `04-matmul-tiled` fixes.
- Try stride 1 (pass `1048576 1`) - it should perform close to identical to
  the coalesced kernel, because "stride 1" *is* the coalesced access
  pattern. That's a useful sanity check that the slowdown you're seeing at
  higher strides is really about the pattern, not some fixed per-kernel
  overhead.
