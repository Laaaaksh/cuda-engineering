# Stage 4 — Matrix multiply: naive, then tiled

**You'll be able to:** explain why a naive GPU matmul is memory-bound, and
implement shared-memory tiling to fix it.

**Time:** 3–5 hours - this is the first stage where the "why" takes real
sitting-with-it time, not just reading.

**Build:** [`code/03-matmul-naive`](../code/03-matmul-naive) then
[`code/04-matmul-tiled`](../code/04-matmul-tiled). Run both at the same `n`
and compare GFLOP/s.

## Read

1. **UIUC ECE408 lectures on tiling** ([lumetta.web.engr.illinois.edu/408-Sum25](https://lumetta.web.engr.illinois.edu/408-Sum25/))
   - the PMPP-derived matmul-tiling lectures walk through exactly the
     naive → shared-memory-tiled progression this stage's code follows,
     with diagrams of which elements get reused.
2. **_Programming Massively Parallel Processors_, 4th ed.**, the tiled
   matrix multiplication chapter, if you're working from the book.
3. **[Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like
   Performance"](https://siboehm.com/articles/22/CUDA-MMM)** (2022-12-31,
   benchmarked on an A6000) - don't try to implement everything in this post
   yet. Read it now for the map of *how much further* tiling can go
   (register-level tiling, vectorized loads, warp tiling) - then come back
   to it as a capstone project after Stage 7.

## Why the naive version is slow

Every thread in `code/03-matmul-naive` recomputes its output element from
scratch, reading `A`'s row and `B`'s column directly from global memory, one
element at a time, for the full inner loop. The problem isn't that any
single read is malformed - it's redundancy: many different threads, across
many different warps and blocks, read the *same* elements of `A` and `B`
over and over, each time paying full global-memory latency. For an n×n
matmul, each element of `A` and `B` is read `n` times from global memory in
the naive version.

## What tiling does

`code/04-matmul-tiled` has each block cooperatively load one tile of `A` and
one tile of `B` into shared memory, once, then every thread in the block
reuses those tiles for `TILE` multiply-adds before moving to the next tile.
Shared memory is on-chip and roughly two orders of magnitude faster to reach
than global memory - the whole gain comes from turning `n` redundant global
reads per element into `n / TILE` reads.

Read the comments in `code/04-matmul-tiled/matmul_tiled.cu` closely,
especially around the two `__syncthreads()` calls - they're barriers, not
formalities, and the sample's README explains what breaks if you remove
either one.

## Do

1. Build and run both samples at `n = 512`, then `n = 1024`, then `n = 2048`.
   Record GFLOP/s for each.
2. In `matmul_tiled.cu`, change `TILE` from 16 to 8, then to 32. Rebuild,
   rerun. Bigger isn't automatically better - you'll see why more precisely
   in Stage 7, but you can already observe it here.

## Checkpoint

- In one sentence: what does shared memory buy you that registers and
  global memory can't, for this specific problem?
- Why does the tiled kernel still need a bounds check (`row < n && col < n`)
  even though the naive one does too - what's different about *where* the
  bounds check has to apply when you're also zero-padding tiles at the
  matrix's edges?

Next: [Stage 5 — reductions](05-reductions.md).
