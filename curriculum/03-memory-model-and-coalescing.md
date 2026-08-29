# Stage 3 — Memory model and coalescing

**You'll be able to:** name CUDA's memory spaces and their relative speed,
explain what a coalesced access pattern is, and measure the cost of an
uncoalesced one.

**Time:** 2–3 hours.

**Build:** [`code/07-coalescing-demo`](../code/07-coalescing-demo) - run it
at several stride values and record the slowdown at each.

## Read

1. **[CUDA Programming Guide §2.3.4.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)**
   (NVIDIA, v13.3) - the current official coalescing section, with a
   matrix-transpose coalesced-vs-uncoalesced worked example and the
   32-byte-transaction model current GPUs actually use.
2. **["Unlock GPU Performance: Global Memory Access in CUDA"](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)**
   (NVIDIA Developer Blog, 2025-09-29) - modern examples on Hopper-class
   GPUs, and shows the actual Nsight Compute metric
   (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`) you'd check to confirm
   a kernel is or isn't coalescing well. This supersedes NVIDIA's older
   2013 coalescing post - that one is still online but its own banner now
   points readers to this one.
3. **[Lei Mao, "CUDA Coalesced Memory Access"](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/)**
   (2023-03, benchmarked on an RTX 3090) - a shorter, code-first
   explanation if the official docs feel too abstract on a first read.
4. **[CUDA C++ Best Practices Guide, §10.2 "Coalesced Access to Global
   Memory"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)**
   (NVIDIA, v13.3) - practical guidance on top of the conceptual model above.

## Memory spaces, roughly fastest to slowest

- **Registers** - per-thread, fastest, extremely limited.
- **Shared memory** - per-block, on-chip, roughly two orders of magnitude
  faster than global memory, and the thing [Stage 4](04-matmul-naive-to-tiled.md)
  uses to fix the naive matmul's redundant reads.
- **L1/L2 cache** - automatic, not directly addressable.
- **Global memory** - the large pool you `cudaMalloc` from. Visible to every
  thread in the grid. By far the most common bottleneck for a first kernel,
  and the subject of this stage.

## Do

Run [`code/07-coalescing-demo`](../code/07-coalescing-demo) with stride 1,
2, 4, 8, 16, 32, 64 and record the slowdown at each. Stride 1 should perform
close to the coalesced baseline - it's the same access pattern by
definition. Plot (even just in a spreadsheet) slowdown vs. stride; the shape
of that curve is a more convincing lesson than any single number.

## Checkpoint

- In your own words: what makes an access pattern "coalesced"? (Not "fast" -
  the actual condition on addresses.)
- `code/03-matmul-naive`, which you haven't optimized yet, is slow mainly
  because of *redundant* global reads (Stage 4's subject), not because any
  single read in it is badly coalesced. Look at `A[row * n + k]` and
  `B[k * n + col]` in the inner loop and think about how many *different*
  threads, across the whole kernel run, end up reading each element of `A`
  and of `B` from global memory. That redundancy - not the addressing
  pattern of one read - is what Stage 4's shared-memory tiling eliminates.



Next: [Stage 4 — naive to tiled matmul](04-matmul-naive-to-tiled.md).
