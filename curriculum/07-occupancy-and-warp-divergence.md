# Stage 7 — Warps, divergence, and why occupancy isn't the goal

**You'll be able to:** explain why the grid/block model isn't the hardware,
predict when a branch will cause warp divergence, and explain why chasing
100% occupancy can make a kernel slower, not faster.

**Time:** 2–3 hours.

**Build:** run `code/05-reduction-naive` and `code/06-reduction-optimized`
at the same `n`, on a real GPU if you have one, and confirm the sequential
kernel is faster. Then explain in your own words - one paragraph - why, in
terms of warps.

## The grid/block model is not the hardware

The thread hierarchy you write (`threadIdx`, `blockIdx`, `blockDim`,
`gridDim`) is a programming convenience, not a description of physical
hardware. NVIDIA's own Programming Guide says this explicitly:

> "The use of multi-dimensional thread blocks and grids is for convenience
> only and does not affect performance."
>
> "The details of which thread blocks are scheduled to execute on which SMs
> cannot be controlled or queried by the application and no ordering
> guarantees are made by the scheduler."

([docs.nvidia.com/cuda/cuda-programming-guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html))

What actually happens on the hardware: a thread block is assigned, whole,
to one Streaming Multiprocessor (SM) - it's never split across SMs. Once
resident, the block is broken into **warps** of 32 consecutive threads, and
the warp - not the thread, not the block - is the unit the hardware actually
schedules and executes in lockstep.
([background explainer, fetched for this repo](https://medium.com/@yunjiangster/understanding-streaming-multiprocessors-sm-blocks-threads-and-warps-in-cuda-programming-7e763c7d2563))

This is why `code/05-reduction-naive` and `code/06-reduction-optimized` can
have identical algorithms, identical thread counts, identical output - and
different speed. The difference is entirely about warps.

## Warp divergence

All 32 threads in a warp execute the same instruction at the same time. If
an `if` inside a kernel sends some threads down one path and others down
another, the warp can't actually run both paths in parallel - it runs one
path with the other threads masked off, then the other path with the first
threads masked off. A warp that diverges pays for both branches, serially.
([modal.com/gpu-glossary/perf/warp-divergence](https://modal.com/gpu-glossary/perf/warp-divergence))

Look back at `code/05-reduction-naive/reduction_naive.cu`: the condition
`tid % (2*s) == 0` is true for scattered thread IDs within a warp, so most
warps run both sides of that `if` on most iterations.
`code/06-reduction-optimized/reduction_optimized.cu` changes the condition
to `tid < s` - true for a contiguous run of the lowest thread IDs - so for
any given warp, either every thread satisfies it or none do. Same algorithm
shape, same number of steps, no more divergence.

## Why "maximize occupancy" is the wrong target

Occupancy is the ratio of active warps on an SM to the maximum the hardware
supports. It's tempting to treat it as the optimization target, because
higher occupancy gives the scheduler more warps to hide memory latency
with - if one warp stalls on a load, another warp can run instead.

But occupancy is a means to hide latency, not a goal in itself, and past a
certain point more occupancy buys nothing:

- Vasily Volkov's widely-cited GTC 2010 talk, hosted by NVIDIA, showed
  measured cases where performance was *maximized* at occupancies as low as
  ~12.5% - by giving each thread more independent work and more registers
  instead of packing in more threads.
  ([nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf))
- A more recent NVIDIA developer blog post makes the same point with a
  different method: once an SM's active-cycle throughput passes roughly
  80%, the kernel is limited by instruction issue rate, not by latency -
  and pushing occupancy higher from there buys at most a few percent.
  ([developer.nvidia.com/blog/the-peak-performance-analysis-method-for-optimizing-any-gpu-workload](https://developer.nvidia.com/blog/the-peak-performance-analysis-method-for-optimizing-any-gpu-workload/))

Concretely: `code/04-matmul-tiled` uses more shared memory and more
registers per thread than the naive version. That can *lower* occupancy
(fewer blocks fit on an SM at once) while still being faster overall,
because the thing it optimized - global memory traffic - mattered more than
having every possible warp slot filled. Occupancy is one input to a
performance model, not the score.

Next: [Stage 8 — where CUDA sits now](08-beyond-raw-cuda.md).
