# Stage 2 — Thread hierarchy, and a first look at what's underneath

**You'll be able to:** index 1D and 2D problems correctly with
`blockIdx`/`threadIdx`/`blockDim`, and describe - at a first-pass level -
how the grid/block model you write maps onto SMs and warps on real
hardware.

**Time:** 1–2 hours.

**Build:** nothing new to build yet - this stage is the conceptual bridge
between Stage 1's 1D vector add and Stage 4's 2D matrix multiply. Re-read
[`code/01-hello-kernel`](../code/01-hello-kernel) and predict, on paper,
what it would print with a 2D launch configuration before you try it.

## Read

1. **[CUDA Programming Guide §2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)**
   (NVIDIA, v13.3) again, specifically the parts on multi-dimensional grids
   and blocks - `dim3`, and how `blockIdx.y`/`threadIdx.y` extend the same
   formula from Stage 1 into two dimensions.
2. **[UIUC ECE408 lecture slides](https://lumetta.web.engr.illinois.edu/408-Sum25/)**
   (Steve Lumetta, built on the PMPP curriculum, Summer 2025 offering) -
   the early lectures walk through 1D/2D indexing with worked diagrams.
   Free, public, and this is the backbone course this curriculum leans on
   most for structured sequencing.
3. If you want a book alongside the free material: **_Programming Massively
   Parallel Processors_, 4th edition** (Hwu, Kirk, El Hajj; Morgan
   Kaufmann, published 2022-05-28) - chapters on data-parallel execution and
   the memory/thread model. It's still the standard textbook recommendation
   for this material; its weak spot is that it predates recent Hopper/
   Blackwell-era tooling, which is why this curriculum leans on current blog
   posts and NVIDIA docs for the profiler-facing content in later stages.

## The one-paragraph version of what's underneath

The `blockIdx`/`threadIdx`/`gridDim`/`blockDim` model is a programmer
convenience - NVIDIA's own docs say explicitly that the dimensionality you
choose "does not affect performance." What actually happens: your grid gets
split into blocks, each block gets assigned whole to one Streaming
Multiprocessor (SM), and each block then gets split into 32-thread **warps**,
which are what the hardware actually schedules. You don't need the full
depth of this yet - that's [Stage 7](07-occupancy-and-warp-divergence.md),
after you've felt the performance consequences firsthand in Stages 4 and 5.
For now, the useful takeaway is narrower: **your choice of block size is a
real decision, not a formality** - it determines how work is grouped into
the warps the hardware schedules.

## Do

Take `code/01-hello-kernel/hello_kernel.cu` and modify it to use a 2D grid
and block (`dim3 blocks(2, 2); dim3 threads(4, 4);`), printing `blockIdx.x`,
`blockIdx.y`, `threadIdx.x`, `threadIdx.y`, and a flattened global index
computed as:

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Predict the range of `row` and `col` before you run it, then check.

## Checkpoint

- Given `dim3 blocks(4, 1)` and `dim3 threads(8, 1)`, what's the global
  thread index formula, and how many threads run total?
- Why does the 2D indexing formula for matrix problems use `row`/`col`
  computed from `blockIdx.y`/`blockDim.y` and `blockIdx.x`/`blockDim.x`
  respectively, rather than the other way around? (Hint: which one usually
  matches how the matrix is laid out in memory - see
  [Stage 3](03-memory-model-and-coalescing.md).)

Next: [Stage 3 — memory model and coalescing](03-memory-model-and-coalescing.md).
