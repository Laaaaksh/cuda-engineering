# Stage 6 — Reading a profiler

**You'll be able to:** run Nsight Systems and Nsight Compute on a kernel,
and read enough of the output to tell whether it's memory-bound or
compute-bound, and whether warp divergence or occupancy is the likely
culprit.

**Time:** 2–4 hours.

**Build:** profile `code/03-matmul-naive` and `code/04-matmul-tiled` at the
same problem size, and write down which specific metric in the output
changed between them. "It got faster" doesn't count - name the metric.

## Why the first run is incomprehensible

Nsight Compute's default report has dozens of sections and hundreds of
metrics. Nobody reads all of it on their first kernel. The tool knows this:
it ships a rule-based **Guided Analysis** that automatically flags likely
bottlenecks in plain language, and every metric has a hover tooltip
explaining what it means
([docs.nvidia.com/nsight-compute/ProfilingGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
Start there - the "Estimated Speedup" and bottleneck callouts at the top of
a report - before trying to read the raw tables underneath.

## Two tools, two jobs

- **Nsight Systems** (`nsys`): a whole-program timeline. CPU threads, GPU
  kernels, memory copies, API calls, all on one timeline so you can see
  gaps - a kernel waiting on a copy, a CPU stall, idle GPU time between
  launches. Use this first, to find *which* kernel or which gap is worth
  digging into.
  ([developer.nvidia.com/nsight-systems/get-started](https://developer.nvidia.com/nsight-systems/get-started),
  current version 2026.4.1 at the time of writing.)
- **Nsight Compute** (`ncu`): a deep, per-kernel profiler. Once you know
  which kernel matters, `ncu` tells you why it's slow - occupancy, memory
  throughput vs. peak, warp stall reasons, instruction mix.
  ([docs.nvidia.com/nsight-compute](https://docs.nvidia.com/nsight-compute/))

The old single tool for this, `nvprof`/the Visual Profiler, is deprecated
and was removed starting with CUDA 13.0
([NVIDIA forum announcement, dated 2026-01-21](https://forums.developer.nvidia.com/t/announcement-cuda-nvprof-and-visual-profiler-are-deprecated/358159)).
If you find a tutorial that uses `nvprof`, the concepts usually still apply
but the commands won't run on a current toolkit - NVIDIA's
[Nsight Compute CLI docs](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
include a short "Nvprof Transition Guide" section pointing at further
archived material.

## A minimal workflow

```bash
# 1. Timeline first: is the GPU busy the whole time, or waiting?
nsys profile -o report ./your_program

# 2. Then drill into one kernel by name:
ncu --kernel-name matmulNaive -o report ./your_program
```

Open the `.nsys-rep` / `.ncu-rep` files in the Nsight Systems/Compute GUI
(or read the CLI summary `ncu` prints directly). None of the samples in
`code/` have been profiled on real hardware as part of building this repo -
see [`code/README.md`](../code/README.md) for why - so treat this stage's
exercise as your first real profiling session, not a "reproduce this number"
task.

## What to look for on the matmul pair

Comparing `code/03-matmul-naive` and `code/04-matmul-tiled` at the same `n`
in Nsight Compute, expect the naive kernel to show:

- Lower achieved memory throughput relative to the GPU's peak, despite
  moving redundant data - it's bottlenecked on the *pattern* of access, not
  raw bandwidth.
- A Guided Analysis callout pointing at memory as the bound, not compute.

And the tiled kernel to show markedly higher achieved compute throughput
relative to memory - the shared-memory reuse from Stage 4 shows up directly
as a shifted bottleneck, not just a smaller wall-clock number.

Next: [Stage 7 — occupancy and warp divergence](07-occupancy-and-warp-divergence.md).
