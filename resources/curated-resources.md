# Curated resources

Every entry here was actually opened and checked at the time of writing
(August 2026) — not recalled from memory. Dates are the resource's own
stated publish/update date where one exists. Where something has aged, that
is said plainly, along with what to read instead or alongside it. Curriculum
stages link the specific entries relevant to them; this page is the full,
browsable list with the reasoning behind each recommendation.

If a link here breaks, or you find something better, please open an issue
using the "Resource suggestion" template — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Official NVIDIA documentation

| Resource | What it covers | Current as of | Verdict |
|---|---|---|---|
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | The current kernel-writing reference — thread hierarchy, memory model, coalescing (§2.3.4.1) | v13.3, updated 2026-08-27 | **Use this one.** NVIDIA restructured docs for CUDA 13; this replaced the old "CUDA C++ Programming Guide." |
| ~~CUDA C++ Programming Guide~~ | The old reference most search results still point to | Superseded | **Don't use as primary.** Its own banner states it "has been replaced... no longer being updated as of CUDA 13.0." Still online for legacy-project reference only. |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Practical tuning guidance — coalescing (§10.2), occupancy (§11.1) | v13.3 | Current. Pair with the Programming Guide rather than reading in place of it — this is the "now that you know the concept, here's the tuning checklist" doc. |
| [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) | Install and verify a toolchain | v13.3, updated 2026-06-25 | Current. Start here for [Stage 0](../curriculum/00-toolchain-setup.md). |
| [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) | Driver/toolkit version compatibility model | Current | The authoritative source for why version-mismatch errors happen. See [common-pitfalls.md](common-pitfalls.md). |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | How to read profiler output, metric reference | v2026.2.1 | Current, and the right place to actually learn to read a report — includes the Guided Analysis rule engine and per-metric tooltips built for exactly the beginner-incomprehensibility problem. |
| [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) | Whole-program timeline profiling | v2026.4.1 | Current. Use before Nsight Compute, to find *which* kernel to drill into. |
| ~~nvprof / Visual Profiler~~ | The old profiling tool | **Removed in CUDA 13.0** | Dead. [NVIDIA's own deprecation announcement](https://forums.developer.nvidia.com/t/announcement-cuda-nvprof-and-visual-profiler-are-deprecated/358159) (2026-01-21) confirms removal. If a tutorial uses `nvprof`, the concepts likely still apply but the commands won't run — NVIDIA's [Nsight Compute CLI docs](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) include a "Nvprof Transition Guide" section (though it mostly just points further to archived docs). |
| ~~CUDA Occupancy Calculator (Excel)~~ | The old spreadsheet tool for computing occupancy by hand | Deprecated | Dead as a workflow. Its [archived page](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-occupancy-calculator/) now points to Nsight Compute's built-in occupancy calculator instead. |
| [NVIDIA CUDA Refresher blog series](https://developer.nvidia.com/blog/tag/cuda-refresher/) | "What is a GPU," "what is a thread block," etc. | 2020, A100-era | Aged. Fine as a very first on-ramp before Stage 1; doesn't cover coalescing or occupancy, and its performance framing predates several architecture generations. |

## Books

| Resource | What it covers | Current as of | Verdict |
|---|---|---|---|
| [_Programming Massively Parallel Processors_, 4th ed.](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Hwu, Kirk, El Hajj (Morgan Kaufmann) | The standard textbook: data-parallel execution, memory hierarchy, tiling, occupancy | Published 2022-05-28 | **Still the standard recommendation.** Confirmed as the named textbook of the actively-maintained [GPU MODE lecture series](https://github.com/gpu-mode/lectures) (pushed as recently as 2026-06-15). Its gap: written before recent Hopper/Blackwell-era tooling and workflows — fill that with the current blog posts and Nsight docs linked above/below, which this curriculum leans on for anything profiler-facing. |
| _Programming Massively Parallel Processors_, 5th ed. | Announced update — filtering, wavefront parallelism, LLM-relevant kernel patterns, cooperative groups, NCCL/NVSHMEM | Publisher pages state 2026-06-03 | **Not yet available at the time of writing** (two independent publisher pages agree on this date, which is still ahead of when this was written). Worth checking for again rather than building around yet. |
| _CUDA by Example_ — Sanders & Kandrot (Addison-Wesley, 2010) | An early, gentle introduction | 2010 | **Aged.** Its texture-reference-heavy chapters are obsolete under current CUDA. Optional as a very early primer only if the official docs feel too dense on day one — don't use it past Stage 1. No link given here since no specific URL for it was verified while researching this repo. |
| _Professional CUDA C Programming_ — Cheng, Grossman, McKercher (Wiley, 2014) | Streams, concurrency, deeper C++ integration | 2014 | **Aged** — predates Volta, Ampere, Hopper, and current Nsight tooling entirely. Its streams/concurrency chapter is the only part worth reading today, and only as a supplement, not a primary text. |
| _CUDA for Deep Learning_ — Elliot Arledge (Manning) | Flash Attention, PyTorch C++ extensions, Ampere/Hopper/Blackwell, Nsight Compute workflows | In MEAP since Jan 2026, estimated full release Oct 2026 | **Not finished yet.** The most hardware-current book found in this research, but incomplete at the time of writing — worth tracking for a future "applied/LLM kernels" module beyond what this curriculum covers. |

## University courses (free materials)

| Resource | What it covers | Current as of | Verdict |
|---|---|---|---|
| [UIUC ECE408 / CS483 — Applied Parallel Programming](https://lumetta.web.engr.illinois.edu/408-Sum25/) (Steve Lumetta, built on the PMPP curriculum) | Full semester: indexing, tiling, memory, reductions, with lecture recordings | Summer 2025 offering, actively maintained (2023 instructor additions confirmed on top of base material) | **The backbone course this curriculum leans on most** for structured sequencing. 25 lecture PDFs and 16 public recordings, free. Note: the old `courses.engr.illinois.edu/ece408` URL is dead — use the link above. |
| [Oxford "Programming in CUDA"](https://people.maths.ox.ac.uk/gilesm/cuda/) (Mike Giles) | Warp shuffles, reductions, profiling-adjacent practicals | Actively run; next session 2026-07-20 to 2026-07-24, slides refreshed for Ampere/Hopper/Blackwell | Current and free. Good second course for depth on warp-level primitives; Practical 3 uses Nsight Systems directly. |
| [CMU 15-418/618 — Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/~418/) | Architectural "why" behind GPU parallelism, as part of a broader systems course | Live Spring 2026 offering; slides public, videos gated to CMU students | Current, but GPU/CUDA is only a two-lecture module in a much broader course — treat as later architectural context, not a first course. |
| [ETH Zurich GPU training](https://github.com/eth-cscs/gpu-training) | Hands-on CUDA and OpenACC exercises | Live public repo | Useful hands-on supplement alongside the courses above. |
| Georgia Tech CS 7295 (GPU Hardware and Software) | Full graduate GPU course | Spring 2026 syllabus public | **Not actually accessible** — it's a paid OMSCS course; materials live behind Canvas for enrolled students. Listed only so you know it exists, not as something you can follow along with. |
| ~~Stanford CS193G~~ | An early, well-known CUDA course | 2010-era | **Dead as a live course.** Only a 2010 GitHub mirror survives; treat any code in it as historical, not current CUDA. |
| ~~University of Sheffield COM4521~~ | A CUDA course | — | **Dead link.** The official page 404s; only a stale 2020 student mirror exists. Not worth chasing. |

## Blog and tutorial series

| Resource | What it covers | Date | Verdict |
|---|---|---|---|
| [Mark Harris, "An Even Easier Introduction to CUDA"](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) | First kernel, from a CPU function | Originally 2017, updated 2025-05-02 | **Still the right starting point.** Kept current by NVIDIA's own updates. |
| [NVIDIA Developer Blog, "Unlock GPU Performance: Global Memory Access in CUDA"](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/) | Coalescing with modern (Hopper-class) examples and real Nsight Compute metric names | 2025-09-29 | Current. Supersedes NVIDIA's older 2013 coalescing post, which now banners readers to this one directly. |
| [Lei Mao, "CUDA Coalesced Memory Access"](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) | A shorter, code-first coalescing explanation | 2023-03, RTX 3090-benchmarked | Current and a good second pass if the official docs feel abstract. |
| [Lei Mao, "CUDA Reduction"](https://leimao.github.io/blog/CUDA-Reduction/) | A modern warp-shuffle reduction, rewritten from Harris's classic deck | 2024-07 | Current, and a good next step after this repo's two reduction samples. |
| [Mark Harris, "Optimizing Parallel Reduction in CUDA"](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | The canonical naive→optimized reduction progression | Undated PDF, pre-Fermi benchmarks | **Concepts hold up; numbers don't.** This is the source this curriculum's reduction samples are patterned after. Its divergence and addressing arguments are architecture-independent; its specific timing figures are from hardware over a decade old — read it for the reasoning, not the benchmark table. |
| [Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"](https://siboehm.com/articles/22/CUDA-MMM) | Naive → tiled → register-tiled → vectorized matmul, benchmarked on an A6000 | 2022-12-31 | Current and excellent. This repo's matmul samples stop at "shared-memory tiling"; this post is the natural next several steps, and a good capstone project after [Stage 7](../curriculum/07-occupancy-and-warp-divergence.md). |
| [Aleksa Gordić, "Inside NVIDIA GPUs: Anatomy of high-performance matmul kernels"](https://aleksagordic.com/blog/matmul) | PTX/SASS-level matmul analysis on H100, real Nsight Compute wave-quantization output | 2025-09-29 | Current and advanced — read this only after Boehm's post, if you want to go past what this curriculum covers into instruction-level tuning. |
| [Modal GPU Glossary](https://modal.com/gpu-glossary/) | Reference definitions: warp divergence, occupancy, and more | Current, references Hopper/Blackwell | Good as a glossary to check a term against, not meant to be read start to end. |

## Video

| Resource | What it covers | Date | Verdict |
|---|---|---|---|
| [GPU MODE](https://www.youtube.com/@GPUMODE) (formerly CUDA MODE) — full lecture series | Wide-ranging, from first kernels to profiling to matmul optimization, mostly PyTorch-adjacent | Ongoing since 2024, still active (2026 uploads confirmed) | The most actively maintained current video series on this material. Two standouts for this curriculum specifically: **Lecture 8, "CUDA Performance Checklist"** (Mark Saroufim, ~2024-09-11) covers coalescing, occupancy, and Nsight Compute together across worked case studies; **Lecture 44, "NVIDIA Profiling"**, presented by NVIDIA's own Nsight team, is the best "how to read the output" walkthrough found in this research. |
| [freeCodeCamp, "CUDA Programming Course"](https://www.youtube.com/watch?v=86FAWCzIe_4) (Elliot Arledge) | ~12-hour zero-to-matmul-optimization course, with a [companion repo](https://github.com/Infatoshi/cuda-course) | Published 2024-09-24 | Current and a solid full-course alternative if you prefer one long video to the stage-by-stage reading list above. |
| GTC sessions — e.g. "CUDA Techniques to Maximize Compute/Instruction Throughput" (GTC25), "Profiling Python and AI Workloads with Nsight Compute" (GTC26) | Conference-depth talks on throughput tuning and profiling | 2025 and 2026 sessions confirmed scheduled/listed on NVIDIA's on-demand platform | Current, but session abstracts are behind a JS-rendered catalog that couldn't be fully read in this research — worth browsing NVIDIA's GTC on-demand library directly for the current year's sessions on these topics rather than trusting a fixed list here. |

## Suggested reading order

This is denser than the curriculum's per-stage lists — use those first;
come back here when you want the full picture or a next step beyond what a
stage asks for.

1. CUDA Quick Start Guide → "An Even Easier Introduction to CUDA" (or GPU
   MODE's Python-first lectures if you're coming from PyTorch).
2. UIUC ECE408 slides as the structured backbone, alongside the current
   CUDA Programming Guide for syntax reference.
3. PMPP 4th edition, chapters on the memory hierarchy and tiling, alongside
   the Best Practices Guide's coalescing and occupancy sections.
4. The Modal GPU Glossary plus Lei Mao's coalescing and reduction posts and
   the 2025 "Unlock GPU Performance" post, for current worked examples.
5. GPU MODE Lecture 8, to tie coalescing, occupancy, and a first real Nsight
   Compute session together.
6. The Nsight Compute Profiling Guide plus GPU MODE Lecture 44, as a
   dedicated pass at reading profiler output.
7. Simon Boehm's matmul post, as a capstone project.
8. Oxford's course or CMU 15-418, for architectural depth beyond this
   curriculum's scope.

## What's covered in the curriculum stages instead

The higher-level ecosystem (Triton, CUTLASS, torch.compile, and where raw
CUDA still earns its place) is covered with full citations in
[`curriculum/08-beyond-raw-cuda.md`](../curriculum/08-beyond-raw-cuda.md)
rather than duplicated here.
