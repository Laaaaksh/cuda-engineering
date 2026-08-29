# Stage 8 — Where CUDA sits now

**You'll be able to:** name the current higher-level tools (Triton,
CUTLASS, torch.compile) and their tradeoffs, and make a reasonable call
about when to hand-write a kernel versus reach for one of them.

**Time:** 2–3 hours, mostly reading and running one Triton tutorial.

**Build:** work through Triton's official "Vector Add" tutorial and compare
the code to `code/02-vector-add`. Same problem, note what the compiler is
doing for you that you did by hand.

## The honest framing

Everything in this repo so far is how GEMM (matrix multiply), reductions,
and memory access patterns actually work at the hardware level. That
understanding doesn't go away once you use a higher-level tool - it's what
lets you read that tool's generated code, diagnose why it's slow, and know
when it's the wrong tool. But almost nobody hand-writes production GEMM
kernels from scratch anymore. Here's what most people reach for instead, and
when raw CUDA is still the right call.

## The stack, roughly bottom to top

- **cuBLAS / cuDNN** - NVIDIA's own hand-tuned libraries for standard linear
  algebra and deep learning primitives. If your problem is a standard
  operation (a GEMM, a convolution), these are usually within a few percent
  of hand-tuned CUTLASS with none of the integration work. Don't hand-roll
  what's already in here.
  ([developer.nvidia.com/cublas](https://developer.nvidia.com/cublas),
  [developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads))

- **CUTLASS / CuTe** - NVIDIA's open-source, templated building blocks for
  writing GEMM-shaped kernels when cuBLAS's fixed operations don't fit your
  case, but you still want NVIDIA's tiling, pipelining, and tensor-core
  strategies rather than reinventing them. `code/04-matmul-tiled` in this
  repo is the single idea (shared-memory tiling) that CUTLASS generalizes
  into a full, composable hierarchy - register tiling, double buffering,
  tensor-core instructions, and more. Actively developed: version 4.8.0 as
  of August 2026, with CUTLASS 4 adding a Python-native DSL (`CuTe DSL`, in
  public beta) for writing these kernels without dropping to C++ templates.
  ([github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass),
  [docs.nvidia.com/cutlass/latest/overview.html](https://docs.nvidia.com/cutlass/latest/overview.html))

- **Triton** - a Python-embedded language and compiler for writing custom
  GPU kernels at close to hand-tuned performance with far less code. You
  write tile-level operations; the compiler handles a lot of what
  `code/04-matmul-tiled` did by hand in this repo - memory coalescing,
  scheduling, some of the tiling strategy. It's the default backend
  `torch.compile` generates code through. Actively developed (commits as
  recently as the day before this was written); version 3.7.1 as of June
  2026. Start with the official tutorial series: Vector Add → Fused Softmax
  → Matrix Multiplication.
  ([triton-lang.org/main/getting-started/tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html),
  [github.com/triton-lang/triton](https://github.com/triton-lang/triton))

- **torch.compile / TorchInductor** - if you're writing PyTorch code, this
  is usually the first thing to reach for before writing any custom kernel
  at all: it traces your model and generates fused Triton kernels
  automatically. PyTorch 2.13 (mid-2026) added an alternative CuTeDSL
  codegen backend for GEMM/RMSNorm specifically, on top of the existing
  Triton path - a direct sign of the CUTLASS and Triton stories converging.
  ([docs.pytorch.org/docs/main/user_guide/torch_compiler](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html))

- **Notable others**: **ThunderKittens** (Stanford Hazy Research's compact
  CUDA framework for hand-writing fast attention-style kernels, v2.0 as of
  January 2026, used in production at several AI infra companies) and
  **Mojo** (Modular's systems language for heterogeneous hardware, reached
  1.0 and went open-source under Apache 2.0 in August 2026) are both worth
  knowing about if you go deeper into kernel authoring, though neither is a
  prerequisite for this curriculum.
  ([hazyresearch.stanford.edu](https://hazyresearch.stanford.edu),
  [docs.modular.com/mojo](https://docs.modular.com/mojo))

## When to still write raw CUDA

The people making this argument in public are worth reading directly rather
than taking a curriculum's word for it:

- NVIDIA's own framing puts Triton between cuDNN (least control, most
  optimized) and CUTLASS (most control, least automatic) - reach for raw
  CUDA/CUTLASS when you need control Triton's abstractions don't expose.
  ([developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/),
  2025-02-05)
- Modular's own kernel-optimization guidance is blunt about the order to try
  things in: vendor libraries first, then a compiler stack, then Triton for
  custom-but-not-extreme cases, and raw CUDA only once a specific bottleneck
  demands hardware-level control that nothing else exposes. "Most teams
  shouldn't start at the bottom of the stack."
  ([handbook.modular.com/kernel-optimization/kernel-optimization-tools](https://handbook.modular.com/kernel-optimization/kernel-optimization-tools))
- Chris Lattner's argument for the other side: at hyperscaler cost and
  scale, the performance gap Triton trades away for productivity can be "the
  difference between a $1B and $800M cloud bill" - i.e. the case for raw
  CUDA/CUTLASS is strongest exactly where margins are counted in
  percentages of a huge number.
  ([modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls),
  2025-03-26)

**The practical rule this curriculum leaves you with:** default to a
library (cuBLAS/cuDNN) or a compiler (torch.compile). Reach for Triton when
you need a custom fused kernel and want to stay in Python. Reach for CUTLASS
or raw CUDA when you've profiled (Stage 6) a specific bottleneck that needs
control none of the above expose, or when you're doing this to actually
learn the hardware - which, if you've made it this far, you now can.

## What this repo doesn't cover

This curriculum stops at single-GPU, single-kernel fundamentals. It does
not cover multi-GPU programming (NCCL, NVSHMEM), CUDA graphs, tensor-core
programming (WMMA/MMA instructions) in any depth, or the numerics of
low-precision training. Those are real next steps, not gaps in what's
written above - a person who's done all eight stages here has the
foundation to go learn any of them from official docs directly.
