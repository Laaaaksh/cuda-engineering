# Stage 1 — Your first kernel

**You'll be able to:** write, launch, and time a kernel; explain the
host/device split and why every CUDA program copies data across it.

**Time:** 2–3 hours.

**Build:** [`code/01-hello-kernel`](../code/01-hello-kernel) and
[`code/02-vector-add`](../code/02-vector-add) - build and run both (or
compile-check with Docker if you don't have a GPU yet; see each sample's
README).

## Read or watch, in this order

1. **["An Even Easier Introduction to CUDA"](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)**
   — Mark Harris, NVIDIA. Originally 2017, updated 2025-05-02. The standard
   first CUDA tutorial: a CPU function turned into a kernel, one step at a
   time. Still the right starting point.
2. **[CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)**
   (NVIDIA, v13.3) if you haven't already installed anything - pairs with
   [Stage 0](00-toolchain-setup.md).
3. **[CUDA Programming Guide, §2.3 "Writing SIMT Kernels"](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)**
   (NVIDIA, v13.3, current as of Aug 2026) - the up-to-date reference for
   kernel syntax and the thread hierarchy. Use this over any tutorial's
   partial explanation, and over the *old* "CUDA C++ Programming Guide" -
   NVIDIA replaced that document for CUDA 13 and its own banner says it's no
   longer updated.
4. Prefer Python and PyTorch already? **[GPU MODE Lecture 3: "Getting
   Started with CUDA for Python Programmers"](https://www.youtube.com/@GPUMODE)**
   (Jeremy Howard) covers the same first-kernel ground from a
   `torch`-first angle.

## Do

Build and run [`code/01-hello-kernel`](../code/01-hello-kernel) first - it's
pure thread-hierarchy, no data. Then
[`code/02-vector-add`](../code/02-vector-add), which adds the full
allocate → copy → launch → copy back → verify → free lifecycle you'll repeat
in every later sample.

Both samples' READMEs have specific things to try (breaking the bounds
check, timing the memcpys) - do those, don't just read the code.

## Checkpoint

You should be able to answer, without looking anything up:

- Why does `vector_add.cu` allocate memory twice (once with `malloc`, once
  with `cudaMalloc`)?
- What does the `if (i < n)` check in each kernel protect against, and what
  actually happens if you remove it?
- Kernel launches are asynchronous - what would happen if you read the
  result from `h_c` immediately after the launch, with no `cudaMemcpy` or
  sync in between?

Next: [Stage 2 — thread hierarchy vs. hardware](02-thread-hierarchy-and-hardware.md).
