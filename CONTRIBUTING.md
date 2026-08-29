# Contributing to cuda-engineering

Thanks for considering a contribution. This is a curriculum plus runnable
samples, open source under the MIT license.

## Getting started

```bash
git clone https://github.com/<your-username>/cuda-engineering.git   # your fork
cd cuda-engineering
```

You need an NVIDIA CUDA toolkit to build the samples. If you don't have a GPU,
you can still compile-check every sample with Docker (compilation doesn't
need a GPU, only *running* the binary does):

```bash
docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  bash -c 'for d in code/*/; do make -C "$d" clean && make -C "$d"; done'
```

If you do have an NVIDIA GPU, build and run a sample directly:

```bash
cd code/02-vector-add
make
./vector_add.out
```

## Contribution workflow

1. Fork the repo, clone your fork (command above).
2. Create a descriptively named branch off `main`.
3. Make focused commits.
4. If you touched anything under `code/`, prove it still compiles - paste the
   `nvcc` output (Docker command above, or your own toolchain) in the PR. If
   you have a GPU, also say what you ran and what it printed.
5. If you touched `resources/curated-resources.md`, open every link you're
   adding or changing and confirm it resolves before submitting. Never add a
   link you haven't personally opened.
6. Open a pull request against `main`.

A PR can merge only when CI's `Compile samples` job passes and review
feedback is resolved.

## What contributions are useful

- Fixing a wrong claim, a stale "current" statement, or a dead link.
- A new sample that isolates one concept the way the existing ones do (see
  "Adding a sample" below) - open an issue first so scope is agreed before you
  write it.
- Corrections from testing on a GPU or CUDA version this repo hasn't been
  checked against - say which GPU/driver/toolkit combination you used.
- Curriculum sequencing feedback: if a stage assumes something the previous
  stage didn't actually teach, that's a real bug in a course, not a nitpick.

## Adding a sample

Each directory under `code/` demonstrates exactly one idea. Follow the
existing pattern:

- One `.cu` file, one `Makefile`, one `README.md` explaining what the sample
  shows, what to look for in the output, and how it connects to the
  curriculum stage that references it.
- Comments in the code explain *why* a line matters for the concept being
  taught, not what CUDA API calls do (link to the docs for that).
- The Makefile should build with a single `make` and clean with `make clean`,
  matching the `-gencode` flags already used in sibling samples so CI's
  no-GPU compile check keeps working.
- Prefer extending an existing stage's "prove it stuck" exercise over adding
  a new curriculum stage - new stages change the sequencing for everyone.

## Code style

- Comments explain *why*, not *what* - the reader can read CUDA syntax, they
  need help with the parts that are non-obvious (memory hierarchy, warp
  behavior, why a synchronization is there).
- Match the error-checking pattern already used in `code/common/` rather than
  inventing a new one.

## Reporting issues

Open a GitHub issue before starting anything larger than a typo fix, so scope
is agreed first. Use the bug report template for something broken, and the
resource suggestion template for anything about `resources/curated-resources.md`.
