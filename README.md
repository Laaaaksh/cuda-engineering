<div align="center">

# cuda-engineering

**A sequenced path from "I can program" to "I can reason about occupancy,
memory coalescing, and profiler output" — with runnable code at every step.**

[![CI](https://github.com/Laaaaksh/cuda-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/Laaaaksh/cuda-engineering/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](code/README.md)

**[Curriculum](curriculum/README.md) • [Code samples](code/README.md) • [Resources](resources/curated-resources.md) • [Common pitfalls](resources/common-pitfalls.md) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)**

</div>

## What this is

CUDA has no shortage of material - NVIDIA's own docs, a decade of blog
posts, a few books, university lecture series, forum answers. What it
lacks is a **path**: something that tells you what order to read things in,
which of them have aged, and what to actually build to check the concept
landed. An unordered list of 200 links is the problem this repo exists to
not be.

This repo is nine sequenced stages, each with:

- **What you'll be able to do** at the end of it, stated concretely.
- **What to read or watch**, in order - a handful of things, not a pile,
  with the reasoning in [`resources/curated-resources.md`](resources/curated-resources.md).
- **A small, complete, commented CUDA program to build and run**, in
  [`code/`](code) - a first kernel, vector add, matmul done naively then
  fixed with shared-memory tiling, a reduction done the way everyone writes
  it first then the way that removes its warp divergence, and a demo where
  the cost of memory coalescing is a number you measure, not a claim you
  read.
- **Checkpoint questions** to answer before moving on.

Start at [`curriculum/README.md`](curriculum/README.md).

## What this doesn't cover

This stops at single-GPU, single-kernel fundamentals. It does not cover
multi-GPU programming, CUDA graphs in depth, tensor-core (WMMA/MMA)
programming, or low-precision training numerics. [Stage
8](curriculum/08-beyond-raw-cuda.md) covers where to go from here - Triton,
CUTLASS, torch.compile - and when reaching for one of those beats writing a
kernel by hand.

## GPU access — read this before anything else

**This repository was built and every sample in it compile-checked without
an NVIDIA GPU available.** `nvcc` doesn't need a GPU to compile, only to run
the result - so every `.cu` file here was verified with:

```bash
docker run --rm -v "$PWD:/work" -w /work/code/0N-sample-name \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

against CUDA Toolkit 12.6.2, and CI repeats this on every push. **No sample
has been run on real hardware, and the performance claims in each sample's
README (e.g. "the tiled kernel is faster," "the strided kernel is slower")
describe well-documented, widely-cited GPU behavior but are not measurements
taken in this repository.** If you run these on a GPU, a PR or issue with
your numbers is genuinely useful - see [`CONTRIBUTING.md`](CONTRIBUTING.md).
Full detail in [`code/README.md`](code/README.md#gpu-access).

If you don't have a GPU either, [Stage 0](curriculum/00-toolchain-setup.md)
covers Colab and hourly cloud rental, with current pricing checked at the
time of writing.

## Repository layout

```
curriculum/   9 sequenced stages - the path itself
code/         7 runnable, commented CUDA samples, one per concept
resources/    honest, dated curation of everything external cited above
```

## Contributing

Contributions are welcome - a wrong claim, a stale "current" statement, a
dead link, a GPU run of a sample with real numbers to report. See
[CONTRIBUTING.md](CONTRIBUTING.md). Please read
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) - please don't open
a public issue for it.

## License

MIT - see [LICENSE](LICENSE).
