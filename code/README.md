# Code samples

Seven small, self-contained CUDA programs, each isolating one idea. Every
sample is a single `.cu` file with its own `Makefile` and `README.md`
explaining what to look for.

| # | Sample | Idea |
|---|--------|------|
| [01](01-hello-kernel) | `hello_kernel.cu` | Thread hierarchy: grid, block, thread |
| [02](02-vector-add) | `vector_add.cu` | Full host↔device lifecycle; compute is cheap, data movement isn't |
| [03](03-matmul-naive) | `matmul_naive.cu` | The obvious kernel is memory-bound, not compute-bound |
| [04](04-matmul-tiled) | `matmul_tiled.cu` | Shared memory removes redundant global reads |
| [05](05-reduction-naive) | `reduction_naive.cu` | Warp divergence from interleaved addressing |
| [06](06-reduction-optimized) | `reduction_optimized.cu` | Fixing divergence with sequential addressing |
| [07](07-coalescing-demo) | `coalescing_demo.cu` | The measured cost of an uncoalesced access pattern |

Each pairs with a stage in [`curriculum/`](../curriculum) - see the
"Companion to" link at the top of each sample's README.

## Building

Every sample builds the same way:

```bash
cd code/0N-sample-name
make
./sample_name.out
```

Each `Makefile` targets compute capabilities 70/75/80/86/89 (Volta through
Ada) so the same binary runs on a V100, a Colab T4, an A100, or a consumer
30/40-series card without editing anything.

## GPU access

**These samples were written and compile-checked in an environment with no
NVIDIA GPU attached.** `nvcc` doesn't need a GPU to compile - only to run the
result - so every sample here was verified with:

```bash
docker run --rm -v "$PWD:/work" -w /work/code/0N-sample-name \
  nvidia/cuda:12.6.2-devel-ubuntu22.04 make
```

against CUDA Toolkit 12.6.2, and CI (`.github/workflows/ci.yml`) repeats this
compile check on every push and PR. **None of these binaries have been run on
real hardware or checked for correct output or the performance claims made in
each README** (e.g. "the tiled kernel is faster," "the strided kernel is
slower"). Those claims describe well-established, widely-documented GPU
behavior (see the sources in
[`resources/curated-resources.md`](../resources/curated-resources.md)), but
they are asserted, not measured, in this repository as it stands. If you run
these on a GPU, please open a PR or issue with your numbers - see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## `code/common/`

`check.cuh` is a shared header with `CUDA_CHECK`/`CUDA_CHECK_LAST` macros
that every sample uses to catch CUDA errors at the line that caused them,
instead of silently continuing with a failed allocation or kernel launch.
