# Stage 0 — Get a toolchain working

**You'll be able to:** compile and run a CUDA program, on your own hardware
or someone else's, and diagnose the version-mismatch error you will
eventually hit.

**Time:** 30–90 minutes, mostly waiting on installs or instance boot times.

**Build:** nothing yet - this stage ends when `nvcc --version` and (if you
have a GPU) `nvidia-smi` both run without error.

## You don't need to own a GPU

Three realistic paths, cheapest first:

### 1. Google Colab (free, easiest first step)

Colab ships the CUDA toolkit preinstalled. Two ways to compile raw `.cu`
files in a notebook:

- **Direct**, no extra install: write your file with a `%%writefile x.cu`
  cell, then compile and run with `!nvcc x.cu -o x && !./x`.
- **`nvcc4jupyter`**: `pip install nvcc4jupyter`, then `%load_ext
  nvcc4jupyter` gives you a `%%cuda` cell magic that compiles and runs the
  cell's contents directly, and also supports linking libraries and running
  Nsight profiling. ([github.com/andreinechaev/nvcc4jupyter](https://github.com/andreinechaev/nvcc4jupyter))

Set Runtime → Change runtime type → GPU before either will work.

**The gotcha that wastes the most beginner time:** Colab's free-tier GPU is
usually a T4 (compute capability 7.5). If you compile with a newer default
architecture target, or copy a `-arch` flag from a tutorial written for a
different card, you'll get a confusing runtime error rather than a compile
error. Target `sm_75` explicitly for Colab's free tier, or use the
multi-architecture `-gencode` flags this repo's own `Makefile`s use (see
[`code/README.md`](../code/README.md)) so the same binary works regardless
of which GPU Colab hands you.

**Limits, from Google's own FAQ:** free-tier sessions run "at most 12 hours,
depending on availability," and idle disconnects apply. GPU/TPU type "varies
over time" - you don't get to pick which GPU you're assigned on the free
tier. Colab Pro+ removes the idle timeout as long as code is executing and
supports up to 24-hour continuous execution with enough compute credits.
([research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html),
fetched August 2026)

Pricing for Colab Pro/Pro+ is set on Google's own sign-up flow and changes
independently of this document - check it there rather than trusting a
number written down here. At the time of writing, third-party trackers
reported Pro at $9.99/month (100 compute units) and Pro+ at $49.99/month
(500 compute units), but that was not independently confirmed against
Google's own pricing page in this repo's research and should be treated as
approximate.

### 2. Rent a GPU by the hour

For anything Colab's limits get in the way of (longer sessions, a specific
GPU, no notebook environment). Approximate on-demand rates for a single
small GPU, as found at the time of writing (all prices change constantly -
re-check before committing):

| Provider | GPU | ~$/hr | Source |
|---|---|---|---|
| [Vast.ai](https://vast.ai) | Tesla T4 | $0.12 | [computeprices.com/providers/vast](https://computeprices.com/providers/vast) |
| [RunPod](https://runpod.io) | RTX A5000 (24GB) | $0.27 | [runpod.io/pricing](https://www.runpod.io/pricing) |
| [AWS EC2](https://aws.amazon.com) | g4dn.xlarge (1× T4) | $0.53 | [instances.vantage.sh/aws/ec2/g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge) |
| [Lambda Labs](https://lambda.ai) | Quadro RTX 6000 (24GB) | $0.69 | [lambda.ai/service/gpu-cloud](https://lambda.ai/service/gpu-cloud) |

Vast.ai's marketplace model (renting other people's idle hardware) is
consistently the cheapest, at the cost of less consistency in what you get.
RunPod didn't list a T4 on its current on-demand catalog at the time of
writing - it may have moved to newer GPUs only.

Any of these gets you SSH access to a real Linux box with a real GPU; from
there, install the CUDA toolkit like you would on any Linux machine, or use
a provider-supplied image that already has it.

### 3. Your own GPU

If you have an NVIDIA GPU already, install the driver first (from
[nvidia.com/drivers](https://www.nvidia.com/drivers)), then the [CUDA
Toolkit](https://developer.nvidia.com/cuda-downloads). Order matters: the
toolkit installer can bundle a driver, but the versions have to be
compatible with each other and with your GPU's compute capability - see the
next section before you install anything.

## Driver and toolkit versions will eventually fight you

This is the single most common way people get stuck before writing any real
code, so it's worth understanding the model instead of memorizing a fix.

- **`nvcc --version`** reports the CUDA *toolkit* version - the compiler
  you're building with.
- **`nvidia-smi`** reports the *driver* version, and the maximum CUDA
  version that driver supports.
- These two numbers do not have to match exactly, but the driver has to be
  new enough for the toolkit. A newer driver can run code built with an
  older toolkit; an older driver usually cannot run code built with a newer
  toolkit.

The classic failure is `CUDA driver version is insufficient for CUDA
runtime version` - it means the toolkit you compiled with is newer than
what your installed driver supports. Fix: upgrade the driver, or compile
with an older toolkit version that matches the driver you have. (Confirmed
as a live, commonly-hit error via NVIDIA's own developer forums, e.g.
[forums.developer.nvidia.com/t/cuda-driver-version-is-insufficient-for-cuda-runtime-version/231444](https://forums.developer.nvidia.com/t/cuda-driver-version-is-insufficient-for-cuda-runtime-version/231444).)

NVIDIA documents the exact compatibility rules - two mechanisms, "Minor
Version Compatibility" (a driver can run apps built against an older minor
version within the same CUDA major-version family) and "Forward
Compatibility" (a `cuda-compat` package that lets a newer toolkit's output
run on an older base driver, with constraints) - at
[docs.nvidia.com/deploy/cuda-compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).
The architecture-to-toolkit support matrix lives at
[docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
As one concrete data point: CUDA Toolkit 13.3 requires driver ≥580 in
general, and ≥610.43.02 specifically on Linux x86_64 - check the [release
notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
for whatever version you're installing, because these minimums move with
every release.

One more current gotcha, specific to Windows: starting with CUDA 13.1, the
Windows toolkit installer no longer bundles a display driver - you have to
install NVIDIA's driver separately first.

## Checkpoint

You're done with this stage when:

```bash
nvcc --version      # prints a CUDA compilation tools version
nvidia-smi           # prints your GPU and driver version (skip if using Colab)
```

both succeed. Next: [Stage 1 — your first kernel](01-first-kernel.md).
