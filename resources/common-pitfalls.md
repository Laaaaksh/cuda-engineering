# The things that actually stop people

Collected in one place, so you can check here before assuming you've found
a new problem. Each of these is covered in more depth wherever the
curriculum stage links to it - this page is the index of "oh, it's this
again."

## Toolchain and versions

**"CUDA driver version is insufficient for CUDA runtime version."** You
compiled with a newer CUDA toolkit than your installed driver supports.
`nvcc --version` tells you the toolkit version; `nvidia-smi` tells you the
driver version and the maximum CUDA version it supports. Fix: update the
driver, or recompile against an older toolkit that matches your driver.
This is one of the most commonly reported CUDA errors on NVIDIA's own
developer forums — see the
[full explanation in Stage 0](../curriculum/00-toolchain-setup.md#driver-and-toolkit-versions-will-eventually-fight-you).

**Compiling for the wrong architecture.** If you hardcode `-arch=sm_86` (an
Ampere consumer card) and run on a Colab T4 (`sm_75`), you'll get a runtime
error, not a compile error — the binary was never valid for that hardware in
the first place. Every sample in `code/` in this repo compiles for a
`-gencode` list spanning `sm_70` through `sm_89` for exactly this reason -
copy that pattern rather than a single `-arch` flag.

**A driver install that silently didn't work.** After installing a driver,
`nvidia-smi` should print your GPU. If it doesn't, the install failed or a
reboot is needed — don't try to debug CUDA code before this works.

## The first profiler run

Nsight Compute's default report has dozens of sections and can look like
noise on your first kernel. It's built with this in mind: the **Guided
Analysis** feature at the top of a report auto-flags the likely bottleneck
in plain language, and hovering any metric shows a tooltip explaining what
it means. Read that summary first; don't try to absorb every table on
your first run. Full walkthrough: [Stage 6](../curriculum/06-profiling.md).

**If you're following an older tutorial that uses `nvprof`** — it was
removed starting with CUDA 13.0. The concepts transfer directly to Nsight
Systems/Compute; the commands don't. See Stage 6 for current commands.

## Misconceptions almost everyone starts with

**"The grid/block hierarchy I write describes the hardware."** It doesn't.
NVIDIA's own docs are explicit that the dimensionality of your grid and
block choice "does not affect performance" and that which SM runs which
block "cannot be controlled or queried by the application." A block is
assigned whole to one SM; once there, it's split into 32-thread warps, and
the warp - not the thread or the block - is what's actually scheduled. Full
explanation: [Stage 7](../curriculum/07-occupancy-and-warp-divergence.md).

**"An `if` statement in a kernel is basically free, like on a CPU."** Not
when it splits a warp. All 32 threads in a warp run in lockstep; if some
take one branch and some take another, the warp runs both paths serially,
masking off the threads that don't apply to each. `code/05-reduction-naive`
is built specifically to make this cost visible as a number.

**"More occupancy is always better."** Occupancy (active warps per SM,
relative to the hardware max) helps the scheduler hide memory latency by
switching to another warp when one stalls — but past a point, more
occupancy buys nothing, and a kernel can be fastest at low occupancy if each
thread has more independent work and more registers instead. Vasily
Volkov's widely-cited 2010 GTC talk demonstrated cases with peak performance
at occupancies as low as ~12.5%; a more recent NVIDIA blog post makes the
same point differently — once a kernel is issue-rate-limited (not
latency-limited), pushing occupancy further yields only marginal gains.
Full explanation, with citations: [Stage 7](../curriculum/07-occupancy-and-warp-divergence.md).

**"If it compiles and gives a plausible answer, it's correct."** CUDA has
no bounds checking. A kernel that reads or writes past a buffer's end can
appear to work by luck - the memory nearby happened to be allocated and
unused - and then break unpredictably when something changes. Use
`compute-sanitizer` (ships with the toolkit) to catch this class of bug
directly instead of hoping a wrong answer surfaces on its own; see the note
in [`code/02-vector-add`](../code/02-vector-add)'s README.

**"A missing `__syncthreads()` just makes things a bit slower."** It can
produce a data race instead — a wrong answer, and possibly one that's only
wrong sometimes, depending on scheduling that varies run to run. See the
note in [`code/04-matmul-tiled`](../code/04-matmul-tiled)'s README for a
concrete example you can break on purpose and observe.
