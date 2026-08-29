// First CUDA program: launch a grid of threads and have each one print who
// it is. The point isn't the printf - it's seeing the two-level thread
// hierarchy (grid of blocks, block of threads) turn into actual numbers.
#include <cstdio>

#include "../common/check.cuh"

// __global__ marks a function that runs on the GPU ("device") and is called
// from the CPU ("host"). This is a *kernel*: when you launch it, every
// thread in the grid runs this same function body, distinguished only by
// the built-in threadIdx/blockIdx/blockDim variables below.
__global__ void helloFromGPU() {
    // blockIdx: which block this thread's block is, within the grid.
    // threadIdx: which thread this is, within its block.
    // blockDim: how many threads are in each block (same for every block
    // here, since we launched a 1-D grid of equal-sized blocks).
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from block %d, thread %d (global thread %d)\n", blockIdx.x,
           threadIdx.x, globalId);
}

int main() {
    const int numBlocks = 2;
    const int threadsPerBlock = 4;

    printf("Launching %d blocks of %d threads each (%d threads total)\n\n",
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);

    // Kernel launch syntax: <<<gridDim, blockDim>>>. This one launches
    // numBlocks blocks, each with threadsPerBlock threads. Launches are
    // asynchronous - control returns to the CPU immediately - so we
    // explicitly wait for the GPU to finish before the program exits.
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();

    // Output order across threads is NOT guaranteed. That's not a bug in
    // this program - it's the first lesson: threads run concurrently and
    // you don't get to assume an order unless you synchronize for it.
    CUDA_CHECK_LAST();

    return 0;
}
