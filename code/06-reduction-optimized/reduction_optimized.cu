// Same sum reduction as 05-reduction-naive, same algorithm shape (a
// shared-memory tree, halving the active set each step) - but with
// sequential addressing instead of interleaved, which removes the warp
// divergence that kernel had. Compare the two kernels: the only change is
// which threads are "active" at each step and how they compute their
// partner index.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

#define BLOCK_SIZE 256

__global__ void reduceOptimized(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Sequential addressing: s starts at half the block and halves each
    // step. `tid < s` is true for a contiguous block of the *lowest*
    // thread IDs - and thread IDs within a warp are consecutive, so for
    // any warp, either all 32 threads satisfy `tid < s` or none do, until s
    // drops below 32. That means no warp splits down two paths: every warp
    // is either fully active or fully idle at each step, so there's no
    // divergence to pay for (once s < 32 you're down to a single warp
    // total, which is unavoidable - the tree only has one node left to
    // combine).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main(int argc, char **argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    const size_t bytes = (size_t)n * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    srand(0);
    double cpuSum = 0.0;
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        cpuSum += h_input[i];
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    reduceOptimized<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float gpuSum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpuSum, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    double relError = std::fabs(gpuSum - cpuSum) / cpuSum;
    bool ok = relError < 1e-3;

    printf("n = %d, blocks = %d, block size = %d\n", n, blocks, BLOCK_SIZE);
    printf("Kernel time: %.4f ms\n", ms);
    printf("GPU sum = %f, CPU sum = %f, relative error = %e\n", gpuSum, cpuSum, relError);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");
    printf("Compare against code/05-reduction-naive at the same n - same math, ");
    printf("only the addressing pattern changed.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);

    return ok ? 0 : 1;
}
