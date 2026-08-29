// Sum reduction, done the way most people write it the first time:
// interleaved addressing with a modulo test. It's correct. It's also the
// textbook example of warp divergence - see the comment in the kernel.
// 06-reduction-optimized fixes this with the same algorithm shape, one
// changed line.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

#define BLOCK_SIZE 256

__global__ void reduceNaive(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing: each step, threads whose tid is a multiple of
    // 2*s add the element s away into themselves. The problem is *which*
    // threads that is. A warp is 32 consecutive threads executing in
    // lockstep; `tid % (2*s) == 0` is true for scattered threads within a
    // warp, not a clean first-half/second-half split. The hardware runs
    // both the "if true" and "if false" paths for every warp that contains
    // a mix, masking off the threads that don't apply to each path - so a
    // warp with even one active and one inactive thread pays for both
    // paths serially. That's warp divergence, and this loop causes it on
    // almost every iteration.
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
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
    reduceNaive<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);

    return ok ? 0 : 1;
}
