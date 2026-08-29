// Naive matrix multiply: C = A * B, all square N x N, row-major.
// One thread computes one output element. It's the obvious translation of
// the textbook triple loop onto the GPU - and it's slow, for a specific,
// measurable reason that 04-matmul-tiled fixes.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

__global__ void matmulNaive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        // Every iteration re-reads A[row][k] and B[k][col] straight from
        // global memory. Neighboring threads in the same block recompute
        // most of the same reads - e.g. every thread in this thread's row
        // reads the exact same row of A, one element at a time, from
        // scratch. Nothing here is reused: it all comes back from global
        // memory, which is ~100x slower to reach than an on-chip cache.
        // That redundant global traffic is exactly what tiling (see
        // 04-matmul-tiled) eliminates.
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmulCPU(const float *A, const float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main(int argc, char **argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 512;
    const size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    srand(0);
    for (int i = 0; i < n * n; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmulNaive<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Only verify a corner of the result at larger sizes - the CPU triple
    // loop is O(n^3) too and would dominate runtime at n=2048+.
    int checkN = std::min(n, 256);
    matmulCPU(h_A, h_B, h_C_ref, n); // still full n, so indices below are valid
    bool ok = true;
    for (int i = 0; i < checkN && ok; i++) {
        for (int j = 0; j < checkN; j++) {
            float diff = std::fabs(h_C[i * n + j] - h_C_ref[i * n + j]);
            if (diff > 1e-2f) {
                printf("Mismatch at (%d,%d): got %f, expected %f\n", i, j,
                       h_C[i * n + j], h_C_ref[i * n + j]);
                ok = false;
                break;
            }
        }
    }

    double flops = 2.0 * n * n * n;
    double gflops = flops / (ms / 1000.0) / 1e9;

    printf("n = %d\n", n);
    printf("Kernel time: %.4f ms (%.2f GFLOP/s)\n", ms, gflops);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return ok ? 0 : 1;
}
