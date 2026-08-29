// Vector addition: c[i] = a[i] + b[i]. The "hello world" of data-parallel
// GPU programming - every element is independent, so it maps directly onto
// one thread per element. This sample introduces the full lifecycle every
// later sample repeats: allocate on host, allocate on device, copy input
// over, launch, copy result back, verify, free.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The grid is usually launched with more threads than elements (grid
    // size rounds up to a whole number of blocks), so every kernel that
    // maps threads to data needs this bounds check - without it, threads
    // past the end of the array read/write out of bounds.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20; // ~1M elements
    const size_t bytes = n * sizeof(float);

    // Host ("h_") buffers live in normal CPU memory.
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device ("d_") buffers live in GPU memory. The CPU cannot dereference
    // these pointers directly - it can only pass them to CUDA API calls and
    // to kernels.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify against a value we can check without re-running the kernel:
    // h_a[i] + h_b[i] == i + 2*i == 3*i.
    bool ok = true;
    for (int i = 0; i < n; i++) {
        float expected = 3.0f * i;
        if (std::fabs(h_c[i] - expected) > 1e-3f) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            ok = false;
            break;
        }
    }

    printf("n = %d, threads/block = %d, blocks = %d\n", n, threadsPerBlock, blocks);
    printf("Kernel time: %.4f ms\n", ms);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
