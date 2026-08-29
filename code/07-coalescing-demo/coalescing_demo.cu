// Same amount of work, same number of threads, same total bytes touched -
// the only difference between the two kernels below is whether consecutive
// threads read consecutive memory addresses. That's memory coalescing, and
// this sample exists to make its cost visible as a number, not just a
// claim.
//
// A warp issues one memory request per instruction, for all 32 threads at
// once. The memory controller services that request in fixed-size segments
// (32 or 128 bytes depending on architecture and access type). If the 32
// threads' addresses fall inside one or two of those segments, the warp's
// request costs one or two transactions. If the 32 threads' addresses are
// scattered across memory, the same logical request can cost up to 32
// separate transactions - the hardware does the same total useful work but
// pays for it in far more trips to DRAM.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

// coalesced: thread i touches element i. Consecutive threads -> consecutive
// addresses -> one memory transaction covers an entire warp's request.
__global__ void copyCoalesced(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

// strided: thread i touches element i * stride. Consecutive threads now
// land `stride` elements apart, so a 32-thread warp's request spans up to
// 32 separate cache lines instead of one contiguous run.
__global__ void copyStrided(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long idx = (long)i * stride;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

float timeCoalesced(const float *d_in, float *d_out, int n) {
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    copyCoalesced<<<blocks, threadsPerBlock>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

float timeStrided(const float *d_in, float *d_out, int n, int stride, int numThreads) {
    const int threadsPerBlock = 256;
    const int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    copyStrided<<<blocks, threadsPerBlock>>>(d_in, d_out, n, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main(int argc, char **argv) {
    // Both kernels touch exactly numThreads elements. The coalesced kernel
    // uses an array of exactly that size; the strided kernel uses an array
    // `stride` times larger so its numThreads touches land `stride` apart.
    const int numThreads = (argc > 1) ? std::atoi(argv[1]) : (1 << 20); // ~1M
    const int stride = (argc > 2) ? std::atoi(argv[2]) : 32;

    const int nStrided = numThreads * stride;
    const size_t bytesCoalesced = (size_t)numThreads * sizeof(float);
    const size_t bytesStrided = (size_t)nStrided * sizeof(float);

    float *h_in = (float *)malloc(bytesStrided);
    for (int i = 0; i < nStrided; i++) {
        h_in[i] = static_cast<float>(i % 1000);
    }

    float *d_in_c, *d_out_c;
    CUDA_CHECK(cudaMalloc(&d_in_c, bytesCoalesced));
    CUDA_CHECK(cudaMalloc(&d_out_c, bytesCoalesced));
    CUDA_CHECK(cudaMemcpy(d_in_c, h_in, bytesCoalesced, cudaMemcpyHostToDevice));

    float *d_in_s, *d_out_s;
    CUDA_CHECK(cudaMalloc(&d_in_s, bytesStrided));
    CUDA_CHECK(cudaMalloc(&d_out_s, bytesStrided));
    CUDA_CHECK(cudaMemset(d_out_s, 0, bytesStrided));
    CUDA_CHECK(cudaMemcpy(d_in_s, h_in, bytesStrided, cudaMemcpyHostToDevice));

    // Warm-up run of each kernel so the timed run doesn't include one-time
    // context/driver costs.
    timeCoalesced(d_in_c, d_out_c, numThreads);
    timeStrided(d_in_s, d_out_s, nStrided, stride, numThreads);

    float msCoalesced = timeCoalesced(d_in_c, d_out_c, numThreads);
    float msStrided = timeStrided(d_in_s, d_out_s, nStrided, stride, numThreads);

    // Verify both against the CPU. The strided kernel only writes elements
    // at multiples of `stride`; check those.
    float *h_out_c = (float *)malloc(bytesCoalesced);
    CUDA_CHECK(cudaMemcpy(h_out_c, d_out_c, bytesCoalesced, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < numThreads; i++) {
        if (std::fabs(h_out_c[i] - h_in[i] * 2.0f) > 1e-3f) {
            ok = false;
            break;
        }
    }

    float *h_out_s = (float *)malloc(bytesStrided);
    CUDA_CHECK(cudaMemcpy(h_out_s, d_out_s, bytesStrided, cudaMemcpyDeviceToHost));
    for (long t = 0; t < numThreads && ok; t++) {
        long idx = t * stride;
        if (idx >= nStrided) break;
        if (std::fabs(h_out_s[idx] - h_in[idx] * 2.0f) > 1e-3f) {
            ok = false;
            break;
        }
    }

    printf("threads = %d, stride = %d\n", numThreads, stride);
    printf("Coalesced kernel: %.4f ms\n", msCoalesced);
    printf("Strided kernel:   %.4f ms\n", msStrided);
    printf("Slowdown: %.2fx\n", msStrided / msCoalesced);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");
    printf("Both kernels do the same number of loads, stores, and multiplies. ");
    printf("The only difference is the access pattern.\n");

    CUDA_CHECK(cudaFree(d_in_c));
    CUDA_CHECK(cudaFree(d_out_c));
    CUDA_CHECK(cudaFree(d_in_s));
    CUDA_CHECK(cudaFree(d_out_s));
    free(h_in);
    free(h_out_c);
    free(h_out_s);

    return ok ? 0 : 1;
}
