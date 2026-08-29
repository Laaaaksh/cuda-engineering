// Tiled matrix multiply: same C = A * B as 03-matmul-naive, but each thread
// block cooperatively loads a TILE x TILE tile of A and B into shared
// memory once, then every thread in the block reuses those tiles from
// on-chip memory instead of re-reading global memory for every multiply.
//
// Shared memory is roughly two orders of magnitude faster to access than
// global memory and is shared by every thread in a block - so this is the
// direct fix for the redundant global reads called out in
// 03-matmul-naive.cu. Compare the two samples' GFLOP/s output.
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/check.cuh"

#define TILE 16

__global__ void matmulTiled(const float *A, const float *B, float *C, int n) {
    // __shared__ memory is allocated per block and visible to every thread
    // in that block - this is the on-chip scratchpad the naive kernel never
    // uses.
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Slide a TILE-wide window across the shared K dimension. Each pass
    // loads one tile of A and one tile of B, synchronizes so the whole
    // block sees the completed tiles, then every thread does TILE
    // multiply-adds purely from shared memory before moving to the next
    // tile.
    int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Each thread loads exactly one element of each tile - the loads
        // are coalesced (see curriculum/03-memory-model-and-coalescing.md)
        // because consecutive threadIdx.x values read consecutive
        // addresses. Out-of-range tiles (n not a multiple of TILE) are
        // zero-padded so the math stays correct at any n.
        tileA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        // Barrier: every thread in the block must finish loading before
        // any thread starts reading tileA/tileB. Without this, a fast
        // thread could read a tile slot a slower thread hasn't written yet
        // - a data race, not a performance detail.
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Second barrier: every thread must finish reading this tile
        // before any thread starts overwriting it with the next tile.
        __syncthreads();
    }

    if (row < n && col < n) {
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

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmulTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_LAST();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int checkN = std::min(n, 256);
    matmulCPU(h_A, h_B, h_C_ref, n);
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

    printf("n = %d, tile = %d\n", n, TILE);
    printf("Kernel time: %.4f ms (%.2f GFLOP/s)\n", ms, gflops);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");
    printf("Compare against code/03-matmul-naive at the same n - same GPU, ");
    printf("same problem, only the memory access pattern changed.\n");

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
