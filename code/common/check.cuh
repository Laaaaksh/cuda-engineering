// Shared across every sample in code/. CUDA calls return an error code
// instead of throwing, and the default is to silently ignore it - which
// turns a bad kernel launch into a confusing wrong-answer bug three lines
// later instead of a clear error where it happened. Wrap every CUDA call
// in CUDA_CHECK so failures point at the right line.
#pragma once

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));              \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// Kernel launches don't return an error directly - the launch itself can
// fail asynchronously. Call this right after a launch to catch it.
#define CUDA_CHECK_LAST()                                                    \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                     \
        CUDA_CHECK(cudaDeviceSynchronize());                                \
    } while (0)
