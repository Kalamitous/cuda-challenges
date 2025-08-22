#include <cuda_runtime.h>

__global__ void kernel(const float* input, float* output, size_t n, size_t m) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < n) {
        output[i * n + j] = max(0.0f, input[i * n + j]);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    kernel<<<gridDim, blockDim>>>(input, output, n, m);
}