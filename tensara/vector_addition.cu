#include <cuda_runtime.h>

__global__ void kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_input1, d_input2, d_output, n);
}