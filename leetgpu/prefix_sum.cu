#include <cuda_runtime.h>

// naive solution; todo: hillis steele scan
__global__ void prefix_sum(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < idx + 1; ++i) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    prefix_sum<<<blocks, threads>>>(input, output, N);
} 