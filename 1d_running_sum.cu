#include <cuda_runtime.h>

__global__ void kernel(const float* input, size_t W, float* output, size_t N) {
    extern __shared__ float smem[];

    int r = W / 2;
    int smem_size = blockDim.x + r * 2;
    int tile_start = blockIdx.x * blockDim.x - r;

    for (int smem_idx = threadIdx.x; smem_idx < smem_size; smem_idx += blockDim.x) {
        int input_idx = tile_start + smem_idx;
        if (input_idx >= 0 && input_idx < N) {
            smem[smem_idx] = input[input_idx];
        } else {
            smem[smem_idx] = 0.0f;
        }
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < W; ++j) {
            sum += smem[threadIdx.x + j];
        }
        output[i] = sum;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, size_t W, float* output, size_t N) {   
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int smem_size = (threads + (W / 2) * 2) * sizeof(float);
    kernel<<<blocks, threads, smem_size>>>(input, W, output, N); 
}