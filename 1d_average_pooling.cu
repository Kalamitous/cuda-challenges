#include <cuda_runtime.h>

__global__ void kernel(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t H_out) {
    extern __shared__ float smem[];

    int smem_len = stride * (blockDim.x - 1) + kernel_size;
    int tile_start = stride * (blockIdx.x * blockDim.x) - padding;

    for (int tid = threadIdx.x; tid < smem_len; tid += blockDim.x) {
        int gi = tile_start + tid;
        if (gi >= 0 && gi < H) {
            smem[tid] = input[gi];
        } else {
            smem[tid] = 0.0f;
        }
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H_out) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_size; ++m) {
            sum += smem[stride * threadIdx.x + m];
        }
        output[i] = sum / kernel_size;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {  
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int threads = 256;
    int blocks = (H_out + threads - 1) / threads;
    int smem_size = (stride * (threads - 1) + kernel_size) * sizeof(float);
    kernel<<<blocks, threads, smem_size>>>(input, kernel_size, stride, padding, output, H, H_out);
}