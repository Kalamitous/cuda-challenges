#include <cuda_runtime.h>

__global__ void image_histogram(const float* image, int num_bins, float* histogram, int N) {
    extern __shared__ float count[];

    for (int i = threadIdx.x; i < num_bins; ++i) {
        count[i] = 0.0f;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int val = image[idx];
        atomicAdd(&count[val], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], count[i]);
    }
}

// Note: image, histogram are all device pointers to float32 arrays
extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) { 
    int N = width * height;   
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    size_t smem_size = num_bins * sizeof(float);
    image_histogram<<<blocks, threads, smem_size>>>(image, num_bins, histogram, N);
}