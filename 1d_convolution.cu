#include <cuda_runtime.h>

__constant__ float B_c[8191];

__global__ void kernel(const float* A, const float* B, float* C, size_t N, size_t K) {
    // each block is responsible for blockDim.x outputs

    // A_s is length (r + blockDim.x + r)
    extern __shared__ float A_s[];

    int r = K / 2;
    int smem_size = blockDim.x + r * 2;
    int block_start = blockIdx.x * blockDim.x;
    int tile_start = blockIdx.x * blockDim.x - r;

    for (int j = threadIdx.x; j < smem_size; j += blockDim.x) {
        int a_idx = tile_start + j;
        if (a_idx >= 0 && a_idx < N) {
            A_s[j] = A[a_idx];
        } else {
            A_s[j] = 0.0f;
        }
    }
    __syncthreads();

    int i = block_start + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum += A_s[threadIdx.x + j] * B_c[j];
        }
        C[i] = sum;
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(B_c, B, K * sizeof(float));

    dim3 blockDim(1024);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t smem_size = (blockDim.x + (K / 2) * 2) * sizeof(float);
    kernel<<<gridDim, blockDim, smem_size>>>(A, B, C, N, K);
}