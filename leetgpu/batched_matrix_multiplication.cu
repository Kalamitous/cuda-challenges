#include <cuda_runtime.h>

const int tile_dim = 32;

__global__ void batched_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float A_s[tile_dim][tile_dim];
    __shared__ float B_s[tile_dim][tile_dim];
    
    int output_row = blockIdx.y * tile_dim + threadIdx.y;
    int output_col = blockIdx.x * tile_dim + threadIdx.x;
    int batch = blockIdx.z;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
        int global_row = output_row;
        int global_col = tile_start + threadIdx.x;
        int a_idx = batch * M * K + global_row * K + global_col;
        if (global_row < M && global_col < K) {
            A_s[threadIdx.y][threadIdx.x] = A[a_idx];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        global_row = tile_start + threadIdx.y;
        global_col = output_col;
        int b_idx = batch * K * N + global_row * N + global_col;
        if (global_row < K && global_col < N) {
            B_s[threadIdx.y][threadIdx.x] = B[b_idx];
        } else {
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_dim; ++k) {
            sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (output_row < M && output_col < N) {
        int c_idx = batch * M * N + output_row * N + output_col;
        C[c_idx] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 blockDim(tile_dim, tile_dim);
    dim3 gridDim((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim, BATCH);
    batched_matmul<<<gridDim, blockDim>>>(A, B, C, M, N, K);
} 