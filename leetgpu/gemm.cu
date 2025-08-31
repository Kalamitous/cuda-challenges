#include <cuda_runtime.h>
#include <cuda_fp16.h>

const int tile_dim = 32;

__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half A_s[tile_dim * tile_dim];
    __shared__ half B_s[tile_dim * tile_dim];

    int g_row = blockIdx.y * tile_dim + threadIdx.y;
    int g_col = blockIdx.x * tile_dim + threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < (K + tile_dim - 1) / tile_dim; ++tile) {
        int t_row = g_row;
        int t_col = tile * tile_dim + threadIdx.x;
        if (t_row < M && t_col < K) {
            A_s[threadIdx.y * tile_dim + threadIdx.x] = A[t_row * K + t_col];
        } else {
            A_s[threadIdx.y * tile_dim + threadIdx.x] = 0.0f;
        }
        
        t_row = tile * tile_dim + threadIdx.y;
        t_col = g_col;
        if (t_row < K && t_col < N) {
            B_s[threadIdx.y * tile_dim + threadIdx.x] = B[t_row * N + t_col];
        } else {
            B_s[threadIdx.y * tile_dim + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < tile_dim; ++k) {
            sum += __half2float(A_s[threadIdx.y * tile_dim + k]) * __half2float(B_s[k * tile_dim + threadIdx.x]);
        }
        __syncthreads();
    }

    if (g_row < M && g_col < N) {
        C[g_row * N + g_col] = __float2half(alpha * sum + beta * __half2float(C[g_row * N + g_col]));
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(tile_dim, tile_dim);
    dim3 gridDim((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
