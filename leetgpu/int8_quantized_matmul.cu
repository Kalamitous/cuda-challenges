#include <cuda_runtime.h>

const int tile_dim = 32;

__global__ void matmul(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float effective_scale, int zero_point_A, int zero_point_B, int zero_point_C) {
    __shared__ int8_t A_s[tile_dim][tile_dim];
    __shared__ int8_t B_s[tile_dim][tile_dim];

    const char4* A_v = reinterpret_cast<const char4*>(A);
    const char4* B_v = reinterpret_cast<const char4*>(B);

    int row = blockIdx.y * tile_dim + threadIdx.y;
    int col = blockIdx.x * tile_dim + threadIdx.x * 4;

    int32_t sum[4] = {0, 0, 0, 0};
    for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
        int A_row = row;
        int A_col = tile_start + threadIdx.x * 4;
        if (K % 4 == 0 && A_row < M && A_col + 3 < K) { // (A_row * K + A_col) needs to be divisible by 4
            char4 v = A_v[(A_row * K + A_col) / 4];
            A_s[threadIdx.y][threadIdx.x * 4 + 0] = v.x;
            A_s[threadIdx.y][threadIdx.x * 4 + 1] = v.y;
            A_s[threadIdx.y][threadIdx.x * 4 + 2] = v.z;
            A_s[threadIdx.y][threadIdx.x * 4 + 3] = v.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (A_row < M && A_col + i < K) {
                    A_s[threadIdx.y][threadIdx.x * 4 + i] = A[A_row * K + A_col + i];
                } else {
                    A_s[threadIdx.y][threadIdx.x * 4 + i] = 0;
                }
            }
        }

        int B_row = tile_start + threadIdx.y;
        int B_col = col;
        if (N % 4 == 0 && B_row < K && B_col + 3 < N) { // (B_row * N + B_col) needs to be divisible by 4
            char4 v = B_v[(B_row * N + B_col) / 4];
            B_s[threadIdx.y][threadIdx.x * 4 + 0] = v.x;
            B_s[threadIdx.y][threadIdx.x * 4 + 1] = v.y;
            B_s[threadIdx.y][threadIdx.x * 4 + 2] = v.z;
            B_s[threadIdx.y][threadIdx.x * 4 + 3] = v.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (B_row < K && B_col + i < N) {
                    B_s[threadIdx.y][threadIdx.x * 4 + i] = B[B_row * N + B_col + i];
                } else {
                    B_s[threadIdx.y][threadIdx.x * 4 + i] = 0;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_dim; ++k) {
            int a_val = int(A_s[threadIdx.y][k]) - zero_point_A;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int b_val = int(B_s[k][threadIdx.x * 4 + i]) - zero_point_B;
                sum[i] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (row < M && col + i < N) {
            float scaled = float(sum[i]) * effective_scale;
            C[row * N + col + i] = min(max(__float2int_rn(scaled) + zero_point_C, -128), 127);
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    dim3 blockDim(tile_dim / 4, tile_dim);
    dim3 gridDim((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    matmul<<<gridDim, blockDim>>>(A, B, C, M, N, K, scale_A * scale_B / scale_C, zero_point_A, zero_point_B, zero_point_C);
} 