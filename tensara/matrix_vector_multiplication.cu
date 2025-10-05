#include <cuda_runtime.h>

const int tile_dim = 32;
const int coarse_factor = 2;

// __global__ void mat_vec_mul_coarsened(const float* A, const float* B, float* C, size_t M, size_t K) {
//     __shared__ float Bs[tile_dim * coarse_factor];

//     int idx = blockIdx.x * tile_dim + threadIdx.x * coarse_factor;

//     float sum[coarse_factor] = {0.0f};
//     for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
//         #pragma unroll
//         for (int i = 0; i < coarse_factor; ++i) {
//             int col = tile_start + threadIdx.x * coarse_factor + i;
//             Bs[threadIdx.x * coarse_factor + i] = (col < K) ? B[col] : 0.0f;
//         }
//         __syncthreads();

//         #pragma unroll
//         for (int i = 0; i < coarse_factor; ++i) {
//             int row = idx + i;
//             if (row >= M) continue;

//             #pragma unroll
//             for (int j = 0; j < tile_dim; ++j) {
//                 int col = tile_start + j;
//                 if (col < K) {
//                     sum[i] += A[row * K + col] * Bs[j];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     #pragma unroll
//     for (int i = 0; i < coarse_factor; ++i) {
//         int row = idx + i;
//         if (row < M) {
//             C[row] = sum[i];
//         }
//     }
// }

__global__ void mat_vec_mul(const float* A, const float* B, float* C, size_t M, size_t K) {
    __shared__ float Bs[tile_dim];

    int idx = blockIdx.x * tile_dim + threadIdx.x;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
        int col = tile_start + threadIdx.x;
        Bs[threadIdx.x] = (col < K) ? B[col] : 0.0f;
        __syncthreads();

        if (idx < M) {
            #pragma unroll
            for (int i = 0; i < tile_dim; ++i) {
                if (tile_start + i < K) {
                    sum += A[idx * K + tile_start + i] * Bs[i];
                }
            }
        }
        __syncthreads();
    }

    if (idx < M) {
        C[idx] = sum;
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    // int threads = tile_dim / coarse_factor;
    int threads = tile_dim;
    int blocks = (m + tile_dim - 1) / tile_dim;
    mat_vec_mul<<<blocks, threads>>>(input_a, input_b, output_c, m, k);
}