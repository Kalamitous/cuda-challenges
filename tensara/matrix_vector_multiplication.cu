#include <cuda_runtime.h>

const int tile_dim = 256;
const int coarse_factor = 4;

__global__ void mat_vec_mul_coarsened(const float* A, const float* B, float* C, size_t M, size_t K) {
    __shared__ float B_s[tile_dim];

    int idx = blockIdx.x * (blockDim.x * coarse_factor) + threadIdx.x;

    float sum[coarse_factor] = {0.0f};
    for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
        int b_col = tile_start + threadIdx.x;
        B_s[threadIdx.x] = (b_col < K) ? B[b_col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < coarse_factor; i++) {
            int a_row = idx + i * blockDim.x;
            if (a_row < M) {
                #pragma unroll
                for (int k = 0; k < tile_dim; ++k) {
                    int a_col = tile_start + k;
                    if (a_col < K) {
                        sum[i] += A[a_row * K + a_col] * B_s[k];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < coarse_factor; i++) {
        int c_col = idx + i * blockDim.x;
        if (c_col < M) C[c_col] = sum[i];
    }
}

// __global__ void mat_vec_mul(const float* A, const float* B, float* C, size_t M, size_t K) {
//     __shared__ float B_s[tile_dim];

//     int idx = blockIdx.x * tile_dim + threadIdx.x;

//     float sum = 0.0f;
//     for (int tile_start = 0; tile_start < K; tile_start += tile_dim) {
//         int b_col = tile_start + threadIdx.x;
//         B_s[threadIdx.x] = (b_col < K) ? B[b_col] : 0.0f;
//         __syncthreads();

//         if (idx < M) {
//             #pragma unroll
//             for (int k = 0; k < tile_dim; ++k) {
//                 int a_col = tile_start + k;
//                 if (a_col < K) {
//                     sum += A[idx * K + a_col] * B_s[k];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     if (idx < M) C[idx] = sum;
// }

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int threads = tile_dim;

    int blocks = (m + threads * coarse_factor - 1) / (threads * coarse_factor);
    mat_vec_mul_coarsened<<<blocks, threads>>>(input_a, input_b, output_c, m, k);

    // int blocks = (m + tile_dim - 1) / tile_dim;
    // mat_vec_mul<<<blocks, threads>>>(input_a, input_b, output_c, m, k);
}