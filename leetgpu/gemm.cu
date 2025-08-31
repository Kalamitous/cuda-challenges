#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_SIZE = 32;
const int TILE_DIM = 32;

// ref: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
__global__ void gemm_wmma(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = warpM * WMMA_M;
        int a_col = k;

        int b_row = k;
        int b_col = warpN * WMMA_N;

        wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
        wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;

    wmma::load_matrix_sync(c_frag, C + c_row * N + c_col, N, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(alpha * acc_frag.x[i] + beta * __half2float(c_frag.x[i]));
    }

    wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
}

__global__ void gemm_tiled(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half A_s[TILE_DIM * TILE_DIM];
    __shared__ half B_s[TILE_DIM * TILE_DIM];

    int g_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int g_col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; ++tile) {
        int t_row = g_row;
        int t_col = tile * TILE_DIM + threadIdx.x;
        if (t_row < M && t_col < K) {
            A_s[threadIdx.y * TILE_DIM + threadIdx.x] = A[t_row * K + t_col];
        } else {
            A_s[threadIdx.y * TILE_DIM + threadIdx.x] = 0.0f;
        }
        
        t_row = tile * TILE_DIM + threadIdx.y;
        t_col = g_col;
        if (t_row < K && t_col < N) {
            B_s[threadIdx.y * TILE_DIM + threadIdx.x] = B[t_row * N + t_col];
        } else {
            B_s[threadIdx.y * TILE_DIM + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += __half2float(A_s[threadIdx.y * TILE_DIM + k]) * __half2float(B_s[k * TILE_DIM + threadIdx.x]);
        }
        __syncthreads();
    }

    if (g_row < M && g_col < N) {
        C[g_row * N + g_col] = __float2half(alpha * sum + beta * __half2float(C[g_row * N + g_col]));
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // wmma can only be used if matrix dimensions are multiples of 16
    bool use_wmma = (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0);
    if (use_wmma) {
        dim3 blockDim(32, 4);
        dim3 gridDim(
            (M + (WMMA_M * blockDim.x / WARP_SIZE) - 1) / (WMMA_M * blockDim.x / WARP_SIZE),
            (N + (WMMA_N * blockDim.y) - 1) / (WMMA_N * blockDim.y)
        );
        gemm_wmma<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        gemm_tiled<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    }
}