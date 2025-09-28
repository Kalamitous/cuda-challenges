#include <cuda_runtime.h>

// todo: implement coarsening for matrix_identity, pivot_normalize, pivot_column_eliminate

template <int TileDim, int CoarseFactor>
__global__ void matrix_transpose(const float* A, float* B, int M, int N) {
    __shared__ float A_s[TileDim][TileDim + 1];

    int row = blockIdx.y * TileDim + threadIdx.y;
    int col = blockIdx.x * TileDim + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < TileDim; i += TileDim / CoarseFactor) {
        if (row < M && col + i < N) {
            A_s[threadIdx.y][threadIdx.x + i] = A[row * N + col + i];
        } else {
            A_s[threadIdx.y][threadIdx.x + i] = 0.0f;
        }
    }
    __syncthreads();

    row = blockIdx.x * TileDim + threadIdx.y;
    col = blockIdx.y * TileDim + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < TileDim; i += TileDim / CoarseFactor) {
        if (row < N && col + i < M) {
            B[row * M + col + i] = A_s[threadIdx.x + i][threadIdx.y];
        }
    }
}

template <int TileDim, int CoarseFactor>
__global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float A_s[TileDim][TileDim];
    __shared__ float B_s[TileDim][TileDim];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TileDim + ty * CoarseFactor;
    int col = blockIdx.x * TileDim + tx * CoarseFactor;

    float sum[CoarseFactor][CoarseFactor] = {0.0f};
    for (int tile_start = 0; tile_start < K; tile_start += TileDim) {
        #pragma unroll
        for (int i = 0; i < CoarseFactor; ++i) {
            #pragma unroll
            for (int j = 0; j < CoarseFactor; ++j) {
                int a_row = row + i;
                int a_col = tile_start + tx * CoarseFactor + j;
                if (a_row < M && a_col < K) {
                    A_s[ty * CoarseFactor + i][tx * CoarseFactor + j] = A[a_row * K + a_col];
                } else {
                    A_s[ty * CoarseFactor + i][tx * CoarseFactor + j] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < CoarseFactor; ++i) {
            #pragma unroll
            for (int j = 0; j < CoarseFactor; ++j) {
                int b_row = tile_start + ty * CoarseFactor + i;
                int b_col = col + j;
                if (b_row < K && b_col < N) {
                    B_s[ty * CoarseFactor + i][tx * CoarseFactor + j] = B[b_row * N + b_col];
                } else {
                    B_s[ty * CoarseFactor + i][tx * CoarseFactor + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TileDim; ++k) {
            #pragma unroll
            for (int i = 0; i < CoarseFactor; ++i) {
                #pragma unroll
                for (int j = 0; j < CoarseFactor; ++j) {
                    sum[i][j] += A_s[ty * CoarseFactor + i][k] * B_s[k][tx * CoarseFactor + j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < CoarseFactor; ++i) {
        #pragma unroll
        for (int j = 0; j < CoarseFactor; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = sum[i][j];
            }
        }
    }
}

__global__ void matrix_identity(float* A, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        A[row * N + col] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void pivot_normalize(float* A, float* B, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pivot;
    if (threadIdx.x % 32 == 0) {
        pivot = A[K * N + K];
    }
    pivot = __shfl_sync(0xffffffff, pivot, 0);

    if (col < N) {
        A[K * N + col] /= pivot;
        B[K * N + col] /= pivot;
    }
}

__global__ void pivot_column_eliminate(float* A, float* B, int N, int K) {
    extern __shared__ float sdata[];
    float* factor = sdata;
    float* A_s = &factor[blockDim.y];
    float* B_s = &A_s[blockDim.x];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        factor[threadIdx.y] = (row < N) ? A[row * N + K] : 0.0f;
    }
    if (threadIdx.y == 0) {
        A_s[threadIdx.x] = (col < N) ? A[K * N + col] : 0.0f;
        B_s[threadIdx.x] = (col < N) ? B[K * N + col] : 0.0f;
    }
    __syncthreads();

    if (row < N && col < N && row != K) {
        A[row * N + col] -= factor[threadIdx.y] * A_s[threadIdx.x];
        B[row * N + col] -= factor[threadIdx.y] * B_s[threadIdx.x];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *XT, *XTy, *XTX, *XTX_inv;
    cudaMalloc(&XT, n_features * n_samples * sizeof(float));
    cudaMalloc(&XTy, n_features * sizeof(float));
    cudaMalloc(&XTX, n_features * n_features * sizeof(float));
    cudaMalloc(&XTX_inv, n_features * n_features * sizeof(float));

    dim3 blockDim1(16 / 2, 16);
    dim3 gridDim1((n_features + blockDim1.x - 1) / (blockDim1.x),
                  (n_samples + blockDim1.y - 1) / blockDim1.y);
    matrix_transpose<16, 2><<<gridDim1, blockDim1>>>(X, XT, n_samples, n_features);
    cudaDeviceSynchronize();

    dim3 blockDim2(16 / 2, 16 / 2);
    dim3 gridDim2(1, (n_features + blockDim2.y - 1) / blockDim2.y);
    matrix_multiply<16, 2><<<gridDim2, blockDim2>>>(XT, y, XTy, n_features, 1, n_samples);
    cudaDeviceSynchronize();

    dim3 blockDim3(16 / 2, 16 / 2);
    dim3 gridDim3((n_features + blockDim3.x - 1) / blockDim3.x,
                  (n_features + blockDim3.y - 1) / blockDim3.y);
    matrix_multiply<16, 2><<<gridDim3, blockDim3>>>(XT, X, XTX, n_features, n_features, n_samples);
    matrix_identity<<<gridDim3, blockDim3>>>(XTX_inv, n_features);
    cudaDeviceSynchronize();

    dim3 blockDim4(256);
    dim3 gridDim4((n_features + blockDim4.x - 1) / blockDim4.x);
    size_t sdata_size = (blockDim3.y + blockDim3.x * 2) * sizeof(float);
    for (int k = 0; k < n_features; ++k) {
        pivot_normalize<<<gridDim4, blockDim4>>>(XTX, XTX_inv, n_features, k);
        cudaDeviceSynchronize();
        pivot_column_eliminate<<<gridDim3, blockDim3, sdata_size>>>(XTX, XTX_inv, n_features, k);
        cudaDeviceSynchronize();
    }

    matrix_multiply<16, 2><<<gridDim3, blockDim3>>>(XTX_inv, XTy, beta, n_features, 1, n_features);

    cudaFree(XT);
    cudaFree(XTy);
    cudaFree(XTX);
    cudaFree(XTX_inv);
}