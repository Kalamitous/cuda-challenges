// todo: optimize

#include <cuda_runtime.h>

__global__ void matrix_transpose(const float* A, float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

__global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// matrix inversion via gauss-jordan elimination

__global__ void matrix_identity(float* A, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        A[row * N + col] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void pivot_normalize(float* A, float* B, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float pivot = A[K * N + K];
        A[K * N + col] /= pivot;
        B[K * N + col] /= pivot;
    }
}

__global__ void pivot_column_eliminate(float* A, float* B, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N && row != K) {
        float factor = A[row * N + K];
        A[row * N + col] -= factor * A[K * N + col];
        B[row * N + col] -= factor * B[K * N + col];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *XT, *XTy, *XTX, *XTX_inv;
    cudaMalloc(&XT, n_features * n_samples * sizeof(float));
    cudaMalloc(&XTy, n_features * sizeof(float));
    cudaMalloc(&XTX, n_features * n_features * sizeof(float));
    cudaMalloc(&XTX_inv, n_features * n_features * sizeof(float));

    dim3 blockDim1(16, 16);
    dim3 gridDim1((n_features + blockDim1.x - 1) / blockDim1.x,
                 (n_samples + blockDim1.y - 1) / blockDim1.y);
    matrix_transpose<<<gridDim1, blockDim1>>>(X, XT, n_samples, n_features);
    cudaDeviceSynchronize();

    dim3 blockDim2(1, 256);
    dim3 gridDim2(1, (n_features + blockDim2.y - 1) / blockDim2.y);
    matrix_multiply<<<gridDim2, blockDim2>>>(XT, y, XTy, n_features, 1, n_samples);
    cudaDeviceSynchronize();

    dim3 blockDim3(16, 16);
    dim3 gridDim3((n_features + blockDim3.x - 1) / blockDim3.x,
                  (n_features + blockDim3.y - 1) / blockDim3.y);
    matrix_multiply<<<gridDim3, blockDim3>>>(XT, X, XTX, n_features, n_features, n_samples);
    matrix_identity<<<gridDim3, blockDim3>>>(XTX_inv, n_features);
    cudaDeviceSynchronize();

    dim3 blockDim4(256);
    dim3 gridDim4((n_features + blockDim4.x - 1) / blockDim4.x);
    for (int k = 0; k < n_features; ++k) {
        pivot_normalize<<<gridDim4, blockDim4>>>(XTX, XTX_inv, n_features, k);
        cudaDeviceSynchronize();
        pivot_column_eliminate<<<gridDim3, blockDim3>>>(XTX, XTX_inv, n_features, k);
        cudaDeviceSynchronize();
    }

    matrix_multiply<<<gridDim3, blockDim3>>>(XTX_inv, XTy, beta, n_features, 1, n_features);

    cudaFree(XT);
    cudaFree(XTy);
    cudaFree(XTX);
    cudaFree(XTX_inv);
}
