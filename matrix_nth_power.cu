#include <cuda_runtime.h>

const int TILE_SIZE = 32;

__global__ void identity_matrix(float* O, size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        if (row == col) {
            O[row * N + col] = 1.0f;
        } else {
            O[row * N + col] = 0.0f;
        }
    }
}

__global__ void matrix_multiply(const float* A, const float* B, float* C, size_t N) {
    __shared__ float As[TILE_SIZE * TILE_SIZE];
    __shared__ float Bs[TILE_SIZE * TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tr = row;
        int tc = tile * TILE_SIZE + threadIdx.x;
        if (tr < N && tc < N) {
            As[threadIdx.y * TILE_SIZE + threadIdx.x] = A[tr * N + tc];
        } else {
            As[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }
        tr = tile * TILE_SIZE + threadIdx.y;
        tc = col;
        if (tr < N && tc < N) {
            Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = B[tr * N + tc];
        } else {
            Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y * TILE_SIZE + k] * Bs[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Note: input_matrix, output_matrix are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const size_t n, float* output_matrix, size_t size) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE);

    size_t bytes = size * size * sizeof(float);
    float *res, *base, *tmp;
    cudaMalloc(&res, bytes);
    cudaMalloc(&base, bytes);
    cudaMalloc(&tmp, bytes);

    // res = I
    identity_matrix<<<gridDim, blockDim>>>(res, size);
    // base = A
    cudaMemcpy(base, input_matrix, bytes, cudaMemcpyDeviceToDevice);

    int pow = n;
    while (pow > 0) {
        if (pow % 2 == 1) {
            // res = res * base
            matrix_multiply<<<gridDim, blockDim>>>(res, base, tmp, size);
            std::swap(res, tmp);
        }
        pow /= 2;
        if (pow > 0) {
            // base = base * base
            matrix_multiply<<<gridDim, blockDim>>>(base, base, tmp, size);
            std::swap(base, tmp);
        }
    }

    cudaMemcpy(output_matrix, res, bytes, cudaMemcpyDeviceToDevice);

    cudaFree(res);
    cudaFree(base);
    cudaFree(tmp);
}