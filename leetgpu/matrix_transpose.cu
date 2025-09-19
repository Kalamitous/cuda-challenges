#include <cuda_runtime.h>

const int tile_dim = 16;

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[tile_dim][tile_dim + 1]; // pad to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // transpose block offset only
    x = blockIdx.y * blockDim.y + threadIdx.x; 
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (y < cols && x < rows) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(tile_dim, tile_dim);
    dim3 blocksPerGrid((cols + tile_dim - 1) / tile_dim,
                       (rows + tile_dim - 1) / tile_dim);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}