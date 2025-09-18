#include <cuda_runtime.h>

__constant__ float kernel_c[256];

__global__ void gaussian_blur(const float* input, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float sdata[];
    int tile_rows = blockDim.y + kernel_rows / 2 * 2;
    int tile_cols = blockDim.x + kernel_cols / 2 * 2;
    
    for (int ty = threadIdx.y; ty < tile_rows; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_cols; tx += blockDim.x) {
            int i_row = blockIdx.y * blockDim.y + ty - kernel_rows / 2;
            int i_col = blockIdx.x * blockDim.x + tx - kernel_cols / 2;
            if (i_row >= 0 && i_row < input_rows && i_col >= 0 && i_col < input_cols) {
                sdata[ty * tile_cols + tx] = input[i_row * input_cols + i_col];
            } else {
                sdata[ty * tile_cols + tx] = 0.0f;
            }
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (int m = 0; m < kernel_rows; ++m) {
        for (int n = 0; n < kernel_cols; ++n) {
            int t_row = threadIdx.y + m;
            int t_col = threadIdx.x + n;
            sum += sdata[t_row * tile_cols + t_col] * kernel_c[m * kernel_cols + n];
        }
    }
    
    int o_row = blockIdx.y * blockDim.y + threadIdx.y;
    int o_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_row >= 0 && o_row < input_rows && o_col >= 0 && o_col < input_cols) {
        output[o_row * input_cols + o_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(kernel_c, kernel, kernel_rows * kernel_cols * sizeof(float));

    dim3 blockDim(32, 32);
    dim3 gridDim((input_cols + blockDim.x - 1) / blockDim.x, (input_rows + blockDim.y - 1) / blockDim.y);
    size_t sdata_size = (blockDim.x + kernel_cols / 2 * 2) * (blockDim.y + kernel_rows / 2 * 2) * sizeof(float);
    gaussian_blur<<<gridDim, blockDim, sdata_size>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
