#include <cuda_runtime.h>

const int TILE_DIM = 32;

__constant__ float c_kernel[1024];

__global__ void convolution(const float* input, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float s_data[];

    int s_data_rows = TILE_DIM + kernel_rows - 1;
    int s_data_cols = TILE_DIM + kernel_cols - 1;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    int output_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int output_col = blockIdx.x * TILE_DIM + threadIdx.x;

    int input_tile_row = blockIdx.y * TILE_DIM;
    int input_tile_col = blockIdx.x * TILE_DIM;

    for (int local_row = threadIdx.y; local_row < s_data_rows; local_row += TILE_DIM) {
        for (int local_col = threadIdx.x; local_col < s_data_cols; local_col += TILE_DIM) {
            int global_row = input_tile_row + local_row;
            int global_col = input_tile_col + local_col;
            if (global_row < input_rows && global_col < input_cols) {
                s_data[local_row * s_data_cols + local_col] = input[global_row * input_cols + global_col];
            } else {
                s_data[local_row * s_data_cols + local_col] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (output_row < output_rows && output_col < output_cols) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_rows; ++m) {
            for (int n = 0; n < kernel_cols; ++n) {
                int local_row = threadIdx.y + m;
                int local_col = threadIdx.x + n;
                sum += s_data[local_row * s_data_cols + local_col] * c_kernel[m * kernel_cols + n];
            }
        }
        output[output_row * output_cols + output_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    cudaMemcpyToSymbol(c_kernel, kernel, kernel_rows * kernel_cols * sizeof(float));
    
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((output_cols + TILE_DIM - 1) / TILE_DIM, (output_rows + TILE_DIM - 1) / TILE_DIM);
    size_t sdata_size = (TILE_DIM + kernel_rows - 1) * (TILE_DIM + kernel_cols - 1) * sizeof(float);
    convolution<<<gridDim, blockDim, sdata_size>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}