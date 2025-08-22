#include <cuda_runtime.h>

__global__ void kernel(const float* input_image, int kernel_size, float* output_image, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float I_s[];

    int k = kernel_size / 2;
    int tile_width = blockDim.x + k * 2;
    int tile_height = blockDim.y + k * 2;
    int by = blockIdx.y * blockDim.y;
    int bx = blockIdx.x * blockDim.x;
    for (int sy = threadIdx.y; sy < tile_height; sy += blockDim.y) {
        for (int sx = threadIdx.x; sx < tile_width; sx += blockDim.x) {
            int input_r = by - k + sy;
            int input_c = bx - k + sx;
            if (input_r >= 0 && input_r < height && input_c >= 0 && input_c < width) {
                I_s[sy * tile_width + sx] = input_image[input_r * width + input_c];
            } else {
                I_s[sy * tile_width + sx] = 0;
            }
        }
    }
    __syncthreads();

    if (row >= height || col >= width) return;

    float sum = 0.0f;
    for (int r = 0; r < kernel_size; ++r) {
        for (int c = 0; c < kernel_size; ++c) {
            sum += I_s[(r + threadIdx.y) * tile_width + (c + threadIdx.x)];
        }
    }

    int r_lo = max(-k, -row);
    int r_hi = min(k, height - 1 - row);
    int c_lo = max(-k, -col);
    int c_hi = min(k, width - 1 - col);
    int n = (r_hi - r_lo + 1) * (c_hi - c_lo + 1);
    output_image[row * width + col] = sum / n;
}

// Note: input_image, output_image are all device pointers to float32 arrays
extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width) { 
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    int smem_size = (blockDim.x + kernel_size - 1) * (blockDim.y + kernel_size - 1) * sizeof(float);
    kernel<<<gridDim, blockDim, smem_size>>>(input_image, kernel_size, output_image, height, width);
}