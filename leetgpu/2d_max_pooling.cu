#include <cuda_runtime.h>

__global__ void max_pooling_2d_tiling(const float* input, float* output,
                               int N, int C, int H, int W, int H_out, int W_out,
                               int kernel_size, int stride, int padding) {
    extern __shared__ float sdata[];
    int tile_h = blockDim.y * stride + kernel_size - 1;
    int tile_w = blockDim.x * stride + kernel_size - 1;
    
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    
    int h_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int h_tile_start = blockIdx.y * blockDim.y * stride - padding;
    int w_tile_start = blockIdx.x * blockDim.x * stride - padding;

    int input_start = (n * C + c) * (H * W);

    for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x) {
            int h_idx = h_tile_start + ty;
            int w_idx = w_tile_start + tx;

            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                sdata[ty * tile_w + tx] = input[input_start + h_idx * W + w_idx];
            } else {
                sdata[ty * tile_w + tx] = -INFINITY;
            }
        }
    }
    __syncthreads();

    if (h_out_idx >= H_out || w_out_idx >= W_out) return;

    float val = -INFINITY;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int sh_idx = threadIdx.y * stride + kh;
            int sw_idx = threadIdx.x * stride + kw;

            val = fmaxf(val, sdata[sh_idx * tile_w + sw_idx]);
        }
    }

    output[(n * C + c) * (H_out * W_out) + h_out_idx * W_out + w_out_idx] = val;
}

__global__ void max_pooling_2d(const float* input, float* output,
                               int N, int C, int H, int W, int H_out, int W_out,
                               int kernel_size, int stride, int padding) {
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    if (n >= N) return;
    
    int h_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_out_idx >= H_out || w_out_idx >= W_out) return;

    int h_start = h_out_idx * stride - padding;
    int w_start = w_out_idx * stride - padding;

    int input_start = (n * C + c) * (H * W);

    float val = -INFINITY;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_idx = h_start + kh;
            int w_idx = w_start + kw;

            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                int input_idx = input_start + h_idx * W + w_idx;
                val = fmaxf(val, input[input_idx]);
            }
        }
    }

    output[(n * C + c) * (H_out * W_out) + h_out_idx * W_out + w_out_idx] = val;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (W_out + blockDim.x - 1) / blockDim.x,
        (H_out + blockDim.y - 1) / blockDim.y,
        N * C
    );

    if (stride < kernel_size) {
        size_t sdata_size = (blockDim.y * stride + kernel_size - 1) * (blockDim.x * stride + kernel_size - 1) * sizeof(float);
        max_pooling_2d_tiling<<<gridDim, blockDim, sdata_size>>>(input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else {
        // no reuse of inputs if stride >= kernel_size, so shared memory tiling would just be additional overhead
        max_pooling_2d<<<gridDim, blockDim>>>(input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    }
}
