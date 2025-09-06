#include <cuda_runtime.h>

const int threads = 256;
const int warp_size = 32;

__global__ void sum(const float* y_samples, float* result, float a, float b, int n_samples) {
    __shared__ float sdata[threads / warp_size];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val;
    if (idx < n_samples) {
        val = y_samples[idx];
    } else {
        val = 0.0f;
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (threadIdx.x < threads / warp_size) {
        val = sdata[threadIdx.x];
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(result, ((b - a) / n_samples) * val);
        }
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    sum<<<blocks, threads>>>(y_samples, result, a, b, n_samples);
}
