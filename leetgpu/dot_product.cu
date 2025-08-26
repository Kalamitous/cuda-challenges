#include <cuda_runtime.h>

const int warp_size = 32;

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val;
    if (idx < N) {
        val = A[idx] * B[idx];
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

    if (threadIdx.x < blockDim.x / warp_size) {
        val = sdata[threadIdx.x];
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(result, val);
        }
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t sdata_size = (threads / warp_size) * sizeof(float);
    dot_product<<<blocks, threads, sdata_size>>>(A, B, result, N);
}