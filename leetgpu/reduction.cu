#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void reduction(const float* input, float* output, int N) {
    __shared__ float smem[WARP_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? input[i] : 0.0f;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // first thread of each warp holds warp-level local sum

    if (N > WARP_SIZE) {
        if (threadIdx.x % WARP_SIZE == 0) {
            smem[threadIdx.x / WARP_SIZE] = val;
        }
        __syncthreads();
        // smem holds all warp-level local sums in block

        if (threadIdx.x < WARP_SIZE) {
            val = smem[threadIdx.x];
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
        }
        // first thread in block now holds block-level local sum 

        if (N > blockDim.x) {
            if (threadIdx.x == 0) {
                atomicAdd(output, val);
            }
            // global sum is sum of all block-level local sums
        } else {
            if (i == 0) {
                *output = val;
            }
            // global sum is block-level local sum since N <= blockDim.x
        }
    } else {
        if (i == 0) {
            *output = val;
        }
        // global sum is warp-level local sum since N is <= WARP_SIZE
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threads = WARP_SIZE * WARP_SIZE;
    int blocks = (N + threads - 1) / threads;
    reduction<<<blocks, threads>>>(input, output, N);
}