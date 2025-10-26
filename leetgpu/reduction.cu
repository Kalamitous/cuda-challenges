#include <cuda_runtime.h>

const int warp_size = 32;

// more atomicAdds can result in rounding errors
const int threads = 256;
const int coarse_factor = 8;
const int inputs_per_thread = threads * coarse_factor;

__global__ void reduction(const float* input, float* output, int N) {
    __shared__ float sdata[threads / warp_size];

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int idx = blockIdx.x * inputs_per_thread + threadIdx.x;

    float val = 0.0f;
    #pragma unroll
    for (int stride = 0; stride < inputs_per_thread; stride += threads) {
        int i = idx + stride;
        val += (i < N) ? input[i] : 0.0f;
    }

    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < threads / warp_size) ? sdata[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) atomicAdd(output, val);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int blocks = (N + inputs_per_thread - 1) / inputs_per_thread;
    reduction<<<blocks, threads>>>(input, output, N);
}