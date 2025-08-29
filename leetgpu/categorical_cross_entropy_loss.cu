#include <cuda_runtime.h>

const int warp_size = 32;

__global__ void cross_entropy_loss(const float* logits, const int* true_labels, float* loss, int N, int C) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val;
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < C; ++k) {
            sum += expf(logits[idx * C + k]);
        }
        val = (logf(sum) - logits[idx * C + true_labels[idx]]) / N;
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
            atomicAdd(loss, val);
        }
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t sdata_size = (threads / warp_size) * sizeof(float);
    cross_entropy_loss<<<blocks, threads, sdata_size>>>(logits, true_labels, loss, N, C);
}