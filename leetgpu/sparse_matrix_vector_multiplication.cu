#include <cuda_runtime.h>

const int warp_size = 32;

__global__ void mul(const float* A, const float* x, float* y, int M, int N) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;

    float val = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        val += A[idx * N + i] * x[i];
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (tid < blockDim.x / warp_size) {
        val = sdata[tid];
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            y[idx] = val;
        }
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threads = 256;
    int blocks = M;
    size_t sdata_size = (threads / warp_size) * sizeof(float);
    mul<<<blocks, threads, sdata_size>>>(A, x, y, M, N);
} 