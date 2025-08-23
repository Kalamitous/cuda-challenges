#include <cuda_runtime.h>
#include <math_constants.h>

// self-imposed challenge of not being able to change block dim
const int THREADS_PER_BLOCK = 256;
const int WARP_SIZE = 32;

__global__ void find_max(const float* input, int* output, int N) {
    __shared__ float max[THREADS_PER_BLOCK / WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float val;
    if (idx < N) {
        val = input[idx];
    } else {
        val = -CUDART_INF_F;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane_id == 0) {
        max[warp_id] = val;
    }
    __syncthreads();
    if (threadIdx.x < THREADS_PER_BLOCK / WARP_SIZE) {
        val = max[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    if (threadIdx.x == 0) {
        atomicMax(output, __float_as_int(val));
    }
}

__global__ void compute_sum(const float* input, float* output, int N, int* max) {
    __shared__ float sum[THREADS_PER_BLOCK / WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float val;
    if (idx < N) {
        val = expf(input[idx] - __int_as_float(*max));
    } else {
        val = 0.0f;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) {
        sum[warp_id] = val;
    }
    __syncthreads();
    if (threadIdx.x < THREADS_PER_BLOCK / WARP_SIZE) {
        val = sum[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

__global__ void softmax_kernel(const float* input, float* output, int N, int* max, float* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    output[idx] = expf(input[idx] - __int_as_float(*max)) / *sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *max;
    cudaMalloc(&max, sizeof(int));
    cudaMemset(max, 0, sizeof(int));

    float *sum;
    cudaMalloc(&sum, sizeof(float));
    cudaMemset(sum, 0, sizeof(float));

    find_max<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, max, N);
    cudaDeviceSynchronize();
    compute_sum<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, sum, N, max);
    cudaDeviceSynchronize();
    softmax_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, output, N, max, sum);
    cudaDeviceSynchronize();

    cudaFree(max);
    cudaFree(sum);
}