#include <cuda_runtime.h>

const int warp_size = 32;
const int threads = 256;

__global__ void compute_mean(const float* input, float* mean, int N, int C) {
    __shared__ float sdata[threads / warp_size];
    
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int j = blockIdx.x;

    float val = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val += input[i * C + j];
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < threads / warp_size) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) mean[j] = val / N;
    }
}

__global__ void compute_variance(const float* input, float* mean, float* var, int N, int C) {
    __shared__ float sdata[threads / warp_size];
    
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int j = blockIdx.x;

    float val = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float delta = input[i * C + j] - mean[j];
        val += delta * delta;
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < threads / warp_size) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) var[j] = val / N;
    }
}

__global__ void normalize(const float* input, const float* mean, const float* var, const float* gamma,
                          const float* beta, float* output, int N, int C, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < C) {
        float x_norm = (input[i * C + j] - mean[j]) / sqrtf(var[j] + eps);
        output[i * C + j] = gamma[j] * x_norm + beta[j];
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    float *mean, *var;
    cudaMalloc(&mean, C * sizeof(float));
    cudaMalloc(&var, C * sizeof(float));

    compute_mean<<<C, threads>>>(input, mean, N, C);
    cudaDeviceSynchronize();

    compute_variance<<<C, threads>>>(input, mean, var, N, C);
    cudaDeviceSynchronize();

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (C + blockDim.y - 1) / blockDim.y);
    normalize<<<gridDim, blockDim>>>(input, mean, var, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();

    cudaFree(mean);
    cudaFree(var);
}
