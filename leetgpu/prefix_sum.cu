#include <cuda_runtime.h>

// Hillis–Steele inclusive scan w/ double-buffering & host-recursive grid-level accumulation
// todo: warp-level accumulation before block-level accumulation

const int threads = 1024;

__global__ void prefix_sum(const float* input, float* output, float* partial_sums, int N) {
    __shared__ float buf1[threads];
    __shared__ float buf2[threads];

    float* input_s = buf1;
    float* output_s = buf2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    input_s[threadIdx.x] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
        if (threadIdx.x >= stride) {
            output_s[threadIdx.x] = input_s[threadIdx.x] + input_s[threadIdx.x - stride];
        } else {
            output_s[threadIdx.x] = input_s[threadIdx.x];
        }
        __syncthreads();

        float* tmp = input_s;
        input_s = output_s;
        output_s = tmp;
    }

    if (partial_sums && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = input_s[threadIdx.x];
    }

    if (idx < N) {
        output[idx] = input_s[threadIdx.x];
    }
}

__global__ void add_block_prefix_sum(float* output, const float* block_prefix_sums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 || idx >= N) return;
    
    __shared__ float prefix_sum;
    if (threadIdx.x == 0) {
        prefix_sum = block_prefix_sums[blockIdx.x - 1];
    }
    __syncthreads();

    output[idx] += prefix_sum;
}

void recurse(const float* input, float* output, int N) {
    int blocks = (N + threads - 1) / threads;

    if (blocks == 1) {
        return prefix_sum<<<1, threads>>>(input, output, nullptr, N);
    }

    float* block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(float));
    prefix_sum<<<blocks, threads>>>(input, output, block_sums, N);

    float* block_prefix_sums;
    cudaMalloc(&block_prefix_sums, blocks * sizeof(float));
    recurse(block_sums, block_prefix_sums, blocks);

    add_block_prefix_sum<<<blocks, threads>>>(output, block_prefix_sums, N);

    cudaFree(block_sums);
    cudaFree(block_prefix_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    recurse(input, output, N);
} 