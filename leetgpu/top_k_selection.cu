#include <cuda_runtime.h>

// todo: using radix sort would allow us to minimize sort steps
// i.e. there is a way to not need to sort the entire input to get the top k
// would also let us not need to pad input to power of 2

const int threads = 256;

__global__ void bitonic_sort_step(float* data, int j, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;
    if (ixj > i) {
        float a = data[i];
        float b = data[ixj];
        if ((i & k) == 0) {
            if (a < b) {
                data[i] = b;
                data[ixj] = a;
            }
        } else {
            if (a > b) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
}

__global__ void copy_k_elements(const float* input, float* output, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        output[i] = input[i];
    }
}

__global__ void fill_pad(float* input, int start, int end, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + start < end) {
        input[start + i] = val;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int N_pow2 = 1;
    while (N_pow2 < N) {
        N_pow2 <<= 1;
    }
    
    float* input_copy;
    cudaMalloc(&input_copy, N_pow2 * sizeof(float));
    cudaMemcpy(input_copy, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    fill_pad<<<(N_pow2 - N + threads - 1) / threads, threads>>>(input_copy, N, N_pow2, -INFINITY);

    for (int _k = 2; _k <= N_pow2; _k *= 2) {
        for (int j = _k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<(N_pow2 + threads - 1) / threads, threads>>>(input_copy, j, _k);
        }
    }

    copy_k_elements<<<(k + threads - 1) / threads, threads>>>(input_copy, output, k);

    cudaFree(input_copy);
}