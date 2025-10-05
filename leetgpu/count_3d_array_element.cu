#include <cuda_runtime.h>

const int warp_size = 32;

// __global__ void count(const int* input, int* output, int N, int M, int K, int P) {
//     extern __shared__ float sdata[];
//     int sdata_len = blockDim.x * blockDim.y / warp_size;

//     int n = blockIdx.z;
//     int m = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.x * blockDim.x + threadIdx.x;

//     int local_id = threadIdx.y * blockDim.x + threadIdx.x;
//     int warp_id = local_id / warp_size;
//     int lane_id = local_id % warp_size;

//     int val = (m < M && k < K) ? input[n * M * K + m * K + k] == P : 0;
//     for (int offset = warp_size / 2; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     }
//     sdata[warp_id] = val;
//     __syncthreads();

//     if (warp_id == 0 && lane_id < sdata_len) {
//         val = sdata[lane_id];
//         for (int offset = warp_size / 2; offset > 0; offset /= 2) {
//             val += __shfl_down_sync(0xffffffff, val, offset);
//         }
//         if (lane_id == 0) {
//             atomicAdd(output, val);
//         }
//     }
// }

__global__ void count_flattened(const int* input, int* output, int N, int M, int K, int P) {
    extern __shared__ float sdata[];
    int sdata_len = blockDim.x / warp_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx / (M * K);
    int m = (idx / K) % M;
    int k = idx % K;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int val = (m < M && k < K) ? input[n * M * K + m * K + k] == P : 0;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    sdata[warp_id] = val;
    __syncthreads();

    if (threadIdx.x < sdata_len) {
        val = sdata[lane_id];
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    // dim3 blockDim(warp_size, warp_size);
    // dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, N);
    // int sdata_size = (blockDim.x * blockDim.y / warp_size) * sizeof(float);
    // count<<<gridDim, blockDim, sdata_size>>>(input, output, N, M, K, P);
    
    int threads = warp_size * warp_size;
    int blocks = (N * M * K + threads - 1) / threads;
    int sdata_size = (threads / warp_size) * sizeof(float);
    count_flattened<<<blocks, threads, sdata_size>>>(input, output, N, M, K, P);
}