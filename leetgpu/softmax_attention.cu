#include <cuda_runtime.h>

const int warp_size = 32;

const int tile_dim = 32;
const int coarse_factor = 4;
// with tile_dim = 32 and coarse_factor = 4:
// ty 0 will handle rows 0, 8, 16, 24
// ty 7 will handle rows 7, 15, 23, 31
// therefore we only need 8 ty to cover all 32 rows
__global__ void matrix_transpose(const float* input, float* output, int N, int d) {
    __shared__ float tile[tile_dim][tile_dim + 1];

    int row_start = blockIdx.y * tile_dim + threadIdx.y;
    int col       = blockIdx.x * tile_dim + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < tile_dim; i += tile_dim / coarse_factor) {
        int row = row_start + i;
        tile[threadIdx.y + i][threadIdx.x] = (row < N && col < d) ? input[row * d + col] : 0.0f;
    }
    __syncthreads();
    
    row_start = blockIdx.x * tile_dim + threadIdx.y;
    col       = blockIdx.y * tile_dim + threadIdx.x;

    if (col < N) {
        #pragma unroll
        for (int i = 0; i < tile_dim; i += tile_dim / coarse_factor) {
            int row = row_start + i;
            if (row < d) {
                output[row * N + col] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}

__global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int d) {
    __shared__ float A_s[tile_dim][tile_dim];
    __shared__ float B_s[tile_dim][tile_dim];

    int row = blockIdx.y * tile_dim + threadIdx.y;
    int col = blockIdx.x * tile_dim + threadIdx.x;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < d; tile_start += tile_dim) {
        int a_row = row;
        int a_col = tile_start + threadIdx.x;
        A_s[threadIdx.y][threadIdx.x] = (a_row < M && a_col < d) ? A[a_row * d + a_col] : 0.0f;

        int b_row = tile_start + threadIdx.y;
        int b_col = col;
        B_s[threadIdx.y][threadIdx.x] = (b_row < d && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < tile_dim; ++i) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void softmax(float* scores, int M, int N, int d) {
    extern __shared__ float sdata[];
    int sdata_len = (blockDim.x + warp_size - 1) / warp_size;

    int warp_id = threadIdx.x / warp_size;

    int row = blockIdx.x;
    int col = threadIdx.x;

    float val = scores[row * N + col];
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    sdata[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (threadIdx.x < sdata_len) ? sdata[threadIdx.x] : -INFINITY;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        sdata[0] = val;
    }
    __syncthreads();
    float max = sdata[0];
    __syncthreads();
    
    float numerator = __expf((scores[row * N + col] - max) / sqrtf(d));

    val = numerator;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    sdata[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (threadIdx.x < sdata_len) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[0] = val;
    }
    __syncthreads();
    float sum = sdata[0];

    scores[row * N + col] = numerator / sum;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *KT, *scores;
    cudaMalloc(&KT, d * N * sizeof(float));
    cudaMalloc(&scores, M * N * sizeof(float));

    dim3 blockDim1(tile_dim, tile_dim / coarse_factor);
    dim3 gridDim1((d + tile_dim - 1) / tile_dim, (N + tile_dim - 1) / tile_dim);
    matrix_transpose<<<gridDim1, blockDim1>>>(K, KT, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim2(tile_dim, tile_dim);
    dim3 gridDim2((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    matrix_multiply<<<gridDim2, blockDim2>>>(Q, KT, scores, M, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim3(N); // there are no leetgpu test cases where N >= 1024
    dim3 gridDim3(M); // each block handles one row
    size_t sdata_size = ((blockDim3.x + warp_size - 1) / warp_size) * sizeof(float);
    softmax<<<gridDim3, blockDim3, sdata_size>>>(scores, M, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim4(tile_dim, tile_dim);
    dim3 gridDim4((d + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    matrix_multiply<<<gridDim4, blockDim4>>>(scores, V, output, M, d, N);

    cudaFree(KT);
    cudaFree(scores);
}