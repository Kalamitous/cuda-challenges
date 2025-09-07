#include <cuda_runtime.h>

const int threads = 256;
const int dim = 3;
const int tile_dim = threads;

__global__ void nearest_neighbor(const float* points, int* indices, int N) {
    __shared__ float tile[tile_dim * dim];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float x = points[idx * dim];
    float y = points[idx * dim + 1];
    float z = points[idx * dim + 2];
    
    int min_idx;
    float min_dist = INFINITY;
    for (int t = 0; t < N; t += tile_dim) {
        int global_idx = t + threadIdx.x;
        if (global_idx < N) {
            tile[threadIdx.x * dim] = points[global_idx * dim];
            tile[threadIdx.x * dim + 1] = points[global_idx * dim + 1];
            tile[threadIdx.x * dim + 2] = points[global_idx * dim + 2];
        }
        __syncthreads();

        int tile_max = min(tile_dim, N - t);
        for (int j = 0; j < tile_max; ++j) {
            int compare_idx = t + j;
            if (compare_idx == idx) continue;
            
            float jx = tile[j * dim];
            float jy = tile[j * dim + 1];
            float jz = tile[j * dim + 2];

            float dx = jx - x;
            float dy = jy - y;
            float dz = jz - z;

            float dist = dx * dx + dy * dy + dz * dz;
            if (dist < min_dist) {
                min_idx = compare_idx;
                min_dist = dist;
            }
        }
        __syncthreads();
    }

    if (idx < N) {
        indices[idx] = min_idx;
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int blocks = (N + threads - 1) / threads;
    nearest_neighbor<<<blocks, threads>>>(points, indices, N);
}