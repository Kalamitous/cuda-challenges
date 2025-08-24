#include <cuda_runtime.h>

// reference: https://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

__global__ void bitonic_sort_step(float* data, int j, int k) {
    // shared memory is not useful here since each thread accesses unique elements in data

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j; // the index j distance away from i

    // avoids duplicate comparison when future i = current j for example
    if (ixj > i) {
        float a = data[i];
        float b = data[ixj];
        if (i & k == 0) {
            // sort ascending
            if (a > b) {
                data[i] = b;
                data[ixj] = a;
            }
        } else {
            // sort descending
            if (a < b) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // k selects the bit pos that determines ascending/descending exchange
    // k is also the length of the subsequences we are sorting
    // k <= N because the final iteration merges the two halves of the full sequence
    for (int k = 2; k <= N; k *= 2) {
        // j is the distance between the elements we compare
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<threads, blocks>>>(data, j, k);
        }
    }
}