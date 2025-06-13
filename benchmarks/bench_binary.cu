#include "bench_binary.cuh"
#include <chrono>

__global__ void vectorAdd(const float* a, const float* b, float* result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        result[i] = a[i] + b[i];
    }
}

__global__ void initializeRandomArray(float* array, int n, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed, tid, 0, &state);
    
    for (int i = tid; i < n; i += stride) {
        array[i] = curand_uniform(&state);
    }
}

CudaDataPtr create_cuda_data_ptr(size_t num_elements) {
    size_t size = num_elements * sizeof(float);
    float* device_ptr;
    
    cudaError_t error = cudaMalloc(&device_ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory: " + std::string(cudaGetErrorString(error)));
    }
    
    auto cuda_deleter = [](void* ptr) {
        if (ptr) {
            cudaFreeAsync(ptr, nullptr);
        }
    };
    
    return CudaDataPtr(device_ptr, cuda_deleter);
}

void launch_vector_add(const float* a, const float* b, float* result, int n) {
    size_t numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    numBlocks = std::min(numBlocks, 1024UL);
    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(a, b, result, n);
}

void launch_initialize_random_array(float* array, int n, unsigned long long seed) {
    size_t numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    numBlocks = std::min(numBlocks, 1024UL); 
    initializeRandomArray<<<numBlocks, THREADS_PER_BLOCK>>>(array, n, seed);
}


