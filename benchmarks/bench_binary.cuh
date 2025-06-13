#ifndef BENCH_BINARY_CUH
#define BENCH_BINARY_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>
#include <functional>

#define VECTOR_SIZE (512 * 512)
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(const float* a, const float* b, float* result, int n);
__global__ void initializeRandomArray(float* array, int n, unsigned long long seed);

class CudaDataPtr {
private:
    std::shared_ptr<void> ptr;

public:
    CudaDataPtr() = default;
    CudaDataPtr(void* data_ptr, std::function<void(void*)> dealloc)
        : ptr(data_ptr, [dealloc](void* data) {
            try {
                dealloc(data);
            } catch (const std::exception& e) {
                std::cerr << "CudaDataPtr deleter error: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "CudaDataPtr deleter threw non-std::exception\n";
            }
        }) {}

    void* data() const { return ptr.get(); }
    float* data_as_float() const { return static_cast<float*>(ptr.get()); }
};

CudaDataPtr create_cuda_data_ptr(size_t num_elements);

// Wrapper functions to launch CUDA kernels
void launch_vector_add(const float* a, const float* b, float* result, int n);
void launch_initialize_random_array(float* array, int n, unsigned long long seed);

#endif // BENCH_BINARY_CUH
