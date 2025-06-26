#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdatomic.h>
#include <atomic>
#include <cuda/std/array>
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/data_ptr.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);
DataPtr empty_cuda(size_t byte_size);
void fill_cuda(void* ptr, size_t size, Scalar t, DataType type);
void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);

template <typename U, typename V>
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out);
template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out);

template <typename T>
__global__ void cudaVecFillKernel(size_t size, T* out, T value);
template <typename T>
void cudaVecFill(size_t size, T* out, T value);

void vecCopyHostToDevice(const void* src, void* dest, size_t size,
                         DataType src_dtype, DataType dest_dtype);
/// @endinternal

class CudaStreamManager {
 public:
  static CudaStreamManager& instance() {
    static CudaStreamManager mgr;
    return mgr;
  } // TODO(nx2372): WARNING ERROR

  void onFree(size_t bytes) { // TODO(nx2372): I have to fix this later, I don't even know if half of these are effective
    size_t old_size = counter_.fetch_add(bytes, std::memory_order_relaxed);
    if (old_size + bytes >= kSyncThreshold &&
        !sync_.test_and_set(std::memory_order_acquire)) {
      LMP_CUDA_CHECK(cudaStreamSynchronize(nullptr));
      LMP_CUDA_CHECK(cudaDeviceSynchronize());

      int num_devices;
      LMP_CUDA_CHECK(cudaGetDeviceCount(&num_devices));

      for (size_t i = 0; i < num_devices; i++) {
        cudaMemPool_t pool;
        LMP_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, i));
        cudaMemPoolTrimTo(pool, kCacheThreshold);
      }

      sync_.clear(std::memory_order_release);
      counter_.store(0, std::memory_order_relaxed);
    }
  }

 private: 
  CudaStreamManager() {
    int num_devices;
    LMP_CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    for (size_t i = 0; i < num_devices; i++) {
      cudaMemPool_t pool;
      LMP_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, i));
      size_t cache_thresh = kCacheThreshold;
      LMP_CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                              &cache_thresh));
    }
  };

  std::atomic<size_t> counter_{0};
  std::atomic_flag sync_ = ATOMIC_FLAG_INIT;
  static const size_t kSyncThreshold = 16ULL << 20;     // 16Mib
  static const size_t kCacheThreshold = 64ULL << 20;  
};

}  // namespace lmp::tensor::detail::cuda