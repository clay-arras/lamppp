#include <cstring>
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/native/copy.cuh"
#include "lamppp/tensor/supported_types.hpp"

namespace lmp::tensor::detail::native {

LMP_DEFINE_DISPATCH(copy_fn, copy_stub);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;
          vecCopy(size, static_cast<const src_type*>(src),
                  static_cast<dest_type*>(dest));
        });
      });

      // memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(src_type)),
                          "copy_cpu to CUDA: cudaMalloc for tmp failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(tmp, src, size * sizeof(src_type),
                                     cudaMemcpyHostToDevice),
                          "copy_cpu to CUDA: cudaMemcpy HtoD for tmp failed.");

          cudaVecCopy<src_type, dest_type>(size,
                                           static_cast<const src_type*>(tmp),
                                           static_cast<dest_type*>(dest));

          LMP_CUDA_ASSERT(cudaGetLastError(),
                          "copy_cpu to CUDA: vecCopy kernel failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cpu to CUDA: cudaFree for tmp failed.");
        });
      });
      break;
    }
  }
}

void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(dest_type)),
                          "copy_cuda to CPU: cudaMalloc for tmp failed.");

          cudaVecCopy<src_type, dest_type>(size,
                                           static_cast<const src_type*>(src),
                                           static_cast<dest_type*>(tmp));
          LMP_CUDA_ASSERT(
              cudaGetLastError(),
              "copy_cuda to CPU: vecCopy kernel launch or execution failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                                     cudaMemcpyDeviceToHost),
                          "copy_cuda to CPU: cudaMemcpy DtoH failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cuda to CPU: cudaFree for tmp failed.");
        });
      });
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(dest_type)),
                          "copy_cuda to CUDA: cudaMalloc for tmp failed.");

          cudaVecCopy<src_type, dest_type>(size,
                                           static_cast<const src_type*>(src),
                                           static_cast<dest_type*>(tmp));

          LMP_CUDA_ASSERT(cudaGetLastError(),
                          "copy_cuda to CUDA: vecCopy kernel failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                                     cudaMemcpyDeviceToDevice),
                          "copy_cuda to CUDA: cudaMemcpy DtoD failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cuda to CUDA: cudaFree for tmp failed.");
        });
      });
      break;
    }
  }
}

template <typename U, typename V>
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = static_cast<V>(in[i]);
  }
}

// TODO: need to make it more clear WHEN something is size and when something is byteSize; should be in function signature
template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  cudaVecCopyKernel<U, V><<<blocks, threads>>>(size, in, out);
}

template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    out[i] = in[i];
  }
}

#include "lamppp/tensor/supported_types.hpp"

#define INSTANTIATE_COPY(arg1_type, arg2_type) \
  template void cudaVecCopy<arg1_type, arg2_type>( \
      size_t, const arg1_type*, arg2_type*); \
  template void vecCopy<arg1_type, arg2_type>( \
      size_t, const arg1_type*, arg2_type*);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_COPY, LMP_LIST_TYPES, LMP_LIST_TYPES)
#undef INSTANTIATE_COPY

LMP_REGISTER_DISPATCH(copy_stub, DeviceType::CPU, copy_cpu);
LMP_REGISTER_DISPATCH(copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace lmp::tensor::detail::native