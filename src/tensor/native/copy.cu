#include <cstring>
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/dispatch_stub.hpp"
#include "include/lamppp/tensor/dispatch_type.hpp"
#include "include/lamppp/tensor/native/copy.cuh"

namespace lmp::tensor::detail::native {

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      assert(false && "Not implemented");
      memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          cudaMalloc(&tmp, size * sizeof(src_type));
          cudaMemcpy(tmp, src, size * sizeof(src_type), cudaMemcpyHostToDevice);
          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(tmp),
                                       static_cast<dest_type*>(dest));
          cudaFree(tmp);
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
          cudaMalloc(&tmp, size * sizeof(dest_type));
          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));
          cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                     cudaMemcpyDeviceToHost);
          cudaFree(tmp);
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
          cudaMalloc(&tmp, size * sizeof(dest_type));
          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));
          cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                     cudaMemcpyDeviceToDevice);
          cudaFree(tmp);
        });
      });
      break;
    }
  }
}

template <typename U, typename V>
__global__ void vecCopyKernel(size_t size, const U* in, V* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = static_cast<V>(in[i]);
  }
}

// TODO: need to make it more clear WHEN something is size and when something is byteSize; should be in function signature
template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecCopyKernel<U, V><<<blocks, threads>>>(size, in, out);
}

// clang-format off
// NOTE: this is the most scuffed code I've ever wrote. this is so so bad
// TODO: make this fill from supported_types.def
#define TYPES(X, ...)   \
  X(__VA_ARGS__, int)   \
  X(__VA_ARGS__, float) \
  X(__VA_ARGS__, double)

#define CAST(U, V) template void vecCopy<U, V>(size_t, const U*, V*);

#define EXPAND(X) X
#define TYPES1() TYPES
#define TYPES2(...) TYPES1 EXPAND(())(__VA_ARGS__)

EXPAND(TYPES(TYPES2, CAST))
// clang-format on

LMP_DEFINE_DISPATCH(copy_stub);

}  // namespace lmp::tensor::detail::native