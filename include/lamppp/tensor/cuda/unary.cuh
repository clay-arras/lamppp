#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cmath>
#include <cuda/std/detail/libcxx/include/array>
#include "lamppp/tensor/cuda/utils.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

using UnaryOpPtrList = ::cuda::std::array<void*, 2>;
using BinaryOpPtrList = ::cuda::std::array<void*, 3>;

template <typename OutType, typename InType>
class LogFunctor {
 public:
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::log(static_cast<double>(in_data[index])));
  }
};

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    fn_(ptr_, i);
  }
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vectorized_unary_kernel<<<blocks, threads>>>(ptr_, fn_, size);
  cudaError_t sync_err = cudaDeviceSynchronize();
  assert(sync_err == cudaSuccess && "unary_kernel_launcher: kernel failed.");
}

template <template <typename, typename> class OpFunctor>
void unary_dispatch_handler(const internal::TensorMetaHandler& meta) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0].type(), [&] {
      using arg_dtype_t = scalar_t;
      unary_kernel_launcher(
          internal::pack_tens<1>(meta.in(), meta.out().data()),
          OpFunctor<out_dtype_t, arg_dtype_t>(), meta.out().size());
    });
  });
}

}  // namespace lmp::tensor::detail::cuda