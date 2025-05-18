#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include <iostream>
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

namespace internal {

template <size_t N>
::cuda::std::array<void*, N + 1> pack_tens(tensor_list tens, void* out) {
  ::cuda::std::array<void*, N + 1> arr;
  assert(tens.size() == N && "pack_tens: tensor list size mismatch with N");
  arr[0] = static_cast<void*>(out);
  // #pragma omp unroll
  for (size_t i = 1; i <= N; i++) {
    arr[i] = static_cast<void*>(tens[i - 1].data());
  }
  return arr;
}

}  // namespace internal

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn& fn_, size_t size) {
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
  if (sync_err != cudaSuccess) {
    std::cerr << "unary_kernel_launcher: CUDA kernel execution failed "
                 "(cudaDeviceSynchronize): "
              << cudaGetErrorString(sync_err) << std::endl;
    assert(sync_err == cudaSuccess &&
           "unary_kernel_launcher: CUDA kernel execution (runtime) failed.");
  }
}

template <template <typename, typename> class OpFunctor>
void unary_gpu_kernel(const internal::TensorMetaHandler& meta) {
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

TensorImpl log_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_gpu_kernel<LogFunctor>(meta);
  return meta.out();
}

}  // namespace lmp::tensor::detail::cuda