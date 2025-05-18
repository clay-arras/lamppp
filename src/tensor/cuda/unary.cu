#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include "lamppp/tensor/cuda/unary.cuh"

namespace lmp::tensor::detail::cuda {

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

template void unary_dispatch_handler<LogFunctor>(
    const internal::TensorMetaHandler&);

}  // namespace lmp::tensor::detail::cuda