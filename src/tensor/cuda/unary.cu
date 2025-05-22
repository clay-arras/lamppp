#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/unary.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ptr_.set_Out(i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], i)));
  }
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vectorized_unary_kernel<<<blocks, threads>>>(ptr_, fn_, size);
  LMP_CUDA_ASSERT(cudaDeviceSynchronize(),
                  "unary_kernel_launcher: kernel failed.");
}

template void unary_dispatch_handler<ExpFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<LogFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<SqrtFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<AbsFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<SinFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<CosFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<TanFunctor>(
    const internal::TensorMetaHandler&);
template void unary_dispatch_handler<ClampFunctor>(
    const internal::TensorMetaHandler&, Scalar&&, Scalar&&);

}  // namespace lmp::tensor::detail::cuda