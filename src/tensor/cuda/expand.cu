#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include "lamppp/tensor/cuda/expand.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         const OffsetUtil<2>* align) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = align->get(i);
    ptr_.set_Out(i, fn_((ptr_.fns[1])(ptr_.data[1], offsets[1]),
                        (ptr_.fns[2])(ptr_.data[2], offsets[2])));
  }
}

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const OffsetUtil<2>* align) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  ListDevicePtr<OffsetUtil<2>> d_align(align, 1);
  vectorized_expand_kernel<<<blocks, threads>>>(ptr_, fn_, size, d_align.get());

  cudaError_t sync_err = cudaDeviceSynchronize();
  // assert(sync_err == cudaSuccess && "expand_kernel_launcher: kernel failed.");
  if (sync_err != cudaSuccess) {
    const char* err_str = cudaGetErrorString(sync_err);
    printf("expand_kernel_launcher: kernel failed with error: %s\n", err_str);
    assert(false);
  }
}

template void expand_dispatch_handler<AddFunctor>(
    const internal::TensorMetaHandler&);

}  // namespace lmp::tensor::detail::cuda