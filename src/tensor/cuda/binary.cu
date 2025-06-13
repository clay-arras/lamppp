#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include "lamppp/tensor/cuda/binary.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/list_ptr.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_binary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    ptr_.set_Out(i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], i),
                        ::cuda::std::get<2>(ptr_.fns)(ptr_.data[2], i)));
  }
}

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  vectorized_binary_kernel<<<blocks, threads>>>(ptr_, fn_, size);

  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "binary_kernel_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(meta.in()[1]->type(), [&] {
        using arg2_dtype_t = scalar_t;
        binary_kernel_launcher(
            internal::CUDAPtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                static_cast<out_dtype_t*>(meta.out().data()),
                static_cast<arg1_dtype_t*>(meta.in()[0]->data()),
                static_cast<arg2_dtype_t*>(meta.in()[1]->data())),
            OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
            meta.out().numel());
      });
    });
  });
}

template void binary_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<LtFunctor>(BinaryMetaHandler&);

}  // namespace lmp::tensor::detail::cuda