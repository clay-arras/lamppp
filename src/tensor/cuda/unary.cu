#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/unary.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    ptr_.set_Out(i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], i)));
  }
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  vectorized_unary_kernel<<<blocks, threads>>>(ptr_, fn_, size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "unary_kernel_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      unary_kernel_launcher(
          internal::CUDAPtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(meta.in()[0]->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel());
    });
  });
}

template void unary_dispatch_handler<NegFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<ExpFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<LogFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SqrtFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<AbsFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SinFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<CosFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<TanFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<ClampFunctor>(UnaryMetaHandler&, Scalar&,
                                                   Scalar&);

}  // namespace lmp::tensor::detail::cuda