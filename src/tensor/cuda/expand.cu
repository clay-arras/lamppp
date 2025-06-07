#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include "lamppp/tensor/cuda/expand.cuh"
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         const CUDAOffsetUtil<kNArgs>* align) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    ::cuda::std::array offsets = align->get(i);
    ptr_.set_Out(i,
                 fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], offsets[1]),
                     ::cuda::std::get<2>(ptr_.fns)(ptr_.data[2], offsets[2])));
  }
}

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const CUDAOffsetUtil<kNArgs>* align) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  ListDevicePtr<CUDAOffsetUtil<kNArgs>> d_align(align, 1);
  vectorized_expand_kernel<<<blocks, threads>>>(ptr_, fn_, size, d_align.get());

  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "expand_kernel_launcher: kernel failed.";
}

template void expand_dispatch_handler<AddFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<SubFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<MulFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<DivFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<PowFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<EqFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<NeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<GeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<GtFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<LeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<LtFunctor>(ExpandMetaHandler&);

}  // namespace lmp::tensor::detail::cuda