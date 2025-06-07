#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

template <size_t NArgs>
CUDAOffsetUtil<NArgs>::CUDAOffsetUtil(
    ::std::array<const TensorImpl*, NArgs> ins, const TensorImpl& outs)
    : OffsetUtil(outs.shape().size()) {
  LMP_INTERNAL_ASSERT(NArgs == ins.size())
      << "NArgs must equal number of input elements";
  std::vector<std::vector<stride_t>> stride_exp(NArgs);

#pragma omp unroll
  for (size_t i = 0; i < NArgs; i++) {
    stride_exp[i] =
        this->init_padded_strides(ins[i]->shape(), ins[i]->strides());
  }

  arg_strides_[0] =
      ListDevicePtr<stride_t>(outs.strides().data(), outs.strides().size());
  arg_pointers_[0] = arg_strides_[0].get();

#pragma omp unroll
  for (size_t i = 1; i <= NArgs; i++) {
    arg_strides_[i] = ListDevicePtr<stride_t>(stride_exp[i - 1].data(),
                                              stride_exp[i - 1].size());
    arg_pointers_[i] = arg_strides_[i].get();
  }
}

template <size_t NArgs>
__device__ ::cuda::std::array<stride_t, NArgs + 1> CUDAOffsetUtil<NArgs>::get(
    size_t idx) const {
  ::cuda::std::array<stride_t, NArgs + 1> result;
  result.fill(0);
  result[0] = idx;

  for (size_t i = 0; i < this->ndim; ++i) {
    stride_t stride = static_cast<const stride_t*>(arg_pointers_[0])[i];
    stride_t this_idx = idx / stride; 
    idx = idx % stride;

#pragma omp unroll
    for (size_t j = 1; j <= NArgs; j++) {
      result[j] +=
          (this_idx * static_cast<const stride_t*>(arg_pointers_[j])[i]);
    }
  }

  return result;
}

template class CUDAOffsetUtil<2>;
template class CUDAOffsetUtil<3>;

namespace {
offset_util_fn<2> offset_util_cuda_2 = offset_util_cuda<2>;
offset_util_fn<3> offset_util_cuda_3 = offset_util_cuda<3>;
}

LMP_REGISTER_DISPATCH(offset_util_stub_2, DeviceType::CUDA,
                      offset_util_cuda_2);
LMP_REGISTER_DISPATCH(offset_util_stub_3, DeviceType::CUDA,
                      offset_util_cuda_3);

}  // namespace lmp::tensor::detail::cuda
