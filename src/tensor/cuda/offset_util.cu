#include <cassert>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

template <size_t NArgs>
OffsetUtil<NArgs>::OffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                              const TensorImpl& outs)
    : ndim(outs.shape().size()) {

  assert(NArgs == ins.size());
  std::vector<std::vector<stride_t>> stride_exp(NArgs);

#pragma omp unroll
  for (size_t i = 0; i < NArgs; i++) {
    stride_exp[i] = init_padded_strides_(ins[i]->shape(), ins[i]->strides());
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
__device__ ::cuda::std::array<stride_t, NArgs + 1> OffsetUtil<NArgs>::get(
    size_t idx) const {
  ::cuda::std::array<stride_t, NArgs + 1> result;
  result.fill(0);
  result[0] = idx;

  for (size_t i = 0; i < ndim; ++i) {
    stride_t this_idx = idx / static_cast<const stride_t*>(arg_pointers_[0])[i];
    idx = idx % static_cast<const stride_t*>(arg_pointers_[0])[i];

#pragma omp unroll
    for (size_t j = 1; j <= NArgs; j++) {
      result[j] +=
          (this_idx * static_cast<const stride_t*>(arg_pointers_[j])[i]);
    }
  }

  return result;
}

template <size_t NArgs>
std::vector<stride_t> OffsetUtil<NArgs>::init_padded_strides_(
    const std::vector<size_t>& shape, const std::vector<stride_t>& stride) {
  assert(ndim > 0);
  assert(shape.size() <= ndim);
  assert(shape.size() == stride.size());

  std::vector<stride_t> padded(ndim, 0);
  const size_t from_back = shape.size();

  for (size_t k = 0; k < from_back; ++k) {
    size_t dst = ndim - 1 - k;
    size_t src = from_back - 1 - k;

    if (shape[src] != 1) {
      padded[dst] = stride[src];
    }
  }
  return padded;
}

template class OffsetUtil<2>;
template class OffsetUtil<3>;  // in case for 3 argument operands, I guess

}  // namespace lmp::tensor::detail::cuda