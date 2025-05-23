#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/align_utils.hpp"

namespace lmp::tensor::detail {

namespace cpu {

template <size_t NArgs>
CPUOffsetUtil<NArgs>::CPUOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                                    const TensorImpl& outs)
    : OffsetUtil<NArgs>(outs.shape().size()) {
  LMP_INTERNAL_ASSERT(NArgs == ins.size(),
                      "NArgs must equal number of input elements");

#pragma omp unroll
  for (size_t i = 0; i < NArgs; i++) {
    arg_strides_[i] =
        this->init_padded_strides_(ins[i]->shape(), ins[i]->strides());
    arg_pointers_[i] = arg_strides_[i].data();
  }
}

template <size_t NArgs>
::std::array<stride_t, NArgs + 1> CPUOffsetUtil<NArgs>::get(size_t idx) const {
  ::std::array<stride_t, NArgs + 1> result;
  result.fill(0);
  result[0] = idx;

  for (size_t i = 0; i < this->ndim; ++i) {
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

}  // namespace cpu

template <size_t NArgs>
std::vector<stride_t> OffsetUtil<NArgs>::init_padded_strides_(
    const std::vector<size_t>& shape, const std::vector<stride_t>& stride) {
  LMP_INTERNAL_ASSERT(ndim > 0, "ndim must be greater than 0");
  LMP_INTERNAL_ASSERT(shape.size() <= ndim,
                      "shape size must be less than or equal to ndim");
  LMP_INTERNAL_ASSERT(shape.size() == stride.size(),
                      "shape size must be equal to stride size");

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

template class cpu::CPUOffsetUtil<2>;
template class cpu::CPUOffsetUtil<3>;

}  // namespace lmp::tensor::detail