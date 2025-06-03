#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/align_utils.hpp"

namespace lmp::tensor::detail {

std::vector<stride_t> OffsetUtil::init_padded_strides(
    const std::vector<size_t>& shape, const std::vector<stride_t>& stride) const {
  LMP_INTERNAL_ASSERT(ndim > 0) << "ndim must be greater than 0";
  LMP_INTERNAL_ASSERT(shape.size() <= ndim)
      << "shape size must be less than or equal to ndim";
  LMP_INTERNAL_ASSERT(shape.size() == stride.size())
      << "shape size must be equal to stride size";

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

namespace cpu {

template <size_t NArgs>
CPUOffsetUtil<NArgs>::CPUOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                                    const TensorImpl& outs)
    : OffsetUtil(outs.shape().size()) {
  LMP_INTERNAL_ASSERT(NArgs == ins.size())
      << "NArgs must equal number of input elements";

  arg_strides_[0] = outs.strides();
#pragma omp unroll
  for (size_t i = 1; i <= NArgs; i++) {
    arg_strides_[i] =
        this->init_padded_strides(ins[i - 1]->shape(), ins[i - 1]->strides());
  }
}

template <size_t NArgs>
::std::array<stride_t, NArgs + 1> CPUOffsetUtil<NArgs>::get(size_t idx) const {
  ::std::array<stride_t, NArgs + 1> result;
  result.fill(0);
  result[0] = idx;

  for (size_t i = 0; i < this->ndim; ++i) {
    stride_t this_idx = idx / (arg_strides_[0])[i];
    idx = idx % (arg_strides_[0])[i];

#pragma omp unroll
    for (size_t j = 1; j <= NArgs; j++) {
      result[j] += (this_idx * (arg_strides_[j])[i]);
    }
  }
  return result;
}

}  // namespace cpu

LMP_DEFINE_DISPATCH(offset_util_fn<2>, offset_util_stub_2);
LMP_DEFINE_DISPATCH(offset_util_fn<3>, offset_util_stub_3);

namespace cpu {

template class CPUOffsetUtil<2>;
template class CPUOffsetUtil<3>;

namespace {
offset_util_fn<2> offset_util_cpu_2 = offset_util_cpu<2>;
offset_util_fn<3> offset_util_cpu_3 = offset_util_cpu<3>;
}

LMP_REGISTER_DISPATCH(offset_util_stub_2, DeviceType::CPU, offset_util_cpu_2);
LMP_REGISTER_DISPATCH(offset_util_stub_3, DeviceType::CPU, offset_util_cpu_3);

}  // namespace cpu

}  // namespace lmp::tensor::detail