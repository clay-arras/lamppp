#pragma once

#include <array>
#include <vector>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail {

/// @internal
/**
 * @brief Offset utility for CPU
 * @details this class is used to get the offset of the elements in the tensor.
 */
class OffsetUtil {
 public:
  explicit OffsetUtil(size_t ndim) : ndim(ndim) {};
  size_t ndim;

 protected:
  std::vector<stride_t> init_padded_strides(
      const std::vector<size_t>& shape, const std::vector<stride_t>& stride) const;
};
/// @endinternal

namespace cpu {

/// @internal
template <size_t NArgs>
class CPUOffsetUtil : public OffsetUtil {
 public:
  explicit CPUOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                         const TensorImpl& outs);
  ::std::array<stride_t, NArgs + 1> get(size_t idx) const;

  ::std::array<std::vector<stride_t>, NArgs + 1> arg_strides_;
};
/// @endinternal

template <size_t NArgs>
std::unique_ptr<OffsetUtil> offset_util_cpu(::std::array<const TensorImpl*, NArgs> ins, 
    const TensorImpl& out) {
  return std::make_unique<cpu::CPUOffsetUtil<NArgs>>(ins, out);
}

}  // namespace cpu

template <size_t NArgs>
using offset_util_fn = std::unique_ptr<OffsetUtil> (*)(::std::array<const TensorImpl*, NArgs>, const TensorImpl&);

LMP_DECLARE_DISPATCH(offset_util_fn<2>, offset_util_stub_2);
LMP_DECLARE_DISPATCH(offset_util_fn<3>, offset_util_stub_3);

};  // namespace lmp::tensor::detail