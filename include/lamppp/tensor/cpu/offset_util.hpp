#pragma once

#include <array>
#include <vector>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail {

// class OffsetUtil {};

template <size_t NArgs>
class OffsetUtil {
 public:
  OffsetUtil(size_t ndim) : ndim(ndim) {};
  static constexpr size_t NVars = NArgs + 1;
  size_t ndim;

 protected:
  std::vector<stride_t> init_padded_strides_(
      const std::vector<size_t>& shape, const std::vector<stride_t>& stride);
};

namespace cpu {

template <size_t NArgs>
class CPUOffsetUtil : public OffsetUtil<NArgs> {
 public:
  explicit CPUOffsetUtil(::std::array<const TensorImpl*, NArgs> ins,
                         const TensorImpl& outs);
  ::std::array<stride_t, NArgs + 1> get(size_t idx) const;

  ::std::array<std::vector<stride_t>, NArgs + 1> arg_strides_;
};

}  // namespace cpu

};  // namespace lmp::tensor::detail