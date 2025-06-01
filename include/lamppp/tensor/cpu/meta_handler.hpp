#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <memory>
#include <vector>
#include "lamppp/tensor/cpu/offset_util.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail {

using tensor_list = std::vector<const lmp::tensor::TensorImpl*>;

template <typename... Args>
class TensorMetaHandler {
 public:
  static constexpr std::size_t NumElem =
      (0 + ... + std::size_t{std::is_same_v<const TensorImpl*, Args>});
  explicit TensorMetaHandler(Args... args);

  inline TensorImpl& out() noexcept { return *outTen; }
  inline tensor_list& in() noexcept { return inTens; }
  inline const OffsetUtil* offset() const noexcept {
    return outOffset.get();
  }

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  std::unique_ptr<OffsetUtil> outOffset;
  std::unique_ptr<TensorImpl> outTen;
  tensor_list inTens;
};

using UnaryMetaHandler = TensorMetaHandler<const TensorImpl*>;
using ExpandMetaHandler =
    TensorMetaHandler<const TensorImpl*, const TensorImpl*>;
using ReductMetaHandler = TensorMetaHandler<const TensorImpl*, size_t>;

}  // namespace lmp::tensor::detail