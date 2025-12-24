#pragma once

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
  static constexpr std::size_t kNumElem =
      (0 + ... + std::size_t{std::is_same_v<const TensorImpl*, Args>});
  explicit TensorMetaHandler(Args... args);

  TensorImpl& out() noexcept { return *outTen_; }
  tensor_list& in() noexcept { return inTens_; }
  const OffsetUtil* offset() const noexcept {
    LMP_INTERNAL_ASSERT(expand_) << "Must have expand = True to get offset";
    return outOffset_.get();
  }
  bool expand() const noexcept { return expand_; }

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  bool expand_{false};
  std::unique_ptr<OffsetUtil> outOffset_;
  std::unique_ptr<TensorImpl> outTen_;
  tensor_list inTens_;
};

using UnaryMetaHandler = TensorMetaHandler<const TensorImpl*>;
using BinaryMetaHandler =
    TensorMetaHandler<const TensorImpl*, const TensorImpl*>;
using ReductMetaHandler = TensorMetaHandler<const TensorImpl*, size_t>;

}  // namespace lmp::tensor::detail