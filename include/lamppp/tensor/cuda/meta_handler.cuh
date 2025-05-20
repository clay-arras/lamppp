#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <memory>
#include <vector>
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda::internal {

class TensorMetaHandler {
 public:
  explicit TensorMetaHandler(tensor_list in);

  TensorImpl out() const noexcept;
  tensor_list in() const noexcept;
  const OffsetUtil<2>* offset() const noexcept;

  void handle_expand_op();
  void handle_unary_op();
  void handle_reduct_op(size_t axis);

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  std::unique_ptr<OffsetUtil<2>> outOffset;
  std::unique_ptr<TensorImpl> outTen;
  tensor_list inTens;
};

}  // namespace lmp::tensor::detail::cuda::internal