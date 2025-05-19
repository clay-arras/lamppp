#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include <memory>
#include <vector>
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda::internal {

template <size_t N>
::cuda::std::array<void*, N + 1> pack_tens(const tensor_list& tens, void* out) {
  ::cuda::std::array<void*, N + 1> arr;
  assert(tens.size() == N && "pack_tens: tensor list size mismatch with N");
  arr[0] = static_cast<void*>(out);
  for (size_t i = 0; i < N; i++) {
    arr[i + 1] = static_cast<void*>(tens[i].data());
  }
  return arr;
}

class TensorMetaHandler {
 public:
  explicit TensorMetaHandler(tensor_list in);

  void handle_binary_op();
  void handle_unary_op();
  TensorImpl out() const;
  tensor_list in() const;

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  std::unique_ptr<TensorImpl> outTen;
  tensor_list inTens;
};

}  // namespace lmp::tensor::detail::cuda::internal