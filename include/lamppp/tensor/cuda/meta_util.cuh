#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include <cuda/std/tuple>
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

template <typename U, typename V>
struct TransformFunctor {
  __device__ __host__ U operator()(void* p, std::size_t i) const {
    return static_cast<U>(static_cast<V*>(p)[i]);
  }
};

template <class OutT, class... SrcTs>
class PtrPack {
 public:
  static constexpr std::size_t N = sizeof...(SrcTs);

  ::cuda::std::array<void*, N + 1> data;
  ::cuda::std::tuple<TransformFunctor<OutT, OutT>,
                     TransformFunctor<OutT, SrcTs>...>
      fns;

  __device__ __host__ constexpr PtrPack(OutT* out, SrcTs*... in)
      : data{static_cast<void*>(out), static_cast<void*>(in)...},
        fns{TransformFunctor<OutT, OutT>{},
            TransformFunctor<OutT, SrcTs>{}...} {}

  __device__ void set_Out(size_t idx, OutT value) {
    static_cast<OutT*>(data[0])[idx] = value;
  }
};

class TensorMetaHandler {
 public:
  explicit TensorMetaHandler(tensor_list in);

  TensorImpl out() const;
  tensor_list in() const;
  const OffsetUtil<2>* offset() const;

  void handle_binary_op();
  void handle_expand_op();
  void handle_unary_op();

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  std::unique_ptr<OffsetUtil<2>> outOffset;
  std::unique_ptr<TensorImpl> outTen;
  tensor_list inTens;
};

}  // namespace lmp::tensor::detail::cuda::internal