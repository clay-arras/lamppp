#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cuda/std/detail/libcxx/include/array>
#include <cuda/std/tuple>

namespace lmp::tensor::detail::cuda::internal {

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

}  // namespace lmp::tensor::detail::cuda::internal