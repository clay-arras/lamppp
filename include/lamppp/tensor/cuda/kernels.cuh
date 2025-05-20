#pragma once

#include <cuda/std/array>
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/basic_ops.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

using UnaryOpPtrList = ::cuda::std::array<void*, 2>;
using BinaryOpPtrList = ::cuda::std::array<void*, 3>;

template <typename T>
struct AddFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 + arg2; }
};

template <typename OutType, typename InType>
struct LogFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::log(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct ExpFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::exp(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct SqrtFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::sqrt(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct AbsFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::abs(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct SinFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::sin(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct CosFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::cos(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct TanFunctor {
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::tan(static_cast<double>(in_data[index])));
  }
};

template <typename OutType, typename InType>
struct ClampFunctor {
  explicit ClampFunctor(Scalar min_val, Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}

  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    double val = static_cast<double>(in_data[index]);
    val = val < min_val_ ? min_val_ : (val > max_val_ ? max_val_ : val);
    out_data[index] = static_cast<OutType>(val);
  }

 private:
  Scalar min_val_, max_val_;
};

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl log_cuda(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);
TensorImpl sqrt_cuda(const TensorImpl& a);
TensorImpl abs_cuda(const TensorImpl& a);
TensorImpl sin_cuda(const TensorImpl& a);
TensorImpl cos_cuda(const TensorImpl& a);
TensorImpl tan_cuda(const TensorImpl& a);
TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val);

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CUDA, add_cuda);

LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CUDA, exp_cuda);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CUDA, sqrt_cuda);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CUDA, abs_cuda);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CUDA, sin_cuda);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CUDA, cos_cuda);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CUDA, tan_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CUDA, clamp_cuda);

}  // namespace lmp::tensor::detail::cuda