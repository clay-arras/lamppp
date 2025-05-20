#pragma once

#include <cuda/std/array>
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/expand_ops.hpp"
#include "lamppp/tensor/functions/reduct_ops.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

using UnaryOpPtrList = ::cuda::std::array<void*, 2>;
using BinaryOpPtrList = ::cuda::std::array<void*, 3>;

template <typename T>
struct AddFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct SubFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 - arg2; }
};
template <typename T>
struct MulFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
template <typename T>
struct DivFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 / arg2; }
};
template <typename T>
struct EqFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 == arg2; }
};
template <typename T>
struct NeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 != arg2; }
};
template <typename T>
struct LeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 <= arg2; }
};
template <typename T>
struct LtFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 < arg2; }
};
template <typename T>
struct GtFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 > arg2; }
};
template <typename T>
struct GeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 >= arg2; }
};
template <typename T>
struct LogFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::log(static_cast<double>(arg)));
  }
};
template <typename T>
struct ExpFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::exp(static_cast<double>(arg)));
  }
};
template <typename T>
struct SqrtFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::sqrt(static_cast<double>(arg)));
  }
};
template <typename T>
struct AbsFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::abs(static_cast<double>(arg)));
  }
};
template <typename T>
struct SinFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::sin(static_cast<double>(arg)));
  }
};
template <typename T>
struct CosFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::cos(static_cast<double>(arg)));
  }
};
template <typename T>
struct TanFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(::tan(static_cast<double>(arg)));
  }
};
template <typename T>
struct ClampFunctor {
  explicit ClampFunctor(Scalar min_val, Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}
  __device__ __host__ T operator()(T arg) {
    return arg < min_val_ ? min_val_ : (arg > max_val_ ? max_val_ : arg);
  }

 private:
  Scalar min_val_, max_val_;
};
template <typename T>
struct SumFunctor {
  static constexpr T identity = 0;
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct MaxFunctor {
  static constexpr T identity = std::numeric_limits<T>::min();
  __device__ __host__ T operator()(T arg1, T arg2) { return max(arg1, arg2); }
};
template <typename T>
struct MinFunctor {
  static constexpr T identity = std::numeric_limits<T>::max();
  __device__ __host__ T operator()(T arg1, T arg2) { return min(arg1, arg2); }
};

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl eq_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl log_cuda(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);
TensorImpl sqrt_cuda(const TensorImpl& a);
TensorImpl abs_cuda(const TensorImpl& a);
TensorImpl sin_cuda(const TensorImpl& a);
TensorImpl cos_cuda(const TensorImpl& a);
TensorImpl tan_cuda(const TensorImpl& a);
TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val);

TensorImpl sum_cuda(const TensorImpl& a, size_t axis);
TensorImpl max_cuda(const TensorImpl& a, size_t axis);
TensorImpl min_cuda(const TensorImpl& a, size_t axis);

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CUDA, add_cuda);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CUDA, sub_cuda);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CUDA, mul_cuda);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CUDA, div_cuda);

LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CUDA, eq_cuda);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CUDA, ne_cuda);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CUDA, le_cuda);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CUDA, lt_cuda);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CUDA, ge_cuda);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CUDA, gt_cuda);

LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CUDA, exp_cuda);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CUDA, sqrt_cuda);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CUDA, abs_cuda);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CUDA, sin_cuda);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CUDA, cos_cuda);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CUDA, tan_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CUDA, clamp_cuda);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CUDA, sum_cuda);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CUDA, max_cuda);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CUDA, min_cuda);

}  // namespace lmp::tensor::detail::cuda