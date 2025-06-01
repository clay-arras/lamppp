#pragma once

#include <cuda/std/array>
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/matrix_ops.hpp"
#include "lamppp/tensor/functions/expand_ops.hpp"
#include "lamppp/tensor/functions/reduct_ops.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

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
struct PowFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return ::std::pow(arg1, arg2); }
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
struct NegFunctor {
  __device__ __host__ T operator()(T arg) { return (-arg); }
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
  static constexpr T identity = std::numeric_limits<T>::lowest();
  __device__ __host__ T operator()(T arg1, T arg2) { return max(arg1, arg2); }
};
template <typename T>
struct MinFunctor {
  static constexpr T identity = std::numeric_limits<T>::max();
  __device__ __host__ T operator()(T arg1, T arg2) { return min(arg1, arg2); }
};
template <typename T>
struct ProdFunctor {
  static constexpr T identity = 1;
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 * arg2; }
};

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl pow_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl eq_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl neg_cuda(const TensorImpl& a);
TensorImpl log_cuda(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);
TensorImpl sqrt_cuda(const TensorImpl& a);
TensorImpl abs_cuda(const TensorImpl& a);
TensorImpl sin_cuda(const TensorImpl& a);
TensorImpl cos_cuda(const TensorImpl& a);
TensorImpl tan_cuda(const TensorImpl& a);
TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val);

TensorImpl transpose_cuda(const TensorImpl& a);
TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl sum_cuda(const TensorImpl& a, size_t axis);
TensorImpl max_cuda(const TensorImpl& a, size_t axis);
TensorImpl min_cuda(const TensorImpl& a, size_t axis);
TensorImpl prod_cuda(const TensorImpl& a, size_t axis);

}  // namespace lmp::tensor::detail::cuda