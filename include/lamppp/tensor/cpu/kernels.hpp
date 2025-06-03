#pragma once

#include <cmath>
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/native/unary_ops.hpp"
#include "lamppp/tensor/native/matrix_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename T>
struct AddFunctor {
  T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct SubFunctor {
  T operator()(T arg1, T arg2) { return arg1 - arg2; }
};
template <typename T>
struct MulFunctor {
  T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
template <typename T>
struct DivFunctor {
  T operator()(T arg1, T arg2) { return arg1 / arg2; }
};
template <typename T>
struct PowFunctor {
  T operator()(T arg1, T arg2) { return ::std::pow(arg1, arg2); }
};
template <typename T>
struct EqFunctor {
  T operator()(T arg1, T arg2) { return arg1 == arg2; }
};
template <typename T>
struct NeFunctor {
  T operator()(T arg1, T arg2) { return arg1 != arg2; }
};
template <typename T>
struct LeFunctor {
  T operator()(T arg1, T arg2) { return arg1 <= arg2; }
};
template <typename T>
struct LtFunctor {
  T operator()(T arg1, T arg2) { return arg1 < arg2; }
};
template <typename T>
struct GtFunctor {
  T operator()(T arg1, T arg2) { return arg1 > arg2; }
};
template <typename T>
struct GeFunctor {
  T operator()(T arg1, T arg2) { return arg1 >= arg2; }
};
template <typename T>
struct NegFunctor {
  T operator()(T arg) { return (-arg); }
};
template <typename T>
struct LogFunctor {
  T operator()(T arg) {
    return static_cast<T>(::log(static_cast<double>(arg)));
  }
};
template <typename T>
struct ExpFunctor {
  T operator()(T arg) {
    return static_cast<T>(::exp(static_cast<double>(arg)));
  }
};
template <typename T>
struct SqrtFunctor {
  T operator()(T arg) {
    return static_cast<T>(::sqrt(static_cast<double>(arg)));
  }
};
template <typename T>
struct AbsFunctor {
  T operator()(T arg) {
    return static_cast<T>(::std::abs(static_cast<double>(arg)));
  }
};
template <typename T>
struct SinFunctor {
  T operator()(T arg) {
    return static_cast<T>(::sin(static_cast<double>(arg)));
  }
};
template <typename T>
struct CosFunctor {
  T operator()(T arg) {
    return static_cast<T>(::cos(static_cast<double>(arg)));
  }
};
template <typename T>
struct TanFunctor {
  T operator()(T arg) {
    return static_cast<T>(::tan(static_cast<double>(arg)));
  }
};
template <typename T>
struct ClampFunctor {
  explicit ClampFunctor(Scalar min_val, Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}
  T operator()(T arg) {
    return arg < min_val_ ? min_val_ : (arg > max_val_ ? max_val_ : arg);
  }

 private:
  Scalar min_val_, max_val_;
};
template <typename T>
struct SumFunctor {
  static constexpr T identity = 0;
  T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct MaxFunctor {
  static constexpr T identity = std::numeric_limits<T>::lowest();
  T operator()(T arg1, T arg2) { return ::std::max(arg1, arg2); }
};
template <typename T>
struct MinFunctor {
  static constexpr T identity = std::numeric_limits<T>::max();
  T operator()(T arg1, T arg2) { return ::std::min(arg1, arg2); }
};
template <typename T>
struct ProdFunctor {
  static constexpr T identity = 1;
  T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
/// @endinternal

/// @internal
TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl pow_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b);

TensorImpl neg_cpu(const TensorImpl& a);
TensorImpl log_cpu(const TensorImpl& a);
TensorImpl exp_cpu(const TensorImpl& a);
TensorImpl sqrt_cpu(const TensorImpl& a);
TensorImpl abs_cpu(const TensorImpl& a);
TensorImpl sin_cpu(const TensorImpl& a);
TensorImpl cos_cpu(const TensorImpl& a);
TensorImpl tan_cpu(const TensorImpl& a);
TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val);

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl transpose_cpu(const TensorImpl& a);

TensorImpl sum_cpu(const TensorImpl& a, size_t axis);
TensorImpl max_cpu(const TensorImpl& a, size_t axis);
TensorImpl min_cpu(const TensorImpl& a, size_t axis);
TensorImpl prod_cpu(const TensorImpl& a, size_t axis);
/// @endinternal

}  // namespace lmp::tensor::detail::cpu