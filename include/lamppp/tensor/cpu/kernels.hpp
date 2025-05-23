#pragma once

#include <cmath>
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

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

TensorImpl log_cpu(const TensorImpl& a);
TensorImpl exp_cpu(const TensorImpl& a);
TensorImpl sqrt_cpu(const TensorImpl& a);
TensorImpl abs_cpu(const TensorImpl& a);
TensorImpl sin_cpu(const TensorImpl& a);
TensorImpl cos_cpu(const TensorImpl& a);
TensorImpl tan_cpu(const TensorImpl& a);
TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val);

}  // namespace lmp::tensor::detail::cpu