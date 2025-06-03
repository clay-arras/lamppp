#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

/// @internal
using neg_fn = TensorImpl (*)(const TensorImpl&);
using exp_fn = TensorImpl (*)(const TensorImpl&);
using log_fn = TensorImpl (*)(const TensorImpl&);
using sqrt_fn = TensorImpl (*)(const TensorImpl&);
using abs_fn = TensorImpl (*)(const TensorImpl&);
using sin_fn = TensorImpl (*)(const TensorImpl&);
using cos_fn = TensorImpl (*)(const TensorImpl&);
using tan_fn = TensorImpl (*)(const TensorImpl&);
using clamp_fn = TensorImpl (*)(const TensorImpl&, Scalar, Scalar);

LMP_DECLARE_DISPATCH(neg_fn, neg_stub);
LMP_DECLARE_DISPATCH(exp_fn, exp_stub);
LMP_DECLARE_DISPATCH(log_fn, log_stub);
LMP_DECLARE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DECLARE_DISPATCH(abs_fn, abs_stub);
LMP_DECLARE_DISPATCH(sin_fn, sin_stub);
LMP_DECLARE_DISPATCH(cos_fn, cos_stub);
LMP_DECLARE_DISPATCH(tan_fn, tan_stub);
LMP_DECLARE_DISPATCH(clamp_fn, clamp_stub);
/// @endinternal

/**
 * @brief Negate a tensor
 * @param a The tensor to negate
 * @return A new tensor with the result of the negation
 */
Tensor neg(const Tensor& a);

/**
 * @brief Exponentiate a tensor
 * @param a The tensor to exponentiate
 * @return A new tensor with the result of the exponentiation
 */
Tensor exp(const Tensor& a);

/**
 * @brief Logarithm of a tensor
 * @param a The tensor to take the logarithm of
 * @return A new tensor with the result of the logarithm
 */
Tensor log(const Tensor& a);

/**
 * @brief Square root of a tensor
 * @param a The tensor to take the square root of
 * @return A new tensor with the result of the square root
 */
Tensor sqrt(const Tensor& self);

/**
 * @brief Absolute value of a tensor
 * @param a The tensor to take the absolute value of
 * @return A new tensor with the result of the absolute value
 */
Tensor abs(const Tensor& self);

/**
 * @brief Sine of a tensor
 * @param a The tensor to take the sine of
 * @return A new tensor with the result of the sine
 */
Tensor sin(const Tensor& self);

/**
 * @brief Cosine of a tensor
 * @param a The tensor to take the cosine of
 * @return A new tensor with the result of the cosine
 */
Tensor cos(const Tensor& self);

/**
 * @brief Tangent of a tensor
 * @param a The tensor to take the tangent of
 * @return A new tensor with the result of the tangent
 */
Tensor tan(const Tensor& self);

/**
 * @brief Clamp a tensor
 * @param a The tensor to clamp
 * @param min_val The minimum value
 * @param max_val The maximum value
 * @return A new tensor with the result of the clamping
 */
Tensor clamp(const Tensor& self, Scalar min_val, Scalar max_val);

}  // namespace lmp::tensor::ops
