#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

/// @internal
using add_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using sub_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using mul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using div_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using pow_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using eq_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ne_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ge_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using le_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using gt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using lt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(add_fn, add_stub);
LMP_DECLARE_DISPATCH(sub_fn, sub_stub);
LMP_DECLARE_DISPATCH(mul_fn, mul_stub);
LMP_DECLARE_DISPATCH(div_fn, div_stub);
LMP_DECLARE_DISPATCH(pow_fn, pow_stub);
LMP_DECLARE_DISPATCH(eq_fn, eq_stub);
LMP_DECLARE_DISPATCH(ne_fn, ne_stub);
LMP_DECLARE_DISPATCH(ge_fn, ge_stub);
LMP_DECLARE_DISPATCH(le_fn, le_stub);
LMP_DECLARE_DISPATCH(gt_fn, gt_stub);
LMP_DECLARE_DISPATCH(lt_fn, lt_stub);
/// @endinternal

/**
 * @brief Add two tensors
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the addition
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Subtract two tensors
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the subtraction
 */
Tensor sub(const Tensor& a, const Tensor& b);

/**
 * @brief Multiply two tensors
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the multiplication
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief Divide two tensors
 * @param a The first tensor
 * @param b The second tensor
 * @note if b is very small, this function will return have undefined behavior.
 * @return A new tensor with the result of the division
 */
Tensor div(const Tensor& a, const Tensor& b);

/**
 * @brief Raise a tensor to the power of another tensor
 * @param a The base tensor
 * @param b The exponent tensor
 * @return A new tensor with the result of the power operation
 */
Tensor pow(const Tensor& a, const Tensor& b);

/**
 * @brief Check if two tensors are equal
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the equality check
 */
Tensor eq(const Tensor& a, const Tensor& b);

/**
 * @brief Check if two tensors are not equal
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the inequality check
 */
Tensor ne(const Tensor& a, const Tensor& b);

/**
 * @brief Check if the first tensor is greater than or equal to the second tensor
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the inequality check
 */
Tensor ge(const Tensor& a, const Tensor& b);

/**
 * @brief Check if the first tensor is less than or equal to the second tensor
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the inequality check
 */
Tensor le(const Tensor& a, const Tensor& b);

/**
 * @brief Check if the first tensor is less than the second tensor
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the inequality check
 */
Tensor lt(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops